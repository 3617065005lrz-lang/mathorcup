from __future__ import annotations

"""Estimate temporal critical conditions without fabricating event labels.

The event is ranked against 2021--2024 values at the same hour and within a
plus/minus 15-day seasonal window.  Empirical quantiles are modelling alert
bands, not universal physical thresholds.  Bootstrap and ETI parameter Monte
Carlo quantify their sampling and parameter uncertainty.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent
CONFIG = json.loads((MODEL_DIR / "config" / "model_config.json").read_text(encoding="utf-8"))
PROCESSED = MODEL_DIR / "processed"
OUTPUTS = MODEL_DIR / "outputs"
DYNAMIC_PATH = PROCESSED / "04_多源动态触发逐小时缓存.parquet"

EVENT_TIME = pd.Timestamp(CONFIG["event_last_complete_hour_beijing"])
BACKGROUND_YEARS = CONFIG["background_years"]
RNG = np.random.default_rng(20261026)

FEATURES: dict[str, tuple[str, str, str]] = {
    "eti_melt_72h_mm": ("72小时ETI条件融水", "mm w.e.", "high"),
    "positive_degree_hours_72h": ("72小时正积温", "°C·h", "high"),
    "absorbed_shortwave_72h_mj_m2": ("72小时吸收短波", "MJ/m²", "high"),
    "freezing_level_mean_72h_m": ("72小时平均零度层高度", "m", "high"),
    "albedo_operational": ("事件前可用反照率", "1", "low"),
    "era5_rain_72h_mm": ("ERA5-Land 72小时降水", "mm", "high"),
    "imerg_rain_72h_mm": ("IMERG 72小时降水", "mm", "high"),
    "chirps_rain_prev_15d_mm": ("CHIRPS前15完整日降水", "mm", "high"),
    "chirps_rain_prev_30d_mm": ("CHIRPS前30完整日降水", "mm", "high"),
    "soil_water_l1_mean_72h": ("ERA5-Land 72小时表层土壤水分", "m³/m³", "high"),
    "smap_rootzone_mean_72h": ("SMAP 72小时根区土壤水分", "m³/m³", "high"),
}


def seasonal_background(frame: pd.DataFrame) -> pd.DataFrame:
    event_doy = EVENT_TIME.dayofyear
    doy = frame.index.dayofyear
    return frame.loc[
        frame.index.year.isin(BACKGROUND_YEARS)
        & (frame.index.hour == EVENT_TIME.hour)
        & (np.abs(doy - event_doy) <= 15)
    ].copy()


def empirical_percentile(background: np.ndarray, event: float, direction: str) -> tuple[float, float]:
    values = background[np.isfinite(background)]
    if not np.isfinite(event) or values.size == 0:
        return np.nan, np.nan
    if direction == "high":
        percentile = (np.count_nonzero(values < event) + 0.5 * np.count_nonzero(values == event)) / values.size
        p_value = (np.count_nonzero(values >= event) + 1) / (values.size + 1)
    else:
        percentile = (np.count_nonzero(values > event) + 0.5 * np.count_nonzero(values == event)) / values.size
        p_value = (np.count_nonzero(values <= event) + 1) / (values.size + 1)
    return float(percentile), float(p_value)


def bootstrap_summary(values: np.ndarray, event: float, direction: str, draws: int = 2000) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if values.size == 0 or not np.isfinite(event):
        return {key: np.nan for key in ["q90", "q95", "q90_lo", "q90_hi", "q95_lo", "q95_hi", "pct_lo", "pct_hi"]}
    samples = RNG.choice(values, size=(draws, values.size), replace=True)
    if direction == "high":
        q90_draw = np.quantile(samples, 0.90, axis=1)
        q95_draw = np.quantile(samples, 0.95, axis=1)
        pct_draw = np.mean(samples < event, axis=1)
        q90, q95 = np.quantile(values, [0.90, 0.95])
    else:
        q90_draw = np.quantile(samples, 0.10, axis=1)
        q95_draw = np.quantile(samples, 0.05, axis=1)
        pct_draw = np.mean(samples > event, axis=1)
        q90, q95 = np.quantile(values, [0.10, 0.05])
    return {
        "q90": float(q90),
        "q95": float(q95),
        "q90_lo": float(np.quantile(q90_draw, 0.025)),
        "q90_hi": float(np.quantile(q90_draw, 0.975)),
        "q95_lo": float(np.quantile(q95_draw, 0.025)),
        "q95_hi": float(np.quantile(q95_draw, 0.975)),
        "pct_lo": float(np.quantile(pct_draw, 0.025)),
        "pct_hi": float(np.quantile(pct_draw, 0.975)),
    }


def threshold_table(frame: pd.DataFrame) -> pd.DataFrame:
    background = seasonal_background(frame)
    event = frame.loc[EVENT_TIME]
    rows = []
    for feature, (name_cn, unit, direction) in FEATURES.items():
        values = background[feature].to_numpy(dtype=float)
        event_value = float(event[feature])
        percentile, p_value = empirical_percentile(values, event_value, direction)
        boot = bootstrap_summary(values, event_value, direction)
        finite_values = values[np.isfinite(values)]
        row = {
            "feature": feature,
            "name_cn": name_cn,
            "unit": unit,
            "risk_direction": direction,
            "background_n": int(np.isfinite(values).sum()),
            "event_value": event_value,
            "background_median": float(np.median(finite_values)) if finite_values.size else np.nan,
            "risk_percentile": percentile,
            "empirical_one_sided_p": p_value,
            **boot,
        }
        for year in BACKGROUND_YEARS:
            yearly = background.loc[background.index.year == year, feature].to_numpy(dtype=float)
            row[f"percentile_vs_{year}"] = empirical_percentile(yearly, event_value, direction)[0]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("risk_percentile", ascending=False)


def rolling_sum_by_year(values: np.ndarray, years: np.ndarray, window: int) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    kernel = np.ones(window, dtype=float)
    for year in np.unique(years):
        idx = np.flatnonzero(years == year)
        data = values[idx]
        valid = np.isfinite(data).astype(float)
        filled = np.nan_to_num(data, nan=0.0)
        sums = np.convolve(filled, kernel, mode="valid")
        counts = np.convolve(valid, kernel, mode="valid")
        local = np.full(data.shape, np.nan)
        local[window - 1 :] = np.where(counts == window, sums, np.nan)
        result[idx] = local
    return result


def eti_monte_carlo(frame: pd.DataFrame) -> pd.DataFrame:
    eti = CONFIG["eti_model"]
    draws = int(eti["monte_carlo_draws"])
    years = frame.index.year.to_numpy()
    event_pos = int(np.flatnonzero(frame.index == EVENT_TIME)[0])
    seasonal_mask = seasonal_background(frame).index
    background_positions = frame.index.get_indexer(seasonal_mask)
    sw = frame["shortwave_w_m2"].to_numpy(dtype=float)
    albedo = frame["albedo_operational"].to_numpy(dtype=float)
    temp_dynamic = frame["source_temperature"].to_numpy(dtype=float)
    temp_fixed = frame["source_temperature_fixed65"].to_numpy(dtype=float)

    rows = []
    for draw in range(draws):
        tf = RNG.uniform(*eti["temperature_factor_range"])
        srf = RNG.uniform(*eti["shortwave_factor_range"])
        t0 = RNG.uniform(*eti["temperature_threshold_range_c"])
        use_dynamic = bool(RNG.integers(0, 2))
        temp = temp_dynamic if use_dynamic else temp_fixed
        melt = np.where(temp > t0, tf * np.maximum(temp - t0, 0) + srf * (1 - albedo) * sw, 0.0)
        melt72 = rolling_sum_by_year(melt, years, 72)
        bg = melt72[background_positions]
        bg = bg[np.isfinite(bg)]
        event_value = float(melt72[event_pos])
        percentile, _ = empirical_percentile(bg, event_value, "high")
        rows.append(
            {
                "draw": draw + 1,
                "temperature_factor": tf,
                "shortwave_factor": srf,
                "temperature_threshold_c": t0,
                "temperature_method": "dynamic_lapse" if use_dynamic else "fixed_6p5K_per_km",
                "event_eti_72h_mm": event_value,
                "background_q90_eti_72h_mm": float(np.quantile(bg, 0.90)),
                "background_q95_eti_72h_mm": float(np.quantile(bg, 0.95)),
                "event_risk_percentile": percentile,
                "exceeds_q90": event_value >= np.quantile(bg, 0.90),
                "exceeds_q95": event_value >= np.quantile(bg, 0.95),
            }
        )
    return pd.DataFrame(rows)


def branch_scores(table: pd.DataFrame) -> pd.DataFrame:
    rank = table.set_index("feature")["risk_percentile"]
    thermal = float(rank["eti_melt_72h_mm"])
    rainfall = float(np.nanmedian([rank["era5_rain_72h_mm"], rank["imerg_rain_72h_mm"]]))
    antecedent = float(np.nanmedian([rank["chirps_rain_prev_15d_mm"], rank["smap_rootzone_mean_72h"]]))
    # Geometric mean encodes that a rainfall/moisture branch needs both input and a wet background.
    rain_moisture = float(np.sqrt(max(rainfall, 0) * max(antecedent, 0)))
    # Alternative trigger mechanisms are combined by a probabilistic OR; no fitted weight is introduced.
    dynamic_union = float(1 - (1 - thermal) * (1 - rain_moisture))
    return pd.DataFrame(
        [
            {"branch": "thermal_cryosphere", "score_0_1": thermal, "definition": "ETI 72h historical percentile"},
            {"branch": "short_rainfall", "score_0_1": rainfall, "definition": "median of ERA5-Land/IMERG 72h rainfall percentiles"},
            {"branch": "antecedent_moisture", "score_0_1": antecedent, "definition": "median of CHIRPS P15/SMAP root-zone percentiles"},
            {"branch": "rain_moisture", "score_0_1": rain_moisture, "definition": "geometric mean(short rainfall, antecedent moisture)"},
            {"branch": "dynamic_trigger_union", "score_0_1": dynamic_union, "definition": "1-(1-thermal)*(1-rain_moisture)"},
        ]
    )


def main() -> None:
    frame = pd.read_parquet(DYNAMIC_PATH).sort_index()
    if 2025 in set(frame.index.year):
        raise RuntimeError("2025 must not enter the formal temporal analysis")
    table = threshold_table(frame)
    table.to_csv(OUTPUTS / "12_时间触发临界条件与稳健性.csv", index=False, encoding="utf-8-sig")

    mc = eti_monte_carlo(frame)
    mc.to_parquet(PROCESSED / "05_ETI参数蒙特卡洛结果.parquet", index=False)
    summary_rows = []
    for column in ["event_eti_72h_mm", "background_q95_eti_72h_mm", "event_risk_percentile"]:
        values = mc[column]
        summary_rows.append(
            {
                "metric": column,
                "median": values.median(),
                "q025": values.quantile(0.025),
                "q975": values.quantile(0.975),
            }
        )
    summary_rows.extend(
        [
            {"metric": "share_exceeds_q90", "median": mc["exceeds_q90"].mean(), "q025": np.nan, "q975": np.nan},
            {"metric": "share_exceeds_q95", "median": mc["exceeds_q95"].mean(), "q025": np.nan, "q975": np.nan},
        ]
    )
    mc_summary = pd.DataFrame(summary_rows)
    mc_summary.to_csv(OUTPUTS / "13_ETI蒙特卡洛摘要.csv", encoding="utf-8-sig")

    branches = branch_scores(table)
    branches.to_csv(OUTPUTS / "14_动态触发分支评分.csv", index=False, encoding="utf-8-sig")

    eti_row = table.loc[table["feature"].eq("eti_melt_72h_mm")].iloc[0]
    report = f"""# 时间触发临界条件与不确定性结果

## 口径

- 主事件：{EVENT_TIME}（事件前最后完整小时）。
- 历史背景：2021—2024 年；2025 年未进入背景、阈值或参数选择。
- 同季节比较：历史各年与事件日期相差不超过 15 天、且同为北京时间 {EVENT_TIME.hour}:00 的记录。
- 90%/95% 分位数是经验关注/高危带，不是题目给出的法定阈值，也不宣称为普适物理临界值。
- 单一事件不能支持普通 Logistic 回归或 ROC/AUC；这里报告经验秩、单侧秩概率、bootstrap 区间和参数 Monte Carlo。

## ETI 72 小时结果

- 中心参数事件值：{eti_row['event_value']:.2f} mm w.e.。
- 历史风险百分位：{eti_row['risk_percentile']:.3f}。
- 历史 90% / 95% 经验阈值：{eti_row['q90']:.2f} / {eti_row['q95']:.2f} mm w.e.。
- Monte Carlo 采用 {len(mc)} 组文献约束参数和两种气温订正方法；完整抽样结果保存在 processed 目录。

## 动态分支

{branches.to_string(index=False)}

注意：ETI 数值是“文献参数条件下的等效融水指标”，本研究重点解释其相对异常与稳定性，而不把它伪装成源区实测融水量。
"""
    (OUTPUTS / "15_时间触发结果说明.md").write_text(report, encoding="utf-8")
    print(table[["name_cn", "event_value", "q90", "q95", "risk_percentile", "empirical_one_sided_p"]].to_string(index=False))
    print("\n", branches.to_string(index=False))
    print("\nMonte Carlo:")
    print(mc[["event_eti_72h_mm", "background_q95_eti_72h_mm", "event_risk_percentile", "exceeds_q90", "exceeds_q95"]].describe().to_string())


if __name__ == "__main__":
    main()
