from __future__ import annotations

"""MCD43-ETI 不确定性、站点—源区迁移和灾后独立验证。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_DIR = Path(__file__).resolve().parent.parent
OUTPUTS = MODEL_DIR / "outputs"
PROCESSED = MODEL_DIR / "processed"
MODELING_ROOT = MODEL_DIR.parents[1]
Q1_CACHE = MODELING_ROOT / "第一问" / "05_重构方案_多源事件对照" / "cache"
CONFIG = json.loads((MODEL_DIR / "config" / "model_config.json").read_text(encoding="utf-8"))
EVENT = pd.Timestamp(CONFIG["event_last_complete_hour_beijing"])
RNG = np.random.default_rng(20261027)


def empirical_percentile(values: np.ndarray, value: float) -> float:
    values = values[np.isfinite(values)]
    return float((np.sum(values < value) + 0.5 * np.sum(values == value)) / len(values))


def rolling_sum_by_year(values: np.ndarray, years: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    for year in np.unique(years):
        idx = np.flatnonzero(years == year)
        x = values[idx]
        valid = np.isfinite(x).astype(int)
        filled = np.nan_to_num(x)
        sums = np.convolve(filled, np.ones(window), mode="valid")
        counts = np.convolve(valid, np.ones(window), mode="valid")
        local = np.full(len(x), np.nan)
        local[window - 1 :] = np.where(counts == window, sums, np.nan)
        out[idx] = local
    return out


def attach_mcd43(hourly: pd.DataFrame) -> pd.DataFrame:
    modis = pd.read_csv(OUTPUTS / "37_MODIS反照率与地表温度源区时序.csv", parse_dates=["date"])
    available = modis[["date", "albedo"]].dropna().sort_values("date")
    available["available_date"] = available["date"] + pd.Timedelta(days=1)
    days = pd.DataFrame({"day": pd.to_datetime(hourly.index.normalize().unique())})
    mapping = pd.merge_asof(
        days.sort_values("day"), available.sort_values("available_date"),
        left_on="day", right_on="available_date", direction="backward", tolerance=pd.Timedelta(days=7)
    ).set_index("day")
    out = hourly.copy()
    out["day"] = out.index.normalize()
    out = out.join(mapping[["date", "albedo"]].rename(columns={"date": "mcd43_date"}), on="day")
    out["albedo_mcd43"] = out["albedo"].where(out["albedo"].between(0.05, 0.95), out["albedo_operational"])
    return out


def mcd43_eti_monte_carlo() -> tuple[pd.DataFrame, pd.DataFrame]:
    hourly = attach_mcd43(pd.read_parquet(PROCESSED / "04_多源动态触发逐小时缓存.parquet").sort_index())
    eti = CONFIG["eti_model"]
    years = hourly.index.year.to_numpy()
    event_pos = int(np.flatnonzero(hourly.index == EVENT)[0])
    background_mask = (
        hourly.index.year.isin(CONFIG["background_years"])
        & (hourly.index.hour == EVENT.hour)
        & (np.abs(hourly.index.dayofyear - EVENT.dayofyear) <= 15)
    )
    background_pos = np.flatnonzero(background_mask)
    sw = (hourly["sw_down_center"].clip(lower=0) / 3600).to_numpy(float)
    albedo = hourly["albedo_mcd43"].to_numpy(float)
    temp_dynamic = hourly["source_temperature"].to_numpy(float)
    temp_fixed = hourly["source_temperature_fixed65"].to_numpy(float)
    rows = []
    for draw in range(int(eti["monte_carlo_draws"])):
        tf = RNG.uniform(*eti["temperature_factor_range"])
        srf = RNG.uniform(*eti["shortwave_factor_range"])
        t0 = RNG.uniform(*eti["temperature_threshold_range_c"])
        use_dynamic = bool(RNG.integers(0, 2))
        temp = temp_dynamic if use_dynamic else temp_fixed
        melt = np.where(temp > t0, tf * np.maximum(temp - t0, 0) + srf * (1 - albedo) * sw, 0)
        melt72 = rolling_sum_by_year(melt, years, 72)
        bg = melt72[background_pos]
        bg = bg[np.isfinite(bg)]
        event_value = float(melt72[event_pos])
        q90, q95 = np.quantile(bg, [0.9, 0.95])
        rows.append({
            "draw": draw + 1, "temperature_factor": tf, "shortwave_factor": srf,
            "temperature_threshold_c": t0,
            "temperature_method": "dynamic_lapse" if use_dynamic else "fixed_6p5K_per_km",
            "event_eti_72h_mm": event_value, "background_q90_eti_72h_mm": q90,
            "background_q95_eti_72h_mm": q95,
            "event_risk_percentile": empirical_percentile(bg, event_value),
            "exceeds_q90": event_value >= q90, "exceeds_q95": event_value >= q95,
        })
    draws = pd.DataFrame(rows)
    summary = []
    for col in ["event_eti_72h_mm", "background_q95_eti_72h_mm", "event_risk_percentile"]:
        summary.append({"metric": col, "median": draws[col].median(), "q025": draws[col].quantile(.025), "q975": draws[col].quantile(.975)})
    summary.extend([
        {"metric": "share_exceeds_q90", "median": draws.exceeds_q90.mean(), "q025": np.nan, "q975": np.nan},
        {"metric": "share_exceeds_q95", "median": draws.exceeds_q95.mean(), "q025": np.nan, "q975": np.nan},
    ])
    return draws, pd.DataFrame(summary)


def transfer_reliability() -> pd.DataFrame:
    source = pd.read_parquet(Q1_CACHE / "source_daily_features_official.parquet")
    station = pd.read_parquet(Q1_CACHE / "station_daily_10h_features_strict.parquet")
    source = source[source.date.dt.year.isin(CONFIG["background_years"])]
    station = station[station.Timestamp.dt.year.isin(CONFIG["background_years"])]
    merged = source.merge(station, left_on="date", right_on="Timestamp", how="inner")
    pairs = [
        ("daily_mean_temperature", "src_tmean_24h", "sta_tmean_24h", "temperature can be transferred after elevation correction"),
        ("daily_precipitation", "src_precip_sum_24h", "sta_precip_sum_24h", "weak correlation means station rain cannot replace source rain"),
        ("daily_shortwave", "src_sw_down_24h", "sta_swdown_24h", "radiation is regional support, not source observation"),
    ]
    rows = []
    for name, src, sta, interpretation in pairs:
        x = merged[[src, sta]].dropna()
        rows.append({
            "indicator": name, "n": len(x), "pearson_r": x[src].corr(x[sta], method="pearson"),
            "spearman_rho": x[src].corr(x[sta], method="spearman"),
            "mean_source_minus_station": (x[src] - x[sta]).mean(),
            "rmse_source_vs_station": float(np.sqrt(np.mean((x[src] - x[sta]) ** 2))),
            "model_use_boundary": interpretation,
        })
    return pd.DataFrame(rows)


def remote_sensing_validation() -> pd.DataFrame:
    s2 = pd.read_csv(OUTPUTS / "39_Sentinel2雪冰覆盖核查.csv")
    ls = pd.read_csv(OUTPUTS / "40_Landsat长期背景核查.csv")
    s1 = pd.read_csv(OUTPUTS / "41_Sentinel1后向散射核查.csv", parse_dates=["time_utc"])
    rows = []
    pre = s2[s2.phase == "2026_pre_late"].iloc[0]
    post = s2[s2.phase == "post_validation_only"].iloc[0]
    for metric in ["ndsi_mean", "snow_fraction_ndsi_ge_0p4", "ndvi_mean", "bare_soil_index_mean"]:
        rows.append({"sensor": "Sentinel-2", "metric": metric, "pre": pre[metric], "post": post[metric], "post_minus_pre": post[metric] - pre[metric], "usable": min(pre.valid_fraction, post.valid_fraction) >= 0.2, "role": "post-event validation only"})
    lpre = ls[ls.phase == "2026_pre"].iloc[0]
    lpost = ls[ls.phase == "post_validation_only"].iloc[0]
    for metric in ["ndsi_mean", "snow_fraction_ndsi_ge_0p4", "ndvi_mean", "bare_soil_index_mean"]:
        rows.append({"sensor": "Landsat", "metric": metric, "pre": lpre[metric], "post": lpost[metric], "post_minus_pre": lpost[metric] - lpre[metric], "usable": min(lpre.valid_fraction, lpost.valid_fraction) >= 0.2, "role": "discard if cloud-valid fraction is insufficient"})
    event_utc = EVENT.tz_localize("Asia/Shanghai").tz_convert("UTC")
    for track, group in s1.groupby("track"):
        pre_group = group[group.time_utc < event_utc]
        post_group = group[group.time_utc > event_utc]
        if pre_group.empty or post_group.empty:
            continue
        a = pre_group.sort_values("time_utc").iloc[-1]
        b = post_group.sort_values("time_utc").iloc[0]
        for metric in ["vv_db_median", "vh_db_median", "vv_minus_vh_db"]:
            rows.append({"sensor": f"Sentinel-1 {track}", "metric": metric, "pre": a[metric], "post": b[metric], "post_minus_pre": b[metric] - a[metric], "usable": True, "role": "same-track GRD post-event validation; not InSAR"})
    return pd.DataFrame(rows)


def md_table(frame: pd.DataFrame) -> str:
    x = frame.copy()
    for col in x.select_dtypes(include=["float"]).columns:
        x[col] = x[col].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    lines = ["| " + " | ".join(x.columns) + " |", "|" + "|".join(["---"] * len(x.columns)) + "|"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in x.itertuples(index=False, name=None))
    return "\n".join(lines)


def main() -> None:
    draws, mc = mcd43_eti_monte_carlo()
    transfer = transfer_reliability()
    validation = remote_sensing_validation()
    draws.to_parquet(PROCESSED / "12_MCD43_ETI参数蒙特卡洛.parquet", index=False)
    mc.to_csv(OUTPUTS / "47_MCD43_ETI蒙特卡洛摘要.csv", index=False, encoding="utf-8-sig")
    transfer.to_csv(OUTPUTS / "48_站点源区迁移可靠性.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(OUTPUTS / "49_灾后遥感独立验证.csv", index=False, encoding="utf-8-sig")

    eligible = pd.DataFrame([
        {"candidate": "2026-08-26", "status": "verified main event", "same_mechanism": True, "use": "main target; no fitting label"},
        {"candidate": "2025-07-08", "status": "verified flood/debris-flow event", "same_mechanism": False, "use": "excluded from background; qualitative external contrast only"},
        {"candidate": "other screenshot dates", "status": "not verified by the current official evidence package", "same_mechanism": False, "use": "not used as positive labels"},
    ])
    eligible.to_csv(OUTPUTS / "50_ROC_Youden候选事件可用性审计.csv", index=False, encoding="utf-8-sig")
    report = f"""# 第二问补充实验总结

## MCD43-ETI 参数不确定性

{md_table(mc)}

上表同时随机化温度因子、短波因子、融化阈温以及两种直减率方案。因此它比只报告一组 ETI 参数更适合进入论文。

## 站点—源区迁移可靠性

{md_table(transfer)}

这一检验定量解释了为什么站点资料可以提供区域热力背景，但不能把站点降水直接当成源区降水。

## 灾后遥感独立验证

{md_table(validation)}

所有 post 数据只检验“事件后表面是否改变”，不回填模型。Landsat 灾后有效像元率若不足 20%，则明确标记为不可用，不强行解释。

## ROC—Youden / Bayes Logistic 的使用决定

{md_table(eligible)}

当前仍不具备用于拟合 Bayes Logistic 的多个同机理、同空间、可对齐的正事件。因此不复制截图论文中的 ROC—Youden 结果，而保留无标签的历史分位、参数蒙特卡洛和空间分块交叉验证。
"""
    (OUTPUTS / "51_第二问补充实验总结.md").write_text(report, encoding="utf-8")
    print(mc.to_string(index=False))
    print(transfer.to_string(index=False))
    print(validation.groupby(["sensor", "usable"]).size().to_string())


if __name__ == "__main__":
    main()
