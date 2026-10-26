from __future__ import annotations

"""Integrate static susceptibility and dynamic trigger into one risk rule.

The upgraded model uses a mechanism OR between thermal and rain-moisture
triggers, followed by a geometric coupling with spatial susceptibility.  Risk
classes are derived from the same continuous score's 2021--2024 reference
distribution, so the map and the numeric index cannot contradict each other.
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
SPATIAL_PATH = PROCESSED / "07_空间易发性评分格网.parquet"
MC_PATH = PROCESSED / "05_ETI参数蒙特卡洛结果.parquet"
EVENT_TIME = pd.Timestamp(CONFIG["event_last_complete_hour_beijing"])
BACKGROUND_YEARS = CONFIG["background_years"]


def seasonal_background(frame: pd.DataFrame, years: list[int] | None = None) -> pd.DataFrame:
    years = BACKGROUND_YEARS if years is None else years
    event_doy = EVENT_TIME.dayofyear
    return frame.loc[
        frame.index.year.isin(years)
        & (frame.index.hour == EVENT_TIME.hour)
        & (np.abs(frame.index.dayofyear - event_doy) <= 15)
    ].copy()


def cdf_values(reference: pd.Series, values: pd.Series | float) -> np.ndarray | float:
    ref = np.sort(reference.dropna().to_numpy(dtype=float))
    if np.isscalar(values):
        return float(np.searchsorted(ref, float(values), side="right") / len(ref)) if len(ref) else np.nan
    arr = pd.Series(values).to_numpy(dtype=float)
    result = np.full(arr.shape, np.nan, dtype=float)
    valid = np.isfinite(arr)
    result[valid] = np.searchsorted(ref, arr[valid], side="right") / len(ref)
    return result


def dynamic_scores(frame: pd.DataFrame, years: list[int] | None = None) -> tuple[pd.DataFrame, dict[str, float]]:
    bg = seasonal_background(frame, years)
    event = frame.loc[EVENT_TIME]
    components = {
        "thermal": "eti_melt_72h_mm",
        "era_rain": "era5_rain_72h_mm",
        "imerg_rain": "imerg_rain_72h_mm",
        "antecedent_rain": "chirps_rain_prev_15d_mm",
        "rootzone_moisture": "smap_rootzone_mean_72h",
    }
    ranked = pd.DataFrame(index=bg.index)
    event_rank: dict[str, float] = {}
    for key, feature in components.items():
        ranked[key] = cdf_values(bg[feature], bg[feature])
        event_rank[key] = cdf_values(bg[feature], float(event[feature]))

    ranked["short_rainfall"] = ranked[["era_rain", "imerg_rain"]].median(axis=1, skipna=True)
    ranked["antecedent_moisture"] = ranked[["antecedent_rain", "rootzone_moisture"]].median(axis=1, skipna=True)
    ranked["rain_moisture"] = np.sqrt(ranked["short_rainfall"] * ranked["antecedent_moisture"])
    ranked["dynamic_upgraded"] = 1 - (1 - ranked["thermal"]) * (1 - ranked["rain_moisture"])
    ranked["dynamic_baseline"] = ranked[["thermal", "short_rainfall", "antecedent_moisture"]].mean(axis=1)

    event_rank["short_rainfall"] = float(np.nanmedian([event_rank["era_rain"], event_rank["imerg_rain"]]))
    event_rank["antecedent_moisture"] = float(
        np.nanmedian([event_rank["antecedent_rain"], event_rank["rootzone_moisture"]])
    )
    event_rank["rain_moisture"] = float(
        np.sqrt(event_rank["short_rainfall"] * event_rank["antecedent_moisture"])
    )
    event_rank["dynamic_upgraded"] = float(
        1 - (1 - event_rank["thermal"]) * (1 - event_rank["rain_moisture"])
    )
    event_rank["dynamic_baseline"] = float(
        np.mean([event_rank["thermal"], event_rank["short_rainfall"], event_rank["antecedent_moisture"]])
    )
    return ranked, event_rank


def reference_thresholds(static_score: np.ndarray, dynamic_score: np.ndarray, model: str) -> dict[str, float]:
    values = []
    for dynamic in dynamic_score[np.isfinite(dynamic_score)]:
        if model == "upgraded":
            values.append(np.sqrt(static_score * dynamic))
        else:
            values.append((static_score + dynamic) / 2.0)
    reference = np.concatenate(values)
    return {f"q{int(q * 100)}": float(np.quantile(reference, q)) for q in [0.50, 0.80, 0.90, 0.95]}


def classify(values: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    return np.select(
        [values >= thresholds["q95"], values >= thresholds["q80"], values >= thresholds["q50"]],
        ["极高", "高", "中"],
        default="低",
    )


def reliability_table(
    frame: pd.DataFrame,
    spatial: pd.DataFrame,
    upgraded_thresholds: dict[str, float],
    event_dynamic: float,
) -> pd.DataFrame:
    source = spatial["event_source_aoi_label"].astype(bool).to_numpy()
    s = spatial["static_mechanism_percentile"].to_numpy()
    risk = np.sqrt(s * event_dynamic)
    rows = []
    for quantile in [0.80, 0.90, 0.95]:
        threshold = upgraded_thresholds[f"q{int(quantile * 100)}"]
        high = risk >= threshold
        rows.append(
            {
                "test": "risk_threshold_sensitivity",
                "setting": f"q{int(quantile * 100)}",
                "dynamic_score": event_dynamic,
                "risk_threshold": threshold,
                "source_aoi_coverage": float((high & source).sum() / source.sum()),
                "high_zone_in_source_aoi": float((high & source).sum() / high.sum()) if high.sum() else np.nan,
            }
        )
    for omitted_year in BACKGROUND_YEARS:
        years = [year for year in BACKGROUND_YEARS if year != omitted_year]
        ranked, event = dynamic_scores(frame, years)
        thresholds = reference_thresholds(s, ranked["dynamic_upgraded"].to_numpy(), "upgraded")
        loo_risk = np.sqrt(s * event["dynamic_upgraded"])
        high = loo_risk >= thresholds["q80"]
        rows.append(
            {
                "test": "leave_one_year_out",
                "setting": f"omit_{omitted_year}",
                "dynamic_score": event["dynamic_upgraded"],
                "risk_threshold": thresholds["q80"],
                "source_aoi_coverage": float((high & source).sum() / source.sum()),
                "high_zone_in_source_aoi": float((high & source).sum() / high.sum()) if high.sum() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    dynamic = pd.read_parquet(DYNAMIC_PATH).sort_index()
    spatial = pd.read_parquet(SPATIAL_PATH).copy()
    if 2025 in set(dynamic.index.year):
        raise RuntimeError("2025 leaked into formal model")

    background_dynamic, event = dynamic_scores(dynamic)
    spatial["static_mechanism_percentile"] = spatial["static_mechanism_score"].rank(pct=True)
    spatial["static_baseline_percentile"] = spatial["static_equal_indicator_baseline"].rank(pct=True)
    s_up = spatial["static_mechanism_percentile"].to_numpy(dtype=float)
    s_base = spatial["static_baseline_percentile"].to_numpy(dtype=float)

    upgraded_thresholds = reference_thresholds(
        s_up, background_dynamic["dynamic_upgraded"].to_numpy(dtype=float), "upgraded"
    )
    baseline_thresholds = reference_thresholds(
        s_base, background_dynamic["dynamic_baseline"].to_numpy(dtype=float), "baseline"
    )
    spatial["event_risk_upgraded"] = np.sqrt(s_up * event["dynamic_upgraded"])
    spatial["event_risk_baseline"] = (s_base + event["dynamic_baseline"]) / 2.0
    spatial["event_risk_level_upgraded"] = classify(spatial["event_risk_upgraded"].to_numpy(), upgraded_thresholds)
    spatial["event_risk_level_baseline"] = classify(spatial["event_risk_baseline"].to_numpy(), baseline_thresholds)
    spatial.to_parquet(PROCESSED / "08_事件时空综合风险格网.parquet", index=False)

    background_dynamic.to_parquet(PROCESSED / "09_历史同期动态评分.parquet")
    component_table = pd.DataFrame(
        [{"component": key, "event_score_0_1": value} for key, value in event.items()]
    )
    component_table.to_csv(OUTPUTS / "21_事件动态分支评分.csv", index=False, encoding="utf-8-sig")

    threshold_rows = []
    for model, values in [("baseline", baseline_thresholds), ("upgraded", upgraded_thresholds)]:
        for name, value in values.items():
            threshold_rows.append({"model": model, "reference_quantile": name, "risk_threshold": value})
    pd.DataFrame(threshold_rows).to_csv(OUTPUTS / "22_统一风险分级阈值.csv", index=False, encoding="utf-8-sig")

    level_summary = (
        spatial.groupby(["event_risk_level_upgraded", "event_source_aoi_label"], as_index=False)
        .size()
        .rename(columns={"size": "pixels"})
    )
    level_summary.to_csv(OUTPUTS / "23_事件风险等级空间统计.csv", index=False, encoding="utf-8-sig")

    reliability = reliability_table(dynamic, spatial, upgraded_thresholds, event["dynamic_upgraded"])
    reliability.to_csv(OUTPUTS / "24_时空模型稳健性检验.csv", index=False, encoding="utf-8-sig")

    mc = pd.read_parquet(MC_PATH)
    rain_moisture = event["rain_moisture"]
    mc["dynamic_upgraded"] = 1 - (1 - mc["event_risk_percentile"]) * (1 - rain_moisture)
    source_median_s = float(spatial.loc[spatial["event_source_aoi_label"].eq(1), "static_mechanism_percentile"].median())
    mc["source_median_joint_risk"] = np.sqrt(source_median_s * mc["dynamic_upgraded"])
    mc["source_median_risk_level"] = classify(mc["source_median_joint_risk"].to_numpy(), upgraded_thresholds)
    mc_levels = mc["source_median_risk_level"].value_counts(normalize=True).rename_axis("risk_level").reset_index(name="share")
    mc_levels.to_csv(OUTPUTS / "25_综合风险蒙特卡洛等级频率.csv", index=False, encoding="utf-8-sig")

    event_dynamic_percentile = float(
        np.mean(background_dynamic["dynamic_upgraded"].to_numpy() < event["dynamic_upgraded"])
    )
    source = spatial["event_source_aoi_label"].eq(1)
    source_levels = spatial.loc[source, "event_risk_level_upgraded"].value_counts(normalize=True)
    report = f"""# 第二问统一时空风险模型结果

## 唯一判断规则

1. 热力冰冻圈分支：72 小时 ETI 相对 2021—2024 同期背景的经验百分位 H。
2. 降雨—水分分支：短时降水百分位与前期湿润度百分位的几何平均 W。
3. 动态触发：D = 1 - (1-H)(1-W)，表示两类替代触发机制至少一类增强。
4. 空间易发性：S 为地形—物源—沟道连通性机制分数的空间百分位。
5. 综合风险：R = sqrt(SD)。风险等级直接由 R 的 2021—2024 时空参考分布分位数划分。

因此二维解释与最终标签使用同一个 R，不存在一套矩阵和另一套指数各自判定的问题。

## 2026 事件结果

- 热力分支 H = {event['thermal']:.3f}。
- 降雨—水分分支 W = {event['rain_moisture']:.3f}。
- 动态触发 D = {event['dynamic_upgraded']:.3f}，处于历史同期动态评分的第 {event_dynamic_percentile:.3f} 分位。
- 升级模型风险分级阈值：中/高 {upgraded_thresholds['q50']:.3f}/{upgraded_thresholds['q80']:.3f}，极高 {upgraded_thresholds['q95']:.3f}。
- 源区 AOI 内各等级比例：{source_levels.to_dict()}。
- ETI 参数 Monte Carlo 下，源区中位空间易发性位置的等级频率：{dict(zip(mc_levels['risk_level'], mc_levels['share']))}。

## 解释边界

- 这是单事件条件下的机制评价模型，不是由大量事件训练出的发生概率模型。
- 等级阈值是历史背景的经验分位阈值，不是题目给定阈值或通用工程规范。
- 历史灾点模型只作辅助；跨境源区主结论来自可解释的地形、冰川状态和沟道连通性。
"""
    (OUTPUTS / "26_第二问统一模型结果说明.md").write_text(report, encoding="utf-8")
    print(component_table.to_string(index=False))
    print("\nThresholds", upgraded_thresholds)
    print("\nSource AOI levels", source_levels.to_dict())
    print("\n", reliability.to_string(index=False))
    print("\nMonte Carlo levels", mc_levels.to_string(index=False))


if __name__ == "__main__":
    main()
