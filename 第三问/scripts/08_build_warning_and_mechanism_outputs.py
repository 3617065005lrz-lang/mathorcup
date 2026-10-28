from __future__ import annotations

"""Add a single-rule warning ladder and explicit dual-mechanism diagnostics."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_DIR = Path(__file__).resolve().parents[1]
SCRIPTS = MODEL_DIR / "scripts"
PROCESSED = MODEL_DIR / "processed"
OUTPUTS = MODEL_DIR / "outputs"
PERSISTENCE_H = 12
LEVEL_NAMES = {0: "常态", 1: "蓝色", 2: "黄色", 3: "橙色", 4: "红色"}


def load_trigger_module():
    path = SCRIPTS / "04_run_trigger_forecast.py"
    spec = importlib.util.spec_from_file_location("q3_trigger_model_for_warning", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def warning_thresholds(module) -> tuple[pd.DataFrame, dict[int, float]]:
    history = module.add_historical_features(
        module.attach_mcd43(pd.read_parquet(module.Q2_DYNAMIC).sort_index())
    )
    bg_scores, _, _, _, _ = module.build_references(history)
    spatial = pd.read_parquet(module.SPATIAL_GRID)
    static = spatial["static_mechanism_score"].rank(pct=True).to_numpy(float)
    joint_reference = np.sqrt(static[:, None] * bg_scores.dynamic.to_numpy(float)[None, :])
    quantiles = {1: 0.80, 2: 0.90, 3: 0.95, 4: 0.99}
    values = {level: float(np.nanquantile(joint_reference, q)) for level, q in quantiles.items()}
    rows = [
        {
            "warning_level_code": 0,
            "warning_level": LEVEL_NAMES[0],
            "reference_quantile_lower": 0.0,
            "risk_threshold_lower": 0.0,
            "definition": "联合风险低于历史时空参考q80",
        }
    ]
    for level, q in quantiles.items():
        rows.append(
            {
                "warning_level_code": level,
                "warning_level": LEVEL_NAMES[level],
                "reference_quantile_lower": q,
                "risk_threshold_lower": values[level],
                "definition": f"联合风险中位数连续{PERSISTENCE_H} h不低于历史时空参考q{int(q*100)}",
            }
        )
    return pd.DataFrame(rows), values


def classify(values: pd.Series, thresholds: dict[int, float]) -> np.ndarray:
    x = values.to_numpy(float)
    level = np.zeros(len(x), dtype=int)
    for code in [1, 2, 3, 4]:
        level[x >= thresholds[code]] = code
    return level


def first_persistent_time(group: pd.DataFrame, level: int) -> pd.Timestamp | pd.NaT:
    mask = group.warning_level_code.to_numpy(int) >= level
    if len(mask) < PERSISTENCE_H:
        return pd.NaT
    run = np.convolve(mask.astype(int), np.ones(PERSISTENCE_H, dtype=int), mode="valid")
    hit = np.flatnonzero(run == PERSISTENCE_H)
    return pd.Timestamp(group.valid_time_beijing.iloc[int(hit[0])]) if len(hit) else pd.NaT


def persistent_max_level(group: pd.DataFrame) -> int:
    out = 0
    for level in [1, 2, 3, 4]:
        if pd.notna(first_persistent_time(group, level)):
            out = level
    return out


def nearest_row(group: pd.DataFrame, target: pd.Timestamp) -> pd.Series:
    pos = np.abs(group.valid_time_beijing - target).argmin()
    return group.iloc[int(pos)]


def main() -> None:
    score_path = PROCESSED / "04_逐时风险情景样本.parquet"
    summary_path = OUTPUTS / "04_触发时间区间与覆盖.csv"
    if not score_path.exists() or not summary_path.exists():
        raise FileNotFoundError("Run 04_run_trigger_forecast.py first")

    module = load_trigger_module()
    threshold_table, thresholds = warning_thresholds(module)
    threshold_table.to_csv(OUTPUTS / "16_统一预警等级阈值.csv", index=False, encoding="utf-8-sig")

    scores = pd.read_parquet(score_path)
    scores = scores.loc[scores.stochastic].copy()
    agg = (
        scores.groupby(["nominal_issue_h", "issue_time_beijing", "valid_time_beijing"])
        .agg(
            thermal_q025=("thermal_score", lambda x: x.quantile(0.025)),
            thermal_median=("thermal_score", "median"),
            thermal_q975=("thermal_score", lambda x: x.quantile(0.975)),
            rain_water_q025=("rain_moisture_score", lambda x: x.quantile(0.025)),
            rain_water_median=("rain_moisture_score", "median"),
            rain_water_q975=("rain_moisture_score", lambda x: x.quantile(0.975)),
            joint_q025=("source_joint_risk", lambda x: x.quantile(0.025)),
            joint_median=("source_joint_risk", "median"),
            joint_q975=("source_joint_risk", lambda x: x.quantile(0.975)),
        )
        .reset_index()
    )
    agg["warning_level_code"] = classify(agg.joint_median, thresholds)
    agg["warning_level"] = agg.warning_level_code.map(LEVEL_NAMES)
    agg["dominant_mechanism"] = np.where(
        agg.thermal_median >= agg.rain_water_median,
        "热力—冰冻圈",
        "降雨—水分",
    )
    agg["mechanism_margin"] = agg.thermal_median - agg.rain_water_median
    agg.to_csv(OUTPUTS / "17_逐时预警等级与双机制分支.csv", index=False, encoding="utf-8-sig")

    trigger_summary = pd.read_csv(summary_path, parse_dates=["trigger_median"])
    trigger_summary = trigger_summary.loc[trigger_summary.model.eq("main") & trigger_summary.stochastic]
    trigger_by_issue = trigger_summary.set_index("nominal_issue_h")
    warning_rows = []
    mechanism_rows = []
    for (nominal, issue), group in agg.groupby(["nominal_issue_h", "issue_time_beijing"]):
        group = group.sort_values("valid_time_beijing").reset_index(drop=True)
        issue = pd.Timestamp(issue)
        peak_time = pd.Timestamp(trigger_by_issue.loc[nominal, "trigger_median"])
        at_event = nearest_row(group, module.EVENT_TRIGGER_BJ)
        at_peak = nearest_row(group, peak_time)
        onsets = {level: first_persistent_time(group, level) for level in [1, 2, 3, 4]}
        max_level = persistent_max_level(group)
        warning_rows.append(
            {
                "nominal_issue_h": nominal,
                "issue_time_beijing": issue,
                "actual_lead_h": (module.EVENT_TRIGGER_BJ - issue) / pd.Timedelta(hours=1),
                "highest_persistent_level_code": max_level,
                "highest_persistent_level": LEVEL_NAMES[max_level],
                "first_blue_time": onsets[1],
                "first_yellow_time": onsets[2],
                "first_orange_time": onsets[3],
                "first_red_time": onsets[4],
                "blue_lead_to_event_h": (module.EVENT_TRIGGER_BJ - onsets[1]) / pd.Timedelta(hours=1) if pd.notna(onsets[1]) else np.nan,
                "yellow_lead_to_event_h": (module.EVENT_TRIGGER_BJ - onsets[2]) / pd.Timedelta(hours=1) if pd.notna(onsets[2]) else np.nan,
                "orange_lead_to_event_h": (module.EVENT_TRIGGER_BJ - onsets[3]) / pd.Timedelta(hours=1) if pd.notna(onsets[3]) else np.nan,
                "red_lead_to_event_h": (module.EVENT_TRIGGER_BJ - onsets[4]) / pd.Timedelta(hours=1) if pd.notna(onsets[4]) else np.nan,
            }
        )
        mechanism_rows.append(
            {
                "nominal_issue_h": nominal,
                "issue_time_beijing": issue,
                "event_reference_time": at_event.valid_time_beijing,
                "thermal_median_at_event": at_event.thermal_median,
                "rain_water_median_at_event": at_event.rain_water_median,
                "dominant_mechanism_at_event": at_event.dominant_mechanism,
                "mechanism_margin_at_event": at_event.mechanism_margin,
                "forecast_peak_time": at_peak.valid_time_beijing,
                "thermal_median_at_peak": at_peak.thermal_median,
                "rain_water_median_at_peak": at_peak.rain_water_median,
                "dominant_mechanism_at_peak": at_peak.dominant_mechanism,
                "mechanism_margin_at_peak": at_peak.mechanism_margin,
            }
        )

    warning_summary = pd.DataFrame(warning_rows).sort_values("nominal_issue_h", ascending=False)
    # A single long-lead issue can fluctuate strongly.  Record, without
    # replacing the raw level, the level supported by two consecutive issue
    # cycles.  This is a stability confirmation, not a second risk index.
    raw_codes = warning_summary.highest_persistent_level_code.to_numpy(int)
    confirmed = np.zeros(len(raw_codes), dtype=int)
    if len(raw_codes) > 1:
        confirmed[1:] = np.minimum(raw_codes[:-1], raw_codes[1:])
    warning_summary["two_issue_confirmed_level_code"] = confirmed
    warning_summary["two_issue_confirmed_level"] = pd.Series(confirmed).map(LEVEL_NAMES).to_numpy()
    warning_summary["confirmed_level_issue_lead_h"] = np.where(
        confirmed > 0,
        warning_summary.actual_lead_h,
        np.nan,
    )
    mechanism_summary = pd.DataFrame(mechanism_rows).sort_values("nominal_issue_h", ascending=False)
    warning_summary.to_csv(OUTPUTS / "18_五起报预警时间汇总.csv", index=False, encoding="utf-8-sig")
    mechanism_summary.to_csv(OUTPUTS / "19_双机制分支诊断.csv", index=False, encoding="utf-8-sig")

    event_thermal_dominance = int(mechanism_summary.dominant_mechanism_at_event.eq("热力—冰冻圈").sum())
    report = f"""# 第三问预警等级与双机制分支补充

## 统一预警规则

预警等级仍使用同一个时空联合风险指数，不另建一套分类规则。联合风险中位数连续{PERSISTENCE_H}小时超过2021—2024历史时空参考分布的q80、q90、q95、q99时，依次进入蓝色、黄色、橙色和红色。它是历史相对风险等级，不是法定阈值，也不是泥石流发生概率。

为避免单次远期预报跳变造成过度预警，另报告相邻两次起报共同支持的最高等级，即两次等级的较低者。该步骤只做滚动稳定性确认，不另建第二套风险判断规则。

{threshold_table.to_string(index=False)}

## 五个起报时次

{warning_summary.to_string(index=False)}

## 机制判别

在五个滚动起报对真实启动时刻的重建中，有{event_thermal_dominance}/5次由热力—冰冻圈分支占主导。降雨—水分分支保留为独立辅助分支，不因2026事件缺少强降雨而被删除，也不被强行提高权重。

{mechanism_summary.to_string(index=False)}

## 解释边界

- q80在高易发源区可能频繁超过，因此蓝色首先表示“持续关注”，不能直接解释为低虚警报警。
- 首次预警时间由预报期未来12小时持续状态决定，是真实起报时可计算的滚动预警结果。
- 目前没有足够同机理事件估计蓝、黄、橙、红四级的真实命中率和虚警率；不报告ROC、AUC或Youden阈值。
- 空间概率核心区和外包络区需等待第四问动力学情景输出，再通过统一接口回接第三问；当前A—B—C—D传播模型保留为可解释基线。
"""
    (OUTPUTS / "20_预警与机制补充说明.md").write_text(report, encoding="utf-8")
    print(warning_summary.to_string(index=False))
    print(mechanism_summary.to_string(index=False))


if __name__ == "__main__":
    main()
