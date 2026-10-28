from __future__ import annotations

"""Build an honest baseline/upgrade and reliability audit for Question 3."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_DIR = Path(__file__).resolve().parents[1]
OUTPUTS = MODEL_DIR / "outputs"
SCRIPTS = MODEL_DIR / "scripts"
TRUE_TRIGGER = pd.Timestamp("2026-08-26 10:52:21.900")


def interval_score_hours(lower: pd.Timestamp, upper: pd.Timestamp, truth: pd.Timestamp, alpha: float = 0.05) -> float:
    """Central interval score: narrow is better, with a strong miss penalty."""
    width = (upper - lower) / pd.Timedelta(hours=1)
    penalty = 0.0
    if truth < lower:
        penalty = 2 / alpha * ((lower - truth) / pd.Timedelta(hours=1))
    elif truth > upper:
        penalty = 2 / alpha * ((truth - upper) / pd.Timedelta(hours=1))
    return float(width + penalty)


def load_trigger_module():
    path = SCRIPTS / "04_run_trigger_forecast.py"
    spec = importlib.util.spec_from_file_location("q3_trigger_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def baseline_upgrade(summary: pd.DataFrame) -> pd.DataFrame:
    main = summary.loc[summary.model.eq("main")].copy()
    deterministic = main.loc[~main.stochastic].set_index("nominal_issue_h")
    ensemble = main.loc[main.stochastic].set_index("nominal_issue_h")
    rows = []
    for nominal in sorted(set(deterministic.index) & set(ensemble.index), reverse=True):
        base = deterministic.loc[nominal]
        up = ensemble.loc[nominal]
        rows.append(
            {
                "nominal_issue_h": nominal,
                "actual_lead_h": up.actual_lead_h,
                "baseline_trigger_time": base.trigger_median,
                "baseline_abs_error_h": base.median_abs_error_h,
                "ensemble_trigger_median": up.trigger_median,
                "ensemble_median_abs_error_h": up.median_abs_error_h,
                "point_error_improvement_h": base.median_abs_error_h - up.median_abs_error_h,
                "scenario_trigger_support": up.trigger_support,
                "raw_q025": up.trigger_q025,
                "raw_q975": up.trigger_q975,
                "raw_interval_width_h": up.interval_width_h,
                "raw_covers_truth": up.covers_true_trigger,
                "raw_interval_score_h": interval_score_hours(up.trigger_q025, up.trigger_q975, TRUE_TRIGGER),
                "operational_q025": up.operational_q025,
                "operational_q975": up.operational_q975,
                "operational_interval_width_h": up.operational_width_h,
                "operational_covers_truth": up.covers_true_operational,
                "operational_interval_score_h": interval_score_hours(up.operational_q025, up.operational_q975, TRUE_TRIGGER),
                "raw_width_to_actual_lead_ratio": up.interval_width_h / up.actual_lead_h,
                "warning_at_issue_share": up.warning_at_issue_share,
            }
        )
    return pd.DataFrame(rows)


def window_sensitivity(summary: pd.DataFrame) -> pd.DataFrame:
    data = summary.loc[summary.stochastic].copy()
    rows = []
    for model, group in data.groupby("model"):
        rows.append(
            {
                "model": model,
                "antecedent_window_h": "mixed(72/72/120)" if model == "main" else int(model.split("_")[1]),
                "rolling_issue_count": len(group),
                "median_abs_error_h": group.median_abs_error_h.median(),
                "mean_abs_error_h": group.median_abs_error_h.mean(),
                "raw_coverage_count": int(group.covers_true_trigger.sum()),
                "operational_coverage_count": int(group.covers_true_operational.sum()),
                "median_raw_width_h": group.interval_width_h.median(),
                "median_operational_width_h": group.operational_width_h.median(),
                "mean_trigger_support": group.trigger_support.mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["model"])


def score_reliability(module) -> tuple[pd.DataFrame, dict[str, float]]:
    history = module.add_historical_features(
        module.attach_mcd43(pd.read_parquet(module.Q2_DYNAMIC).sort_index())
    )
    bg_scores, refs, thresholds, _, source_static = module.build_references(history)
    nearest_pos = history.index.get_indexer([module.EVENT_TRIGGER_BJ], method="nearest")[0]
    event_row = history.iloc[nearest_pos]
    thermal = float(module.cdf(refs["thermal_72"], np.array([event_row["eti_mcd43_72h_mm"]]))[0])
    rain = float(module.cdf(refs["rain_72"], np.array([event_row["era5_rain_72h_mm"]]))[0])
    antecedent_rain = float(module.cdf(refs["p15"], np.array([event_row["chirps_rain_prev_15d_mm"]]))[0])
    rootzone = float(module.cdf(refs["soil_120"], np.array([event_row["smap_rootzone_mean_120h"]]))[0])
    moisture = float(np.median([antecedent_rain, rootzone]))
    rain_moisture = float(np.sqrt(max(rain, 0) * max(moisture, 0)))
    dynamic = float(1 - (1 - thermal) * (1 - rain_moisture))
    event_joint = float(np.sqrt(source_static * dynamic))
    background_source_joint = np.sqrt(source_static * bg_scores.dynamic.to_numpy(float))
    audit = pd.DataFrame(
        [
            {"audit_item": "2026_event_nearest_hour", "value": str(history.index[nearest_pos]), "interpretation": "closest available hourly reconstruction to verified onset"},
            {"audit_item": "event_thermal_percentile", "value": thermal, "interpretation": "MCD43-ETI72 percentile against 2021-2024 seasonal background"},
            {"audit_item": "event_short_rain_percentile", "value": rain, "interpretation": "rain is secondary, not forced to dominate"},
            {"audit_item": "event_antecedent_rain_percentile", "value": antecedent_rain, "interpretation": "15-day moisture proxy"},
            {"audit_item": "event_rootzone_percentile", "value": rootzone, "interpretation": "SMAP root-zone state"},
            {"audit_item": "event_dynamic_score", "value": dynamic, "interpretation": "mechanism union score, not probability"},
            {"audit_item": "event_source_hotspot_joint_risk", "value": event_joint, "interpretation": "source hotspot gate times dynamic score"},
            {"audit_item": "q80_joint_threshold", "value": thresholds["main"], "interpretation": "Q2 historical space-time screening threshold"},
            {"audit_item": "seasonal_daily_source_hotspot_exceedance_share", "value": float(np.mean(background_source_joint >= thresholds["main"])), "interpretation": "diagnostic only; high value means q80 is not a rare-event alarm threshold"},
        ]
    )
    values = {
        "thermal": thermal,
        "rain": rain,
        "antecedent": antecedent_rain,
        "rootzone": rootzone,
        "dynamic": dynamic,
        "event_joint": event_joint,
        "threshold": thresholds["main"],
        "background_exceedance": float(np.mean(background_source_joint >= thresholds["main"])),
    }
    return audit, values


def main() -> None:
    source = OUTPUTS / "04_触发时间区间与覆盖.csv"
    if not source.exists():
        raise FileNotFoundError(source)
    summary = pd.read_csv(
        source,
        parse_dates=[
            "issue_time_beijing",
            "trigger_q025",
            "trigger_median",
            "trigger_q975",
            "operational_q025",
            "operational_q975",
        ],
    )
    comparison = baseline_upgrade(summary)
    sensitivity = window_sensitivity(summary)
    audit, audit_values = score_reliability(load_trigger_module())
    comparison.to_csv(OUTPUTS / "12_基线与集合升级模型对照.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(OUTPUTS / "13_五时间窗敏感性汇总.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(OUTPUTS / "14_可靠性与解释边界审计.csv", index=False, encoding="utf-8-sig")

    short = comparison.loc[comparison.nominal_issue_h.isin([24, 48])]
    all_op_coverage = int(comparison.operational_covers_truth.sum())
    all_raw_coverage = int(comparison.raw_covers_truth.sum())
    report = f"""# 第三问模型对照与可靠性结论

## 1. 基线与升级模型

确定性GEFS均值是基线，GEFS均值/离散度生成的1000个相关天气—参数情景是升级模型。升级的主要作用是给出不确定性范围，而不是保证点预测一定优于基线。五个滚动起报中，原始情景区间覆盖真实启动时刻{all_raw_coverage}/5次；计入6小时原生分辨率半步误差后的操作区间覆盖{all_op_coverage}/5次。24 h和48 h两次短临起报的操作区间覆盖为{int(short.operational_covers_truth.sum())}/2，但集合中位时刻仍平均偏差{short.ensemble_median_abs_error_h.mean():.2f}小时。

这五次是同一事件的五个滚动起报，彼此不独立，因此不能把{all_op_coverage}/5写成普适“准确率”或频率学覆盖率。

## 2. 风险阈值能做什么

事件最近整点的热力百分位为{audit_values['thermal']:.3f}，短时降雨百分位为{audit_values['rain']:.3f}，动态评分为{audit_values['dynamic']:.3f}；说明模型保持“热力主导、降雨辅助”的机理结构。源区热点联合风险为{audit_values['event_joint']:.3f}，高于Q2的q80筛查阈值{audit_values['threshold']:.3f}。

但源区本身静态易发性很高，历史同期每日样本在该热点处超过q80的比例约{audit_values['background_exceedance']:.1%}。所以q80只适合表示“已进入持续关注状态”，不能解释为罕见事件的独立报警阈值，也不能据此声称低虚警率。触发时刻由未来持续风险峰值给出，越阈仅作为有无触发支持的门控。

## 3. 可写入论文的结论

- 24—48 h：可以给出约小时级危险窗口和区间，但中位时刻仍存在数小时偏差。
- 72 h：区间能够包住真值，但区间较宽，适合趋势预警，不适合精确报时。
- 120—168 h：主要用于监测风险形势；120 h区间过宽，168 h未覆盖真值，不宜声称具备精确触发时刻能力。
- A—B—C—D传播模型可以把触发区间传递为口岸到达区间；速度来自本事件灾后反演，因此属于参数校准后的事件回放，不是跨事件独立验证。

## 4. 不能写的结论

- 不能把情景支持比例写成“泥石流发生概率”。
- 不能把4/5写成80%预测准确率。
- 不能把经6小时分辨率扩展后的操作区间冒充原始95%情景区间。
- 不能声称模型已经构成可直接部署的现实预警系统；同机理事件样本和历史业务预报档案仍不足。

## 基线—升级逐起报对照

{comparison.to_string(index=False)}

## 五窗口敏感性

{sensitivity.to_string(index=False)}
"""
    (OUTPUTS / "15_模型对照与可靠性报告.md").write_text(report, encoding="utf-8")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
