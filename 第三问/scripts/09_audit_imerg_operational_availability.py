from __future__ import annotations

"""Audit what IMERG Late observations were actually available at each issue time.

IMERG Late is an independent rainfall-evidence branch, not a future rainfall
forecast.  NASA gives a nominal latency of about 14 h, so an issue at time t may
only use observations ending no later than t-14 h.  Missing coverage is reported
as unavailable and is never converted to zero rainfall.
"""

from pathlib import Path

import numpy as np
import pandas as pd


MODEL_DIR = Path(__file__).resolve().parents[1]
PROJECT = MODEL_DIR.parent
OUTPUTS = MODEL_DIR / "outputs"
IMERG_CACHE = (
    PROJECT
    / "第一问"
    / "05_重构方案_多源事件对照"
    / "cache"
    / "imerg_source_halfhourly.parquet"
)
Q2_THRESHOLDS = (
    PROJECT
    / "第二问"
    / "07_正式模型_外源数据融合"
    / "outputs"
    / "12_时间触发临界条件与稳健性.csv"
)
WARNING_SUMMARY = OUTPUTS / "18_五起报预警时间汇总.csv"
EVENT_TRIGGER = pd.Timestamp("2026-08-26 10:52:21.900")
NOMINAL_LATE_LATENCY_H = 14
NASA_LATENCY_URL = "https://gpm.nasa.gov/resources/faq/what-determines-latency-imerg"


def accumulation(
    imerg: pd.DataFrame, cutoff: pd.Timestamp, window_h: int
) -> tuple[float, int, int, float, bool]:
    start = cutoff - pd.Timedelta(hours=window_h)
    sample = imerg.loc[(imerg.index > start) & (imerg.index <= cutoff)]
    expected = window_h * 2
    valid = int(sample["precip_amount_mm"].notna().sum())
    completeness = valid / expected
    complete = completeness >= 0.95
    value = float(sample["precip_amount_mm"].sum()) if complete else np.nan
    return value, valid, expected, completeness, complete


def main() -> None:
    for path in [IMERG_CACHE, Q2_THRESHOLDS, WARNING_SUMMARY]:
        if not path.exists():
            raise FileNotFoundError(path)

    imerg = pd.read_parquet(IMERG_CACHE).sort_index()
    imerg.index = pd.to_datetime(imerg.index)
    q2 = pd.read_csv(Q2_THRESHOLDS)
    q90_72 = float(q2.loc[q2.feature.eq("imerg_rain_72h_mm"), "q90"].iloc[0])
    q95_72 = float(q2.loc[q2.feature.eq("imerg_rain_72h_mm"), "q95"].iloc[0])
    issues = pd.read_csv(WARNING_SUMMARY, parse_dates=["issue_time_beijing"])

    rows: list[dict[str, object]] = []
    for row in issues.sort_values("nominal_issue_h", ascending=False).itertuples():
        issue = pd.Timestamp(row.issue_time_beijing)
        cutoff = issue - pd.Timedelta(hours=NOMINAL_LATE_LATENCY_H)
        rain24, n24, e24, c24, ok24 = accumulation(imerg, cutoff, 24)
        rain72, n72, e72, c72, ok72 = accumulation(imerg, cutoff, 72)
        if ok72:
            branch = "达到q95" if rain72 >= q95_72 else "达到q90" if rain72 >= q90_72 else "未达到q90"
            supports = bool(rain72 >= q90_72)
            note = "可用完整72 h Late Run，仅作为已发生降雨证据"
        else:
            branch = "资料窗不完整"
            supports = np.nan
            note = "不得把缺测解释为零降雨或弱降雨"
        rows.append(
            {
                "nominal_issue_h": int(row.nominal_issue_h),
                "issue_time_beijing": issue,
                "nominal_late_latency_h": NOMINAL_LATE_LATENCY_H,
                "latest_observable_time_beijing": cutoff,
                "rain24_available": ok24,
                "rain24_mm": rain24,
                "rain24_valid_halfhours": n24,
                "rain24_expected_halfhours": e24,
                "rain24_completeness": c24,
                "rain72_available": ok72,
                "rain72_mm": rain72,
                "rain72_valid_halfhours": n72,
                "rain72_expected_halfhours": e72,
                "rain72_completeness": c72,
                "historical_q90_72h_mm": q90_72,
                "historical_q95_72h_mm": q95_72,
                "rainfall_branch_supports_high_trigger": supports,
                "rainfall_evidence_level": branch,
                "interpretation": note,
            }
        )

    audit = pd.DataFrame(rows)
    audit.to_csv(OUTPUTS / "21_IMERG起报可用性审计.csv", index=False, encoding="utf-8-sig")

    event72, event_n, event_e, event_c, event_ok = accumulation(imerg, EVENT_TRIGGER, 72)
    complete_issues = audit.loc[audit.rain72_available]
    high_count = int(complete_issues.rainfall_branch_supports_high_trigger.eq(True).sum())
    report = f"""# IMERG业务可用性与独立降雨分支审计

## 1. 为什么需要单独审计

当前缓存是IMERG Late Run V07半小时产品。NASA说明Late Run的名义延迟约为14小时（近期常约12小时），因此它可以作为独立的近实时降雨证据，却不能在起报时使用尚未发布的未来资料。本文采用更保守的14小时截断：起报时刻`t`只允许使用`t-14 h`及以前的IMERG。

官方说明：{NASA_LATENCY_URL}

## 2. 五次起报的可用资料

{audit.to_string(index=False)}

在具有完整72小时可用窗的{len(complete_issues)}次起报中，达到历史q90（{q90_72:.2f} mm）的次数为{high_count}次。资料窗不完整的起报仅标记为“不可判”，不能解释成无雨。

## 3. 事件回溯核验

按真实启动时刻回溯的IMERG 72小时累计降雨为{event72:.3f} mm（完整度{event_c:.1%}，{event_n}/{event_e}个半小时），低于历史q90={q90_72:.2f} mm和q95={q95_72:.2f} mm。这是对“2026事件非强降雨主导”的独立支持，但不证明降雨对坡面水分或河道汇流完全没有作用。

## 4. 论文边界

- IMERG Late是观测型证据，不承担未来1—7天降雨预报；未来降雨仍由GEFS情景提供。
- 该分支只验证降雨是否达到异常触发水平，不与热力分支竞争一套新的最终等级。
- 不能把Late Run的14小时延迟忽略，也不能用事后完整72小时雨量冒充较早起报时已知的信息。
- 不能把缺失窗口填成零降雨。
"""
    (OUTPUTS / "22_IMERG业务解释与降雨分支核验.md").write_text(report, encoding="utf-8")
    print(audit.to_string(index=False))
    print(f"event retrospective 72 h rainfall={event72:.3f} mm")


if __name__ == "__main__":
    main()
