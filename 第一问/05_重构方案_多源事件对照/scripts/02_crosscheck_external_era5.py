from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REBUILD_DIR = SCRIPT_DIR.parent
EXTERNAL_PROCESSED = REBUILD_DIR / "external" / "gyirong-2026" / "data" / "processed"
OUTPUT_DIR = REBUILD_DIR / "outputs"

# 使用事件前最后一个完整日，避免daily_series中的8月26日整日统计含有灾后时段。
EVENT_LAST_COMPLETE_DAY = pd.Timestamp("2026-08-25")
MAIN_CONTROL_YEARS = list(range(2016, 2025))
SENSITIVITY_CONTROL_YEARS = list(range(2016, 2026))
WINDOW_DAYS = [1, 3, 5, 7, 10, 30]


def robust_z(value: float, controls: np.ndarray) -> float:
    median = float(np.median(controls))
    mad = float(np.median(np.abs(controls - median)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(controls, ddof=1))
    return (value - median) / scale if scale > 1e-12 else 0.0


def percentile(value: float, controls: np.ndarray) -> float:
    # 中秩经验百分位，避免小样本中把并列值全部记为低或高。
    below = float(np.sum(controls < value))
    equal = float(np.sum(controls == value))
    return 100.0 * (below + 0.5 * equal) / len(controls)


def window_values(
    daily: pd.DataFrame,
    year: int,
    days: int,
) -> dict[str, float]:
    end = EVENT_LAST_COMPLETE_DAY.replace(year=year)
    start = end - pd.Timedelta(days=days - 1)
    frame = daily[(daily["date"] >= start) & (daily["date"] <= end)].copy()
    if len(frame) != days:
        return {}
    return {
        "源区平均气温": float(frame["t_source"].mean()),
        "源区最高气温": float(frame["t_source_max"].max()),
        "源区正积温": float(frame["t_source"].clip(lower=0).sum()),
        "源区气温距平": float(frame["t_source_anom"].mean()),
        "区域累计降水": float(frame["precip"].sum()),
        "0℃等温线平均高度": float(frame["freezing_level"].mean()),
        "动态递减率中位数": float(frame["lapse_slope"].median() * 1000.0),
        "递减率拟合R2中位数": float(frame["lapse_r2"].median()),
    }


def evaluate_baseline(daily: pd.DataFrame, control_years: list[int], scenario: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    expected_direction = {
        "源区平均气温": "高",
        "源区最高气温": "高",
        "源区正积温": "高",
        "源区气温距平": "高",
        "区域累计降水": "双侧；高值支持降雨作用，低值支持非极端降雨",
        "0℃等温线平均高度": "高",
        "动态递减率中位数": "不排序；仅诊断订正参数",
        "递减率拟合R2中位数": "不排序；仅诊断订正质量",
    }

    for days in WINDOW_DAYS:
        event = window_values(daily, 2026, days)
        controls_by_year = {
            year: window_values(daily, year, days) for year in control_years
        }
        controls_by_year = {
            year: values for year, values in controls_by_year.items() if values
        }
        for feature, event_value in event.items():
            controls = np.array(
                [values[feature] for values in controls_by_year.values()], dtype=float
            )
            rows.append(
                {
                    "基准场景": scenario,
                    "窗口天数": days,
                    "指标": feature,
                    "2026事件值": event_value,
                    "历史中位数": float(np.median(controls)),
                    "历史最小值": float(np.min(controls)),
                    "历史最大值": float(np.max(controls)),
                    "稳健Z": robust_z(event_value, controls),
                    "经验百分位": percentile(event_value, controls),
                    "历史年份数": len(controls),
                    "预期解释方向": expected_direction[feature],
                    "时间口径": "截至8月25日的完整日；不含8月26日灾后时段",
                }
            )
    return pd.DataFrame(rows)


def build_report(results: pd.DataFrame, hourly_diagnostics: dict[str, object]) -> str:
    main = results[results["基准场景"] == "主基准_2016至2024"]
    ranked = main[
        ~main["指标"].isin(["动态递减率中位数", "递减率拟合R2中位数"])
    ].copy()
    # 一日窗口且日均温为正时，一日PDD与一日平均气温数值完全重复；报告中只展示气温。
    ranked = ranked[~((ranked["指标"] == "源区正积温") & (ranked["窗口天数"] == 1))]
    ranked["尾部异常度"] = (ranked["经验百分位"] - 50.0).abs()
    ranked = ranked.sort_values(["尾部异常度", "稳健Z"], ascending=[False, False])

    lines = [
        "# 外部ERA5-Land成品独立交叉核验",
        "",
        "本报告只利用外部公开项目已经发布的处理后序列，对第一问变量设计进行预核验，不是最终排名。",
        "",
        "## 严格时间口径",
        "",
        "- 日尺度特征统一截止到2026年8月25日，避免8月26日整日均值混入灾后时段。",
        "- 小时尺度24/72/168小时结果沿用外部项目截止到8月26日10:00的诊断。",
        "- 主基准使用2016—2024；加入2025只作为基准敏感性。",
        "",
        "## 主基准下尾部异常较强的组合",
        "",
    ]
    for row in ranked.head(15).to_dict("records"):
        lines.append(
            f"- {row['指标']}（{row['窗口天数']}日）：事件值={row['2026事件值']:.3f}，"
            f"历史中位数={row['历史中位数']:.3f}，稳健Z={row['稳健Z']:.2f}，"
            f"经验百分位={row['经验百分位']:.1f}%。"
        )

    lines.extend(["", "## 小时尺度外部诊断", ""])
    for item in hourly_diagnostics.get("precip_windows", []):
        lines.append(
            f"- 灾前{item['hours']}小时区域降水={item['total']:.2f} mm，"
            f"外部项目历史百分位={item['percentile']:.1f}%。"
        )
    for item in hourly_diagnostics.get("freezing_windows", []):
        lines.append(
            f"- 灾前{item['hours']}小时0℃等温线平均高度={item['mean']:.0f} m，"
            f"外部项目历史百分位={item['percentile']:.1f}%。"
        )

    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 外部项目历史期只有2016—2025，短于计划采用的2001—2024主背景。",
            "- 其气温通过81个ERA5-Land格点拟合逐日动态递减率后订正到5300m，仍然是区域再分析估计，不是源区实测。",
            "- 一日窗口内若日均温为正，一日PDD与一日平均气温代数重复；最终筛选必须按物理组去冗余。",
            "- 公开研究报告的源区近邻格点7日均温约9.43℃、PDD约65.94℃·d，而本项目成品经动态递减率订正到5300m后，截止8月25日的7日均温约4.44℃、PDD约31.06℃·d。二者绝对值差异主要来自空间支持、高程订正和截止口径，说明最终模型应重视历史百分位及多空间方案稳定性，不采用单一绝对温度阈值。",
            "- 其降水为空间走廊加权平均，与源区点值、IMERG格点和题目观测站降水含义不同。",
            "- 本结果可以检验指标是否值得保留，不能直接作为最终五因素排序。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(EXTERNAL_PROCESSED / "daily_series.csv", parse_dates=["date"])
    main_result = evaluate_baseline(daily, MAIN_CONTROL_YEARS, "主基准_2016至2024")
    sensitivity_result = evaluate_baseline(
        daily, SENSITIVITY_CONTROL_YEARS, "敏感性_2016至2025"
    )
    results = pd.concat([main_result, sensitivity_result], ignore_index=True)
    results.to_csv(
        OUTPUT_DIR / "05_外部ERA5日尺度交叉核验.csv",
        index=False,
        encoding="utf-8-sig",
    )

    diagnostics = json.loads(
        (EXTERNAL_PROCESSED / "event_hour_diagnostics.json").read_text(encoding="utf-8")
    )
    report = build_report(results, diagnostics)
    (OUTPUT_DIR / "06_外部ERA5交叉核验报告.md").write_text(report, encoding="utf-8")

    print(f"交叉核验完成：{OUTPUT_DIR}")
    print(f"结果行数：{len(results)}")


if __name__ == "__main__":
    main()

