from __future__ import annotations

import importlib.util
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
VALIDATION_DIR = SCRIPT_DIR.parent
QUESTION_DIR = VALIDATION_DIR.parent
ANALYSIS_SCRIPT = (
    QUESTION_DIR
    / "02_特征构造与主分析"
    / "scripts"
    / "01_run_2026_main_analysis.py"
)
OUTPUT_DIR = VALIDATION_DIR / "outputs"

IMPACT_TIME = pd.Timestamp("2025-07-08 05:30:00")
PRIMARY_CUTOFF = pd.Timestamp("2025-07-08 04:00:00")
SENSITIVITY_CUTOFF = pd.Timestamp("2025-07-08 05:00:00")
CONTROL_YEARS = [2021, 2022, 2023, 2024]
ANALYSIS_YEARS = CONTROL_YEARS + [2025]

STATION_COORD = (28.86, 85.29)
SOURCE_COORD = (28.402561, 85.646705)
IMPACT_COORD = (28.2797, 85.3778)

FROZEN_FEATURES = {
    "vwc_delta_24h": {
        "稳定性顺序": 1,
        "最终特征": "剖面土壤含水量24小时变化",
        "所属物理因素": "土壤湿润状态",
        "单位": "m³/m³",
        "预期方向": "偏低",
        "方向符号": -1,
    },
    "temp_mean_24h": {
        "稳定性顺序": 2,
        "最终特征": "24小时平均2米气温",
        "所属物理因素": "热量累积",
        "单位": "°C",
        "预期方向": "偏高",
        "方向符号": 1,
    },
    "positive_netrad_24h": {
        "稳定性顺序": 3,
        "最终特征": "24小时累计正净辐射",
        "所属物理因素": "辐射能量",
        "单位": "kWh/m²",
        "预期方向": "偏高",
        "方向符号": 1,
    },
    "precip_sum_72h": {
        "稳定性顺序": 4,
        "最终特征": "72小时累计降水",
        "所属物理因素": "降水与水分输入",
        "单位": "mm",
        "预期方向": "偏低",
        "方向符号": -1,
    },
    "rh_mean_24h": {
        "稳定性顺序": 5,
        "最终特征": "24小时平均2米相对湿度",
        "所属物理因素": "大气湿度与水汽",
        "单位": "%",
        "预期方向": "偏低",
        "方向符号": -1,
    },
}

CUTOFFS = {
    "主截止04:00": PRIMARY_CUTOFF,
    "敏感性截止05:00": SENSITIVITY_CUTOFF,
}
EVALUATION_HOURS = [12, 24, 48]
DATE_BANDS = [0, 3, 7, 15]
QC_MODES = ["strict", "expanded"]


def load_analysis_module():
    spec = importlib.util.spec_from_file_location("question1_analysis", ANALYSIS_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载主分析脚本：{ANALYSIS_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVENT_TIME = IMPACT_TIME
    module.EVENT_CUTOFF = PRIMARY_CUTOFF
    module.CONTROL_YEARS = CONTROL_YEARS
    module.ANALYSIS_YEARS = ANALYSIS_YEARS
    module.FEATURE_CATALOG = {
        name: module.FEATURE_CATALOG[name] for name in FROZEN_FEATURES
    }
    return module


def haversine_km(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    lat1, lon1 = map(radians, point_a)
    lat2, lon2 = map(radians, point_b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    value = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6371.0088 * asin(sqrt(value))


def evaluate_scenario(
    module,
    features: pd.DataFrame,
    mode: str,
    cutoff_label: str,
    cutoff: pd.Timestamp,
    evaluation_hours: int,
    date_band: int,
) -> pd.DataFrame:
    module.EVENT_CUTOFF = cutoff
    metrics = module.evaluate_features(
        features,
        mode,
        evaluation_hours=evaluation_hours,
        control_date_band_days=date_band,
        flux_policy="不涉及FLUX",
    )
    selected = metrics[metrics["衍生指标"].isin(FROZEN_FEATURES)].copy()
    if len(selected) != len(FROZEN_FEATURES):
        missing = sorted(set(FROZEN_FEATURES) - set(selected["衍生指标"]))
        raise RuntimeError(f"场景中缺少冻结特征：{missing}")

    selected["截止时刻口径"] = cutoff_label
    selected["灾前截止时刻"] = cutoff
    selected["预期方向"] = selected["衍生指标"].map(
        lambda name: FROZEN_FEATURES[name]["预期方向"]
    )
    selected["方向符号"] = selected["衍生指标"].map(
        lambda name: FROZEN_FEATURES[name]["方向符号"]
    )
    selected["方向一致"] = selected["偏离方向"] == selected["预期方向"]
    selected["方向性历史百分位"] = np.where(
        selected["方向符号"] > 0,
        selected["事件中位数历史百分位"],
        1 - selected["事件中位数历史百分位"],
    )
    selected["强支持"] = selected["方向一致"] & (
        selected["方向性历史百分位"] >= 0.80
    )
    selected["方向性稳健差异"] = (
        selected["方向符号"] * selected["稳健标准化差异"]
    )
    selected["稳定性顺序"] = selected["衍生指标"].map(
        lambda name: FROZEN_FEATURES[name]["稳定性顺序"]
    )
    selected["单位"] = selected["衍生指标"].map(
        lambda name: FROZEN_FEATURES[name]["单位"]
    )
    return selected.sort_values("稳定性顺序")


def scenario_summary(all_results: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "场景编号",
        "截止时刻口径",
        "灾前截止时刻",
        "清洗版本",
        "评价窗口小时",
        "同期日期带天数",
    ]
    rows = []
    for values, frame in all_results.groupby(keys, sort=True):
        row = dict(zip(keys, values))
        row.update(
            {
                "方向一致项数": int(frame["方向一致"].sum()),
                "强支持项数": int(frame["强支持"].sum()),
                "等权方向百分位得分": float(frame["方向性历史百分位"].mean()),
                "等权方向性稳健差异": float(frame["方向性稳健差异"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("场景编号")


def feature_stability(all_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature, frame in all_results.groupby("衍生指标"):
        meta = FROZEN_FEATURES[feature]
        rows.append(
            {
                "稳定性顺序": meta["稳定性顺序"],
                "最终特征": meta["最终特征"],
                "衍生指标": feature,
                "预期方向": meta["预期方向"],
                "场景数": len(frame),
                "方向一致比例": float(frame["方向一致"].mean()),
                "强支持比例": float(frame["强支持"].mean()),
                "方向性历史百分位中位数": float(frame["方向性历史百分位"].median()),
                "方向性稳健差异中位数": float(frame["方向性稳健差异"].median()),
                "最小事件窗口覆盖率": float(frame["事件窗口覆盖率"].min()),
            }
        )
    return pd.DataFrame(rows).sort_values("稳定性顺序")


def validation_grade(primary: pd.DataFrame, scenarios: pd.DataFrame) -> tuple[str, str]:
    primary_direction = int(primary["方向一致"].sum())
    primary_strong = int(primary["强支持"].sum())
    rate_at_least_four = float((scenarios["方向一致项数"] >= 4).mean())
    rate_at_least_three = float((scenarios["方向一致项数"] >= 3).mean())
    if primary_direction >= 4 and primary_strong >= 3 and rate_at_least_four >= 0.75:
        return "强验证", "达到预设强验证判据"
    if primary_direction >= 3 and primary_strong >= 2 and rate_at_least_three >= 0.75:
        return "部分验证", "达到预设部分验证判据，但未达到强验证判据"
    return "未获得支持", "未达到预设部分验证判据"


def write_report(
    primary: pd.DataFrame,
    all_results: pd.DataFrame,
    scenarios: pd.DataFrame,
    stability: pd.DataFrame,
) -> None:
    grade, grade_note = validation_grade(primary, scenarios)
    primary_direction = int(primary["方向一致"].sum())
    primary_strong = int(primary["强支持"].sum())
    rate_four = float((scenarios["方向一致项数"] >= 4).mean())
    rate_three = float((scenarios["方向一致项数"] >= 3).mean())
    source_distance = haversine_km(STATION_COORD, SOURCE_COORD)
    impact_distance = haversine_km(STATION_COORD, IMPACT_COORD)

    lines = [
        "# 2025年Purepu冰湖溃决事件独立验证报告",
        "",
        "## 验证结论",
        "",
        f"**预设判据结果：{grade}。** {grade_note}。",
        "",
        f"- 主场景方向一致：{primary_direction}/5。",
        f"- 主场景强支持：{primary_strong}/5。",
        f"- 48个敏感性场景中达到至少4/5方向一致的比例：{rate_four:.1%}。",
        f"- 48个敏感性场景中达到至少3/5方向一致的比例：{rate_three:.1%}。",
        "",
        "该等级只表示2025年区域站点异常与2026年冻结特征方向的复现程度，不表示五个特征是冰湖溃决的直接原因。",
        "",
        "## 事件与时间口径",
        "",
        f"- 首次报告影响时间：{IMPACT_TIME}，北京时间。",
        "- 该时间是洪水在中尼边境口岸的首次报告影响时间，源区开始排水的准确时刻未知。",
        f"- 主验证截止：{PRIMARY_CUTOFF}，排除05:00—05:59事件所在小时。",
        f"- 补充截止：{SENSITIVITY_CUTOFF}，用于检验整点归属造成的敏感性。",
        "- 源区：Purepu冰川冰面湖，28.402561°N、85.646705°E、海拔约5160 m。",
        "- 灾害机制：冰面湖快速排水引发GLOF，并沿河形成跨境山洪泥石流灾害链。",
        "",
        "## 主场景逐项结果",
        "",
        "| 顺序 | 冻结特征 | 2025事件中位数 | 历史同期中位数 | 2025方向 | 2026预期方向 | 方向性百分位 | 稳健差异 | 结论 |",
        "|---:|---|---:|---:|---|---|---:|---:|---|",
    ]
    for _, row in primary.sort_values("稳定性顺序").iterrows():
        conclusion = "强支持" if row["强支持"] else ("方向支持" if row["方向一致"] else "方向不一致")
        lines.append(
            f"| {int(row['稳定性顺序'])} | {row['指标中文名']} | "
            f"{row['事件中位数']:.6g} {row['单位']} | {row['对照中位数']:.6g} {row['单位']} | "
            f"{row['偏离方向']} | {row['预期方向']} | {row['方向性历史百分位']:.1%} | "
            f"{row['稳健标准化差异']:.3f} | {conclusion} |"
        )

    lines.extend(
        [
            "",
            "## 五个特征的跨场景稳定性",
            "",
            "| 顺序 | 冻结特征 | 预期方向 | 方向一致比例 | 强支持比例 | 方向性百分位中位数 |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    for _, row in stability.iterrows():
        lines.append(
            f"| {int(row['稳定性顺序'])} | {row['最终特征']} | {row['预期方向']} | "
            f"{row['方向一致比例']:.1%} | {row['强支持比例']:.1%} | "
            f"{row['方向性历史百分位中位数']:.1%} |"
        )

    lines.extend(
        [
            "",
            "## 空间代表性",
            "",
            f"- 按经纬度球面距离估算，观测站至Purepu源区约{source_distance:.1f} km。",
            f"- 观测站至边境口岸受灾点约{impact_distance:.1f} km。",
            "- 因此本验证检验的是区域热量、水汽、辐射和地表状态背景能否复现，不是源区局地触发条件的直接验证。",
            "",
            "## 论文使用原则",
            "",
            f"- 可以如实写为：按预设判据，2025年独立事件得到“{grade}”。",
            "- 必须同时报告不一致的特征，不能只保留验证成功的指标。",
            "- 2025年是冰面湖快速排水型GLOF，2026年是高位冰岩失稳型灾害链；跨事件一致性主要说明区域背景特征可能具有一定可迁移性，不等于两次事件启动机制相同。",
            "- 源区排水时刻未知，05:30是下游首次报告影响时间；论文中不能将其写成精确溃决启动时刻。",
            "",
            "## 资料来源",
            "",
            "- [HiRISK《RHA Report No. CN1》](https://hirisk.org/wp-content/uploads/2025/07/RHA_CN1_PurepuGLOF-2.pdf)",
            "- [NDRRMA《Rasuwa Glacial Flood Situation Report #1》](https://ndrrma.gov.np/mediafiles/publications/Rasuwagadhi_Flood_Sitrep1_08072025_.pdf)",
            "",
            "## 输出文件",
            "",
            "- `01_2025主验证五特征.csv`：主场景逐项数值。",
            "- `02_2025验证敏感性全部场景.csv`：48个场景、每个场景5个特征的完整结果。",
            "- `03_2025验证场景汇总.csv`：每个场景的一致项数和等权得分。",
            "- `04_2025特征稳定性汇总.csv`：每个冻结特征的跨场景复现情况。",
        ]
    )
    (OUTPUT_DIR / "05_2025独立验证报告.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    module = load_analysis_module()
    feature_cache: dict[str, pd.DataFrame] = {}
    for mode in QC_MODES:
        raw = module.load_mode(mode, exclude_flux_review=False)
        feature_cache[mode] = module.build_features(raw)

    rows = []
    scenario_id = 0
    for cutoff_label, cutoff in CUTOFFS.items():
        for evaluation_hours in EVALUATION_HOURS:
            for date_band in DATE_BANDS:
                for mode in QC_MODES:
                    scenario_id += 1
                    result = evaluate_scenario(
                        module,
                        feature_cache[mode],
                        mode,
                        cutoff_label,
                        cutoff,
                        evaluation_hours,
                        date_band,
                    )
                    result.insert(0, "场景编号", scenario_id)
                    rows.append(result)

    all_results = pd.concat(rows, ignore_index=True)
    primary = all_results[
        (all_results["截止时刻口径"] == "主截止04:00")
        & (all_results["清洗版本"] == "strict")
        & (all_results["评价窗口小时"] == 24)
        & (all_results["同期日期带天数"] == 0)
    ].copy()
    scenarios = scenario_summary(all_results)
    stability = feature_stability(all_results)

    primary.to_csv(
        OUTPUT_DIR / "01_2025主验证五特征.csv", index=False, encoding="utf-8-sig"
    )
    all_results.to_csv(
        OUTPUT_DIR / "02_2025验证敏感性全部场景.csv",
        index=False,
        encoding="utf-8-sig",
    )
    scenarios.to_csv(
        OUTPUT_DIR / "03_2025验证场景汇总.csv", index=False, encoding="utf-8-sig"
    )
    stability.to_csv(
        OUTPUT_DIR / "04_2025特征稳定性汇总.csv", index=False, encoding="utf-8-sig"
    )
    write_report(primary, all_results, scenarios, stability)
    print(f"2025年独立验证完成，共{scenario_id}个场景。")
    print(f"结果目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
