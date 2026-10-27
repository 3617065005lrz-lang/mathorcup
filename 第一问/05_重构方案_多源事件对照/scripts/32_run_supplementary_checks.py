from __future__ import annotations

"""第一问三项补充分析。

1. LCL只作为湿度/边界层候选指标，检查其异常性和与VPD、RH的冗余；
2. 输出2026事件前40天GMM天气型序列；
3. 对2025事件做排序完成后的跨机理方向核查。

2025不进入GMM训练、历史标准化、显著性门控、权重或TOPSIS排序。
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
CACHE = ROOT / "cache"
OUTPUT = ROOT / "outputs"

EVENT_2026 = pd.Timestamp("2026-08-26 10:00:00")
EVENT_2025 = pd.Timestamp("2025-07-08 05:00:00")
CONTROL_YEARS = [2021, 2022, 2023, 2024]

spec = spec_from_file_location("event_ranking", SCRIPT_DIR / "05_run_event_control_ranking.py")
ranking = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(ranking)


def lcl_sensitivity() -> pd.DataFrame:
    station = pd.read_parquet(CACHE / "station_daily_10h_features_strict.parquet")
    station["Timestamp"] = pd.to_datetime(station["Timestamp"])
    historical = station[station["Timestamp"].dt.year.isin(CONTROL_YEARS)]
    dictionary = pd.read_csv(OUTPUT / "07_站点衍生指标字典.csv")
    lcl_dictionary = dictionary[
        dictionary["清洗版本"].eq("strict")
        & dictionary["衍生指标"].astype(str).str.startswith("sta_lcl_")
    ].drop_duplicates(["清洗版本", "衍生指标"])
    classified, _, _ = ranking.fit_weather_types("strict", n_components=6, diagnostics=False)
    lcl_windows = ranking.evaluate_layer(
        station, lcl_dictionary, classified, "strict", "区域观测站"
    )

    # Rank LCL only in a non-destructive sensitivity copy.  Formal outputs stay
    # free of duplicated T/RH information.
    formal_windows = pd.read_csv(OUTPUT / "10_全部窗口双对照评价.csv")
    sensitivity_windows = pd.concat(
        [formal_windows[formal_windows["清洗版本"].eq("strict")], lcl_windows],
        ignore_index=True,
    )
    sensitivity_bases = ranking.aggregate_base_metrics(sensitivity_windows)
    sensitivity_rank, _ = ranking.topsis_rank(sensitivity_bases, ["基础指标"])
    lcl_rank = sensitivity_rank[
        sensitivity_rank["基础指标"].isin(["sta_lcl_mean", "sta_lcl_p90"])
    ][["基础指标", "排名", "TOPSIS贴近度", "显著性门控"]]
    result = lcl_windows.merge(lcl_rank, on="基础指标", how="left")

    corr_rows = []
    for hours in [24, 48, 72, 120, 168]:
        lcl = f"sta_lcl_mean_{hours}h"
        vpd = f"sta_VPD_mean_{hours}h"
        rh = f"sta_RH_mean_{hours}h"
        corr_rows.append(
            {
                "窗口小时": hours,
                "与同窗口VPD的Spearman": historical[[lcl, vpd]].corr(method="spearman").iloc[0, 1],
                "与同窗口RH的Spearman": historical[[lcl, rh]].corr(method="spearman").iloc[0, 1],
            }
        )
    result = result.merge(pd.DataFrame(corr_rows), on="窗口小时", how="left")
    result.to_csv(OUTPUT / "33_LCL补充敏感性.csv", index=False, encoding="utf-8-sig")
    return result


def weather_sequence() -> pd.DataFrame:
    station = pd.read_parquet(CACHE / "station_daily_10h_features_strict.parquet")
    station["Timestamp"] = pd.to_datetime(station["Timestamp"])
    classified, centers, _ = ranking.fit_weather_types("strict", n_components=6, diagnostics=False)
    keep = station[["Timestamp", "sta_tmean_24h", "sta_precip_sum_24h", "sta_RH_mean_24h"]].merge(
        classified, on="Timestamp", how="left"
    )
    start = EVENT_2026 - pd.Timedelta(days=39)
    sequence = keep[keep["Timestamp"].between(start, EVENT_2026)].copy()
    sequence["距事件截止日_天"] = (
        (sequence["Timestamp"] - EVENT_2026).dt.total_seconds() / 86400
    ).astype(int)
    sequence = sequence.rename(
        columns={
            "Timestamp": "日期时刻",
            "sta_tmean_24h": "24小时平均气温_C",
            "sta_precip_sum_24h": "24小时累计降水_mm",
            "sta_RH_mean_24h": "24小时平均相对湿度_pct",
        }
    )
    sequence.to_csv(OUTPUT / "34_2026灾前40天天气型演变.csv", index=False, encoding="utf-8-sig")
    return sequence


def cross_mechanism_2025() -> pd.DataFrame:
    station = pd.read_parquet(CACHE / "station_daily_05h_features_strict.parquet")
    station["Timestamp"] = pd.to_datetime(station["Timestamp"])
    event = station[station["Timestamp"].eq(EVENT_2025)]
    if event.empty:
        raise RuntimeError("2025-07-08 05:00端点不存在，无法进行跨机理核查")
    historical = station[station["Timestamp"].dt.year.isin(CONTROL_YEARS)].copy()
    controls = historical[ranking.seasonal_mask(historical["Timestamp"], EVENT_2025, 15)]

    indicators = {
        "空气温度与热量累积": "sta_pdd_72h",
        "土壤湿润状态": "sta_VWC_change_72h",
        "辐射与地表能量": "sta_netrad_mean_72h",
        "大气湿度与干湿状态": "sta_VPD_p90_72h",
        "边界层稳定结构": "sta_theta_gradient_mean_72h",
    }
    formal_factors = pd.read_csv(OUTPUT / "12_物理因素初排.csv")
    formal_factors = formal_factors[
        formal_factors["清洗版本"].eq("strict")
    ].set_index("物理因素")
    rows = []
    for factor, indicator in indicators.items():
        value = float(event[indicator].iloc[0])
        stats = ranking.robust_comparison(value, controls[indicator])
        direction = "偏高" if value > stats["对照中位数"] else "偏低"
        direction_2026 = str(formal_factors.loc[factor, "代表指标主导方向"])
        rows.append(
            {
                "物理因素": factor,
                "72小时代表指标": indicator,
                "2025事件端点": EVENT_2025,
                "2025事件值": value,
                "2021至2024同期中位数": stats["对照中位数"],
                "2025稳健Z": stats["稳健Z"],
                "2025历史百分位": stats["历史百分位"],
                "2025偏离方向": direction,
                "2026主事件方向": direction_2026,
                "方向一致": direction == direction_2026,
                "证据角色": "事后跨机理定性核查；不参与筛选、拟合、门控、权重或排序",
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT / "35_2025跨机理方向核查.csv", index=False, encoding="utf-8-sig")
    return result


def write_report(lcl: pd.DataFrame, sequence: pd.DataFrame, cross: pd.DataFrame) -> None:
    base_rows = lcl.drop_duplicates("基础指标").sort_values("排名")
    median_corr_vpd = float(lcl["与同窗口VPD的Spearman"].abs().median())
    median_corr_rh = float(lcl["与同窗口RH的Spearman"].abs().median())
    type_counts = (
        sequence.groupby(["天气型编号", "解释性名称"], dropna=False)
        .size()
        .sort_values(ascending=False)
    )
    event_type = int(sequence.loc[sequence["距事件截止日_天"].eq(0), "天气型编号"].iloc[0])
    last10 = sequence[sequence["距事件截止日_天"].ge(-9)]
    event_type_days = int(last10["天气型编号"].eq(event_type).sum())
    consistent = int(cross["方向一致"].sum())

    lines = [
        "# 第一问补充分析：LCL、天气型演变与2025跨机理核查",
        "",
        "## 1. LCL敏感性",
        "",
        "LCL由题目站点2米气温和相对湿度推得，仍属于区域边界层背景，不解释为源区云底实测。它与VPD和相对湿度共享原始信息，因此只进入“大气湿度与干湿状态”组，不单独占据一个主要因素名额。",
        "",
        f"历史背景中，LCL与同窗口VPD的绝对Spearman相关中位数为{median_corr_vpd:.3f}，与相对湿度的绝对相关中位数为{median_corr_rh:.3f}。",
        "",
        "| 基础指标 | 基础指标排名 | TOPSIS贴近度 | 门控 |",
        "|---|---:|---:|---|",
    ]
    for _, row in base_rows.iterrows():
        lines.append(
            f"| {row['基础指标']} | {int(row['排名'])} | {row['TOPSIS贴近度']:.3f} | {row['显著性门控']} |"
        )
    lines.extend(
        [
            "",
            "## 2. 2026事件前40天天气型演变",
            "",
            f"事件截止日属于天气型{event_type}；最后10天中有{event_type_days}天被分入同一类型。GMM只用于描述天气背景和选择公平对照，不解释为灾害机制。",
            "",
            "40天类型频数：" + "；".join(
                f"天气型{int(key[0])}（{key[1]}）{count}天"
                for key, count in type_counts.items()
            ) + "。",
            "",
            "## 3. 2025跨机理方向核查",
            "",
            f"五类因素中有{consistent}/5与2026主事件方向一致。该结果只说明指标在另一场不同机理灾害附近是否呈现相似背景，不构成训练、独立验证或模型准确率证据。无论一致与否均保留报告。",
            "",
            "## 使用边界",
            "",
            "- 2025年没有进入2021—2024历史背景、GMM训练、标准化、显著性门控、熵权、CRITIC或TOPSIS。",
            "- 2025事件采用北京时间05:30前的05:00最后完整小时；站点不等于灾害源区。",
            "- LCL与VPD、相对湿度共享气温和湿度输入；本数据中它与相对湿度高度相关，只能作为湿度因素的替代/佐证指标。",
            "- 不基于2025核查结果反向修改2026前五因素。",
        ]
    )
    (OUTPUT / "36_三项补充分析报告.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    lcl = lcl_sensitivity()
    sequence = weather_sequence()
    cross = cross_mechanism_2025()
    write_report(lcl, sequence, cross)
    print("supplementary checks completed")
    print(cross[["物理因素", "2025偏离方向", "2026主事件方向", "方向一致"]].to_string(index=False))


if __name__ == "__main__":
    main()
