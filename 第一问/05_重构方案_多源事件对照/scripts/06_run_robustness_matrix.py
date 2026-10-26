from __future__ import annotations

"""对第一问物理因素排序执行预注册式稳健性矩阵。

变化维度：严格/扩展清洗、GMM天气型数、同期日期带、时间窗口集合、
熵权—CRITIC组合比例。输出的是进入前五的频率和排名区间，不反向挑选
“最好看”的单一场景。
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
WORK_DIR = SCRIPT_DIR.parent
CACHE_DIR = WORK_DIR / "cache"
OUTPUT_DIR = WORK_DIR / "outputs"

spec = spec_from_file_location("event_ranking", SCRIPT_DIR / "05_run_event_control_ranking.py")
ranking = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(ranking)

GMM_COMPONENT_OPTIONS = [4, 5, 6]
CALENDAR_BAND_OPTIONS = [7, 15, 30]
WINDOW_SETS = {
    "全部窗口": [24, 48, 72, 120, 168],
    "短中期窗口": [24, 48, 72],
    "中长期窗口": [72, 120, 168],
}
ENTROPY_SHARE_OPTIONS = [0.0, 0.5, 1.0]
SOURCE_OPTIONS = {
    "中心格点_动态递减率_4991m": "source_daily_features_official.parquet",
    "3x3平均_动态递减率_4991m": "source_daily_features_official_mean3.parquet",
    "中心格点_固定6.5Kkm_4991m": "source_daily_features_official_fixed65.parquet",
    "中心格点_动态递减率_5300m": "source_daily_features_official_elev5300.parquet",
}


def main() -> None:
    station_dictionary = pd.read_csv(OUTPUT_DIR / "07_站点衍生指标字典.csv")
    station_dictionary = station_dictionary.drop_duplicates(["清洗版本", "衍生指标"])
    source_dictionary = pd.read_csv(OUTPUT_DIR / "23_源区ERA5正式衍生指标字典.csv")

    scenario_rows = []
    scenario_id = 0
    for mode in ["strict", "expanded"]:
        station = pd.read_parquet(CACHE_DIR / f"station_daily_10h_features_{mode}.parquet")
        for components in GMM_COMPONENT_OPTIONS:
            classified, _, _ = ranking.fit_weather_types(
                mode, n_components=components, diagnostics=False
            )
            for band in CALENDAR_BAND_OPTIONS:
                station_windows = ranking.evaluate_layer(
                    station,
                    station_dictionary[station_dictionary["清洗版本"].eq(mode)],
                    classified,
                    mode,
                    "区域观测站",
                    calendar_band_days=band,
                )
                for source_name, source_file in SOURCE_OPTIONS.items():
                    source = pd.read_parquet(CACHE_DIR / source_file)
                    source_windows = ranking.evaluate_layer(
                        source,
                        source_dictionary,
                        classified,
                        mode,
                        "源区ERA5正式重算",
                        calendar_band_days=band,
                    )
                    all_windows = pd.concat([station_windows, source_windows], ignore_index=True)
                    for window_name, hours in WINDOW_SETS.items():
                        subset = all_windows[all_windows["窗口小时"].isin(hours)].copy()
                        bases = ranking.aggregate_base_metrics(subset)
                        for entropy_share in ENTROPY_SHARE_OPTIONS:
                            scenario_id += 1
                            base_ranked, _ = ranking.topsis_rank(
                                bases, ["基础指标"], entropy_share=entropy_share
                            )
                            factors = ranking.aggregate_factors(base_ranked)
                            factor_ranked, _ = ranking.topsis_rank(
                                factors, ["因素组ID"], entropy_share=entropy_share
                            )
                            factor_ranked.insert(0, "场景编号", scenario_id)
                            factor_ranked.insert(1, "源区空间与订正", source_name)
                            factor_ranked.insert(2, "GMM天气型数", components)
                            factor_ranked.insert(3, "同期日期带天数", band)
                            factor_ranked.insert(4, "窗口集合", window_name)
                            factor_ranked.insert(5, "熵权占比", entropy_share)
                            scenario_rows.append(factor_ranked)

    scenarios = pd.concat(scenario_rows, ignore_index=True)
    total_scenarios = scenarios["场景编号"].nunique()
    summary_rows = []
    for factor, group in scenarios.groupby("物理因素"):
        top5 = group["排名"] <= 5
        representative = group["代表基础指标"].value_counts()
        direction = group["代表指标主导方向"].value_counts()
        gate = group["代表指标显著性门控"].value_counts()
        summary_rows.append(
            {
                "物理因素": factor,
                "总场景数": total_scenarios,
                "进入前五次数": int(top5.sum()),
                "进入前五比例": float(top5.mean()),
                "排名中位数": float(group["排名"].median()),
                "最好排名": int(group["排名"].min()),
                "最差排名": int(group["排名"].max()),
                "平均TOPSIS贴近度": float(group["TOPSIS贴近度"].mean()),
                "最常见代表指标": representative.index[0],
                "代表指标一致比例": float(representative.iloc[0] / len(group)),
                "主导方向": direction.index[0],
                "方向一致比例": float(direction.iloc[0] / len(group)),
                "最常见显著性门控": gate.index[0],
                "通过显著性门控比例": float((group["代表指标显著性门控"] != "未通过").mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["进入前五比例", "排名中位数", "最差排名", "平均TOPSIS贴近度"],
        ascending=[False, True, True, False],
    ).reset_index(drop=True)
    summary.insert(0, "稳健顺序", range(1, len(summary) + 1))

    scenarios.to_csv(OUTPUT_DIR / "18_稳健性矩阵全部场景.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "19_物理因素稳健性汇总.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# 第一问物理因素排序稳健性矩阵",
        "",
        f"共运行{total_scenarios}个预设场景：2种清洗口径 × 3种GMM天气型数 × 3种日期带 × 4种源区空间/高程订正 × 3种窗口集合 × 3种熵权占比。",
        "",
        "## 解释规则",
        "",
        "- 进入前五比例是主判断；排名中位数和最差排名用于区分同频因素。",
        "- 代表指标可能随场景变化，但论文结论写物理因素，并附最常见代表指标及其原始字段映射。",
        "- 不根据结果删除不利场景，也不把TOPSIS贴近度解释为因果贡献率。",
        "- 源区ERA5-Land已使用本项目正式下载数据重算；空间支持和高程订正另列扩展敏感性。",
        "",
        "## 稳健性结果",
        "",
        "| 稳健顺序 | 物理因素 | 前五比例 | 门控通过率 | 排名中位数 | 排名范围 | 最常见代表指标 | 代表一致性 | 主导方向 | 方向一致性 |",
        "|---:|---|---:|---:|---:|---:|---|---:|---|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {int(row['稳健顺序'])} | {row['物理因素']} | {row['进入前五比例']:.1%} | {row['通过显著性门控比例']:.1%} | "
            f"{row['排名中位数']:.1f} | {int(row['最好排名'])}—{int(row['最差排名'])} | "
            f"{row['最常见代表指标']} | {row['代表指标一致比例']:.1%} | {row['主导方向']} | {row['方向一致比例']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## 论文使用门槛",
            "",
            "- 建议把前五比例≥80%的因素称为‘稳健主要关联因素’。",
            "- 前五比例50%—80%的因素称为‘次稳健关联因素’，放在敏感性讨论中。",
            "- 低于50%的因素不得仅凭主场景排名写入最终前五。",
            "- 上述80%/50%是本研究预先规定的稳定性分级线，不是题目给定阈值，也不是自然灾害行业阈值。",
        ]
    )
    (OUTPUT_DIR / "20_稳健性矩阵报告.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"完成{total_scenarios}个场景。稳健前五：")
    print(summary.head(5)[["稳健顺序", "物理因素", "进入前五比例", "排名中位数", "最常见代表指标"]].to_string(index=False))


if __name__ == "__main__":
    main()
