from __future__ import annotations

"""Prepare small, stable tables used only by the formal plotting script.

This script deliberately separates model outputs from rendering.  It does not
draw figures and does not revisit raw meteorological files.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs"
CACHE = ROOT / "plot_cache"
CACHE.mkdir(exist_ok=True)


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUT / name)


def main() -> None:
    main_rank = read("12_物理因素初排.csv")
    robust = read("19_物理因素稳健性汇总.csv")
    year_boot = read("30_年份整块自助法汇总.csv")
    imerg = read("25_IMERG独立降水核验.csv")
    gmm = read("27_GMM随机初值稳定性.csv")

    # Figure 1: formal main-scene ranking, one row per physical factor.
    cols = [
        "排名", "物理因素", "代表基础指标", "代表指标主导方向",
        "代表指标显著性门控", "TOPSIS贴近度",
    ]
    formal_main = main_rank[main_rank["清洗版本"].eq("strict")].copy()
    if len(formal_main) != 12:
        raise RuntimeError(f"正式严格清洗主场景应有12个物理因素，实际为{len(formal_main)}个")
    formal_main[cols].sort_values("排名").to_csv(
        CACHE / "fig01_主场景因素排序.csv", index=False, encoding="utf-8-sig"
    )

    # Figure 2: two complementary stability tests aligned by physical factor.
    robust_cols = [
        "物理因素", "进入前五比例", "排名中位数", "通过显著性门控比例",
    ]
    boot_cols = [
        "物理因素", "进入前五概率", "排名中位数", "排名95%区间下限",
        "排名95%区间上限", "门控通过概率",
    ]
    stability = robust[robust_cols].merge(
        year_boot[boot_cols], on="物理因素", how="outer", suffixes=("_648场景", "_年份自助")
    )
    stability.to_csv(
        CACHE / "fig02_因素稳健性对照.csv", index=False, encoding="utf-8-sig"
    )

    # Figure 3: independent precipitation check; cumulative windows only.
    rainfall = imerg[imerg["指标"].eq("累计降水")].sort_values("窗口小时")
    rainfall.to_csv(
        CACHE / "fig03_IMERG累计降水核验.csv", index=False, encoding="utf-8-sig"
    )

    # Figure 4: GMM stability diagnostics.
    gmm.sort_values(["清洗版本", "天气型数"]).to_csv(
        CACHE / "fig04_GMM稳定性诊断.csv", index=False, encoding="utf-8-sig"
    )

    # Final five follow one rule: strong evidence in the main scene, recurring
    # support in both robustness experiments, and no duplicated long-window
    # temperature/soil-temperature evidence.  Soil temperature remains an
    # important corroborating indicator but is not counted as an independent
    # fifth factor because of its collinearity with accumulated heat.
    final_five = pd.DataFrame(
        [
            [1, "空气温度与热量累积", "核心因素", "偏高", "热力积累与冰体软化/融化背景"],
            [2, "土壤湿润状态", "核心因素", "偏低", "区域陆面偏干，反证强降雨型触发"],
            [3, "辐射与地表能量", "核心因素", "偏高", "提供持续热量输入"],
            [4, "大气湿度与干湿状态", "次稳健因素", "偏高", "高VPD表征暖干边界层背景"],
            [5, "边界层稳定结构", "候选第五因素", "偏高", "补充近地层热力结构且避免重复计算地温"],
        ],
        columns=["建议顺序", "物理因素", "证据等级", "事件偏离方向", "论文物理解释"],
    )
    final_five.to_csv(
        OUTPUT / "32_最终五因素建议.csv", index=False, encoding="utf-8-sig"
    )
    final_five.to_csv(
        CACHE / "fig05_最终五因素.csv", index=False, encoding="utf-8-sig"
    )

    # Figure 5: daily evolution of a common 72 h indicator for each selected
    # factor.  Each event value is robust-standardised against 2021--2024
    # values lying within +/-15 calendar days of the corresponding month/day.
    station = pd.read_parquet(ROOT / "cache" / "station_daily_10h_features_strict.parquet")
    station["Timestamp"] = pd.to_datetime(station["Timestamp"])
    historical = station[station["Timestamp"].dt.year.isin([2021, 2022, 2023, 2024])].copy()
    event = station[
        station["Timestamp"].between("2026-08-19 10:00", "2026-08-26 10:00")
    ].copy()
    indicators = {
        "空气温度与热量累积": "sta_pdd_72h",
        "土壤湿润状态": "sta_VWC_change_72h",
        "辐射与地表能量": "sta_netrad_mean_72h",
        "大气湿度与干湿状态": "sta_VPD_p90_72h",
        "边界层稳定结构": "sta_theta_gradient_mean_72h",
    }
    trajectory_rows: list[dict[str, object]] = []
    reference_year = 2000
    for _, row in event.iterrows():
        target = row["Timestamp"]
        target_ref = pd.Timestamp(reference_year, target.month, target.day)
        hist_ref = pd.to_datetime(
            {
                "year": np.full(len(historical), reference_year),
                "month": historical["Timestamp"].dt.month,
                "day": historical["Timestamp"].dt.day,
            }
        )
        controls = historical[(hist_ref - target_ref).abs().dt.days <= 15]
        for factor, indicator in indicators.items():
            values = pd.to_numeric(controls[indicator], errors="coerce").dropna().to_numpy()
            median = float(np.median(values))
            q25, q75 = np.quantile(values, [0.25, 0.75])
            scale = float((q75 - q25) / 1.349)
            if not np.isfinite(scale) or scale <= 1e-12:
                scale = float(np.std(values, ddof=1))
            value = float(row[indicator])
            trajectory_rows.append(
                {
                    "时刻": target,
                    "距截止时刻_小时": int((target - pd.Timestamp("2026-08-26 10:00")).total_seconds() / 3600),
                    "物理因素": factor,
                    "72小时指标": indicator,
                    "事件值": value,
                    "历史同期中位数": median,
                    "稳健Z": (value - median) / scale if scale > 0 else np.nan,
                    "历史样本数": len(values),
                }
            )
    pd.DataFrame(trajectory_rows).to_csv(
        CACHE / "fig06_前五因素灾前演化.csv", index=False, encoding="utf-8-sig"
    )

    # Figure 6: 40-day weather-regime evolution.  Script 32 creates this
    # supplementary result without changing the formal ranking.
    weather_sequence = read("34_2026灾前40天天气型演变.csv")
    weather_sequence.to_csv(
        CACHE / "fig07_灾前40天天气型演变.csv", index=False, encoding="utf-8-sig"
    )

    print(f"plot cache prepared: {CACHE}")
    print(f"tables: {len(list(CACHE.glob('*.csv')))}")


if __name__ == "__main__":
    main()
