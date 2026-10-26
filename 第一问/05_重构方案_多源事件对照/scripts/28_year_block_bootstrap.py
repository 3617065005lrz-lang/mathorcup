from __future__ import annotations

"""Year-block bootstrap for the formal Question 1 factor ranking.

The four historical years (2021--2024) are resampled as whole blocks.  Weather
types are held fixed at the formal six-component strict-QC solution so this
experiment isolates sensitivity to the short historical period rather than
mixing it with GMM relabelling uncertainty (already tested separately).
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import re

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
WORK_DIR = SCRIPT_DIR.parent
CACHE_DIR = WORK_DIR / "cache"
OUTPUT_DIR = WORK_DIR / "outputs"
YEARS = [2021, 2022, 2023, 2024]
N_BOOTSTRAP = 1000
RANDOM_SEED = 2026
CALENDAR_BAND = 15

spec = spec_from_file_location("event_ranking", SCRIPT_DIR / "05_run_event_control_ranking.py")
ranking = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(ranking)


def comparison(event_value: float, controls: np.ndarray) -> dict[str, float]:
    values = np.asarray(controls, dtype=float)
    values = values[np.isfinite(values)]
    if not np.isfinite(event_value) or len(values) < ranking.MIN_CONTROL_COUNT:
        return {
            "对照样本数": len(values), "对照中位数": np.nan, "稳健Z": np.nan,
            "稳健效应强度": np.nan, "历史百分位": np.nan, "百分位极端度": np.nan,
            "经验双侧P": np.nan, "蒙特卡洛双侧P": np.nan,
        }
    median = float(np.median(values))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.median(np.abs(values - median)) * 1.4826)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(values, ddof=1))
    z = (event_value - median) / scale if np.isfinite(scale) and scale > 1e-12 else np.nan
    less = int((values < event_value).sum())
    equal = int((values == event_value).sum())
    percentile = (less + 0.5 * equal) / len(values)
    event_deviation = abs(event_value - median)
    p_value = (1 + int((np.abs(values - median) >= event_deviation).sum())) / (len(values) + 1)
    return {
        "对照样本数": len(values),
        "对照中位数": median,
        "稳健Z": z,
        "稳健效应强度": abs(z) / (1 + abs(z)) if np.isfinite(z) else np.nan,
        "历史百分位": percentile,
        "百分位极端度": abs(2 * percentile - 1),
        "经验双侧P": p_value,
        # Exact empirical probability is used in the outer bootstrap. Running an
        # inner 1000-draw Monte Carlo would only add simulation noise and cost.
        "蒙特卡洛双侧P": p_value,
    }


def concatenate_years(groups: dict[int, np.ndarray], sampled: np.ndarray) -> np.ndarray:
    parts = [groups[int(year)] for year in sampled if int(year) in groups]
    return np.concatenate(parts) if parts else np.array([], dtype=float)


def prepare_layer(
    frame: pd.DataFrame,
    dictionary: pd.DataFrame,
    classified: pd.DataFrame,
    layer: str,
) -> list[dict[str, object]]:
    if layer == "区域观测站":
        date_column = "Timestamp"
        target = ranking.STATION_EVENT_ENDPOINT
        data = frame.merge(classified[["Timestamp", "天气型编号"]], on="Timestamp", how="left")
    else:
        date_column = "date"
        target = ranking.SOURCE_EVENT_ENDPOINT
        weather = classified[["Timestamp", "天气型编号"]].rename(columns={"Timestamp": "date"})
        data = frame.merge(weather, on="date", how="left")

    event_row = data[data[date_column].eq(target)]
    if event_row.empty:
        raise RuntimeError(f"event endpoint absent for {layer}")
    event_type = int(event_row["天气型编号"].iloc[0])
    historical = data[data[date_column].dt.year.isin(YEARS)].copy()
    calendar = historical[ranking.seasonal_mask(historical[date_column], target, CALENDAR_BAND)]
    regime, scope = ranking.choose_regime_controls(
        data, date_column, target, event_type, CALENDAR_BAND
    )

    prepared = []
    for _, item in dictionary.iterrows():
        feature = item["衍生指标"]
        if feature not in data.columns:
            continue
        event_value = float(pd.to_numeric(event_row[feature], errors="coerce").iloc[0])
        calendar_groups = {
            year: pd.to_numeric(
                calendar.loc[calendar[date_column].dt.year.eq(year), feature], errors="coerce"
            ).dropna().to_numpy(dtype=float)
            for year in YEARS
        }
        regime_groups = {
            year: pd.to_numeric(
                regime.loc[regime[date_column].dt.year.eq(year), feature], errors="coerce"
            ).dropna().to_numpy(dtype=float)
            for year in YEARS
        }
        prepared.append(
            {
                "清洗版本": "strict",
                "数据层": layer,
                "衍生指标": feature,
                "基础指标": re.sub(r"_\d+h$", "", feature),
                "因素组ID": item["因素组ID"],
                "物理因素": item["物理因素"],
                "指标中文名": item["指标中文名"],
                "对应原始字段": item["对应原始字段"],
                "窗口小时": int(item["窗口小时"]),
                "单位": item["单位"],
                "事件值": event_value,
                "事件天气型": event_type,
                "同天气型对照口径": scope,
                "calendar_groups": calendar_groups,
                "regime_groups": regime_groups,
            }
        )
    return prepared


def evaluate_prepared(prepared: list[dict[str, object]], sampled: np.ndarray) -> pd.DataFrame:
    rows = []
    for item in prepared:
        event_value = float(item["事件值"])
        calendar = comparison(event_value, concatenate_years(item["calendar_groups"], sampled))
        regime = comparison(event_value, concatenate_years(item["regime_groups"], sampled))
        row = {key: value for key, value in item.items() if not key.endswith("_groups")}
        row["偏离方向"] = "偏高" if event_value > calendar["对照中位数"] else "偏低"
        row.update({f"同期{k}": v for k, v in calendar.items()})
        row.update({f"同天气型{k}": v for k, v in regime.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    station_dictionary = pd.read_csv(OUTPUT_DIR / "07_站点衍生指标字典.csv")
    station_dictionary = station_dictionary.drop_duplicates(["清洗版本", "衍生指标"])
    source_dictionary = pd.read_csv(OUTPUT_DIR / "23_源区ERA5正式衍生指标字典.csv")
    station = pd.read_parquet(CACHE_DIR / "station_daily_10h_features_strict.parquet")
    source = pd.read_parquet(CACHE_DIR / "source_daily_features_official.parquet")
    classified, _, _ = ranking.fit_weather_types("strict", n_components=6, diagnostics=False)

    prepared = prepare_layer(
        station,
        station_dictionary[station_dictionary["清洗版本"].eq("strict")],
        classified,
        "区域观测站",
    )
    prepared += prepare_layer(source, source_dictionary, classified, "源区ERA5正式重算")

    rng = np.random.default_rng(RANDOM_SEED)
    samples = rng.choice(YEARS, size=(N_BOOTSTRAP, len(YEARS)), replace=True)
    result_parts = []
    for iteration, sampled in enumerate(samples, 1):
        windows = evaluate_prepared(prepared, sampled)
        bases = ranking.aggregate_base_metrics(windows)
        base_ranked, _ = ranking.topsis_rank(bases, ["基础指标"], entropy_share=0.5)
        factors = ranking.aggregate_factors(base_ranked)
        factor_ranked, _ = ranking.topsis_rank(factors, ["因素组ID"], entropy_share=0.5)
        factor_ranked.insert(0, "自助迭代", iteration)
        factor_ranked.insert(1, "抽样年份", "+".join(map(str, sampled)))
        result_parts.append(factor_ranked)
        if iteration % 100 == 0:
            print(f"year-block bootstrap {iteration}/{N_BOOTSTRAP}", flush=True)

    all_results = pd.concat(result_parts, ignore_index=True)
    all_results.to_csv(OUTPUT_DIR / "29_年份整块自助法全部结果.csv", index=False, encoding="utf-8-sig")
    rows = []
    for factor, group in all_results.groupby("物理因素"):
        rows.append(
            {
                "物理因素": factor,
                "进入前五概率": float((group["排名"] <= 5).mean()),
                "排名中位数": float(group["排名"].median()),
                "排名95%区间下限": float(group["排名"].quantile(0.025)),
                "排名95%区间上限": float(group["排名"].quantile(0.975)),
                "TOPSIS中位数": float(group["TOPSIS贴近度"].median()),
                "TOPSIS_95%下限": float(group["TOPSIS贴近度"].quantile(0.025)),
                "TOPSIS_95%上限": float(group["TOPSIS贴近度"].quantile(0.975)),
                "门控通过概率": float((group["代表指标显著性门控"] != "未通过").mean()),
                "最常见代表指标": group["代表基础指标"].value_counts().index[0],
                "主导方向": group["代表指标主导方向"].value_counts().index[0],
            }
        )
    summary = pd.DataFrame(rows).sort_values(
        ["进入前五概率", "排名中位数", "TOPSIS中位数"], ascending=[False, True, False]
    ).reset_index(drop=True)
    summary.insert(0, "稳健顺序", np.arange(1, len(summary) + 1))
    summary.to_csv(OUTPUT_DIR / "30_年份整块自助法汇总.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# 历史年份整块自助法排名区间",
        "",
        f"对2021—2024四个历史年进行{N_BOOTSTRAP}次有放回整块抽样。GMM天气型固定为正式6类模型，避免将年份不确定性和聚类重标记混在同一实验中。",
        "",
        "| 顺序 | 物理因素 | 前五概率 | 排名中位数 | 排名95%区间 | TOPSIS中位数 | TOPSIS 95%区间 | 门控通过率 | 方向 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {int(row['稳健顺序'])} | {row['物理因素']} | {row['进入前五概率']:.1%} | "
            f"{row['排名中位数']:.1f} | {row['排名95%区间下限']:.0f}—{row['排名95%区间上限']:.0f} | "
            f"{row['TOPSIS中位数']:.3f} | {row['TOPSIS_95%下限']:.3f}—{row['TOPSIS_95%上限']:.3f} | "
            f"{row['门控通过概率']:.1%} | {row['主导方向']} |"
        )
    lines.extend(
        [
            "",
            "排名区间反映四个历史年份有限所造成的不确定性，不是总体参数的经典置信区间。",
        ]
    )
    (OUTPUT_DIR / "31_年份整块自助法报告.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
