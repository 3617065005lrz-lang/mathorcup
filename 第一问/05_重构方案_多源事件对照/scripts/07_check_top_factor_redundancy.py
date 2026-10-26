from __future__ import annotations

"""检查稳健前五代表指标是否只是同一信号的重复表达。"""

from itertools import combinations
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
WORK_DIR = SCRIPT_DIR.parent
CACHE_DIR = WORK_DIR / "cache"
OUTPUT_DIR = WORK_DIR / "outputs"
WINDOWS = [24, 48, 72, 120, 168]


def in_season(timestamp: pd.Timestamp) -> bool:
    anchor = pd.Timestamp(2000, 8, 26)
    current = pd.Timestamp(2000, timestamp.month, timestamp.day)
    return abs((current - anchor).days) <= 30


def main() -> None:
    summary = pd.read_csv(OUTPUT_DIR / "19_物理因素稳健性汇总.csv").head(5)
    representatives = dict(zip(summary["物理因素"], summary["最常见代表指标"]))
    station = pd.read_parquet(CACHE_DIR / "station_daily_10h_features_strict.parquet")
    source = pd.read_parquet(CACHE_DIR / "source_daily_features_official.parquet").rename(
        columns={"date": "Timestamp"}
    )
    history = station.merge(source, on="Timestamp", how="left", suffixes=("", "_source"))
    history = history[
        history["Timestamp"].dt.year.isin([2021, 2022, 2023, 2024])
        & history["Timestamp"].map(in_season)
    ].copy()

    rows = []
    for window in WINDOWS:
        available = {
            factor: f"{base}_{window}h"
            for factor, base in representatives.items()
            if f"{base}_{window}h" in history.columns
        }
        for left, right in combinations(available, 2):
            pair = history[[available[left], available[right]]].dropna()
            rows.append(
                {
                    "窗口小时": window,
                    "因素A": left,
                    "代表指标A": available[left],
                    "因素B": right,
                    "代表指标B": available[right],
                    "有效样本数": len(pair),
                    "Spearman相关系数": pair.corr(method="spearman").iloc[0, 1] if len(pair) >= 12 else float("nan"),
                }
            )
    result = pd.DataFrame(rows)
    result["绝对相关"] = result["Spearman相关系数"].abs()
    result.to_csv(OUTPUT_DIR / "21_稳健前五冗余诊断.csv", index=False, encoding="utf-8-sig")
    maximum = result.loc[result["绝对相关"].idxmax()]
    high = result[result["绝对相关"] >= 0.80]
    lines = [
        "# 稳健前五代表指标冗余诊断",
        "",
        "使用2021—2024年8月26日前后30天的站点—正式源区联合日端点，分别在24、48、72、120和168小时尺度计算代表指标之间的Spearman相关。",
        "",
        f"- 最大绝对相关为{maximum['绝对相关']:.3f}：{maximum['因素A']}与{maximum['因素B']}，窗口{int(maximum['窗口小时'])}小时。",
        f"- |rho|≥0.80的组合数为{len(high)}。",
        "- 本检查只识别统计冗余，不把相关性解释为因果关系。",
        "- 源区代表指标已纳入本次相关性诊断。",
    ]
    if high.empty:
        lines.append("- 当前未发现达到0.80阈值的严重重复，因此暂不因相关性合并这五个物理因素。")
    else:
        lines.append("- 存在严重重复，论文中应合并对应因素或仅保留物理解释更直接的一项。")
    (OUTPUT_DIR / "22_稳健前五冗余诊断报告.md").write_text("\n".join(lines), encoding="utf-8")
    print("冗余诊断完成：", lines[-1])


if __name__ == "__main__":
    main()
