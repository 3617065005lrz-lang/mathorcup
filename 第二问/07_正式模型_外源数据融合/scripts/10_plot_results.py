from __future__ import annotations

"""Render final Question 2 figures from cached outputs only.

This script never rebuilds source caches or reruns models.  It writes PNG only
and intentionally contains no flowchart.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent
CONFIG = json.loads((MODEL_DIR / "config" / "model_config.json").read_text(encoding="utf-8"))
PROCESSED = MODEL_DIR / "processed"
OUTPUTS = MODEL_DIR / "outputs"
FIGURES = MODEL_DIR / "figures"

plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def temporal_percentiles() -> None:
    table = pd.read_csv(OUTPUTS / "12_时间触发临界条件与稳健性.csv").sort_values("risk_percentile")
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = np.where(table["risk_percentile"] >= 0.8, "#d95f02", "#2b8cbe")
    ax.barh(table["name_cn"], table["risk_percentile"], color=colors)
    ax.axvline(0.90, color="#b2182b", linestyle="--", linewidth=1.2, label="历史90%经验关注带")
    ax.axvline(0.50, color="#777777", linestyle=":", linewidth=1.0)
    ax.set_xlim(0, 1)
    ax.set_xlabel("2026事件值在2021—2024同期背景中的风险百分位")
    ax.set_title("事件前关键触发指标的历史相对位置")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "图1_时间指标历史百分位.png")


def eti_uncertainty() -> None:
    mc = pd.read_parquet(PROCESSED / "05_ETI参数蒙特卡洛结果.parquet")
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(
        min(mc["event_eti_72h_mm"].min(), mc["background_q95_eti_72h_mm"].min()),
        max(mc["event_eti_72h_mm"].max(), mc["background_q95_eti_72h_mm"].max()),
        35,
    )
    ax.hist(mc["event_eti_72h_mm"], bins=bins, alpha=0.65, color="#2b8cbe", label="事件ETI 72 h")
    ax.hist(mc["background_q95_eti_72h_mm"], bins=bins, alpha=0.55, color="#d95f02", label="历史同期95%阈值")
    ax.set_xlabel("等效融水指标（mm w.e.）")
    ax.set_ylabel("Monte Carlo频数")
    ax.set_title("ETI参数不确定性：事件值与历史高危带")
    ax.legend(frameon=False)
    save(fig, "图2_ETI蒙特卡洛不确定性.png")


def add_source_box(ax: plt.Axes) -> None:
    aoi = CONFIG["source_aoi_wgs84"]
    ax.add_patch(
        Rectangle(
            (aoi["west"], aoi["south"]),
            aoi["east"] - aoi["west"],
            aoi["north"] - aoi["south"],
            fill=False,
            edgecolor="#ffdd55",
            linewidth=1.5,
            label="重建源区AOI",
        )
    )


def spatial_maps() -> None:
    spatial = pd.read_parquet(PROCESSED / "08_事件时空综合风险格网.parquet")
    fig, ax = plt.subplots(figsize=(8, 6))
    points = ax.scatter(
        spatial["longitude_wgs84"],
        spatial["latitude_wgs84"],
        c=spatial["static_mechanism_score"],
        s=4,
        cmap="viridis",
        linewidths=0,
    )
    add_source_box(ax)
    fig.colorbar(points, ax=ax, label="静态机制易发性分数")
    ax.set_xlabel("经度（°E）")
    ax.set_ylabel("纬度（°N）")
    ax.set_title("地形—物源—沟道连通性空间易发性")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=True, loc="upper right")
    save(fig, "图3_空间易发性机制分数.png")

    category_order = ["低", "中", "高", "极高"]
    category_colors = {"低": "#d9d9d9", "中": "#fee08b", "高": "#f46d43", "极高": "#a50026"}
    fig, ax = plt.subplots(figsize=(8, 6))
    for category in category_order:
        part = spatial.loc[spatial["event_risk_level_upgraded"].eq(category)]
        if not part.empty:
            ax.scatter(
                part["longitude_wgs84"],
                part["latitude_wgs84"],
                s=4,
                color=category_colors[category],
                linewidths=0,
                label=category,
            )
    add_source_box(ax)
    ax.set_xlabel("经度（°E）")
    ax.set_ylabel("纬度（°N）")
    ax.set_title("2026事件前统一时空风险等级")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=True, ncol=2, loc="upper right")
    save(fig, "图4_2026事件综合风险等级.png")


def model_comparison() -> None:
    table = pd.read_csv(OUTPUTS / "18_空间模型源区合理性检验.csv")
    labels = {
        "static_equal_indicator_baseline": "等指标基线",
        "static_mechanism_score": "机制约束主模型",
        "historical_logistic_probability": "历史灾点Logistic",
    }
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for score, part in table.groupby("score"):
        ax.plot(part["top_share"], part["source_aoi_coverage"], marker="o", label=labels.get(score, score))
    ax.set_xlabel("划为高值区的空间比例")
    ax.set_ylabel("源区AOI覆盖率")
    ax.set_title("不同空间模型对源区的覆盖能力")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    save(fig, "图5_空间基线与升级模型对照.png")


def gfs_skill() -> None:
    skill = pd.read_csv(OUTPUTS / "27_GFS分时效个例核查.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plots = [
        ("temperature_mae_c", "气温MAE（°C）", "#2b8cbe"),
        ("precipitation_mae_3h_mm", "3小时降水MAE（mm）", "#41ab5d"),
        ("shortwave_mae_w_m2", "短波辐射MAE（W/m²）", "#d95f02"),
    ]
    for ax, (column, ylabel, color) in zip(axes, plots):
        ax.bar(skill["lead_band"], skill[column], color=color)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=35)
    fig.suptitle("GFS事件周不同预报时效的个例误差", y=1.03)
    save(fig, "图6_GFS分时效个例误差.png")


def dynamic_branches() -> None:
    table = pd.read_csv(OUTPUTS / "21_事件动态分支评分.csv")
    selected = table.loc[
        table["component"].isin(["thermal", "short_rainfall", "antecedent_moisture", "rain_moisture", "dynamic_upgraded"])
    ].copy()
    names = {
        "thermal": "热力冰冻圈",
        "short_rainfall": "短时降水",
        "antecedent_moisture": "前期湿润",
        "rain_moisture": "降雨—水分",
        "dynamic_upgraded": "综合动态触发",
    }
    selected["label"] = selected["component"].map(names)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(selected["label"], selected["event_score_0_1"], color=["#d95f02", "#2b8cbe", "#41ab5d", "#756bb1", "#b2182b"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("历史相对评分（0—1）")
    ax.set_title("2026事件动态触发分支分解")
    ax.tick_params(axis="x", rotation=20)
    save(fig, "图7_动态触发分支分解.png")


def main() -> None:
    temporal_percentiles()
    eti_uncertainty()
    spatial_maps()
    model_comparison()
    gfs_skill()
    dynamic_branches()
    print(f"PNG figures saved to {FIGURES}")


if __name__ == "__main__":
    main()
