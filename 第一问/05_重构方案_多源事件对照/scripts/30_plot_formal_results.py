from __future__ import annotations

"""Render final Question 1 figures from plot_cache only.

Run script 29 once after model results change.  Re-running this script never
touches raw data or feature caches.  Output format is PNG only.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "plot_cache"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

COLORS = {
    "blue": "#2878B5",
    "orange": "#F28E2B",
    "green": "#59A14F",
    "red": "#E15759",
    "gray": "#7F7F7F",
}


def setup() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, filename: str) -> None:
    fig.savefig(FIGURES / filename, format="png", facecolor="white")
    plt.close(fig)


def plot_main_ranking() -> None:
    data = pd.read_csv(CACHE / "fig01_主场景因素排序.csv").sort_values("排名", ascending=False)
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    colors = [COLORS["orange"] if r <= 5 else "#BFC7D5" for r in data["排名"]]
    ax.barh(data["物理因素"], data["TOPSIS贴近度"], color=colors)
    for i, value in enumerate(data["TOPSIS贴近度"]):
        ax.text(value + 0.012, i, f"{value:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, max(1.0, data["TOPSIS贴近度"].max() + 0.10))
    ax.set_xlabel("TOPSIS贴近度（综合关联评分）")
    ax.set_title("2026年事件主场景物理因素排序")
    save(fig, "图1_主场景物理因素排序.png")


def plot_stability() -> None:
    data = pd.read_csv(CACHE / "fig02_因素稳健性对照.csv")
    data = data.sort_values("进入前五比例", ascending=True)
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(9.5, 6.8))
    ax.scatter(data["进入前五比例"] * 100, y - 0.12, s=45, color=COLORS["blue"], label="648场景")
    ax.scatter(data["进入前五概率"] * 100, y + 0.12, s=45, marker="s", color=COLORS["orange"], label="年份整块自助法")
    ax.axvline(50, color=COLORS["gray"], linestyle="--", linewidth=1)
    ax.set_yticks(y, data["物理因素"])
    ax.set_xlim(-2, 102)
    ax.set_xlabel("进入前五比例（%）")
    ax.set_title("两类稳健性检验下的因素入选五频率")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "图2_因素稳健性对照.png")


def plot_imerg() -> None:
    data = pd.read_csv(CACHE / "fig03_IMERG累计降水核验.csv").sort_values("窗口小时")
    x = np.arange(len(data))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.4, 5.3))
    ax.bar(x - width / 2, data["事件值_mm"], width, color=COLORS["blue"], label="2026事件前")
    ax.bar(x + width / 2, data["历史中位数_mm"], width, color="#BFC7D5", label="2021—2024同期中位数")
    ax.set_xticks(x, [f"{int(v)} h" for v in data["窗口小时"]])
    ax.set_ylabel("累计降水（mm）")
    ax.set_title("IMERG对事件前累计降水的独立核验")
    ax.legend(frameon=False)
    save(fig, "图3_IMERG累计降水核验.png")


def plot_gmm() -> None:
    data = pd.read_csv(CACHE / "fig04_GMM稳定性诊断.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3))
    for label, group in data.groupby("清洗版本"):
        axes[0].plot(group["天气型数"], group["BIC均值"], marker="o", label=label)
        axes[1].plot(group["天气型数"], group["聚类ARI均值"], marker="o", label=label)
    axes[0].axvline(6, color=COLORS["orange"], linestyle="--", linewidth=1)
    axes[1].axvline(6, color=COLORS["orange"], linestyle="--", linewidth=1)
    axes[0].set(title="拟合优度", xlabel="天气型数", ylabel="BIC均值（越低越好）")
    axes[1].set(title="随机初值稳定性", xlabel="天气型数", ylabel="ARI均值（越高越好）")
    axes[0].legend(frameon=False)
    fig.suptitle("GMM天气型数量选择诊断", y=1.02)
    fig.tight_layout()
    save(fig, "图4_GMM天气型稳定性.png")


def plot_precursor_trajectory() -> None:
    data = pd.read_csv(CACHE / "fig06_前五因素灾前演化.csv")
    order = [
        "空气温度与热量累积", "土壤湿润状态", "辐射与地表能量",
        "大气湿度与干湿状态", "边界层稳定结构",
    ]
    colors = ["#D55E00", "#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    for factor, color in zip(order, colors):
        group = data[data["物理因素"].eq(factor)].sort_values("距截止时刻_小时")
        ax.plot(group["距截止时刻_小时"], group["稳健Z"], marker="o", linewidth=1.8, color=color, label=factor)
    ax.axhline(0, color=COLORS["gray"], linewidth=1)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("距2026年8月26日10:00的时间（h）")
    ax.set_ylabel("相对2021—2024同期的稳健Z")
    ax.set_title("最终五因素在事件前的72小时尺度异常演化")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    save(fig, "图5_前五因素灾前演化.png")


def main() -> None:
    setup()
    required = list(CACHE.glob("fig*.csv"))
    if len(required) < 5:
        raise FileNotFoundError("绘图缓存不完整，请先运行 scripts/29_prepare_formal_plot_cache.py")
    plot_main_ranking()
    plot_stability()
    plot_imerg()
    plot_gmm()
    plot_precursor_trajectory()
    print(f"PNG figures written to: {FIGURES}")


if __name__ == "__main__":
    main()
