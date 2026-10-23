from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from plot_style import COLORS, configure_style, polish_axes, save_figure


SCRIPT_DIR = Path(__file__).resolve().parent
VIS_DIR = SCRIPT_DIR.parent
CACHE_DIR = VIS_DIR / "plot_cache"
FIGURE_DIR = VIS_DIR / "figures"


def load_cache() -> dict[str, object]:
    manifest_path = CACHE_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "尚未生成绘图缓存。请先运行 01_prepare_plot_cache.py，再运行绘图脚本。"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("excluded_validation_year") != 2025:
        raise RuntimeError("缓存清单未确认隔离2025年，停止绘图。")
    return {
        "manifest": manifest,
        "base": pd.read_csv(CACHE_DIR / "base_factor_ranking.csv"),
        "stability": pd.read_csv(CACHE_DIR / "factor_stability.csv"),
        "top5": pd.read_csv(CACHE_DIR / "top5_factor_metadata.csv"),
        "grid": pd.read_csv(CACHE_DIR / "top5_sensitivity_grid.csv"),
        "trajectory": pd.read_parquet(CACHE_DIR / "top5_matched_trajectory.parquet"),
        "distribution": pd.read_parquet(
            CACHE_DIR / "top5_event_control_distribution.parquet"
        ),
    }


def figure01_trajectories(cache: dict[str, object], dpi: int) -> None:
    top5 = cache["top5"]
    data = cache["trajectory"]
    fig, axes = plt.subplots(len(top5), 1, figsize=(7.0, 10.2), sharex=True)
    for ax, (_, meta) in zip(axes, top5.iterrows()):
        part = data[data["物理因素"] == meta["物理因素"]].sort_values("相对小时")
        x = part["相对天"].to_numpy(float)
        event = part["事件值"].to_numpy(float)
        median = part["历史中位数"].to_numpy(float)
        q1 = part["历史Q1"].to_numpy(float)
        q3 = part["历史Q3"].to_numpy(float)
        ax.fill_between(x, q1, q3, color="#D9D9D9", alpha=0.75, label="2021—2024同期IQR")
        ax.plot(x, median, color=COLORS["mid"], linewidth=1.2, label="历史同期中位数")
        ax.plot(x, event, color=COLORS["navy"], linewidth=1.7, label="2026事件前")
        ax.axvline(0, color=COLORS["red"], linestyle="--", linewidth=1.0)
        ax.set_ylabel(meta["单位"])
        ax.set_title(f"{meta['物理因素']}：{meta['指标中文名']}", loc="left", fontweight="bold")
        polish_axes(ax, "y")
    axes[0].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.34))
    axes[-1].set_xlabel("距事件前的时间（天；0为2026-08-26 10:00）")
    fig.suptitle("2026年事件前稳定前五因素的七日演化", fontsize=14, fontweight="bold", y=0.995)
    fig.subplots_adjust(hspace=0.48, top=0.94)
    save_figure(fig, FIGURE_DIR, "fig01_top5_trajectories", dpi)


def figure02_stability(cache: dict[str, object], dpi: int) -> None:
    data = cache["stability"].copy().sort_values(
        ["进入前五比例", "因素排名中位数"], ascending=[True, False]
    )
    colors = [COLORS["navy"] if value >= data.nlargest(5, "进入前五比例")["进入前五比例"].min() else "#C7CDD4" for value in data["进入前五比例"]]
    y = np.arange(len(data))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 5.5), gridspec_kw={"width_ratios": [1.25, 1]})
    ax1.barh(y, data["进入前五比例"], color=colors, height=0.68, edgecolor="white")
    for yi, value, available in zip(y, data["进入前五比例"], data["可用场景比例"]):
        ax1.text(min(value + 0.025, 1.03), yi, f"{value:.0%}", va="center", fontsize=8.5)
        if available < 1:
            ax1.text(0.02, yi, f"可用{available:.0%}", va="center", ha="left", fontsize=7.5, color=COLORS["red"])
    ax1.set_yticks(y, data["物理因素"])
    ax1.set_xlim(0, 1.08)
    ax1.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.set_xlabel("48个场景中进入前五的比例")
    ax1.set_title("前五入选稳定性", loc="left", fontweight="bold")
    polish_axes(ax1, "x")

    ax2.hlines(y, data["因素排名最优"], data["因素排名最差"], color="#AAB2BD", linewidth=2)
    ax2.scatter(data["因素排名中位数"], y, color=colors, s=42, zorder=3, edgecolor="white", linewidth=0.5)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlim(10.5, 0.5)
    ax2.set_xticks(range(1, 11))
    ax2.set_xlabel("因素排名（越靠左越优）")
    ax2.set_title("排名中位数及范围", loc="left", fontweight="bold")
    polish_axes(ax2, "x")
    fig.suptitle("物理因素排序的敏感性检验", fontsize=14, fontweight="bold")
    fig.subplots_adjust(wspace=0.12, top=0.88)
    save_figure(fig, FIGURE_DIR, "fig02_factor_stability", dpi)


def figure03_heatmap(cache: dict[str, object], dpi: int) -> None:
    top5 = cache["top5"]
    grid = cache["grid"]
    hours = sorted(grid["评价窗口小时"].unique())
    bands = sorted(grid["同期日期带天数"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.6), constrained_layout=True)
    image = None
    for ax, (_, meta) in zip(axes.flat, top5.iterrows()):
        part = grid[grid["物理因素"] == meta["物理因素"]]
        matrix = part.pivot(index="评价窗口小时", columns="同期日期带天数", values="进入前五比例").reindex(index=hours, columns=bands)
        image = ax.imshow(matrix.to_numpy(float), cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(bands)), ["同日" if x == 0 else f"±{x}天" for x in bands])
        ax.set_yticks(range(len(hours)), [f"{x}h" for x in hours])
        ax.set_title(meta["物理因素"], fontweight="bold")
        ax.set_xlabel("历史同期日期带")
        ax.set_ylabel("评价窗口")
        for i in range(len(hours)):
            for j in range(len(bands)):
                value = matrix.iloc[i, j]
                color = "white" if pd.notna(value) and value < 0.58 else "#222222"
                label = "N/A" if pd.isna(value) else f"{value:.0%}"
                ax.text(j, i, label, ha="center", va="center", color=color, fontsize=8.5)
    axes.flat[-1].axis("off")
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, pad=0.02)
        colorbar.set_label("进入因素前五的比例")
        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    fig.suptitle("稳定前五因素对窗口设定的敏感性", fontsize=14, fontweight="bold")
    save_figure(fig, FIGURE_DIR, "fig03_top5_sensitivity_heatmap", dpi)


def figure04_distributions(cache: dict[str, object], dpi: int) -> None:
    top5 = cache["top5"]
    data = cache["distribution"]
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.8), constrained_layout=True)
    for ax, (_, meta) in zip(axes.flat, top5.iterrows()):
        part = data[data["物理因素"] == meta["物理因素"]]
        history = part.loc[part["样本组"] == "2021—2024历史同期", "数值"].dropna().to_numpy(float)
        event = part.loc[part["样本组"] == "2026事件窗口", "数值"].dropna().to_numpy(float)
        bp = ax.boxplot(
            [history, event],
            positions=[1, 2],
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111111", "linewidth": 1.7},
            whiskerprops={"color": COLORS["dark"]},
            capprops={"color": COLORS["dark"]},
        )
        bp["boxes"][0].set_facecolor("#D9D9D9")
        bp["boxes"][1].set_facecolor("#8FB3D9")
        for xpos, values, color in [(1, history, COLORS["mid"]), (2, event, COLORS["navy"])]:
            jitter = rng.normal(0, 0.045, size=len(values))
            ax.scatter(np.full(len(values), xpos) + jitter, values, s=8, alpha=0.35, color=color, edgecolors="none")
        ax.set_xticks([1, 2], [f"历史同期\nn={len(history)}", f"2026事件\nn={len(event)}"])
        ax.set_ylabel(meta["单位"])
        ax.set_title(meta["物理因素"], fontweight="bold")
        polish_axes(ax, "y")
    axes.flat[-1].axis("off")
    fig.suptitle("灾前24小时与2021—2024历史同期分布对照", fontsize=14, fontweight="bold")
    save_figure(fig, FIGURE_DIR, "fig04_event_vs_history", dpi)


FIGURES = {
    "01": figure01_trajectories,
    "02": figure02_stability,
    "03": figure03_heatmap,
    "04": figure04_distributions,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="读取plot_cache并生成第一问论文图。")
    parser.add_argument("--fig", default="all", choices=["all", *FIGURES.keys()])
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dpi < 300:
        raise ValueError("论文PNG建议至少300 DPI，本脚本不接受低于300的PNG输出。")
    configure_style()
    cache = load_cache()
    selected = FIGURES if args.fig == "all" else {args.fig: FIGURES[args.fig]}
    for key, function in selected.items():
        function(cache, args.dpi)
        print(f"图{key}已生成。")
    print(f"图片目录：{FIGURE_DIR}")


if __name__ == "__main__":
    main()

