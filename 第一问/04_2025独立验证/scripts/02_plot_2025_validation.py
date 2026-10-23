from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
VALIDATION_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = VALIDATION_DIR / "outputs"
FIGURE_DIR = VALIDATION_DIR / "figures"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="读取2025验证结果并生成PNG。")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.dpi < 300:
        raise ValueError("PNG输出不低于300 DPI。")

    primary = pd.read_csv(OUTPUT_DIR / "01_2025主验证五特征.csv")
    stability = pd.read_csv(OUTPUT_DIR / "04_2025特征稳定性汇总.csv")
    primary = primary.sort_values("稳定性顺序", ascending=False)
    stability = stability.sort_values("稳定性顺序", ascending=False)

    configure_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 5.2), constrained_layout=True)

    primary_colors = np.where(
        primary["强支持"].astype(bool),
        "#1F3A93",
        np.where(primary["方向一致"].astype(bool), "#8FB3D9", "#C44E52"),
    )
    y1 = np.arange(len(primary))
    ax1.barh(y1, primary["方向性历史百分位"], color=primary_colors, height=0.62)
    ax1.axvline(0.5, color="#777777", linestyle="--", linewidth=1, label="方向分界")
    ax1.axvline(0.8, color="#E65100", linestyle=":", linewidth=1.2, label="强支持阈值")
    ax1.set_yticks(y1, primary["指标中文名"])
    ax1.set_xlim(0, 1.04)
    ax1.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.set_xlabel("主场景方向性历史百分位")
    ax1.set_title("主场景逐项验证", loc="left", fontweight="bold")
    ax1.legend(frameon=False, loc="lower right")
    ax1.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.45)
    for yi, value in zip(y1, primary["方向性历史百分位"]):
        ax1.text(min(value + 0.025, 1.01), yi, f"{value:.0%}", va="center", fontsize=8.5)

    y2 = np.arange(len(stability))
    height = 0.34
    ax2.barh(
        y2 + height / 2,
        stability["方向一致比例"],
        height=height,
        color="#1F3A93",
        label="方向一致",
    )
    ax2.barh(
        y2 - height / 2,
        stability["强支持比例"],
        height=height,
        color="#E69F00",
        label="强支持",
    )
    ax2.set_yticks(y2, stability["最终特征"])
    ax2.set_xlim(0, 1.04)
    ax2.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.set_xlabel("48个敏感性场景中的比例")
    ax2.set_title("跨场景复现稳定性", loc="left", fontweight="bold")
    ax2.legend(frameon=False, loc="lower right")
    ax2.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.45)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)

    fig.suptitle("2025年Purepu冰湖溃决事件独立验证", fontsize=14, fontweight="bold")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURE_DIR / "fig2025_independent_validation.png"
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"PNG已生成：{output_path}")


if __name__ == "__main__":
    main()
