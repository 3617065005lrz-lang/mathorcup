from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


COLORS = {
    "navy": "#1F3A93",
    "blue": "#1976D2",
    "orange": "#E65100",
    "green": "#2E7D32",
    "purple": "#6A1B9A",
    "red": "#AD1457",
    "dark": "#333333",
    "mid": "#777777",
    "light": "#D9D9D9",
    "pale": "#EEF3F8",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "axes.edgecolor": COLORS["dark"],
            "axes.labelcolor": COLORS["dark"],
            "xtick.color": COLORS["dark"],
            "ytick.color": COLORS["dark"],
            "text.color": COLORS["dark"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def polish_axes(ax: plt.Axes, grid_axis: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["dark"])
    ax.spines["bottom"].set_color(COLORS["dark"])
    if grid_axis:
        ax.grid(
            True,
            axis=grid_axis,
            linestyle="--",
            linewidth=0.5,
            color="#CCCCCC",
            alpha=0.5,
            zorder=0,
        )
    ax.set_axisbelow(True)


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    dpi: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / f"{stem}.png",
        bbox_inches="tight",
        facecolor="white",
        dpi=dpi,
    )
    plt.close(fig)

