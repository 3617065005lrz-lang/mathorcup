"""图1：稳定五特征灾前轨迹。解压完整目录后在IDE中运行本文件。

安装：python -m pip install -r requirements.txt
不弹出窗口：python plot_fig01.py --no-show
数据来自同学GitHub成果，data/目录必须与本脚本一同保留。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, FuncFormatter
import numpy as np
import pandas as pd

# ===== 在IDE中可直接修改这些参数 =====
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR
DPI = 600
SHOW_FIGURE = True
WIDTH_CM, HEIGHT_CM = 17.5, 21.0
COLORS = {
    "event": "#29A8E0",       # Okabe–Ito蓝：2026年事件前的真实轨迹
    "history": "#FF805C",     # Okabe–Ito朱橙虚线：历年匹配时刻中位数
    "history_text": "#C44B2C", # 摘要文字使用加深橙色，提升白底小字可读性
    "iqr": "#FFE0D6",         # 浅橙阴影：历史Q1—Q3，非置信区间
    "window": "#E5E9ED",      # 浅灰背景：最后24小时评价窗口
    "text": "#303940",
    "secondary": "#606D76",
    "grid": "#E6EAED",
}
FEATURES = [
    ("vwc_delta_24h", "剖面土壤含水量24小时变化", "m³/m³", 5),
    ("temp_mean_24h", "24小时平均2米气温", "°C", 2),
    ("positive_netrad_24h", "24小时累计正净辐射", "kWh/m²", 2),
    ("precip_sum_72h", "72小时累计降水", "mm", 2),
    ("rh_mean_24h", "24小时平均2米相对湿度", "%", 2),
]


def font_settings() -> dict:
    available = {font.name for font in font_manager.fontManager.ttflist}
    chinese = next((name for name in ["Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans SC", "SimHei", "PingFang SC"] if name in available), None)
    if chinese is None:
        raise RuntimeError("未找到中文字体。请安装微软雅黑或思源黑体，然后重启IDE。")
    latin = "Arial" if "Arial" in available else "DejaVu Sans"
    return {
        "font.family": [latin, chinese],
        "axes.unicode_minus": False,
        "font.size": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.3,
        "text.color": COLORS["text"],
        "axes.labelcolor": COLORS["text"],
        "xtick.color": COLORS["secondary"],
        "ytick.color": COLORS["secondary"],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": None,  # 固定17.5厘米输出宽度，避免tight裁切改变尺寸
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "svg.fonttype": "path",  # 文字转矢量轮廓，跨电脑不依赖字体且放大清晰
    }


def read_data():
    paths = [DATA_DIR / name for name in ["trajectory.csv", "final_features.csv", "provenance.json"]]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少配套数据文件，请解压完整代码包：\n" + "\n".join(missing))
    trajectory = pd.read_csv(paths[0], encoding="utf-8-sig", float_precision="round_trip")
    summary = pd.read_csv(paths[1], encoding="utf-8-sig", float_precision="round_trip")
    provenance = json.loads(paths[2].read_text(encoding="utf-8"))
    if provenance["control_years"] != [2021, 2022, 2023, 2024] or provenance["excluded_validation_year"] != 2025:
        raise ValueError("年份口径不符合冻结主分析：历史2021—2024，2025独立隔离。")
    for path in paths[:2]:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != provenance["data_sha256"][path.name]:
            warnings.warn(f"{path.name}已被修改，请确认更新后的数据和统计口径一致。", stacklevel=2)
    for code, _, _, _ in FEATURES:
        part = trajectory[trajectory["衍生指标"] == code].sort_values("相对小时")
        if not np.array_equal(part["相对小时"].to_numpy(), np.arange(-167, 1)):
            raise ValueError(f"{code}应当包含相对小时-167到0的168条记录，不能重复或遗漏。")
        if len(summary[summary["衍生指标代码"] == code]) != 1:
            raise ValueError(f"{code}在最终结果表中必须有且只有一条记录。")
        times = pd.to_datetime(part["事件时间"])
        cutoff = pd.Timestamp(provenance["event_cutoff"])
        if not ((times - cutoff).dt.total_seconds().to_numpy() / 3600 == part["相对小时"].to_numpy()).all():
            raise ValueError(f"{code}的事件时间与相对小时不一致。")
        q1, med, q3 = (part[col].to_numpy(float) for col in ["历史Q1", "历史中位数", "历史Q3"])
        valid = np.isfinite(q1) & np.isfinite(med) & np.isfinite(q3)
        if np.any(q1[valid] > med[valid]) or np.any(med[valid] > q3[valid]):
            raise ValueError(f"{code}的历史分位数顺序错误。")
    return trajectory, summary, provenance


def create_figure(trajectory, summary, provenance):
    fig, axes = plt.subplots(5, 1, figsize=(WIDTH_CM / 2.54, HEIGHT_CM / 2.54), sharex=True)
    fig.subplots_adjust(left=0.13, right=0.79, bottom=0.115, top=0.835, hspace=0.62)
    fig.suptitle("五个稳定特征的灾前演化与历史同期对照", x=0.51, y=0.978,
                 fontsize=12, fontweight="bold")
    fig.text(0.51, 0.942, "2026年主事件  |  历史背景：2021—2024年  |  严格QC口径",
             ha="center", fontsize=7.8, color=COLORS["secondary"])
    handles = [
        Line2D([], [], color=COLORS["event"], lw=1.7, label="2026年灾前轨迹"),
        Line2D([], [], color=COLORS["history"], lw=1.0, ls="--", label="历史同期中位数"),
        Patch(facecolor=COLORS["iqr"], label="历史同期IQR"),
        Patch(facecolor=COLORS["window"], label="最后24小时评价窗口"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.51, 0.917),
               frameon=False, ncol=2, columnspacing=2.3, handlelength=2.3, labelspacing=0.65)

    cutoff = pd.Timestamp(provenance["event_cutoff"])
    event_time = pd.Timestamp(provenance["event_time"])
    for i, (ax, (code, title, unit, decimals)) in enumerate(zip(axes, FEATURES)):
        part = trajectory[trajectory["衍生指标"] == code].sort_values("相对小时")
        x = part["相对小时"].to_numpy(float) / 24.0
        event = part["事件值"].to_numpy(float)
        median = part["历史中位数"].to_numpy(float)
        q1, q3 = part["历史Q1"].to_numpy(float), part["历史Q3"].to_numpy(float)
        # 评价窗口为(cutoff-24h, cutoff]，共有24个小时观测。
        ax.axvspan(-1, 0, color=COLORS["window"], alpha=0.7, lw=0, zorder=0)
        ax.fill_between(x, q1, q3, color=COLORS["iqr"], alpha=0.85, lw=0, zorder=1)
        ax.plot(x, median, color=COLORS["history"], lw=1.05, ls=(0, (4, 2.5)), zorder=2)
        ax.plot(x, event, color=COLORS["event"], lw=1.7, solid_capstyle="round", zorder=3)
        if np.isfinite(event[-1]):
            ax.plot(x[-1], event[-1], "o", ms=3.3, color=COLORS["event"], zorder=4)
        ax.axvline(0, color="#90999F", lw=0.75, ls=":", zorder=2)
        ax.set_title(f"({chr(97 + i)})  {title}", loc="left", fontsize=9, fontweight="bold", pad=7)
        ax.set_ylabel(unit, labelpad=6)
        ax.set_xlim(-7, 0.06)
        ax.margins(y=0.15)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
        if code == "vwc_delta_24h":
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3f}"))
        else:
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55, ls=(0, (3, 3)))
        ax.set_axisbelow(True)
        ax.tick_params(length=3, width=0.6)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        for side in ["left", "bottom"]:
            ax.spines[side].set_color("#B1BAC0")
        # 右侧数字直接来自最终结果表；历史值不是逐时历史中位数的再次取中位数。
        row = summary.loc[summary["衍生指标代码"] == code].iloc[0]
        y = ax.get_position().y0 + ax.get_position().height * 0.52
        fig.text(0.835, y + 0.026, f"事件  {row['2026事件窗口中位数']:.{decimals}f}",
                 fontsize=8.1, fontweight="bold", color=COLORS["event"])
        fig.text(0.835, y + 0.001, f"历史  {row['2021-2024历史同期中位数']:.{decimals}f}",
                 fontsize=7.7, color=COLORS["history_text"])
        fig.text(0.835, y - 0.024, f"{row['关联方向']}  /  {unit}", fontsize=7.3, color=COLORS["secondary"])
    fig.text(0.835, 0.86, "评价窗口中位数", fontsize=7.5, fontweight="bold")
    fig.add_artist(Line2D([0.815, 0.815], [0.115, 0.835], transform=fig.transFigure,
                          color="#E0E5E8", lw=0.7))
    axes[-1].set_xticks(np.arange(-7, 1))
    axes[-1].set_xlabel("相对分析截止时刻的时间（天）", labelpad=7)
    fig.text(0.51, 0.054,
             f"0 = {cutoff:%Y-%m-%d %H:%M}；事件参考时间：{event_time:%H:%M}（北京时间）",
             ha="center", fontsize=7.2, color=COLORS["secondary"])
    fig.text(0.51, 0.027, "阴影带为历史四分位范围（IQR），非置信区间；站点特征表示区域环境背景。",
             ha="center", fontsize=7.1, color=COLORS["secondary"])
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument("--no-show", action="store_true", help="仅保存图片，不弹出窗口")
    args = parser.parse_args()
    if args.dpi < 300:
        parser.error("论文PNG的DPI请设为300或更高。")
    trajectory, summary, provenance = read_data()
    with mpl.rc_context(font_settings()):
        fig = create_figure(trajectory, summary, provenance)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output = OUTPUT_DIR / "fig01_pre_event_trajectories.png"
        fig.savefig(output, dpi=args.dpi, bbox_inches=None, facecolor="white")
        svg_output = output.with_suffix(".svg")
        fig.savefig(svg_output, format="svg", bbox_inches=None, facecolor="white")
        manifest = {
            "source_commit": provenance["source_commit"],
            "source_data_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA_DIR.glob("*.csv")},
            "size_cm": [WIDTH_CM, HEIGHT_CM], "dpi": args.dpi,
            "outputs": [output.name, svg_output.name],
            "palette": "Bright sky blue #29A8E0 and coral #FF805C",
            "svg": "纯矢量路径；文字转轮廓；无嵌入位图",
            "versions": {"matplotlib": mpl.__version__, "pandas": pd.__version__, "numpy": np.__version__},
            "transformations": ["按相对小时排序", "小时除以24换算为天", "数值显示舍入；轨迹不平滑、不插值、不重新筛选特征"],
            "uncertainty": "历史匹配时刻Q1—Q3；非置信区间",
            "summary": "最终24小时事件评价窗口与历史匹配窗口中位数，取自冻结结果表",
        }
        (OUTPUT_DIR / "fig01_provenance.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已生成：{output}\n矢量图：{svg_output}\n尺寸：{WIDTH_CM} × {HEIGHT_CM} cm，{args.dpi} DPI")
        if SHOW_FIGURE and not args.no_show:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
