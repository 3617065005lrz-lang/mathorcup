from __future__ import annotations

"""Create Q3 PNG figures only.  This script never rebuilds the model cache."""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_DIR = Path(__file__).resolve().parents[1]
OUTPUTS = MODEL_DIR / "outputs"
FIGURES = MODEL_DIR / "figures"
FIGURES.mkdir(exist_ok=True)
TRUE_TRIGGER = pd.Timestamp("2026-08-26 10:52:21.900")
TRUE_PORT = pd.Timestamp("2026-08-26 10:59:48")


def style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_risk_evolution() -> None:
    data = pd.read_csv(OUTPUTS / "05_集合风险演化.csv", parse_dates=["valid_time_beijing"])
    threshold = float(pd.read_csv(OUTPUTS / "03_触发阈值.csv").query("model == 'main'").q80_threshold.iloc[0])
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, 5))
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for color, nominal in zip(colors, [168, 120, 72, 48, 24]):
        g = data.loc[data.nominal_issue_h.eq(nominal)].sort_values("valid_time_beijing")
        ax.plot(g.valid_time_beijing, g.risk_median, color=color, lw=2, label=f"提前{nominal} h")
        ax.fill_between(g.valid_time_beijing, g.risk_q025, g.risk_q975, color=color, alpha=0.10)
    ax.axhline(threshold, color="#D55E00", ls="--", lw=1.6, label="历史q80关注阈值")
    ax.axvline(TRUE_TRIGGER, color="black", ls=":", lw=1.8, label="核实启动时刻")
    ax.set_ylabel("源区联合风险指数")
    ax.set_xlabel("北京时间")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    ax.legend(ncol=4, fontsize=9)
    ax.set_title("五个滚动起报的风险演化与95%情景带")
    save(fig, "01_五起报风险演化.png")


def plot_trigger_intervals() -> None:
    data = pd.read_csv(
        OUTPUTS / "04_触发时间区间与覆盖.csv",
        parse_dates=["trigger_q025", "trigger_median", "trigger_q975", "operational_q025", "operational_q975"],
    )
    data = data.loc[data.model.eq("main") & data.stochastic].sort_values("nominal_issue_h", ascending=False)
    y = np.arange(len(data))
    origin = pd.Timestamp("2026-08-19 00:00")
    to_h = lambda x: (x - origin) / pd.Timedelta(hours=1)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for i, row in enumerate(data.itertuples()):
        ax.plot([to_h(row.operational_q025), to_h(row.operational_q975)], [i, i], color="#9ecae1", lw=8, solid_capstyle="round")
        ax.plot([to_h(row.trigger_q025), to_h(row.trigger_q975)], [i, i], color="#2171b5", lw=3, solid_capstyle="round")
        ax.scatter(to_h(row.trigger_median), i, color="#08306b", s=35, zorder=3)
    ax.axvline(to_h(TRUE_TRIGGER), color="#D55E00", ls="--", lw=2, label="核实启动时刻")
    ticks = pd.date_range("2026-08-19", "2026-08-28", freq="2D")
    ax.set_xticks([to_h(x) for x in ticks], [x.strftime("%m-%d") for x in ticks])
    ax.set_yticks(y, [f"提前{h} h" for h in data.nominal_issue_h])
    ax.set_xlabel("预测触发窗口（北京时间）")
    ax.set_title("原始95%情景区间与6小时分辨率修正后的操作区间")
    ax.legend()
    save(fig, "02_触发时间区间.png")


def plot_model_comparison() -> None:
    data = pd.read_csv(OUTPUTS / "12_基线与集合升级模型对照.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].plot(data.nominal_issue_h, data.baseline_abs_error_h, "o-", label="确定性基线")
    axes[0].plot(data.nominal_issue_h, data.ensemble_median_abs_error_h, "s-", label="集合中位数")
    axes[0].set_xlabel("名义提前量 / h")
    axes[0].set_ylabel("触发时刻绝对误差 / h")
    axes[0].set_title("点预测误差")
    axes[0].legend()
    axes[1].plot(data.nominal_issue_h, data.raw_interval_width_h, "o-", label="原始95%区间")
    axes[1].plot(data.nominal_issue_h, data.operational_interval_width_h, "s-", label="操作区间")
    axes[1].set_xlabel("名义提前量 / h")
    axes[1].set_ylabel("区间宽度 / h")
    axes[1].set_title("提前量—不确定性权衡")
    axes[1].legend()
    save(fig, "03_基线升级与区间宽度.png")


def plot_propagation() -> None:
    arrival = pd.read_csv(
        OUTPUTS / "08_口岸到达时间区间.csv",
        parse_dates=["arrival_q025", "arrival_median", "arrival_q975", "operational_arrival_q025", "operational_arrival_q975"],
    )
    arrival = arrival.loc[arrival.stochastic].sort_values("nominal_issue_h", ascending=False)
    dist = pd.read_csv(OUTPUTS / "09_触发后前缘距离区间.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    origin = pd.Timestamp("2026-08-19")
    to_h = lambda x: (x - origin) / pd.Timedelta(hours=1)
    for i, row in enumerate(arrival.itertuples()):
        axes[0].plot([to_h(row.operational_arrival_q025), to_h(row.operational_arrival_q975)], [i, i], color="#bcbddc", lw=8)
        axes[0].plot([to_h(row.arrival_q025), to_h(row.arrival_q975)], [i, i], color="#756bb1", lw=3)
        axes[0].scatter(to_h(row.arrival_median), i, color="#3f007d", s=30)
    axes[0].axvline(to_h(TRUE_PORT), color="#D55E00", ls="--", lw=2)
    axes[0].set_yticks(np.arange(len(arrival)), [f"提前{h} h" for h in arrival.nominal_issue_h])
    ticks = pd.date_range("2026-08-19", "2026-08-28", freq="2D")
    axes[0].set_xticks([to_h(x) for x in ticks], [x.strftime("%m-%d") for x in ticks])
    axes[0].set_title("口岸前缘到达区间")
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, 5))
    for color, nominal in zip(colors, [168, 120, 72, 48, 24]):
        g = dist.loc[dist.nominal_issue_h.eq(nominal)].sort_values("elapsed_min")
        axes[1].plot(g.elapsed_min, g.distance_median_km, color=color, lw=2, label=f"{nominal} h")
        axes[1].fill_between(g.elapsed_min, g.distance_q025_km, g.distance_q975_km, color=color, alpha=0.10)
    axes[1].set_xlabel("触发后时间 / min")
    axes[1].set_ylabel("沿沟道前缘距离 / km")
    axes[1].set_title("前缘传播距离情景区间")
    axes[1].legend(ncol=2, fontsize=8)
    save(fig, "04_到达时间与传播距离.png")


def plot_window_sensitivity() -> None:
    data = pd.read_csv(OUTPUTS / "13_五时间窗敏感性汇总.csv")
    order = ["uniform_24", "uniform_48", "uniform_72", "uniform_120", "uniform_168", "main"]
    data["order"] = data.model.map({x: i for i, x in enumerate(order)})
    data = data.sort_values("order")
    labels = ["24 h", "48 h", "72 h", "120 h", "168 h", "主模型"]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(data))
    ax.bar(x - 0.18, data.raw_coverage_count, width=0.36, label="原始区间覆盖次数", color="#6baed6")
    ax.bar(x + 0.18, data.operational_coverage_count, width=0.36, label="操作区间覆盖次数", color="#fd8d3c")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 5.5)
    ax.set_ylabel("五个滚动起报中的覆盖次数")
    ax.set_title("五时间尺度敏感性（同一事件滚动起报，非独立样本）")
    ax.legend()
    save(fig, "05_五时间窗敏感性.png")


def plot_warning_ladder() -> None:
    data = pd.read_csv(OUTPUTS / "17_逐时预警等级与双机制分支.csv", parse_dates=["valid_time_beijing"])
    issues = [168, 120, 72, 48, 24]
    colors = ["#d9d9d9", "#3182bd", "#ffd92f", "#f28e2b", "#d62728"]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for i, nominal in enumerate(issues):
        g = data.loc[data.nominal_issue_h.eq(nominal)].sort_values("valid_time_beijing")
        for level in range(5):
            mask = g.warning_level_code.eq(level)
            ax.scatter(
                g.loc[mask, "valid_time_beijing"],
                np.full(mask.sum(), i),
                s=28,
                marker="s",
                color=colors[level],
                linewidths=0,
            )
    ax.axvline(TRUE_TRIGGER, color="black", ls="--", lw=1.8, label="核实启动时刻")
    ax.set_yticks(np.arange(len(issues)), [f"提前{h} h" for h in issues])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    ax.set_xlabel("北京时间")
    ax.set_title("逐时相对风险等级（连续12小时方可正式升级）")
    handles = [plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=colors[i], markersize=9, label=x) for i, x in enumerate(["常态", "蓝色", "黄色", "橙色", "红色"])]
    handles.append(plt.Line2D([0], [0], color="black", ls="--", label="核实启动时刻"))
    ax.legend(handles=handles, ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    save(fig, "06_滚动预警等级时间轴.png")


def plot_mechanism_diagnosis() -> None:
    data = pd.read_csv(OUTPUTS / "19_双机制分支诊断.csv").sort_values("nominal_issue_h", ascending=False)
    x = np.arange(len(data))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.bar(x - 1.5 * width, data.thermal_median_at_event, width, color="#E66101", label="真实时刻：热力—冰冻圈")
    ax.bar(x - 0.5 * width, data.rain_water_median_at_event, width, color="#5E3C99", label="真实时刻：降雨—水分")
    ax.bar(x + 0.5 * width, data.thermal_median_at_peak, width, color="#FDB863", label="预测峰值：热力—冰冻圈")
    ax.bar(x + 1.5 * width, data.rain_water_median_at_peak, width, color="#B2ABD2", label="预测峰值：降雨—水分")
    ax.set_xticks(x, [f"提前{h} h" for h in data.nominal_issue_h])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("历史经验百分位/分支评分")
    ax.set_title("五个滚动起报的双机制分支诊断")
    ax.legend(ncol=2, fontsize=8)
    save(fig, "07_双机制分支诊断.png")


def plot_imerg_availability() -> None:
    data = pd.read_csv(OUTPUTS / "21_IMERG起报可用性审计.csv")
    data = data.sort_values("nominal_issue_h", ascending=False).reset_index(drop=True)
    x = np.arange(len(data))
    available = data.rain72_available.astype(bool)
    values = data.rain72_mm.fillna(0.0)
    q90 = float(data.historical_q90_72h_mm.iloc[0])
    q95 = float(data.historical_q95_72h_mm.iloc[0])
    fig, ax = plt.subplots(figsize=(9.5, 4.9))
    bars = ax.bar(x, values, color=np.where(available, "#4C78A8", "#d9d9d9"), width=0.62)
    for i, (bar, ok, value) in enumerate(zip(bars, available, data.rain72_mm)):
        if ok:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.7, f"{value:.1f}", ha="center", fontsize=9)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, 1.0, "资料窗\n不完整", ha="center", va="bottom", fontsize=8, color="#666666")
    ax.axhline(q90, color="#F28E2B", ls="--", lw=1.8, label=f"历史q90={q90:.1f} mm")
    ax.axhline(q95, color="#D62728", ls=":", lw=1.8, label=f"历史q95={q95:.1f} mm")
    ax.set_xticks(x, [f"提前{h} h" for h in data.nominal_issue_h])
    ax.set_ylabel("起报时已可获得的IMERG Late 72 h累计降雨 / mm")
    ax.set_title("IMERG Late独立降雨证据（按14小时发布延迟截断）")
    ax.legend()
    save(fig, "08_IMERG起报可用降雨审计.png")


def main() -> None:
    style()
    plot_risk_evolution()
    plot_trigger_intervals()
    plot_model_comparison()
    plot_propagation()
    plot_window_sensitivity()
    plot_warning_ladder()
    plot_mechanism_diagnosis()
    plot_imerg_availability()
    print(f"PNG figures written to: {FIGURES}")


if __name__ == "__main__":
    main()
