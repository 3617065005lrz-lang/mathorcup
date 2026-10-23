from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SCRIPT_DIR.parent
QUESTION_DIR = ANALYSIS_DIR.parent
PREPROCESS_DIR = QUESTION_DIR / "01_数据预处理"
OUTPUT_DIR = ANALYSIS_DIR / "outputs"

EVENT_TIME = pd.Timestamp("2026-08-26 10:30:00")
EVENT_CUTOFF = pd.Timestamp("2026-08-26 10:00:00")
EVALUATION_HOURS = 24
TRAJECTORY_HOURS = 168
CONTROL_YEARS = [2021, 2022, 2023, 2024]
ANALYSIS_YEARS = CONTROL_YEARS + [2026]
ROLLING_MIN_COVERAGE = 0.80
EVALUATION_MIN_COVERAGE = 0.75

SENSITIVITY_EVALUATION_HOURS = [12, 24, 48]
SENSITIVITY_DATE_BANDS = [0, 3, 7, 15]
SENSITIVITY_MODES = ["strict", "expanded"]
SENSITIVITY_FLUX_POLICIES = ["保留复核点", "排除复核点"]

FLUX_REVIEW_COLUMNS = [
    "amb_e(kPa)",
    "amb_e_sat(kPa)",
    "USTAR(m/s)",
    "TKE(m2/s2)",
    "H(W/m2)",
    "LE(W/m2)",
]


FEATURE_CATALOG = {
    "temp_mean_24h": ("热量累积", "24小时平均2米气温", "Ta_2m"),
    "temp_mean_72h": ("热量累积", "72小时平均2米气温", "Ta_2m"),
    "pdd_72h": ("热量累积", "72小时正积温", "Ta_2m"),
    "pdd_168h": ("热量累积", "7日正积温", "Ta_2m"),
    "precip_sum_24h": ("降水与水分输入", "24小时累计降水", "Precip"),
    "precip_sum_72h": ("降水与水分输入", "72小时累计降水", "Precip"),
    "precip_sum_168h": ("降水与水分输入", "7日累计降水", "Precip"),
    "precip_max_24h": ("降水与水分输入", "24小时最大1小时降水", "Precip"),
    "vwc_mean_24h": ("土壤湿润状态", "24小时平均剖面土壤含水量", "VWC五层"),
    "vwc_mean_72h": ("土壤湿润状态", "72小时平均剖面土壤含水量", "VWC五层"),
    "vwc_delta_24h": ("土壤湿润状态", "剖面土壤含水量24小时变化", "VWC五层"),
    "vwc_gradient_24h": ("土壤湿润状态", "24小时平均深浅层含水量梯度", "VWC_0.1m和VWC_1.6m"),
    "soiltemp_mean_24h": ("土壤温度与冻融", "24小时平均浅层土壤温度", "soilT_0.1m"),
    "soiltemp_mean_72h": ("土壤温度与冻融", "72小时平均浅层土壤温度", "soilT_0.1m"),
    "soiltemp_delta_24h": ("土壤温度与冻融", "浅层土壤温度24小时变化", "soilT_0.1m"),
    "freeze_thaw_count_168h": ("土壤温度与冻融", "7日冻融过零次数", "soilT_0.1m"),
    "netrad_mean_24h": ("辐射能量", "24小时平均净辐射", "Rsd、Rsu、Rld、Rlu"),
    "positive_netrad_24h": ("辐射能量", "24小时累计正净辐射", "Rsd、Rsu、Rld、Rlu"),
    "positive_netrad_72h": ("辐射能量", "72小时累计正净辐射", "Rsd、Rsu、Rld、Rlu"),
    "positive_netrad_168h": ("辐射能量", "7日累计正净辐射", "Rsd、Rsu、Rld、Rlu"),
    "rh_mean_24h": ("大气湿度与水汽", "24小时平均2米相对湿度", "RH_2m"),
    "rh_mean_72h": ("大气湿度与水汽", "72小时平均2米相对湿度", "RH_2m"),
    "vpd_ambient_median_24h": ("大气湿度与水汽", "24小时环境水汽压亏缺中位数", "amb_e和amb_e_sat"),
    "vpd_ambient_median_72h": ("大气湿度与水汽", "72小时环境水汽压亏缺中位数", "amb_e和amb_e_sat"),
    "pressure_mean_24h": ("气压与天气过程", "24小时平均气压", "Pressure"),
    "pressure_delta_24h": ("气压与天气过程", "24小时气压变化", "Pressure"),
    "pressure_range_24h": ("气压与天气过程", "24小时气压极差", "Pressure"),
    "wind10_mean_24h": ("风场与湍流", "24小时平均10米风速", "WS_10m"),
    "wind_shear_mean_24h": ("风场与湍流", "24小时平均1—20米风速切变", "WS_1m和WS_20m"),
    "ustar_median_24h": ("风场与湍流", "24小时摩擦速度中位数", "USTAR"),
    "tke_median_24h": ("风场与湍流", "24小时湍动能中位数", "TKE"),
    "temp_gradient_mean_24h": ("边界层稳定结构", "24小时平均1—20米温度梯度", "Ta_1m和Ta_20m"),
    "temp_gradient_range_24h": ("边界层稳定结构", "24小时温度梯度极差", "Ta_1m和Ta_20m"),
    "h_median_24h": ("地表湍流热交换", "24小时感热通量中位数", "H"),
    "le_median_24h": ("地表湍流热交换", "24小时潜热通量中位数", "LE"),
    "turbulent_heat_median_24h": ("地表湍流热交换", "24小时感热与潜热之和中位数", "H和LE"),
}


def rolling(series: pd.Series, window: int, method: str) -> pd.Series:
    minimum = int(np.ceil(window * ROLLING_MIN_COVERAGE))
    obj = series.rolling(window, min_periods=minimum)
    return getattr(obj, method)()


def load_mode(mode: str, exclude_flux_review: bool = False) -> pd.DataFrame:
    path = PREPROCESS_DIR / "processed" / mode
    merged = None
    for group in ["MET", "RADM", "SOIL", "FLUX"]:
        file_path = next(path.glob(f"{group}_Gyirong_2021_2026_{mode}.parquet"))
        frame = pd.read_parquet(file_path)
        keep = [
            column
            for column in frame.columns
            if column == "Timestamp"
            or (column != "source_file" and not column.lower().startswith("qc_"))
        ]
        frame = frame[keep]
        merged = frame if merged is None else merged.merge(frame, on="Timestamp", how="outer")
    merged = merged[merged["Timestamp"].dt.year.isin(ANALYSIS_YEARS)].copy()
    if exclude_flux_review:
        review_path = PREPROCESS_DIR / "outputs" / "09_FLUX复核标记.csv"
        review = pd.read_csv(review_path, parse_dates=["Timestamp"])
        flagged_times = set(review["Timestamp"])
        flag_mask = merged["Timestamp"].isin(flagged_times)
        columns = [column for column in FLUX_REVIEW_COLUMNS if column in merged.columns]
        merged.loc[flag_mask, columns] = np.nan
    return merged.sort_values("Timestamp").reset_index(drop=True)


def build_features_one_year(raw: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame({"Timestamp": raw["Timestamp"]})
    temp = raw["Ta_2m(degree C)"]
    precip = raw["Precip(mm)"]
    vwc_columns = [f"VWC_{depth}m(m3/m3)" for depth in ["0.1", "0.2", "0.4", "0.8", "1.6"]]
    vwc_mean = raw[vwc_columns].mean(axis=1, skipna=False)
    soil_temp = raw["soilT_0.1m(degree C)"]

    net_radiation = (
        raw["Rsd(W/m2)"]
        - raw["Rsu(W/m2)"]
        + raw["Rld(W/m2)"]
        - raw["Rlu(W/m2)"]
    )
    positive_net_radiation = net_radiation.clip(lower=0)
    positive_temp = temp.clip(lower=0)
    vpd_ambient = ((raw["amb_e_sat(kPa)"] - raw["amb_e(kPa)"]) * 10).clip(lower=0)
    wind_shear = (raw["WS_20m(m/s)"] - raw["WS_1m(m/s)"]) / 19.0
    temp_gradient = (raw["Ta_20m(degree C)"] - raw["Ta_1m(degree C)"]) / 19.0
    turbulent_heat = raw["H(W/m2)"] + raw["LE(W/m2)"]
    freeze_thaw = (
        np.sign(soil_temp).ne(np.sign(soil_temp.shift(1)))
        & soil_temp.notna()
        & soil_temp.shift(1).notna()
    ).astype(float)

    f["temp_mean_24h"] = rolling(temp, 24, "mean")
    f["temp_mean_72h"] = rolling(temp, 72, "mean")
    f["pdd_72h"] = rolling(positive_temp, 72, "sum") / 24.0
    f["pdd_168h"] = rolling(positive_temp, 168, "sum") / 24.0

    f["precip_sum_24h"] = rolling(precip, 24, "sum")
    f["precip_sum_72h"] = rolling(precip, 72, "sum")
    f["precip_sum_168h"] = rolling(precip, 168, "sum")
    f["precip_max_24h"] = rolling(precip, 24, "max")

    f["vwc_mean_24h"] = rolling(vwc_mean, 24, "mean")
    f["vwc_mean_72h"] = rolling(vwc_mean, 72, "mean")
    f["vwc_delta_24h"] = vwc_mean - vwc_mean.shift(24)
    vwc_gradient = (raw["VWC_1.6m(m3/m3)"] - raw["VWC_0.1m(m3/m3)"]) / 1.5
    f["vwc_gradient_24h"] = rolling(vwc_gradient, 24, "mean")

    f["soiltemp_mean_24h"] = rolling(soil_temp, 24, "mean")
    f["soiltemp_mean_72h"] = rolling(soil_temp, 72, "mean")
    f["soiltemp_delta_24h"] = soil_temp - soil_temp.shift(24)
    f["freeze_thaw_count_168h"] = rolling(freeze_thaw, 168, "sum")

    f["netrad_mean_24h"] = rolling(net_radiation, 24, "mean")
    f["positive_netrad_24h"] = rolling(positive_net_radiation, 24, "sum") / 1000.0
    f["positive_netrad_72h"] = rolling(positive_net_radiation, 72, "sum") / 1000.0
    f["positive_netrad_168h"] = rolling(positive_net_radiation, 168, "sum") / 1000.0

    f["rh_mean_24h"] = rolling(raw["RH_2m(%)"], 24, "mean")
    f["rh_mean_72h"] = rolling(raw["RH_2m(%)"], 72, "mean")
    f["vpd_ambient_median_24h"] = rolling(vpd_ambient, 24, "median")
    f["vpd_ambient_median_72h"] = rolling(vpd_ambient, 72, "median")

    pressure = raw["Pressure(hPa)"]
    f["pressure_mean_24h"] = rolling(pressure, 24, "mean")
    f["pressure_delta_24h"] = pressure - pressure.shift(24)
    f["pressure_range_24h"] = rolling(pressure, 24, "max") - rolling(pressure, 24, "min")

    f["wind10_mean_24h"] = rolling(raw["WS_10m(m/s)"], 24, "mean")
    f["wind_shear_mean_24h"] = rolling(wind_shear, 24, "mean")
    f["ustar_median_24h"] = rolling(raw["USTAR(m/s)"], 24, "median")
    f["tke_median_24h"] = rolling(raw["TKE(m2/s2)"], 24, "median")

    f["temp_gradient_mean_24h"] = rolling(temp_gradient, 24, "mean")
    f["temp_gradient_range_24h"] = rolling(temp_gradient, 24, "max") - rolling(temp_gradient, 24, "min")

    f["h_median_24h"] = rolling(raw["H(W/m2)"], 24, "median")
    f["le_median_24h"] = rolling(raw["LE(W/m2)"], 24, "median")
    f["turbulent_heat_median_24h"] = rolling(turbulent_heat, 24, "median")
    return f


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """逐年构造滚动指标，防止跨年边界或被隔离的2025年进入窗口。"""
    yearly = []
    for _, frame in raw.groupby(raw["Timestamp"].dt.year, sort=True):
        yearly.append(build_features_one_year(frame.reset_index(drop=True)))
    return pd.concat(yearly, ignore_index=True).sort_values("Timestamp").reset_index(drop=True)


def cliffs_delta(event: np.ndarray, control: np.ndarray) -> float:
    if len(event) == 0 or len(control) == 0:
        return float("nan")
    u = mannwhitneyu(event, control, alternative="two-sided").statistic
    return float(2 * u / (len(event) * len(control)) - 1)


def matched_outside_rate(
    feature_frame: pd.DataFrame,
    feature: str,
    evaluation_hours: int,
) -> float:
    event = feature_frame[
        (feature_frame["Timestamp"] > EVENT_CUTOFF - pd.Timedelta(hours=evaluation_hours))
        & (feature_frame["Timestamp"] <= EVENT_CUTOFF)
    ][["Timestamp", feature]].dropna()
    controls = feature_frame[
        feature_frame["Timestamp"].dt.year.isin(CONTROL_YEARS)
    ][["Timestamp", feature]].dropna()
    event = event.copy()
    controls = controls.copy()
    event["match_key"] = event["Timestamp"].dt.strftime("%m-%d-%H")
    controls["match_key"] = controls["Timestamp"].dt.strftime("%m-%d-%H")
    bounds = controls.groupby("match_key")[feature].agg(
        control_min="min", control_max="max", control_count="count"
    )
    joined = event.join(bounds, on="match_key")
    joined = joined[joined["control_count"] >= 3]
    if joined.empty:
        return float("nan")
    outside = (joined[feature] < joined["control_min"]) | (joined[feature] > joined["control_max"])
    return float(outside.mean())


def evaluate_features(
    features: pd.DataFrame,
    mode: str,
    evaluation_hours: int = EVALUATION_HOURS,
    control_date_band_days: int = 0,
    flux_policy: str = "保留复核点",
) -> pd.DataFrame:
    event_mask = (
        (features["Timestamp"] > EVENT_CUTOFF - pd.Timedelta(hours=evaluation_hours))
        & (features["Timestamp"] <= EVENT_CUTOFF)
    )
    control_mask = pd.Series(False, index=features.index)
    for year in CONTROL_YEARS:
        matched_end = EVENT_CUTOFF.replace(year=year)
        for offset in range(-control_date_band_days, control_date_band_days + 1):
            end = matched_end + pd.Timedelta(days=offset)
            control_mask |= (
                (features["Timestamp"] > end - pd.Timedelta(hours=evaluation_hours))
                & (features["Timestamp"] <= end)
            )

    rows = []
    for feature, (factor, display, raw_source) in FEATURE_CATALOG.items():
        event = features.loc[event_mask, feature].dropna().to_numpy(dtype=float)
        control = features.loc[control_mask, feature].dropna().to_numpy(dtype=float)
        endpoint_count = 2 * control_date_band_days + 1
        expected_per_year = min(
            evaluation_hours * endpoint_count,
            evaluation_hours + 24 * (endpoint_count - 1),
        )
        expected_control = expected_per_year * len(CONTROL_YEARS)
        event_coverage = len(event) / evaluation_hours
        control_coverage = len(control) / expected_control
        if event_coverage < EVALUATION_MIN_COVERAGE or control_coverage < EVALUATION_MIN_COVERAGE:
            continue
        event_median = float(np.median(event))
        control_median = float(np.median(control))
        q25, q75 = np.quantile(control, [0.25, 0.75])
        scale = float((q75 - q25) / 1.349)
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = float(np.std(control, ddof=1))
        robust_effect = (event_median - control_median) / scale if scale > 1e-12 else 0.0
        percentile = float(np.mean(control <= event_median))
        rows.append(
            {
                "清洗版本": mode,
                "FLUX复核点处理": flux_policy,
                "评价窗口小时": evaluation_hours,
                "同期日期带天数": control_date_band_days,
                "物理因素": factor,
                "衍生指标": feature,
                "指标中文名": display,
                "对应原始指标": raw_source,
                "事件窗口覆盖率": event_coverage,
                "对照窗口覆盖率": control_coverage,
                "事件中位数": event_median,
                "对照中位数": control_median,
                "稳健标准化差异": robust_effect,
                "Cliff_delta": cliffs_delta(event, control),
                "匹配时刻超历史范围比例": matched_outside_rate(
                    features, feature, evaluation_hours
                ),
                "事件中位数历史百分位": percentile,
                "偏离方向": "偏高" if event_median > control_median else ("偏低" if event_median < control_median else "相当"),
            }
        )
    result = pd.DataFrame(rows)
    result["绝对稳健差异"] = result["稳健标准化差异"].abs()
    result["绝对Cliff_delta"] = result["Cliff_delta"].abs()
    result["百分位极端度"] = (2 * result["事件中位数历史百分位"] - 1).abs()
    rank_columns = ["绝对稳健差异", "绝对Cliff_delta", "匹配时刻超历史范围比例", "百分位极端度"]
    for column in rank_columns:
        result[f"{column}_排名"] = result[column].rank(ascending=False, method="average")
    result["指标稳健综合名次"] = result[[f"{column}_排名" for column in rank_columns]].mean(axis=1)
    return result.sort_values("指标稳健综合名次")


def combined_factor_ranking(all_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    value_columns = ["绝对稳健差异", "绝对Cliff_delta", "匹配时刻超历史范围比例", "百分位极端度"]
    descriptive_columns = [
        "事件中位数", "对照中位数", "稳健标准化差异", "Cliff_delta",
        "事件窗口覆盖率", "对照窗口覆盖率", "事件中位数历史百分位",
    ]
    combined = (
        all_metrics.groupby(["物理因素", "衍生指标", "指标中文名", "对应原始指标"], as_index=False)[value_columns + descriptive_columns]
        .mean()
    )
    combined["偏离方向"] = np.where(
        combined["事件中位数"] > combined["对照中位数"],
        "偏高",
        np.where(combined["事件中位数"] < combined["对照中位数"], "偏低", "相当"),
    )
    for column in value_columns:
        combined[f"{column}_排名"] = combined[column].rank(ascending=False, method="average")
    combined["跨版本指标综合名次"] = combined[[f"{column}_排名" for column in value_columns]].mean(axis=1)
    combined = combined.sort_values("跨版本指标综合名次")
    representatives = combined.groupby("物理因素", as_index=False).first()
    representatives = representatives.sort_values(
        ["跨版本指标综合名次", "绝对稳健差异", "绝对Cliff_delta"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    representatives["因素排名"] = np.arange(1, len(representatives) + 1)
    return combined, representatives


def mechanism_state(row: pd.Series) -> str:
    high = row["事件中位数"] > row["对照中位数"]
    factor = row["物理因素"]
    indicator = row["指标中文名"]
    if factor == "大气湿度与水汽":
        if "水汽压亏缺" in indicator:
            return "偏干" if high else "偏湿"
        return "偏湿" if high else "偏干"
    labels = {
        "热量累积": ("热量偏高", "热量偏低"),
        "降水与水分输入": ("降水偏多", "降水偏少"),
        "土壤湿润状态": ("土壤水分指标偏高", "土壤水分指标偏低"),
        "土壤温度与冻融": ("土温或冻融指标偏高", "土温或冻融指标偏低"),
        "辐射能量": ("辐射偏强", "辐射偏弱"),
        "气压与天气过程": ("气压过程指标偏高", "气压过程指标偏低"),
        "风场与湍流": ("风场或湍流指标偏高", "风场或湍流指标偏低"),
        "边界层稳定结构": ("温度梯度指标偏高", "温度梯度指标偏低"),
        "地表湍流热交换": ("湍流热通量指标偏高", "湍流热通量指标偏低"),
    }
    high_label, low_label = labels.get(factor, ("偏高", "偏低"))
    return high_label if high else low_label


def summarize_sensitivity(all_scenarios: pd.DataFrame) -> pd.DataFrame:
    scenario_count = all_scenarios["场景编号"].nunique()
    rows = []
    for factor, group in all_scenarios.groupby("物理因素"):
        top5 = group["因素排名"] <= 5
        direction_counts = group["机制状态"].value_counts(dropna=False)
        indicator_counts = group["指标中文名"].value_counts(dropna=False)
        available_scenarios = group["场景编号"].nunique()
        rows.append(
            {
                "物理因素": factor,
                "敏感性场景数": scenario_count,
                "可用场景数": available_scenarios,
                "可用场景比例": float(available_scenarios / scenario_count),
                "进入前五次数": int(top5.sum()),
                "进入前五比例": float(top5.sum() / scenario_count),
                "因素排名中位数": float(group["因素排名"].median()),
                "因素排名最优": int(group["因素排名"].min()),
                "因素排名最差": int(group["因素排名"].max()),
                "主导机制状态": direction_counts.index[0],
                "机制状态一致比例": float(direction_counts.iloc[0] / len(group)),
                "最常入选代表指标": indicator_counts.index[0],
                "代表指标一致比例": float(indicator_counts.iloc[0] / len(group)),
            }
        )
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["进入前五比例", "因素排名中位数", "因素排名最差"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def write_sensitivity_report(summary: pd.DataFrame, scenario_count: int) -> None:
    lines = [
        "# 2026年主分析敏感性检验",
        "",
        "## 数据隔离",
        "",
        "- 历史背景只使用2021—2024年。",
        "- 2026年只作为主分析事件年。",
        "- 2025年数据未进入特征筛选、历史标准化、阈值确定或本敏感性检验；主方案冻结后在独立验证模块中单独使用。",
        "",
        "## 检验设计",
        "",
        f"- 共计算{scenario_count}个场景。",
        f"- 评价窗口：{', '.join(map(str, SENSITIVITY_EVALUATION_HOURS))}小时。",
        f"- 历史同期日期带：精确同日以及前后{', '.join(map(str, SENSITIVITY_DATE_BANDS[1:]))}天。",
        "- 数据版本：严格QC版和扩展QC版。",
        "- FLUX复核点：分别保留和排除。",
        "- 稳定性用因素进入前五的比例、排名范围、机制状态一致性和代表指标一致性衡量。",
        "",
        "## 稳定性排序",
        "",
        "| 稳定性顺序 | 物理因素 | 可用场景比例 | 前五比例 | 排名中位数 | 排名范围 | 主导机制状态 | 状态一致性 | 最常入选代表指标 |",
        "|---:|---|---:|---:|---:|---:|---|---:|---|",
    ]
    for index, row in summary.iterrows():
        lines.append(
            f"| {index + 1} | {row['物理因素']} | {row['可用场景比例']:.1%} | {row['进入前五比例']:.1%} | "
            f"{row['因素排名中位数']:.1f} | {int(row['因素排名最优'])}—{int(row['因素排名最差'])} | "
            f"{row['主导机制状态']} | {row['机制状态一致比例']:.1%} | {row['最常入选代表指标']} |"
        )
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "- 前五比例越高，说明该物理因素不依赖某一个任意窗口或清洗口径。",
            "- 前五比例统一以全部48个预设场景为分母；覆盖率不足而无法计算的场景不计入前五，避免高估缺测较多的因素。",
            "- 机制状态已统一指标含义，例如相对湿度偏低和水汽压亏缺偏高都归为偏干；状态一致性不代表因果方向。",
            "- 该检验用于确定指标排序是否稳健，不把一次事件转化为有标签的监督学习样本。",
        ]
    )
    (OUTPUT_DIR / "08_敏感性检验报告.md").write_text("\n".join(lines), encoding="utf-8")


def write_report(metrics: pd.DataFrame, representatives: pd.DataFrame) -> None:
    lines = [
        "# 2026年事件主分析结果",
        "",
        "## 分析口径",
        "",
        f"- 事件参考时间：{EVENT_TIME}（北京时间）。",
        f"- 灾前截止时刻：{EVENT_CUTOFF}。",
        f"- 排名评价窗口：灾前最后{EVALUATION_HOURS}小时。",
        f"- 指标演化展示窗口：灾前{TRAJECTORY_HOURS}小时。",
        f"- 历史同期对照年份：{', '.join(map(str, CONTROL_YEARS))}。",
        "- 2025年数据未参与特征筛选、历史标准化、阈值确定或本轮敏感性检验；主方案冻结后另行独立验证。",
        "- 排名依据包括稳健标准化差异、Cliff's delta、超出历年同一时刻范围的比例和历史百分位极端度。",
        "- 最终先在物理因素组内选代表衍生指标，再进行因素排序。",
        "",
        "## 因素初排",
        "",
        "| 排名 | 物理因素 | 代表衍生指标 | 对应原始指标 | 事件中位数 | 历史同期中位数 | 方向 |",
        "|---:|---|---|---|---:|---:|---|",
    ]
    for _, row in representatives.iterrows():
        lines.append(
            f"| {int(row['因素排名'])} | {row['物理因素']} | {row['指标中文名']} | {row['对应原始指标']} | "
            f"{row['事件中位数']:.4g} | {row['对照中位数']:.4g} | {row['偏离方向']} |"
        )
    lines.extend(
        [
            "",
            "## 解释限制",
            "",
            "- 这是单一事件相对于历史同期的异常关联排序，不是因果贡献率。",
            "- 气象站距2026年受灾区约65km，结果表示区域环境背景，不能直接替代灾源区局地条件。",
            "- 排名稳定性另见敏感性检验结果；2025独立验证沿用冻结规则，未依据验证结果反向调整指标。",
        ]
    )
    (OUTPUT_DIR / "04_2026主分析结果.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mode_results = []
    event_trajectories = []
    feature_cache = {}
    for mode in SENSITIVITY_MODES:
        raw = load_mode(mode, exclude_flux_review=False)
        features = build_features(raw)
        feature_cache[(mode, "保留复核点")] = features
        metrics = evaluate_features(features, mode)
        mode_results.append(metrics)
        trajectory = features[
            (features["Timestamp"] > EVENT_CUTOFF - pd.Timedelta(hours=TRAJECTORY_HOURS))
            & (features["Timestamp"] <= EVENT_CUTOFF)
        ].copy()
        trajectory.insert(1, "清洗版本", mode)
        event_trajectories.append(trajectory)

    all_metrics = pd.concat(mode_results, ignore_index=True)
    combined, representatives = combined_factor_ranking(all_metrics)

    all_metrics.to_csv(OUTPUT_DIR / "01_候选衍生指标分版本评价.csv", index=False, encoding="utf-8-sig")
    combined.to_csv(OUTPUT_DIR / "02_跨版本候选指标排序.csv", index=False, encoding="utf-8-sig")
    representatives.to_csv(OUTPUT_DIR / "03_物理因素排序.csv", index=False, encoding="utf-8-sig")
    pd.concat(event_trajectories, ignore_index=True).to_parquet(
        OUTPUT_DIR / "05_2026灾前7天衍生指标轨迹.parquet", index=False
    )
    write_report(all_metrics, representatives)

    sensitivity_rows = []
    scenario_id = 0
    for mode in SENSITIVITY_MODES:
        for flux_policy in SENSITIVITY_FLUX_POLICIES:
            cache_key = (mode, flux_policy)
            if cache_key not in feature_cache:
                raw = load_mode(mode, exclude_flux_review=(flux_policy == "排除复核点"))
                feature_cache[cache_key] = build_features(raw)
            features = feature_cache[cache_key]
            for evaluation_hours in SENSITIVITY_EVALUATION_HOURS:
                for date_band in SENSITIVITY_DATE_BANDS:
                    scenario_id += 1
                    scenario_metrics = evaluate_features(
                        features,
                        mode,
                        evaluation_hours=evaluation_hours,
                        control_date_band_days=date_band,
                        flux_policy=flux_policy,
                    )
                    _, scenario_factors = combined_factor_ranking(scenario_metrics)
                    scenario_factors["机制状态"] = scenario_factors.apply(mechanism_state, axis=1)
                    scenario_factors.insert(0, "场景编号", scenario_id)
                    scenario_factors.insert(1, "清洗版本", mode)
                    scenario_factors.insert(2, "FLUX复核点处理", flux_policy)
                    scenario_factors.insert(3, "评价窗口小时", evaluation_hours)
                    scenario_factors.insert(4, "同期日期带天数", date_band)
                    sensitivity_rows.append(scenario_factors)

    sensitivity = pd.concat(sensitivity_rows, ignore_index=True)
    stability = summarize_sensitivity(sensitivity)
    sensitivity.to_csv(OUTPUT_DIR / "06_敏感性检验全部场景.csv", index=False, encoding="utf-8-sig")
    stability.to_csv(OUTPUT_DIR / "07_物理因素稳定性汇总.csv", index=False, encoding="utf-8-sig")
    write_sensitivity_report(stability, scenario_id)
    print(f"主分析与{scenario_id}个敏感性场景已保存至：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
