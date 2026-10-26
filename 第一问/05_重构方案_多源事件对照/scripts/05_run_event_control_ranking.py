from __future__ import annotations

"""2026事件—历史对照的可解释因素初排。

本脚本实现两套对照：
1. 2021—2024年同季节（目标日期前后15天）对照；
2. 在同季节中再限制为与事件相同的GMM天气型。

候选统计量先按“基础指标”合并五个窗口，再按物理因素组聚合，最后用
熵权与CRITIC各占一半确定准则权重、TOPSIS排序。排名对象始终是物理因素，
不是难以解释的抽象成分。源区ERA5派生缓存当前属于暂定外源复核数据。
"""

from pathlib import Path
import os
import re

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

SCRIPT_DIR = Path(__file__).resolve().parent
WORK_DIR = SCRIPT_DIR.parent
CACHE_DIR = WORK_DIR / "cache"
OUTPUT_DIR = WORK_DIR / "outputs"

CONTROL_YEARS = [2021, 2022, 2023, 2024]
EVENT_TIME = pd.Timestamp("2026-08-26 10:52:07")
STATION_EVENT_ENDPOINT = pd.Timestamp("2026-08-26 10:00:00")
SOURCE_EVENT_ENDPOINT = pd.Timestamp("2026-08-26 10:00:00")
CALENDAR_BAND_DAYS = 15
MIN_CONTROL_COUNT = 12
GMM_COMPONENTS = 6
RANDOM_SEED = 2026

WEATHER_FEATURES = [
    "sta_tmean_24h",
    "sta_precip_sum_24h",
    "sta_RH_mean_24h",
    "sta_pressure_mean_24h",
    "sta_wind10_mean_24h",
    "sta_positive_netrad_24h",
]

CRITERIA = [
    "同期稳健效应强度",
    "同期百分位极端度",
    "同天气型稳健效应强度",
    "同天气型百分位极端度",
    "多窗口方向一致性",
    "随机化支持度",
]


def calendar_distance_days(timestamp: pd.Timestamp, target: pd.Timestamp) -> int:
    anchor = pd.Timestamp(2000, target.month, target.day)
    current = pd.Timestamp(2000, timestamp.month, timestamp.day)
    return abs((current - anchor).days)


def seasonal_mask(dates: pd.Series, target: pd.Timestamp, band: int) -> pd.Series:
    return dates.map(lambda value: calendar_distance_days(pd.Timestamp(value), target) <= band)


def fit_weather_types(
    mode: str,
    n_components: int = GMM_COMPONENTS,
    diagnostics: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    station = pd.read_parquet(CACHE_DIR / f"station_daily_10h_features_{mode}.parquet")
    weather_coverage = station[WEATHER_FEATURES].notna().sum(axis=1)
    classifiable = weather_coverage >= 4
    train_mask = station["Timestamp"].dt.year.isin(CONTROL_YEARS) & classifiable
    train = station.loc[train_mask, WEATHER_FEATURES].copy()
    medians = train.median()
    x_train = train.fillna(medians)
    x_all = station.loc[classifiable, WEATHER_FEATURES].fillna(medians)

    # 降水零值多且右偏；仅在天气分型时做log1p，排名仍使用原始毫米量。
    precip_index = WEATHER_FEATURES.index("sta_precip_sum_24h")
    x_train.iloc[:, precip_index] = np.log1p(x_train.iloc[:, precip_index].clip(lower=0))
    x_all.iloc[:, precip_index] = np.log1p(x_all.iloc[:, precip_index].clip(lower=0))

    scaler = RobustScaler()
    z_train = scaler.fit_transform(x_train)
    z_all = scaler.transform(x_all)

    bic_rows = []
    models: dict[int, GaussianMixture] = {}
    component_grid = range(2, 8) if diagnostics else [n_components]
    for components in component_grid:
        model = GaussianMixture(
            n_components=components,
            covariance_type="full",
            n_init=20,
            reg_covar=1e-5,
            random_state=RANDOM_SEED,
        ).fit(z_train)
        models[components] = model
        bic_rows.append({"清洗版本": mode, "天气型数": components, "BIC": model.bic(z_train)})

    model = models[n_components]
    labels = model.predict(z_all)
    probabilities = model.predict_proba(z_all)
    classified = station[["Timestamp"]].copy()
    classified["天气型编号"] = pd.Series(pd.NA, index=classified.index, dtype="Int64")
    classified.loc[classifiable, "天气型编号"] = labels + 1
    classified["最大后验概率"] = np.nan
    classified.loc[classifiable, "最大后验概率"] = probabilities.max(axis=1)

    state = station.loc[classifiable, WEATHER_FEATURES].copy()
    state["天气型编号"] = labels + 1
    centers = state.groupby("天气型编号")[WEATHER_FEATURES].median()
    global_median = station.loc[train_mask, WEATHER_FEATURES].median()
    global_iqr = station.loc[train_mask, WEATHER_FEATURES].quantile(0.75) - station.loc[train_mask, WEATHER_FEATURES].quantile(0.25)
    center_rows = []
    name_map = {
        "sta_tmean_24h": "气温",
        "sta_precip_sum_24h": "降水",
        "sta_RH_mean_24h": "湿度",
        "sta_pressure_mean_24h": "气压",
        "sta_wind10_mean_24h": "风速",
        "sta_positive_netrad_24h": "正净辐射",
    }
    for cluster, row in centers.iterrows():
        deviations = pd.Series(
            {
                column: (row[column] - global_median[column]) / global_iqr[column]
                if np.isfinite(global_iqr[column]) and global_iqr[column] > 0
                else np.nan
                for column in WEATHER_FEATURES
            }
        ).dropna().sort_values()
        if deviations.empty:
            description = "综合天气型"
        else:
            low = name_map[deviations.index[0]] + "偏低"
            high = name_map[deviations.index[-1]] + "偏高"
            description = f"{high}—{low}型"
        record = {"清洗版本": mode, "天气型编号": int(cluster), "解释性名称": description}
        record.update(row.to_dict())
        record["历史样本数"] = int(
            ((classified["天气型编号"] == cluster) & train_mask).fillna(False).sum()
        )
        center_rows.append(record)
    centers_out = pd.DataFrame(center_rows)
    classified = classified.merge(centers_out[["天气型编号", "解释性名称"]], on="天气型编号", how="left")
    return classified, centers_out, pd.DataFrame(bic_rows)


def robust_comparison(event_value: float, controls: pd.Series) -> dict[str, float]:
    controls = pd.to_numeric(controls, errors="coerce").dropna().astype(float)
    if not np.isfinite(event_value) or len(controls) < MIN_CONTROL_COUNT:
        return {
            "对照样本数": len(controls),
            "对照中位数": np.nan,
            "稳健Z": np.nan,
            "稳健效应强度": np.nan,
            "历史百分位": np.nan,
            "百分位极端度": np.nan,
            "经验双侧P": np.nan,
            "蒙特卡洛双侧P": np.nan,
        }
    median = float(controls.median())
    q25, q75 = controls.quantile([0.25, 0.75])
    scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale <= 1e-12:
        mad = float(np.median(np.abs(controls.to_numpy() - median))) * 1.4826
        scale = mad
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(controls.std(ddof=1))
    robust_z = (event_value - median) / scale if np.isfinite(scale) and scale > 1e-12 else np.nan
    less = int((controls < event_value).sum())
    equal = int((controls == event_value).sum())
    percentile = (less + 0.5 * equal) / len(controls)
    extremeness = abs(2.0 * percentile - 1.0)
    deviations = np.abs(controls.to_numpy() - median)
    event_deviation = abs(event_value - median)
    p_value = (1 + int((deviations >= event_deviation).sum())) / (len(controls) + 1)
    # Each control value is itself a rolling feature from a contiguous hourly
    # block. Resampling these historical pseudo-event endpoints therefore keeps
    # the within-window autocorrelation instead of treating hours as independent.
    seed_offset = int(np.round(abs(event_value) * 1_000_000)) % 1_000_003
    rng = np.random.default_rng(RANDOM_SEED + 17 * len(controls) + seed_offset)
    null_draws = rng.choice(deviations, size=1000, replace=True)
    mc_p = (1 + int((null_draws >= event_deviation).sum())) / 1001
    return {
        "对照样本数": len(controls),
        "对照中位数": median,
        "稳健Z": robust_z,
        "稳健效应强度": abs(robust_z) / (1 + abs(robust_z)) if np.isfinite(robust_z) else np.nan,
        "历史百分位": percentile,
        "百分位极端度": extremeness,
        "经验双侧P": p_value,
        "蒙特卡洛双侧P": mc_p,
    }


def safe_median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.median()) if len(numeric) else np.nan


def choose_regime_controls(
    frame: pd.DataFrame,
    date_column: str,
    target_date: pd.Timestamp,
    event_type: int,
    calendar_band_days: int = CALENDAR_BAND_DAYS,
) -> tuple[pd.DataFrame, str]:
    controls = frame[frame[date_column].dt.year.isin(CONTROL_YEARS)].copy()
    seasonal = controls[seasonal_mask(controls[date_column], target_date, calendar_band_days)]
    matched = seasonal[seasonal["天气型编号"].eq(event_type)]
    scope = f"同季节±{calendar_band_days}天且同天气型"
    if len(matched) < MIN_CONTROL_COUNT:
        seasonal30 = controls[seasonal_mask(controls[date_column], target_date, 30)]
        matched = seasonal30[seasonal30["天气型编号"].eq(event_type)]
        scope = "同季节±30天且同天气型（样本扩展）"
    if len(matched) < MIN_CONTROL_COUNT:
        matched = controls[controls["天气型编号"].eq(event_type)]
        scope = "全年同天气型（样本不足回退）"
    return matched, scope


def evaluate_layer(
    frame: pd.DataFrame,
    dictionary: pd.DataFrame,
    classified: pd.DataFrame,
    mode: str,
    layer: str,
    calendar_band_days: int = CALENDAR_BAND_DAYS,
) -> pd.DataFrame:
    if layer == "区域观测站":
        date_column = "Timestamp"
        event_endpoint = STATION_EVENT_ENDPOINT
        target_date = STATION_EVENT_ENDPOINT
        data = frame.merge(classified[["Timestamp", "天气型编号"]], on="Timestamp", how="left")
    else:
        date_column = "date"
        event_endpoint = SOURCE_EVENT_ENDPOINT
        target_date = SOURCE_EVENT_ENDPOINT
        weather = classified[["Timestamp", "天气型编号"]].copy()
        # The formal source cache is evaluated at the same 10:00 Beijing endpoint
        # as the station cache; no previous-complete-day substitution is needed.
        weather["date"] = weather["Timestamp"]
        data = frame.merge(weather[["date", "天气型编号"]], on="date", how="left")

    event_row = data[data[date_column].eq(event_endpoint)]
    if event_row.empty:
        raise RuntimeError(f"找不到事件端点：{layer} {event_endpoint}")
    event_type = int(event_row["天气型编号"].iloc[0])
    historical = data[data[date_column].dt.year.isin(CONTROL_YEARS)].copy()
    calendar_controls = historical[seasonal_mask(historical[date_column], target_date, calendar_band_days)]
    regime_controls, regime_scope = choose_regime_controls(
        data, date_column, target_date, event_type, calendar_band_days
    )

    rows = []
    for _, item in dictionary.iterrows():
        feature = item["衍生指标"]
        if feature not in data.columns:
            continue
        event_value = pd.to_numeric(event_row[feature], errors="coerce").iloc[0]
        calendar = robust_comparison(float(event_value), calendar_controls[feature])
        regime = robust_comparison(float(event_value), regime_controls[feature])
        row = {
            "清洗版本": mode,
            "数据层": layer,
            "衍生指标": feature,
            "基础指标": re.sub(r"_\d+h$", "", feature),
            "因素组ID": item["因素组ID"],
            "物理因素": item["物理因素"],
            "指标中文名": item["指标中文名"],
            "对应原始字段": item["对应原始字段"],
            "窗口小时": int(item["窗口小时"]),
            "单位": item["单位"],
            "事件值": float(event_value),
            "偏离方向": "偏高" if event_value > calendar["对照中位数"] else "偏低",
            "事件天气型": event_type,
            "同天气型对照口径": regime_scope,
        }
        row.update({f"同期{k}": v for k, v in calendar.items()})
        row.update({f"同天气型{k}": v for k, v in regime.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_base_metrics(window_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in window_results.groupby(
        ["清洗版本", "数据层", "基础指标", "因素组ID", "物理因素", "对应原始字段"],
        dropna=False,
    ):
        mode, layer, base, factor_id, factor, fields = keys
        signs = np.sign(pd.to_numeric(group["同期稳健Z"], errors="coerce").dropna())
        direction_consistency = max((signs > 0).mean(), (signs < 0).mean()) if len(signs) else np.nan
        dominant_direction = "偏高" if (signs > 0).sum() >= (signs < 0).sum() else "偏低"
        calendar_p = pd.to_numeric(group["同期经验双侧P"], errors="coerce")
        regime_p = pd.to_numeric(group["同天气型经验双侧P"], errors="coerce")
        calendar_mc = pd.to_numeric(group["同期蒙特卡洛双侧P"], errors="coerce")
        regime_mc = pd.to_numeric(group["同天气型蒙特卡洛双侧P"], errors="coerce")
        paired_signs = (
            np.sign(pd.to_numeric(group["同期稳健Z"], errors="coerce"))
            * np.sign(pd.to_numeric(group["同天气型稳健Z"], errors="coerce"))
        )
        dual_direction = float((paired_signs > 0).mean()) if paired_signs.notna().any() else np.nan
        support_values = np.r_[
            1 - calendar_mc.dropna().to_numpy(),
            1 - regime_mc.dropna().to_numpy(),
        ]
        randomization_support = float(np.median(support_values)) if len(support_values) else np.nan
        dual_mc_significant = int(((calendar_mc <= 0.10) & (regime_mc <= 0.10)).sum())
        if dual_direction >= 0.80 and dual_mc_significant >= 1:
            gate = "强支持"
        elif dual_direction >= 0.60 and randomization_support >= 0.75:
            gate = "中等支持"
        else:
            gate = "未通过"
        rows.append(
            {
                "清洗版本": mode,
                "数据层": layer,
                "基础指标": base,
                "因素组ID": factor_id,
                "物理因素": factor,
                "对应原始字段": fields,
                "窗口数": int(len(group)),
                "主导方向": dominant_direction,
                "多窗口方向一致性": direction_consistency,
                "同期稳健效应强度": safe_median(group["同期稳健效应强度"]),
                "同期百分位极端度": safe_median(group["同期百分位极端度"]),
                "同天气型稳健效应强度": safe_median(group["同天气型稳健效应强度"]),
                "同天气型百分位极端度": safe_median(group["同天气型百分位极端度"]),
                "随机化支持度": randomization_support,
                "双对照方向一致性": dual_direction,
                "显著性门控": gate,
                "同期P中位数": safe_median(calendar_p),
                "同天气型P中位数": safe_median(regime_p),
                "同期蒙特卡洛P中位数": safe_median(calendar_mc),
                "同天气型蒙特卡洛P中位数": safe_median(regime_mc),
                "双对照显著窗口数(P<=0.10)": int(((calendar_p <= 0.10) & (regime_p <= 0.10)).sum()),
                "双对照蒙特卡洛显著窗口数(P<=0.10)": dual_mc_significant,
            }
        )
    return pd.DataFrame(rows)


def normalize_benefit(matrix: pd.DataFrame) -> pd.DataFrame:
    normalized = pd.DataFrame(index=matrix.index)
    for column in matrix.columns:
        values = pd.to_numeric(matrix[column], errors="coerce")
        fill_value = float(values.dropna().median()) if values.notna().any() else 0.0
        values = values.fillna(fill_value)
        lower, upper = float(values.min()), float(values.max())
        normalized[column] = (values - lower) / (upper - lower) if upper > lower else 0.0
    return normalized


def entropy_weights(normalized: pd.DataFrame) -> pd.Series:
    eps = 1e-12
    shifted = normalized + eps
    proportions = shifted.div(shifted.sum(axis=0), axis=1)
    n = len(normalized)
    entropy = -(proportions * np.log(proportions)).sum(axis=0) / np.log(n)
    divergence = 1 - entropy
    if divergence.sum() <= eps:
        return pd.Series(1 / len(divergence), index=divergence.index)
    return divergence / divergence.sum()


def critic_weights(normalized: pd.DataFrame) -> pd.Series:
    std = normalized.std(axis=0, ddof=0)
    correlation = normalized.corr().fillna(0.0)
    information = std * (1 - correlation).sum(axis=1)
    if information.sum() <= 1e-12:
        return pd.Series(1 / len(information), index=information.index)
    return information / information.sum()


def topsis_rank(
    frame: pd.DataFrame,
    id_columns: list[str],
    entropy_share: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = normalize_benefit(frame[CRITERIA])
    entropy = entropy_weights(matrix)
    critic = critic_weights(matrix)
    weights = entropy_share * entropy + (1 - entropy_share) * critic
    weights = weights / weights.sum()
    weighted = matrix * weights
    ideal_best = weighted.max(axis=0)
    ideal_worst = weighted.min(axis=0)
    distance_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    distance_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))
    denominator = distance_best + distance_worst
    score = distance_worst / denominator.replace(0, np.nan)
    ranked = frame.copy()
    ranked["TOPSIS贴近度"] = score.fillna(0.0)
    ranked["排名"] = ranked["TOPSIS贴近度"].rank(ascending=False, method="min").astype(int)
    ranked = ranked.sort_values(["排名", "TOPSIS贴近度"], ascending=[True, False]).reset_index(drop=True)
    weight_table = pd.DataFrame(
        {
            "评价准则": CRITERIA,
            "熵权": entropy.reindex(CRITERIA).to_numpy(),
            "CRITIC权重": critic.reindex(CRITERIA).to_numpy(),
            "组合权重": weights.reindex(CRITERIA).to_numpy(),
        }
    )
    return ranked, weight_table


def aggregate_factors(base_ranked: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (mode, factor_id, factor), group in base_ranked.groupby(["清洗版本", "因素组ID", "物理因素"]):
        best = group.sort_values("TOPSIS贴近度", ascending=False).iloc[0]
        layers = "+".join(sorted(group["数据层"].unique()))
        record = {
            "清洗版本": mode,
            "因素组ID": factor_id,
            "物理因素": factor,
            "证据数据层": layers,
            "候选基础指标数": int(len(group)),
            "代表基础指标": best["基础指标"],
            "代表指标原始字段": best["对应原始字段"],
            "代表指标主导方向": best["主导方向"],
            "代表指标TOPSIS贴近度": best["TOPSIS贴近度"],
            "代表指标显著性门控": best["显著性门控"],
            "代表指标双对照方向一致性": best["双对照方向一致性"],
        }
        for criterion in CRITERIA:
            record[criterion] = safe_median(group[criterion])
        rows.append(record)
    return pd.DataFrame(rows)


def write_report(factor_ranking: pd.DataFrame, centers: pd.DataFrame, weights: pd.DataFrame) -> None:
    strict = factor_ranking[factor_ranking["清洗版本"].eq("strict")].sort_values("排名")
    event_type = int(centers.loc[centers["清洗版本"].eq("strict"), "事件天气型"].iloc[0])
    type_row = centers[(centers["清洗版本"].eq("strict")) & (centers["天气型编号"].eq(event_type))].iloc[0]
    lines = [
        "# 第一问重构方案：2026事件—历史对照初排",
        "",
        "## 当前结论的性质",
        "",
        "- 这是可解释的事件关联初排，不是因果贡献率，也不是最终冻结结论。",
        "- 主历史背景严格只用2021—2024年；2025年未进入GMM、标准化、权重或排序。",
        "- 源区ERA5-Land由本项目下载的CDS大气—能量场与原生网格雪—深层土壤数据正式重算。",
        "- 观测站变量仅解释区域天气与边界层背景，不替代约65 km外的源区局地状态。",
        "",
        "## 对照结构",
        "",
        "1. 历史同期：2021—2024年目标日期前后15天的每日10:00端点。",
        "2. 同天气型：上述同期样本中与事件GMM天气型相同的日子；不足时按冻结规则扩展。",
        f"3. 2026事件被分到天气型{event_type}：{type_row['解释性名称']}，最大后验概率为{type_row['事件最大后验概率']:.3f}。",
        "4. 五个时间窗口为24、48、72、120、168小时；先合并窗口，再合并同一物理因素。",
        "",
        "## 严格清洗版因素排序",
        "",
        "| 排名 | 物理因素 | 代表指标 | 原始字段 | 方向 | 数据层 | TOPSIS贴近度 |",
        "|---:|---|---|---|---|---|---:|",
    ]
    for _, row in strict.iterrows():
        lines.append(
            f"| {int(row['排名'])} | {row['物理因素']} | {row['代表基础指标']} | {row['代表指标原始字段']} | "
            f"{row['代表指标主导方向']} | {row['证据数据层']} | {row['TOPSIS贴近度']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 权重说明",
            "",
            "评价准则的权重不是主观指定：熵权反映准则的区分度，CRITIC同时考虑区分度与准则间冲突性，二者等权组合后进入TOPSIS。等权组合本身将在稳健性阶段改为多组权重敏感性检验。",
            "",
            "| 准则 | 熵权 | CRITIC | 组合权重 |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in weights[weights["清洗版本"].eq("strict")].iterrows():
        lines.append(f"| {row['评价准则']} | {row['熵权']:.3f} | {row['CRITIC权重']:.3f} | {row['组合权重']:.3f} |")
    lines.extend(
        [
            "",
            "## 尚不能越过的边界",
            "",
            "- GMM天气型是对照筛选工具，不是致灾机制。",
            "- 2026事件本体是冰岩崩塌—碎屑流—泥石流—山洪链；地形、裂隙、冻土、冰体加载、灾前形变和沟道松散物属于机理支撑层，不与小时气象因素争夺前五。",
            "- 本主场景随后由648个清洗、天气型、日期带、窗口、权重、空间支持和高程订正场景检验；论文主结论应以稳健性汇总为准。",
        ]
    )
    (OUTPUT_DIR / "15_事件对照因素初排报告.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    station_dictionary = pd.read_csv(OUTPUT_DIR / "07_站点衍生指标字典.csv")
    station_dictionary = station_dictionary.drop_duplicates(["清洗版本", "衍生指标"])
    source_dictionary = pd.read_csv(OUTPUT_DIR / "23_源区ERA5正式衍生指标字典.csv")
    source = pd.read_parquet(CACHE_DIR / "source_daily_features_official.parquet")

    window_results = []
    centers_all = []
    bic_all = []
    for mode in ["strict", "expanded"]:
        classified, centers, bic = fit_weather_types(mode)
        event_weather = classified[classified["Timestamp"].eq(STATION_EVENT_ENDPOINT)].iloc[0]
        centers["事件天气型"] = int(event_weather["天气型编号"])
        centers["事件最大后验概率"] = float(event_weather["最大后验概率"])
        centers_all.append(centers)
        bic_all.append(bic)

        station = pd.read_parquet(CACHE_DIR / f"station_daily_10h_features_{mode}.parquet")
        window_results.append(
            evaluate_layer(
                station,
                station_dictionary[station_dictionary["清洗版本"].eq(mode)],
                classified,
                mode,
                "区域观测站",
            )
        )
        window_results.append(
            evaluate_layer(source, source_dictionary, classified, mode, "源区ERA5正式重算")
        )

    windows = pd.concat(window_results, ignore_index=True)
    bases = aggregate_base_metrics(windows)
    ranked_bases = []
    base_weights = []
    factor_rows = []
    factor_weights = []
    for mode in ["strict", "expanded"]:
        mode_bases = bases[bases["清洗版本"].eq(mode)].reset_index(drop=True)
        ranked, weights = topsis_rank(mode_bases, ["基础指标"])
        weights.insert(0, "清洗版本", mode)
        ranked_bases.append(ranked)
        base_weights.append(weights)
        factors = aggregate_factors(ranked)
        factor_ranked, weights_factor = topsis_rank(factors, ["因素组ID"])
        weights_factor.insert(0, "清洗版本", mode)
        factor_rows.append(factor_ranked)
        factor_weights.append(weights_factor)

    base_ranked_all = pd.concat(ranked_bases, ignore_index=True)
    factor_ranked_all = pd.concat(factor_rows, ignore_index=True)
    weight_all = pd.concat(factor_weights, ignore_index=True)
    centers_out = pd.concat(centers_all, ignore_index=True)

    windows.to_csv(OUTPUT_DIR / "10_全部窗口双对照评价.csv", index=False, encoding="utf-8-sig")
    base_ranked_all.to_csv(OUTPUT_DIR / "11_基础指标跨窗口排序.csv", index=False, encoding="utf-8-sig")
    factor_ranked_all.to_csv(OUTPUT_DIR / "12_物理因素初排.csv", index=False, encoding="utf-8-sig")
    centers_out.to_csv(OUTPUT_DIR / "13_GMM天气型解释.csv", index=False, encoding="utf-8-sig")
    pd.concat(bic_all, ignore_index=True).to_csv(OUTPUT_DIR / "14_GMM_BIC诊断.csv", index=False, encoding="utf-8-sig")
    pd.concat(base_weights, ignore_index=True).to_csv(OUTPUT_DIR / "16_基础指标评价权重.csv", index=False, encoding="utf-8-sig")
    weight_all.to_csv(OUTPUT_DIR / "17_物理因素评价权重.csv", index=False, encoding="utf-8-sig")
    write_report(factor_ranked_all, centers_out, weight_all)
    strict_top = factor_ranked_all[factor_ranked_all["清洗版本"].eq("strict")].nsmallest(5, "排名")
    print("严格版暂定前五：")
    print(strict_top[["排名", "物理因素", "代表基础指标", "代表指标主导方向", "TOPSIS贴近度"]].to_string(index=False))
    print("结果仍需稳健性矩阵检验，不应直接写成最终结论。")


if __name__ == "__main__":
    main()
