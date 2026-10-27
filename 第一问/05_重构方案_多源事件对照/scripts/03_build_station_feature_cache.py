from __future__ import annotations

"""构造题目观测站的可解释多尺度特征缓存。

设计原则：
1. 每个衍生量都能追溯到题目原始字段；
2. 只做有明确物理含义的统计或经典诊断量；
3. 滚动窗口逐年计算，防止跨年；2025仅保留给事后跨机理核查；
4. 排名用每日10:00端点，另存2026事件前逐小时轨迹供画图；
5. 本脚本只生成缓存，不进行排名，也不生成图片。
"""

from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
WORK_DIR = SCRIPT_DIR.parent
QUESTION_DIR = WORK_DIR.parent
PREPROCESS_DIR = QUESTION_DIR / "01_数据预处理"
CACHE_DIR = WORK_DIR / "cache"
OUTPUT_DIR = WORK_DIR / "outputs"

EVENT_CUTOFF = pd.Timestamp("2026-08-26 10:00:00")
WINDOWS = [24, 48, 72, 120, 168]
ANALYSIS_YEARS = [2021, 2022, 2023, 2024, 2025, 2026]
ROLLING_MIN_COVERAGE = 0.80
DAYLIGHT_RSD_THRESHOLD = 50.0
MIN_VECTOR_SHEAR_SQ = 0.25
GRAVITY = 9.80665
KAPPA = 0.286

SENSOR_HEIGHTS = {
    "1m": 1.0,
    "2m": 2.0,
    "4m": 4.0,
    "10m": 10.0,
    "20m": 20.0,
}


def load_station(mode: str) -> pd.DataFrame:
    base = PREPROCESS_DIR / "processed" / mode
    merged: pd.DataFrame | None = None
    for group in ["MET", "RADM", "SOIL", "FLUX"]:
        path = next(base.glob(f"{group}_Gyirong_2021_2026_{mode}.parquet"))
        frame = pd.read_parquet(path)
        keep = [
            column
            for column in frame.columns
            if column == "Timestamp"
            or (column != "source_file" and not column.lower().startswith("qc_"))
        ]
        frame = frame[keep]
        merged = frame if merged is None else merged.merge(frame, on="Timestamp", how="outer")
    assert merged is not None
    merged["Timestamp"] = pd.to_datetime(merged["Timestamp"])
    merged = merged[merged["Timestamp"].dt.year.isin(ANALYSIS_YEARS)].copy()
    return merged.sort_values("Timestamp").reset_index(drop=True)


def saturation_vapor_pressure_hpa(temp_c: pd.Series) -> pd.Series:
    """Magnus公式；用于从题目2m温度和相对湿度得到可解释VPD。"""
    return 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5))


def wind_components(speed: pd.Series, direction_deg: pd.Series) -> tuple[pd.Series, pd.Series]:
    """气象风向（来向）转东向u和北向v。"""
    radians = np.deg2rad(direction_deg)
    return -speed * np.sin(radians), -speed * np.cos(radians)


def base_hourly_variables(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    b = pd.DataFrame({"Timestamp": raw["Timestamp"]})
    catalog: list[dict[str, object]] = []

    def add(
        name: str,
        values: pd.Series,
        factor_id: str,
        factor: str,
        chinese: str,
        raw_fields: str,
        definition: str,
        unit: str,
        priority: str = "B",
    ) -> None:
        b[name] = values.astype(float)
        catalog.append(
            {
                "基础变量": name,
                "因素组ID": factor_id,
                "物理因素": factor,
                "中文名称": chinese,
                "对应原始字段": raw_fields,
                "透明定义": definition,
                "单位": unit,
                "优先级": priority,
            }
        )

    temp = raw["Ta_2m(degree C)"]
    add("temp_2m", temp, "TH01", "空气温度与热量累积", "2米气温", "Ta_2m(degree C)", "原始2米气温", "degC", "A")
    add("positive_temp", temp.clip(lower=0), "TH01", "空气温度与热量累积", "正温度", "Ta_2m(degree C)", "max(Ta_2m,0)", "degC", "A")

    precip = raw["Precip(mm)"].clip(lower=0)
    add("precip", precip, "PR01", "降水与液态水输入", "小时降水", "Precip(mm)", "非负小时降水", "mm", "A")

    net_radiation = raw["Rsd(W/m2)"] - raw["Rsu(W/m2)"] + raw["Rld(W/m2)"] - raw["Rlu(W/m2)"]
    add("net_radiation", net_radiation, "RD01", "辐射与地表能量", "净辐射", "Rsd+Rsu+Rld+Rlu", "Rsd-Rsu+Rld-Rlu", "W m-2", "A")
    add("positive_net_radiation", net_radiation.clip(lower=0), "RD01", "辐射与地表能量", "正净辐射", "Rsd+Rsu+Rld+Rlu", "max(Rn,0)", "W m-2", "A")
    add("shortwave_down", raw["Rsd(W/m2)"], "RD01", "辐射与地表能量", "下行短波辐射", "Rsd(W/m2)", "原始下行短波辐射", "W m-2")
    daylight = raw["Rsd(W/m2)"] >= DAYLIGHT_RSD_THRESHOLD
    albedo = (raw["Rsu(W/m2)"] / raw["Rsd(W/m2)"]).where(daylight)
    albedo = albedo.where(albedo.between(0, 1.2))
    add("albedo", albedo, "RD01", "辐射与地表能量", "白天短波反照率", "Rsu(W/m2)+Rsd(W/m2)", "Rsu/Rsd；仅Rsd>=50且0<=比值<=1.2", "1")

    add("sensible_heat", raw["H(W/m2)"], "SF01", "地表湍流热交换", "感热通量", "H(W/m2)", "原始感热通量", "W m-2", "A")
    add("latent_heat", raw["LE(W/m2)"], "SF01", "地表湍流热交换", "潜热通量", "LE(W/m2)", "原始潜热通量", "W m-2", "A")
    bowen = (raw["H(W/m2)"] / raw["LE(W/m2)"]).where(raw["LE(W/m2)"].abs() >= 10)
    add("bowen_ratio", bowen, "SF01", "地表湍流热交换", "波文比", "H(W/m2)+LE(W/m2)", "H/LE；仅|LE|>=10 W m-2", "1")

    pressure = raw["Pressure(hPa)"]
    theta: dict[str, pd.Series] = {}
    uv: dict[str, tuple[pd.Series, pd.Series]] = {}
    for label in SENSOR_HEIGHTS:
        theta[label] = (raw[f"Ta_{label}(degree C)"] + 273.15) * (1000.0 / pressure) ** KAPPA
        uv[label] = wind_components(raw[f"WS_{label}(m/s)"], raw[f"WD_{label}(degree)"])
    dz = SENSOR_HEIGHTS["20m"] - SENSOR_HEIGHTS["1m"]
    theta_gradient = (theta["20m"] - theta["1m"]) / dz
    add("theta_gradient", theta_gradient, "BL01", "边界层稳定结构", "1—20米位温梯度", "Ta_1m+Ta_20m+Pressure", "[theta(20m)-theta(1m)]/19m", "K m-1", "A")
    du = uv["20m"][0] - uv["1m"][0]
    dv = uv["20m"][1] - uv["1m"][1]
    vector_shear_sq = du.pow(2) + dv.pow(2)
    vector_shear = np.sqrt(vector_shear_sq) / dz
    add("vector_wind_shear", vector_shear, "BL01", "边界层稳定结构", "1—20米风矢量切变", "WS/WD_1m+WS/WD_20m", "sqrt((du)^2+(dv)^2)/19m", "s-1")
    theta_mean = (theta["20m"] + theta["1m"]) / 2
    bulk_ri = (GRAVITY / theta_mean) * (theta["20m"] - theta["1m"]) * dz / vector_shear_sq
    bulk_ri = bulk_ri.where(vector_shear_sq >= MIN_VECTOR_SHEAR_SQ)
    add("bulk_richardson", bulk_ri, "BL01", "边界层稳定结构", "体积Richardson数", "Ta/WS/WD_1m和20m+Pressure", "(g/theta_bar)*delta_theta*delta_z/(delta_u^2+delta_v^2)；风矢量差平方>=0.25", "1", "A")

    add("tke", raw["TKE(m2/s2)"], "TB01", "湍流强度", "湍动能", "TKE(m2/s2)", "原始湍动能", "m2 s-2", "A")
    add("ustar", raw["USTAR(m/s)"], "TB01", "湍流强度", "摩擦速度", "USTAR(m/s)", "原始摩擦速度", "m s-1", "A")
    add("tau", raw["TAU(kg/m*s2)"], "TB01", "湍流强度", "动量通量", "TAU(kg/m*s2)", "原始动量通量", "kg m-1 s-2")

    rh = raw["RH_2m(%)"]
    vpd = saturation_vapor_pressure_hpa(temp) * (1 - rh / 100.0)
    vpd = vpd.where(rh.between(0, 100)).clip(lower=0)
    valid_rh = rh.where(rh.gt(0) & rh.le(100))
    gamma = np.log(valid_rh / 100.0) + 17.67 * temp / (temp + 243.5)
    dewpoint = 243.5 * gamma / (17.67 - gamma)
    lcl = (125.0 * (temp - dewpoint)).clip(lower=0)
    add("relative_humidity", rh, "HM01", "大气湿度与干湿状态", "2米相对湿度", "RH_2m(%)", "原始2米相对湿度", "%")
    add("vpd_derived", vpd, "HM01", "大气湿度与干湿状态", "计算水汽压亏缺", "Ta_2m(degree C)+RH_2m(%)", "es(Ta)*(1-RH/100)，Magnus饱和水汽压公式", "hPa", "A")
    add(
        "lcl_height",
        lcl,
        "HM01",
        "大气湿度与干湿状态",
        "近似抬升凝结高度",
        "Ta_2m(degree C)+RH_2m(%)",
        "先由Magnus公式反算露点Td，再计算LCL=125*(T-Td)",
        "m",
    )

    soil_t_cols = [f"soilT_{depth}m(degree C)" for depth in ["0.1", "0.2", "0.4", "0.8", "1.6"]]
    soil_temp_profile = raw[soil_t_cols].mean(axis=1, skipna=False)
    soil_temp_gradient = (raw["soilT_1.6m(degree C)"] - raw["soilT_0.1m(degree C)"]) / 1.5
    add("soil_temp_shallow", raw["soilT_0.1m(degree C)"], "ST01", "土壤温度与冻融", "0.1米土壤温度", "soilT_0.1m(degree C)", "原始浅层土壤温度", "degC")
    add("soil_temp_profile", soil_temp_profile, "ST01", "土壤温度与冻融", "土壤剖面平均温度", "+".join(soil_t_cols), "五层土温同时有效时的算术平均", "degC")
    add("soil_temp_gradient", soil_temp_gradient, "ST01", "土壤温度与冻融", "深浅层土温梯度", "soilT_0.1m+soilT_1.6m", "(T1.6m-T0.1m)/1.5m", "K m-1")
    crossing = ((raw["soilT_0.1m(degree C)"] >= 0) != (raw["soilT_0.1m(degree C)"].shift(1) >= 0)).astype(float)
    crossing[raw["soilT_0.1m(degree C)"].isna() | raw["soilT_0.1m(degree C)"].shift(1).isna()] = np.nan
    add("freeze_thaw_crossing", crossing, "ST01", "土壤温度与冻融", "浅层冻融过零", "soilT_0.1m(degree C)", "相邻小时是否跨越0摄氏度", "count")

    vwc_cols = [f"VWC_{depth}m(m3/m3)" for depth in ["0.1", "0.2", "0.4", "0.8", "1.6"]]
    vwc_profile = raw[vwc_cols].mean(axis=1, skipna=False)
    vwc_gradient = (raw["VWC_1.6m(m3/m3)"] - raw["VWC_0.1m(m3/m3)"]) / 1.5
    add("vwc_profile", vwc_profile, "SM01", "土壤湿润状态", "剖面平均体积含水量", "+".join(vwc_cols), "五层VWC同时有效时的算术平均", "m3 m-3")
    add("vwc_shallow", raw["VWC_0.1m(m3/m3)"], "SM01", "土壤湿润状态", "0.1米体积含水量", "VWC_0.1m(m3/m3)", "原始浅层体积含水量", "m3 m-3")
    add("vwc_gradient", vwc_gradient, "SM01", "土壤湿润状态", "深浅层含水量梯度", "VWC_0.1m+VWC_1.6m", "(VWC1.6m-VWC0.1m)/1.5m", "m3 m-4")

    add("pressure", pressure, "PS01", "气压与天气过程", "气压", "Pressure(hPa)", "原始站点气压", "hPa")
    add("wind10", raw["WS_10m(m/s)"], "WN01", "风场与通风条件", "10米风速", "WS_10m(m/s)", "原始10米风速", "m s-1")
    direction = np.deg2rad(raw["WD_10m(degree)"])
    add("wind10_sin", np.sin(direction), "WN01", "风场与通风条件", "10米风向正弦分量", "WD_10m(degree)", "sin(WD10)", "1")
    add("wind10_cos", np.cos(direction), "WN01", "风场与通风条件", "10米风向余弦分量", "WD_10m(degree)", "cos(WD10)", "1")
    return b, catalog


def roll(series: pd.Series, window: int, method: str, coverage: float = ROLLING_MIN_COVERAGE) -> pd.Series:
    minimum = max(1, int(np.ceil(window * coverage)))
    obj = series.rolling(window, min_periods=minimum)
    return getattr(obj, method)()


def construct_window_features_one_year(base: pd.DataFrame, catalog: list[dict[str, object]]) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    feature_values: dict[str, pd.Series] = {"Timestamp": base["Timestamp"]}
    definitions: list[dict[str, object]] = []
    cat = {row["基础变量"]: row for row in catalog}

    def add_feature(name: str, values: pd.Series, base_name: str, statistic: str, window: int, unit: str | None = None) -> None:
        feature_values[name] = values
        source = cat[base_name]
        definitions.append(
            {
                "衍生指标": name,
                "因素组ID": source["因素组ID"],
                "物理因素": source["物理因素"],
                "指标中文名": f"{window}小时{source['中文名称']}{statistic}",
                "对应原始字段": source["对应原始字段"],
                "基础变量定义": source["透明定义"],
                "窗口统计": statistic,
                "窗口小时": window,
                "单位": unit or source["单位"],
                "优先级": source["优先级"],
            }
        )

    for window in WINDOWS:
        suffix = f"_{window}h"
        add_feature(f"sta_tmean{suffix}", roll(base["temp_2m"], window, "mean"), "temp_2m", "均值", window)
        add_feature(f"sta_tmax{suffix}", roll(base["temp_2m"], window, "max"), "temp_2m", "最大值", window)
        add_feature(f"sta_pdd{suffix}", roll(base["positive_temp"], window, "sum") / 24.0, "positive_temp", "正积温", window, "degC d")
        add_feature(f"sta_temp_change{suffix}", base["temp_2m"] - base["temp_2m"].shift(window - 1), "temp_2m", "首末变化", window, "degC")

        add_feature(f"sta_precip_sum{suffix}", roll(base["precip"], window, "sum"), "precip", "累计", window)
        add_feature(f"sta_precip_max1h{suffix}", roll(base["precip"], window, "max"), "precip", "窗口内最大1小时", window)
        max3 = roll(base["precip"], 3, "sum", coverage=1.0)
        max6 = roll(base["precip"], 6, "sum", coverage=1.0)
        add_feature(f"sta_precip_max3h{suffix}", roll(max3, window, "max"), "precip", "窗口内最大3小时累计", window)
        add_feature(f"sta_precip_max6h{suffix}", roll(max6, window, "max"), "precip", "窗口内最大6小时累计", window)

        add_feature(f"sta_netrad_mean{suffix}", roll(base["net_radiation"], window, "mean"), "net_radiation", "均值", window)
        add_feature(f"sta_positive_netrad{suffix}", roll(base["positive_net_radiation"], window, "sum") / 1000.0, "positive_net_radiation", "累计", window, "kWh m-2")
        add_feature(f"sta_swdown{suffix}", roll(base["shortwave_down"].clip(lower=0), window, "sum") / 1000.0, "shortwave_down", "累计", window, "kWh m-2")
        add_feature(f"sta_albedo{suffix}", roll(base["albedo"], window, "median", coverage=0.20), "albedo", "白天中位数", window)

        add_feature(f"sta_H_median{suffix}", roll(base["sensible_heat"], window, "median"), "sensible_heat", "中位数", window)
        add_feature(f"sta_H_positive{suffix}", roll(base["sensible_heat"].clip(lower=0), window, "sum") / 1000.0, "sensible_heat", "正值累计", window, "kWh m-2")
        add_feature(f"sta_LE_median{suffix}", roll(base["latent_heat"], window, "median"), "latent_heat", "中位数", window)
        add_feature(f"sta_LE_positive{suffix}", roll(base["latent_heat"].clip(lower=0), window, "sum") / 1000.0, "latent_heat", "正值累计", window, "kWh m-2")
        add_feature(f"sta_bowen_median{suffix}", roll(base["bowen_ratio"], window, "median", coverage=0.50), "bowen_ratio", "中位数", window)

        add_feature(f"sta_theta_gradient_mean{suffix}", roll(base["theta_gradient"], window, "mean"), "theta_gradient", "均值", window)
        add_feature(f"sta_vector_shear_mean{suffix}", roll(base["vector_wind_shear"], window, "mean"), "vector_wind_shear", "均值", window)
        add_feature(f"sta_bulk_ri_median{suffix}", roll(base["bulk_richardson"], window, "median", coverage=0.50), "bulk_richardson", "中位数", window)
        add_feature(f"sta_unstable_fraction{suffix}", roll((base["bulk_richardson"] < 0).astype(float).where(base["bulk_richardson"].notna()), window, "mean", coverage=0.50), "bulk_richardson", "不稳定时数比例(Ri<0)", window)

        for variable, label in [("tke", "TKE"), ("ustar", "ustar"), ("tau", "tau")]:
            add_feature(f"sta_{label}_median{suffix}", roll(base[variable], window, "median"), variable, "中位数", window)
            p90 = base[variable].rolling(
                window, min_periods=int(np.ceil(window * ROLLING_MIN_COVERAGE))
            ).quantile(0.90)
            add_feature(f"sta_{label}_p90{suffix}", p90, variable, "90%分位数", window)

        add_feature(f"sta_RH_mean{suffix}", roll(base["relative_humidity"], window, "mean"), "relative_humidity", "均值", window)
        add_feature(f"sta_VPD_mean{suffix}", roll(base["vpd_derived"], window, "mean"), "vpd_derived", "均值", window)
        add_feature(f"sta_VPD_p90{suffix}", base["vpd_derived"].rolling(window, min_periods=int(np.ceil(window * ROLLING_MIN_COVERAGE))).quantile(0.90), "vpd_derived", "90%分位数", window)
        add_feature(f"sta_lcl_mean{suffix}", roll(base["lcl_height"], window, "mean"), "lcl_height", "均值", window)
        add_feature(f"sta_lcl_p90{suffix}", base["lcl_height"].rolling(window, min_periods=int(np.ceil(window * ROLLING_MIN_COVERAGE))).quantile(0.90), "lcl_height", "90%分位数", window)

        add_feature(f"sta_soilT_shallow_mean{suffix}", roll(base["soil_temp_shallow"], window, "mean"), "soil_temp_shallow", "均值", window)
        add_feature(f"sta_soilT_profile_mean{suffix}", roll(base["soil_temp_profile"], window, "mean"), "soil_temp_profile", "均值", window)
        add_feature(f"sta_soilT_change{suffix}", base["soil_temp_shallow"] - base["soil_temp_shallow"].shift(window - 1), "soil_temp_shallow", "首末变化", window)
        add_feature(f"sta_soilT_gradient_mean{suffix}", roll(base["soil_temp_gradient"], window, "mean"), "soil_temp_gradient", "均值", window)
        add_feature(f"sta_freeze_thaw_count{suffix}", roll(base["freeze_thaw_crossing"], window, "sum"), "freeze_thaw_crossing", "次数", window)

        add_feature(f"sta_VWC_profile_mean{suffix}", roll(base["vwc_profile"], window, "mean"), "vwc_profile", "均值", window)
        add_feature(f"sta_VWC_shallow_mean{suffix}", roll(base["vwc_shallow"], window, "mean"), "vwc_shallow", "均值", window)
        add_feature(f"sta_VWC_change{suffix}", base["vwc_profile"] - base["vwc_profile"].shift(window - 1), "vwc_profile", "首末变化", window)
        add_feature(f"sta_VWC_gradient_mean{suffix}", roll(base["vwc_gradient"], window, "mean"), "vwc_gradient", "均值", window)

        add_feature(f"sta_pressure_mean{suffix}", roll(base["pressure"], window, "mean"), "pressure", "均值", window)
        add_feature(f"sta_pressure_change{suffix}", base["pressure"] - base["pressure"].shift(window - 1), "pressure", "首末变化", window)
        add_feature(f"sta_pressure_range{suffix}", roll(base["pressure"], window, "max") - roll(base["pressure"], window, "min"), "pressure", "极差", window)
        add_feature(f"sta_wind10_mean{suffix}", roll(base["wind10"], window, "mean"), "wind10", "均值", window)
        add_feature(f"sta_wind10_p90{suffix}", base["wind10"].rolling(window, min_periods=int(np.ceil(window * ROLLING_MIN_COVERAGE))).quantile(0.90), "wind10", "90%分位数", window)
        mean_sin = roll(base["wind10_sin"], window, "mean")
        mean_cos = roll(base["wind10_cos"], window, "mean")
        circ_var = 1.0 - np.sqrt(mean_sin.pow(2) + mean_cos.pow(2))
        add_feature(f"sta_wind_direction_variance{suffix}", circ_var, "wind10_sin", "圆方差", window)
    return pd.DataFrame(feature_values), definitions


def build_mode(mode: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = load_station(mode)
    base, base_catalog = base_hourly_variables(raw)
    frames: list[pd.DataFrame] = []
    definitions: list[dict[str, object]] | None = None
    for _, yearly in base.groupby(base["Timestamp"].dt.year, sort=True):
        frame, current_definitions = construct_window_features_one_year(yearly.reset_index(drop=True), base_catalog)
        frames.append(frame)
        if definitions is None:
            definitions = current_definitions
    hourly = pd.concat(frames, ignore_index=True).sort_values("Timestamp").reset_index(drop=True)
    daily = hourly[hourly["Timestamp"].dt.hour.eq(10)].copy()
    daily_05 = hourly[hourly["Timestamp"].dt.hour.eq(5)].copy()
    event = hourly[
        (hourly["Timestamp"] > EVENT_CUTOFF - pd.Timedelta(hours=168))
        & (hourly["Timestamp"] <= EVENT_CUTOFF)
    ].copy()
    dictionary = pd.DataFrame(definitions)
    dictionary.insert(0, "清洗版本", mode)
    return daily, daily_05, event, dictionary


def quality_summary(daily: pd.DataFrame, mode: str) -> pd.DataFrame:
    feature_cols = [column for column in daily.columns if column != "Timestamp"]
    rows = []
    event_row = daily[daily["Timestamp"].eq(EVENT_CUTOFF)]
    for feature in feature_cols:
        controls = daily[daily["Timestamp"].dt.year.isin([2021, 2022, 2023, 2024])][feature]
        rows.append(
            {
                "清洗版本": mode,
                "衍生指标": feature,
                "历史日端点有效数": int(controls.notna().sum()),
                "历史日端点有效率": float(controls.notna().mean()),
                "2026事件端点是否有效": bool(not event_row.empty and event_row[feature].notna().iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dictionaries = []
    quality = []
    for mode in ["strict", "expanded"]:
        daily, daily_05, event, dictionary = build_mode(mode)
        daily.to_parquet(CACHE_DIR / f"station_daily_10h_features_{mode}.parquet", index=False)
        # 2025事件公开技术资料给出的边境冲击时刻约为北京时间05:30；
        # 05:00端点只用于排序完成后的跨机理方向核查，不进入主模型。
        daily_05.to_parquet(CACHE_DIR / f"station_daily_05h_features_{mode}.parquet", index=False)
        event.to_parquet(CACHE_DIR / f"station_event_168h_trajectory_{mode}.parquet", index=False)
        dictionaries.append(dictionary)
        quality.append(quality_summary(daily, mode))
        print(f"{mode}: daily={daily.shape}, event={event.shape}")
    dictionary_all = pd.concat(dictionaries, ignore_index=True)
    dictionary_all.to_csv(OUTPUT_DIR / "07_站点衍生指标字典.csv", index=False, encoding="utf-8-sig")
    quality_all = pd.concat(quality, ignore_index=True)
    quality_all.to_csv(OUTPUT_DIR / "08_站点衍生指标可用性.csv", index=False, encoding="utf-8-sig")
    print(f"缓存已保存至：{CACHE_DIR}")


if __name__ == "__main__":
    main()
