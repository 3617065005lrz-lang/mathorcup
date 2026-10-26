from __future__ import annotations

"""Build the formal source-area ERA5-Land feature cache for Question 1.

The script combines two numerically cross-validated deliveries:

* CDS GRIB: atmosphere, precipitation, radiation, wind and shallow land state;
* Earth Engine native-grid GeoTIFF: snow, snowmelt and deep soil state.

All hourly timestamps are converted from UTC to Beijing time.  Daily model rows
are snapshots at 10:00 Beijing time, so the 2026 event row stops before the
10:52 source failure.  The primary spatial support is the ERA5-Land grid cell
whose centre is nearest the reconstructed source area (85.5 E, 28.3 N).
"""

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import rasterio


SCRIPT_DIR = Path(__file__).resolve().parent
WORK_DIR = SCRIPT_DIR.parent
CACHE_DIR = WORK_DIR / "cache"
OUTPUT_DIR = WORK_DIR / "outputs"

ERA_ROOT = Path(r"D:\MathorCup_Data\ERA5_Land_Gyirong")
CDS_ROOT = ERA_ROOT / "raw_official"
GEE_ROOT = ERA_ROOT / "earth_engine_nativegrid_2021_2026"
GRID_ELEVATION = (
    WORK_DIR / "external" / "gyirong-2026" / "data" / "processed" / "grid_elevation.nc"
)

YEARS = [2021, 2022, 2023, 2024, 2026]
WINDOWS = [24, 48, 72, 120, 168]
SOURCE_LAT = 28.289
SOURCE_LON = 85.530
TARGET_ELEVATION_M = 4991.0
FIXED_LAPSE_K_PER_M = -0.0065
LAPSE_MIN = -0.0095
LAPSE_MAX = -0.0020
LAPSE_MIN_R2 = 0.70

BASE_ELEMENTS = {
    "2T": "t2m",
    "2D": "d2m",
    "SP": "surface_pressure",
    "10U": "u10",
    "10V": "v10",
    "TP": "precipitation",
    "SSRD": "sw_down",
    "STRD": "lw_down",
    "SSR": "net_sw",
    "STR": "net_lw",
    "SSHF": "sensible_heat",
    "SLHF": "latent_heat",
    "FAL": "albedo",
    "SKT": "skin_temperature",
    "ST": "soil_temperature_1",
    "SWVL1": "soil_water_1",
}
ACCUMULATED = {"TP", "SSRD", "STRD", "SSR", "STR", "SSHF", "SLHF"}

GEE_BANDS = [
    "snow_cover",
    "snow_depth",
    "snow_depth_water_equivalent",
    "snowfall",
    "snowmelt",
    "snowfall_hourly",
    "snowmelt_hourly",
    "soil_temperature_level_2",
    "soil_temperature_level_3",
    "soil_temperature_level_4",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
]


def utc_from_epoch(value: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.fromtimestamp(int(value), tz=timezone.utc)).tz_localize(None)


def spatial_indices(dataset: rasterio.DatasetReader) -> tuple[int, int]:
    row, col = dataset.index(SOURCE_LON, SOURCE_LAT)
    if not (0 <= row < dataset.height and 0 <= col < dataset.width):
        raise RuntimeError("source coordinate lies outside the ERA5-Land grid")
    return row, col


def spatial_values(array: np.ndarray, row: int, col: int) -> tuple[float, float]:
    center = float(array[row, col])
    window = array[max(0, row - 1) : row + 2, max(0, col - 1) : col + 2]
    return center, float(np.nanmean(window))


def cds_files() -> list[Path]:
    history = sorted((CDS_ROOT / "q1_history_mapped").glob("*.grib"))
    event = sorted(CDS_ROOT.glob("era5_land_gyirong_2026_*_atmosphere_energy.grib"))
    event_snow = sorted(CDS_ROOT.glob("era5_land_gyirong_2026_*_snow_soil.grib"))
    paths = history + event + event_snow
    if len(history) != 16 or len(event) != 4 or len(event_snow) != 4:
        raise RuntimeError(
            f"CDS inputs incomplete: history={len(history)}/16, "
            f"event atmosphere={len(event)}/4, event snow-soil={len(event_snow)}/4"
        )
    return paths


def read_cds_hourly() -> pd.DataFrame:
    records: dict[str, list[dict[str, object]]] = defaultdict(list)
    reference_profile = None
    for path in cds_files():
        with rasterio.open(path) as dataset:
            profile = (dataset.width, dataset.height, tuple(dataset.transform), tuple(dataset.bounds))
            if reference_profile is None:
                reference_profile = profile
            elif profile != reference_profile:
                raise RuntimeError(f"CDS grid mismatch: {path}")
            row, col = spatial_indices(dataset)
            arrays = dataset.read()
            for band_index in range(1, dataset.count + 1):
                tags = dataset.tags(band_index)
                element = tags.get("GRIB_ELEMENT")
                if element not in BASE_ELEMENTS:
                    continue
                valid = utc_from_epoch(tags["GRIB_VALID_TIME"])
                if valid.year not in YEARS:
                    continue
                array = arrays[band_index - 1].astype("f8", copy=False)
                center, mean3 = spatial_values(array, row, col)
                records[element].append(
                    {
                        "time_utc": valid,
                        "reference": utc_from_epoch(tags["GRIB_REF_TIME"]),
                        "step_seconds": int(tags.get("GRIB_FORECAST_SECONDS", "0")),
                        "center": center,
                        "mean3": mean3,
                        "grid": array.copy() if element == "2T" else None,
                    }
                )

    rows: dict[pd.Timestamp, dict[str, float]] = defaultdict(dict)
    for element, items in records.items():
        # Identical instantaneous variables can occur in two 2026 files. Keep one.
        unique: dict[tuple[pd.Timestamp, pd.Timestamp, int], dict[str, object]] = {}
        for item in items:
            unique[(item["time_utc"], item["reference"], item["step_seconds"])] = item
        items = sorted(unique.values(), key=lambda item: (item["reference"], item["step_seconds"]))
        by_forecast: dict[tuple[pd.Timestamp, int], dict[str, object]] = {
            (item["reference"], item["step_seconds"]): item for item in items
        }
        for item in items:
            value_center = float(item["center"])
            value_mean3 = float(item["mean3"])
            if element in ACCUMULATED:
                step = int(item["step_seconds"])
                previous = by_forecast.get((item["reference"], step - 3600))
                if step == 3600:
                    pass
                elif previous is not None:
                    value_center -= float(previous["center"])
                    value_mean3 -= float(previous["mean3"])
                else:
                    # The first 00 UTC accumulation of each downloaded season lacks
                    # the preceding 23 UTC field. It is deliberately set missing;
                    # no event/control window depends on this boundary hour.
                    value_center = np.nan
                    value_mean3 = np.nan
            name = BASE_ELEMENTS[element]
            rows[item["time_utc"]][f"{name}_center"] = value_center
            rows[item["time_utc"]][f"{name}_mean3"] = value_mean3
            if element == "2T":
                rows[item["time_utc"]]["t2m_grid"] = item["grid"]

    frame = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    frame.index.name = "time_utc"
    return frame


def read_gee_hourly() -> pd.DataFrame:
    rows: list[dict[str, float | pd.Timestamp]] = []
    pattern = re.compile(r"era5_land_gyirong_(\d{8})_\d{8}_snow_deepsoil\.tif$")
    paths = sorted(GEE_ROOT.glob("20??/*.tif"))
    expected = len(YEARS) * 122
    paths = [path for path in paths if int(path.parent.name) in YEARS]
    if len(paths) != expected:
        raise RuntimeError(f"native-grid Earth Engine files incomplete: {len(paths)}/{expected}")
    reference_profile = None
    for path in paths:
        match = pattern.fullmatch(path.name)
        if not match:
            raise RuntimeError(f"unexpected Earth Engine filename: {path.name}")
        day = pd.Timestamp(match.group(1))
        with rasterio.open(path) as dataset:
            profile = (dataset.width, dataset.height, dataset.count, tuple(dataset.transform), tuple(dataset.bounds))
            if reference_profile is None:
                reference_profile = profile
            elif profile != reference_profile:
                raise RuntimeError(f"Earth Engine grid mismatch: {path}")
            if dataset.count != 24 * len(GEE_BANDS):
                raise RuntimeError(f"Earth Engine band mismatch: {path}")
            row, col = spatial_indices(dataset)
            arrays = dataset.read().astype("f8", copy=False)
            for hour in range(24):
                record: dict[str, float | pd.Timestamp] = {"time_utc": day + pd.Timedelta(hours=hour)}
                for variable_index, name in enumerate(GEE_BANDS):
                    array = arrays[hour * len(GEE_BANDS) + variable_index]
                    center, mean3 = spatial_values(array, row, col)
                    record[f"{name}_center"] = center
                    record[f"{name}_mean3"] = mean3
                rows.append(record)
    return pd.DataFrame(rows).set_index("time_utc").sort_index()


def load_elevation_for_grib() -> np.ndarray:
    with rasterio.open(GRID_ELEVATION) as dataset:
        elevation = dataset.read(1).astype("f8")
    # The NetCDF stores latitude south-to-north, while GDAL's GRIB reader returns
    # north-to-south rows. Flip once to put both arrays on the same 9x9 grid.
    elevation = np.flipud(elevation)
    if elevation.shape != (9, 9) or not np.isfinite(elevation).all():
        raise RuntimeError("invalid ERA5 grid elevation field")
    return elevation


def temperature_and_lapse(frame: pd.DataFrame) -> pd.DataFrame:
    elevation = load_elevation_for_grib()
    h = elevation.ravel()
    h_mean = float(h.mean())
    center_elevation = float(elevation[4, 4])
    mean3_elevation = float(elevation[3:6, 3:6].mean())
    corrected = []
    corrected_mean3 = []
    corrected_fixed = []
    corrected_5300 = []
    lapse_used = []
    lapse_raw = []
    lapse_r2 = []
    freezing = []
    for _, row in frame.iterrows():
        grid = row.get("t2m_grid")
        if not isinstance(grid, np.ndarray):
            corrected.append(np.nan)
            corrected_mean3.append(np.nan)
            corrected_fixed.append(np.nan)
            corrected_5300.append(np.nan)
            lapse_used.append(np.nan)
            lapse_raw.append(np.nan)
            lapse_r2.append(np.nan)
            freezing.append(np.nan)
            continue
        temp = grid.astype("f8")
        if np.nanmedian(temp) > 100:
            temp = temp - 273.15
        values = temp.ravel()
        usable = np.isfinite(values) & np.isfinite(h)
        dh = h[usable] - h[usable].mean()
        tv = values[usable]
        slope = float(np.sum((tv - tv.mean()) * dh) / np.sum(dh**2))
        intercept = float(tv.mean() - slope * h[usable].mean())
        predicted = intercept + slope * h[usable]
        ss_tot = float(np.sum((tv - tv.mean()) ** 2))
        r2 = 1.0 - float(np.sum((tv - predicted) ** 2)) / ss_tot if ss_tot > 0 else np.nan
        reliable = LAPSE_MIN <= slope <= LAPSE_MAX and np.isfinite(r2) and r2 >= LAPSE_MIN_R2
        used = slope if reliable else FIXED_LAPSE_K_PER_M
        center = float(row["t2m_center"])
        center = center - 273.15 if center > 100 else center
        corrected.append(center + used * (TARGET_ELEVATION_M - center_elevation))
        mean3 = float(row["t2m_mean3"])
        mean3 = mean3 - 273.15 if mean3 > 100 else mean3
        corrected_mean3.append(mean3 + used * (TARGET_ELEVATION_M - mean3_elevation))
        corrected_fixed.append(center + FIXED_LAPSE_K_PER_M * (TARGET_ELEVATION_M - center_elevation))
        corrected_5300.append(center + used * (5300.0 - center_elevation))
        lapse_used.append(used)
        lapse_raw.append(slope)
        lapse_r2.append(r2)
        # Anchor the fallback line at the spatial mean rather than sea level.
        intercept_used = float(tv.mean() - used * h_mean)
        freezing.append(-intercept_used / used if used < 0 else np.nan)
    result = frame.drop(columns=["t2m_grid"]).copy()
    result["source_temperature"] = corrected
    result["source_temperature_mean3"] = corrected_mean3
    result["source_temperature_fixed65"] = corrected_fixed
    result["source_temperature_5300"] = corrected_5300
    result["lapse_rate_used"] = lapse_used
    result["lapse_rate_raw"] = lapse_raw
    result["lapse_r2"] = lapse_r2
    result["freezing_level"] = freezing
    return result


def prepare_hourly() -> pd.DataFrame:
    cds = temperature_and_lapse(read_cds_hourly())
    gee = read_gee_hourly()
    hourly = cds.join(gee, how="inner")

    # Instantaneous temperatures are Kelvin despite misleading GDAL GRIB comments.
    for column in [
        "t2m_center", "t2m_mean3", "d2m_center", "d2m_mean3",
        "skin_temperature_center", "skin_temperature_mean3",
        "soil_temperature_1_center", "soil_temperature_1_mean3",
        "soil_temperature_level_2_center", "soil_temperature_level_2_mean3",
        "soil_temperature_level_3_center", "soil_temperature_level_3_mean3",
        "soil_temperature_level_4_center", "soil_temperature_level_4_mean3",
    ]:
        if column in hourly and hourly[column].median() > 100:
            hourly[column] = hourly[column] - 273.15

    for support in ["center", "mean3"]:
        t = hourly[f"t2m_{support}"]
        td = hourly[f"d2m_{support}"]
        es = 0.6108 * np.exp(17.27 * t / (t + 237.3))
        ea = 0.6108 * np.exp(17.27 * td / (td + 237.3))
        hourly[f"vpd_{support}"] = (es - ea).clip(lower=0)
        hourly[f"wind_speed_{support}"] = np.hypot(hourly[f"u10_{support}"], hourly[f"v10_{support}"])
        hourly[f"net_radiation_{support}"] = hourly[f"net_sw_{support}"] + hourly[f"net_lw_{support}"]

    # Convert accumulated hourly amounts to convenient physical units.
    for support in ["center", "mean3"]:
        hourly[f"precipitation_{support}"] = hourly[f"precipitation_{support}"].clip(lower=0) * 1000.0
        hourly[f"snowfall_hourly_{support}"] = hourly[f"snowfall_hourly_{support}"].clip(lower=0) * 1000.0
        hourly[f"snowmelt_hourly_{support}"] = hourly[f"snowmelt_hourly_{support}"].clip(lower=0) * 1000.0

    hourly.index = hourly.index + pd.Timedelta(hours=8)
    hourly.index.name = "Timestamp"
    return hourly.sort_index()


def add_definition(
    definitions: list[dict[str, object]], name: str, factor_id: str, factor: str,
    chinese: str, fields: str, formula: str, hours: int, unit: str,
) -> None:
    definitions.append(
        {
            "衍生指标": name,
            "因素组ID": factor_id,
            "物理因素": factor,
            "指标中文名": chinese,
            "对应原始字段": fields,
            "透明定义": formula,
            "窗口小时": hours,
            "单位": unit,
            "数据层": "源区ERA5-Land正式重算",
            "主空间支持": "源区最近0.1度格点（85.5E, 28.3N）",
        }
    )


def build_endpoint_features(
    hourly: pd.DataFrame,
    support: str = "center",
    temperature_column: str = "source_temperature",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = pd.DataFrame(index=hourly.index)
    definitions: list[dict[str, object]] = []

    def add(name, values, factor_id, factor, chinese, fields, formula, hours, unit):
        output[name] = values
        add_definition(definitions, name, factor_id, factor, chinese, fields, formula, hours, unit)

    for hours in WINDOWS:
        min_periods = int(np.ceil(hours * 0.90))
        suffix = f"_{hours}h"
        roll = lambda series: series.rolling(hours, min_periods=min_periods)

        temp = hourly[temperature_column]
        add(f"src_tmean{suffix}", roll(temp).mean(), "TH01", "空气温度与热量累积", f"源区{hours}小时平均订正气温", "2m_temperature+grid_elevation", "动态递减率可靠时采用动态值，否则采用-6.5 K/km，订正到4991 m后取均值", hours, "degC")
        add(f"src_tmax{suffix}", roll(temp).max(), "TH01", "空气温度与热量累积", f"源区{hours}小时最高订正气温", "2m_temperature+grid_elevation", "高程订正气温窗口最大值", hours, "degC")
        add(f"src_pdh{suffix}", roll(temp.clip(lower=0)).sum(), "TH01", "空气温度与热量累积", f"源区{hours}小时正度时", "2m_temperature+grid_elevation", "窗口内max(订正气温,0)逐小时求和", hours, "degC h")
        add(f"src_warm_hours{suffix}", roll((temp > 0).astype(float)).sum(), "TH01", "空气温度与热量累积", f"源区{hours}小时正温持续时间", "2m_temperature+grid_elevation", "窗口内订正气温高于0摄氏度的小时数", hours, "h")
        add(f"src_temp_change{suffix}", temp - temp.shift(hours - 1), "TH01", "空气温度与热量累积", f"源区{hours}小时气温变化", "2m_temperature+grid_elevation", "窗口末端订正气温减窗口起点订正气温", hours, "degC")
        add(f"src_freezing_level{suffix}", roll(hourly["freezing_level"]).mean(), "TH01", "空气温度与热量累积", f"源区区域{hours}小时0摄氏度层高度", "2m_temperature+grid_elevation", "9×9格点温度-高程回归所得0摄氏度层高度的窗口均值", hours, "m")

        net = hourly[f"net_radiation_{support}"]
        add(f"src_positive_netrad{suffix}", roll(net.clip(lower=0)).sum() / 1e6, "RD01", "辐射与地表能量", f"源区{hours}小时累计正净辐射", "surface_net_solar_radiation+surface_net_thermal_radiation", "逐小时正净辐射能量累计", hours, "MJ m-2")
        add(f"src_sw_down{suffix}", roll(hourly[f"sw_down_{support}"].clip(lower=0)).sum() / 1e6, "RD01", "辐射与地表能量", f"源区{hours}小时下行短波", "surface_solar_radiation_downwards", "逐小时下行短波能量累计", hours, "MJ m-2")
        add(f"src_albedo_mean{suffix}", roll(hourly[f"albedo_{support}"]).median(), "RD01", "辐射与地表能量", f"源区{hours}小时反照率", "forecast_albedo", "窗口反照率中位数", hours, "1")

        rain = hourly[f"precipitation_{support}"]
        add(f"src_precip_sum{suffix}", roll(rain).sum(), "PR01", "降水与液态水输入", f"源区{hours}小时累计降水", "total_precipitation", "逐小时降水增量累计", hours, "mm")
        add(f"src_precip_max1h{suffix}", roll(rain).max(), "PR01", "降水与液态水输入", f"源区{hours}小时内最大1小时降水", "total_precipitation", "窗口内逐小时降水最大值", hours, "mm h-1")
        for duration in [3, 6]:
            inner = rain.rolling(duration, min_periods=duration).sum()
            add(f"src_precip_max{duration}h{suffix}", roll(inner).max(), "PR01", "降水与液态水输入", f"源区{hours}小时内最大{duration}小时降水", "total_precipitation", f"窗口内最大连续{duration}小时降水", hours, "mm")

        vpd = hourly[f"vpd_{support}"]
        add(f"src_vpd_mean{suffix}", roll(vpd).mean(), "HM01", "大气湿度与干湿状态", f"源区{hours}小时平均水汽压亏缺", "2m_temperature+2m_dewpoint_temperature", "由气温与露点计算VPD后取均值", hours, "kPa")
        add(f"src_vpd_p90{suffix}", roll(vpd).quantile(0.9), "HM01", "大气湿度与干湿状态", f"源区{hours}小时水汽压亏缺90分位", "2m_temperature+2m_dewpoint_temperature", "由气温与露点计算VPD后取90分位", hours, "kPa")

        wind = hourly[f"wind_speed_{support}"]
        add(f"src_wind_mean{suffix}", roll(wind).mean(), "WN01", "风场与通风条件", f"源区{hours}小时平均风速", "10m_u_component_of_wind+10m_v_component_of_wind", "由u/v计算风速后取均值", hours, "m s-1")
        add(f"src_wind_p90{suffix}", roll(wind).quantile(0.9), "WN01", "风场与通风条件", f"源区{hours}小时风速90分位", "10m_u_component_of_wind+10m_v_component_of_wind", "由u/v计算风速后取90分位", hours, "m s-1")

        melt = hourly[f"snowmelt_hourly_{support}"]
        add(f"src_snowmelt_sum{suffix}", roll(melt).sum(), "SN01", "积雪与融雪状态", f"源区{hours}小时ERA5融雪量", "snowmelt_hourly", "逐小时融雪水当量累计", hours, "mm w.e.")
        cover = hourly[f"snow_cover_{support}"]
        add(f"src_snowcover_mean{suffix}", roll(cover).mean(), "SN01", "积雪与融雪状态", f"源区{hours}小时平均雪盖", "snow_cover", "窗口雪盖百分比均值", hours, "%")
        add(f"src_snowcover_change{suffix}", cover - cover.shift(hours - 1), "SN01", "积雪与融雪状态", f"源区{hours}小时雪盖变化", "snow_cover", "窗口末端雪盖减窗口起点雪盖", hours, "percentage point")
        swe = hourly[f"snow_depth_water_equivalent_{support}"]
        add(f"src_swe_change{suffix}", (swe - swe.shift(hours - 1)) * 1000.0, "SN01", "积雪与融雪状态", f"源区{hours}小时积雪水当量变化", "snow_depth_water_equivalent", "窗口末端积雪水当量减窗口起点", hours, "mm w.e.")

        st1 = hourly[f"soil_temperature_1_{support}"]
        st4 = hourly[f"soil_temperature_level_4_{support}"]
        add(f"src_soilT1_mean{suffix}", roll(st1).mean(), "ST01", "土壤温度与冻融", f"源区{hours}小时浅层土壤温度", "soil_temperature_level_1", "0—7 cm土壤温度窗口均值", hours, "degC")
        add(f"src_soilT1_change{suffix}", st1 - st1.shift(hours - 1), "ST01", "土壤温度与冻融", f"源区{hours}小时浅层地温变化", "soil_temperature_level_1", "窗口末端浅层地温减窗口起点", hours, "degC")
        add(f"src_soilT_gradient{suffix}", roll(st4 - st1).mean(), "ST01", "土壤温度与冻融", f"源区{hours}小时深浅地温差", "soil_temperature_level_1+soil_temperature_level_4", "100—289 cm层温度减0—7 cm层温度的窗口均值", hours, "degC")

        sw1 = hourly[f"soil_water_1_{support}"]
        sw4 = hourly[f"volumetric_soil_water_layer_4_{support}"]
        add(f"src_soilwater1_mean{suffix}", roll(sw1).mean(), "SM01", "土壤湿润状态", f"源区{hours}小时浅层土壤水分", "volumetric_soil_water_layer_1", "0—7 cm体积含水量窗口均值", hours, "m3 m-3")
        add(f"src_soilwater1_change{suffix}", sw1 - sw1.shift(hours - 1), "SM01", "土壤湿润状态", f"源区{hours}小时浅层土壤水分变化", "volumetric_soil_water_layer_1", "窗口末端浅层含水量减窗口起点", hours, "m3 m-3")
        add(f"src_soilwater_gradient{suffix}", roll(sw4 - sw1).mean(), "SM01", "土壤湿润状态", f"源区{hours}小时深浅含水差", "volumetric_soil_water_layer_1+volumetric_soil_water_layer_4", "深层体积含水量减浅层体积含水量的窗口均值", hours, "m3 m-3")

        pressure = hourly[f"surface_pressure_{support}"] / 100.0
        add(f"src_pressure_range{suffix}", roll(pressure).max() - roll(pressure).min(), "PS01", "气压与天气过程", f"源区{hours}小时气压极差", "surface_pressure", "窗口最高气压减最低气压", hours, "hPa")
        add(f"src_pressure_change{suffix}", pressure - pressure.shift(hours - 1), "PS01", "气压与天气过程", f"源区{hours}小时气压变化", "surface_pressure", "窗口末端气压减窗口起点", hours, "hPa")

    endpoints = output[(output.index.hour == 10) & output.index.year.isin(YEARS)].copy()
    endpoints.insert(0, "date", endpoints.index)
    endpoints.reset_index(drop=True, inplace=True)
    return endpoints, pd.DataFrame(definitions)


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hourly = prepare_hourly()
    endpoints, definitions = build_endpoint_features(hourly)
    endpoints_mean3, _ = build_endpoint_features(
        hourly, support="mean3", temperature_column="source_temperature_mean3"
    )
    endpoints_fixed, _ = build_endpoint_features(
        hourly, support="center", temperature_column="source_temperature_fixed65"
    )
    endpoints_5300, _ = build_endpoint_features(
        hourly, support="center", temperature_column="source_temperature_5300"
    )

    hourly.drop(columns=[column for column in hourly.columns if column.endswith("_mean3")]).to_parquet(
        CACHE_DIR / "source_hourly_official_center.parquet"
    )
    endpoints.to_parquet(CACHE_DIR / "source_daily_features_official.parquet", index=False)
    endpoints_mean3.to_parquet(CACHE_DIR / "source_daily_features_official_mean3.parquet", index=False)
    endpoints_fixed.to_parquet(CACHE_DIR / "source_daily_features_official_fixed65.parquet", index=False)
    endpoints_5300.to_parquet(CACHE_DIR / "source_daily_features_official_elev5300.parquet", index=False)
    definitions.to_csv(
        OUTPUT_DIR / "23_源区ERA5正式衍生指标字典.csv", index=False, encoding="utf-8-sig"
    )

    event_time = pd.Timestamp("2026-08-26 10:00:00")
    event = endpoints[endpoints["date"].eq(event_time)]
    year_counts = endpoints.groupby(endpoints["date"].dt.year).size().to_dict()
    diagnostics = {
        "hourly_start_beijing": str(hourly.index.min()),
        "hourly_end_beijing": str(hourly.index.max()),
        "hourly_rows": int(len(hourly)),
        "endpoint_rows_by_year": {str(k): int(v) for k, v in year_counts.items()},
        "feature_count": int(len(definitions)),
        "unique_features": int(definitions["衍生指标"].nunique()),
        "event_endpoint_present": not event.empty,
        "event_feature_missing": int(event.drop(columns=["date"]).isna().sum(axis=1).iloc[0]) if not event.empty else None,
        "source_grid_center": {"longitude": 85.5, "latitude": 28.3},
        "target_elevation_m": TARGET_ELEVATION_M,
        "lapse_fallback_k_per_km": FIXED_LAPSE_K_PER_M * 1000,
    }
    (OUTPUT_DIR / "24_源区正式特征质量报告.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
