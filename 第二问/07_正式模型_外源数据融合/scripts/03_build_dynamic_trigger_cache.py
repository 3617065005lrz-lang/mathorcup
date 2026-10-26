from __future__ import annotations

"""Build interpretable hourly trigger features for Question 2.

Inputs are source-area ERA5-Land, lagged MODIS snow/albedo, IMERG half-hourly
rainfall and CHIRPS daily rainfall.  Every rolling feature is backward-looking;
2025 is excluded by design and no same-day MODIS value is used for the event.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent
MODELING_ROOT = MODEL_DIR.parents[1]
CONFIG = json.loads((MODEL_DIR / "config" / "model_config.json").read_text(encoding="utf-8"))
PROCESSED = MODEL_DIR / "processed"
OUTPUTS = MODEL_DIR / "outputs"

Q1_CACHE = MODELING_ROOT / "第一问" / "05_重构方案_多源事件对照" / "cache"
ERA5_PATH = Q1_CACHE / "source_hourly_official_center.parquet"
IMERG_PATH = Q1_CACHE / "imerg_source_halfhourly.parquet"
MODIS_PATH = PROCESSED / "02_MODIS源区融合每日缓存.parquet"
CHIRPS_PATH = PROCESSED / "03_CHIRPS源区逐日缓存.parquet"
SMAP_PATH = PROCESSED / "04_SMAP源区三小时土壤水分缓存.parquet"

TARGET_YEARS = CONFIG["background_years"] + [pd.Timestamp(CONFIG["event_time_beijing"]).year]
WINDOWS = CONFIG["time_windows_hours"]


def attach_lagged_modis(hourly: pd.DataFrame, modis: pd.DataFrame) -> pd.DataFrame:
    """Attach the latest valid MODIS observation strictly before each date."""
    daily_dates = pd.DataFrame(
        {"date": pd.to_datetime(hourly.index.normalize().unique()).astype("datetime64[ns]")}
    )
    available = modis.copy()
    available["date"] = pd.to_datetime(available["date"]).astype("datetime64[ns]")
    available["available_date"] = available["date"] + pd.Timedelta(days=1)
    keep = [
        "available_date",
        "date",
        "snow_albedo_mean",
        "snow_fraction_40",
        "ndsi_snow_cover_mean_percent",
        "qa_valid_fraction",
        "snow_albedo_pixel_observations",
    ]
    available = available[keep].sort_values("available_date")
    mapped = pd.merge_asof(
        daily_dates.sort_values("date"),
        available,
        left_on="date",
        right_on="available_date",
        direction="backward",
        tolerance=pd.Timedelta(days=CONFIG["modis"]["past_fill_window_days"]),
    )
    mapped = mapped.rename(columns={"date_x": "date", "date_y": "modis_observation_date"})
    mapped["modis_age_days"] = (mapped["date"] - mapped["modis_observation_date"]).dt.days
    mapped = mapped.set_index("date")
    result = hourly.copy()
    result["date"] = result.index.normalize()
    result = result.join(mapped.drop(columns=["available_date"]), on="date")
    result["albedo_operational"] = result["snow_albedo_mean"].where(
        result["snow_albedo_mean"].between(0.05, 0.95), result["albedo_center"]
    )
    result["albedo_source"] = np.where(result["snow_albedo_mean"].between(0.05, 0.95), "MODIS_lagged", "ERA5_fallback")
    result["albedo_operational"] = result["albedo_operational"].clip(0.05, 0.95)
    return result


def add_rolling_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    eti = CONFIG["eti_model"]
    out["shortwave_w_m2"] = out["sw_down_center"].clip(lower=0) / 3600.0
    out["temperature_excess_c"] = (out["source_temperature"] - eti["temperature_threshold_c"]).clip(lower=0)
    out["absorbed_shortwave_w_m2"] = (1.0 - out["albedo_operational"]) * out["shortwave_w_m2"]
    out["eti_melt_mm_h"] = np.where(
        out["source_temperature"] > eti["temperature_threshold_c"],
        eti["temperature_factor_mm_h_c"] * out["temperature_excess_c"]
        + eti["shortwave_factor_mm_h_wm2"] * out["absorbed_shortwave_w_m2"],
        0.0,
    )
    out["eti_melt_mm_h"] = np.maximum(out["eti_melt_mm_h"], 0.0)
    out["positive_degree_hour_c"] = out["source_temperature"].clip(lower=0)

    for window in WINDOWS:
        roller = out.rolling(window=window, min_periods=window)
        out[f"eti_melt_{window}h_mm"] = roller["eti_melt_mm_h"].sum()
        out[f"positive_degree_hours_{window}h"] = roller["positive_degree_hour_c"].sum()
        out[f"absorbed_shortwave_{window}h_mj_m2"] = roller["absorbed_shortwave_w_m2"].sum() * 0.0036
        out[f"era5_rain_{window}h_mm"] = roller["precipitation_center"].sum()
        out[f"era5_snowmelt_{window}h_mm"] = roller["snowmelt_hourly_center"].sum()
        out[f"source_temperature_mean_{window}h_c"] = roller["source_temperature"].mean()
        out[f"source_temperature_max_{window}h_c"] = roller["source_temperature"].max()
        out[f"soil_water_l1_mean_{window}h"] = roller["soil_water_1_center"].mean()
        out[f"soil_water_l2_mean_{window}h"] = roller["volumetric_soil_water_layer_2_center"].mean()
        out[f"freezing_level_mean_{window}h_m"] = roller["freezing_level"].mean()
        if "smap_sm_surface" in out:
            out[f"smap_surface_mean_{window}h"] = roller["smap_sm_surface"].mean()
            out[f"smap_rootzone_mean_{window}h"] = roller["smap_sm_rootzone"].mean()
            out[f"smap_rootzone_pctl_mean_{window}h"] = roller["smap_sm_rootzone_pctl"].mean()
    return out


def attach_smap(hourly: pd.DataFrame) -> pd.DataFrame:
    smap = pd.read_parquet(SMAP_PATH).sort_index().drop(columns=["year", "source_file"], errors="ignore")
    smap = smap.loc[smap.index.year.isin(TARGET_YEARS)]
    left = hourly.sort_index().reset_index().rename(columns={hourly.index.name or "index": "timestamp_beijing"})
    right = smap.reset_index().rename(columns={smap.index.name or "index": "smap_observation_time"})
    left["timestamp_beijing"] = pd.to_datetime(left["timestamp_beijing"]).astype("datetime64[ns]")
    right["smap_observation_time"] = pd.to_datetime(right["smap_observation_time"]).astype("datetime64[ns]")
    merged = pd.merge_asof(
        left,
        right.sort_values("smap_observation_time"),
        left_on="timestamp_beijing",
        right_on="smap_observation_time",
        direction="backward",
        tolerance=pd.Timedelta(hours=4),
    ).set_index("timestamp_beijing")
    merged["smap_age_minutes"] = (merged.index.to_series() - merged["smap_observation_time"]).dt.total_seconds() / 60.0
    return merged


def build_imerg_features() -> pd.DataFrame:
    imerg = pd.read_parquet(IMERG_PATH).sort_index()
    imerg = imerg.loc[imerg.index.year.isin(TARGET_YEARS)].copy()
    pieces = []
    for year, group in imerg.groupby(imerg.index.year):
        group = group.sort_index().copy()
        for hours in [1, 3, 6, 24, 72]:
            n = hours * 2
            group[f"imerg_rain_{hours}h_mm"] = group["precip_amount_mm"].rolling(n, min_periods=n).sum()
            group[f"imerg_coverage_{hours}h"] = group["precip_amount_mm"].rolling(n, min_periods=n).count() / n
        pieces.append(group)
    half_hourly = pd.concat(pieces).sort_index()
    hourly = half_hourly.loc[half_hourly.index.minute == 0].drop(columns=["file"])
    return hourly


def build_chirps_features() -> pd.DataFrame:
    chirps = pd.read_parquet(CHIRPS_PATH).sort_values("date").set_index("date")
    pieces = []
    for year, group in chirps.groupby(chirps.index.year):
        group = group.sort_index().copy()
        # Shift one day: at an event-time forecast, only completed previous days are available.
        previous = group["chirps_source_mm"].shift(1)
        previous_neighbourhood = group["chirps_neighbourhood_mean_mm"].shift(1)
        for days in [3, 7, 15, 30]:
            group[f"chirps_rain_prev_{days}d_mm"] = previous.rolling(days, min_periods=days).sum()
            group[f"chirps_neighbourhood_prev_{days}d_mm"] = previous_neighbourhood.rolling(days, min_periods=days).sum()
        pieces.append(group)
    return pd.concat(pieces).sort_index()


def main() -> None:
    for required in [ERA5_PATH, IMERG_PATH, MODIS_PATH, CHIRPS_PATH, SMAP_PATH]:
        if not required.exists():
            raise FileNotFoundError(required)

    era = pd.read_parquet(ERA5_PATH).sort_index()
    era = era.loc[era.index.year.isin(TARGET_YEARS)].copy()
    if any(year in set(era.index.year) for year in CONFIG["excluded_years"]):
        raise RuntimeError("Excluded year leaked into ERA5 table")
    modis = pd.read_parquet(MODIS_PATH)
    hourly = attach_lagged_modis(era, modis)
    hourly = attach_smap(hourly)

    # Rolling windows must not cross the seasonal boundary between years.
    hourly = pd.concat(
        [add_rolling_features(group.sort_index()) for _, group in hourly.groupby(hourly.index.year)]
    ).sort_index()

    imerg_hourly = build_imerg_features()
    hourly = hourly.join(imerg_hourly, how="left")
    chirps_daily = build_chirps_features()
    hourly = hourly.join(chirps_daily.drop(columns=["year", "month", "is_background", "is_event_year", "source_file"]), on="date")

    event_last_hour = pd.Timestamp(CONFIG["event_last_complete_hour_beijing"])
    hourly["is_event_last_complete_hour"] = hourly.index == event_last_hour
    hourly["is_background"] = hourly.index.year.isin(CONFIG["background_years"])
    hourly["year"] = hourly.index.year
    hourly["month"] = hourly.index.month
    hourly["day"] = hourly.index.day
    hourly["hour"] = hourly.index.hour
    hourly.index.name = "timestamp_beijing"

    out_path = PROCESSED / "04_多源动态触发逐小时缓存.parquet"
    hourly.to_parquet(out_path)

    event = hourly.loc[[event_last_hour]].T.rename(columns={event_last_hour: "event_value"})
    event.to_csv(OUTPUTS / "07_事件时刻动态特征全表.csv", encoding="utf-8-sig")

    feature_names = []
    for window in WINDOWS:
        feature_names.extend(
            [
                f"eti_melt_{window}h_mm",
                f"positive_degree_hours_{window}h",
                f"absorbed_shortwave_{window}h_mj_m2",
                f"era5_rain_{window}h_mm",
                f"soil_water_l1_mean_{window}h",
                f"smap_rootzone_mean_{window}h",
                f"freezing_level_mean_{window}h_m",
            ]
        )
    feature_names += [
        "imerg_rain_3h_mm",
        "imerg_rain_6h_mm",
        "imerg_rain_24h_mm",
        "imerg_rain_72h_mm",
        "chirps_rain_prev_15d_mm",
        "chirps_rain_prev_30d_mm",
        "albedo_operational",
        "snow_fraction_40",
    ]
    event_summary = hourly.loc[event_last_hour, feature_names].rename("event_value").reset_index()
    event_summary.columns = ["feature", "event_value"]
    event_summary.to_csv(OUTPUTS / "08_事件时刻核心动态特征.csv", index=False, encoding="utf-8-sig")

    albedo_share = hourly["albedo_source"].value_counts(dropna=False).rename_axis("source").reset_index(name="hours")
    albedo_markdown = "| source | hours |\n|---|---:|\n" + "\n".join(
        f"| {row.source} | {int(row.hours)} |" for row in albedo_share.itertuples(index=False)
    )
    report = f"""# 多源动态触发缓存报告

- 时间范围：{hourly.index.min()} 至 {hourly.index.max()}（北京时间）。
- 纳入年份：{TARGET_YEARS}；2025 年已严格排除。
- 行数：{len(hourly)}；事件判定截止时刻：{event_last_hour}。
- 气温：ERA5-Land 源区高程订正温度；不是将观测站原值直接当作源区温度。
- 融水：增强温度指数（ETI）条件融水量；中心参数来自文献，仅作为相对融水指标，不宣称为本地实测融水。
- 反照率：仅使用前一日或更早的 MODIS 有效雪反照率，最多向前填充 {CONFIG['modis']['past_fill_window_days']} 天；缺失时回退至 ERA5-Land 反照率并保留来源标记。
- 短时降水：IMERG 与 ERA5-Land；前期湿润背景：CHIRPS 前 15/30 个完整日。
- 土壤水分：ERA5-Land 提供逐小时连续量，SMAP L4 9 km/3 小时作为独立区域湿润背景验证。
- 所有滚动窗口均向后计算，不使用事件后的信息。

## 反照率来源小时数

{albedo_markdown}
"""
    (OUTPUTS / "09_多源动态触发缓存报告.md").write_text(report, encoding="utf-8")
    print(f"Saved {len(hourly)} rows to {out_path}")
    print(event_summary.to_string(index=False))


if __name__ == "__main__":
    main()
