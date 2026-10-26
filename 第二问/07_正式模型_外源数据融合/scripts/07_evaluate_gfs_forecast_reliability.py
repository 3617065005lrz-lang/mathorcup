from __future__ import annotations

"""Evaluate event-week GFS forecast inputs against ERA5-Land reconstruction.

This is a case-study forecast reliability check, not a multi-year operational
skill assessment.  Only forecast cycles available before the event are used.
"""

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent
CONFIG = json.loads((MODEL_DIR / "config" / "model_config.json").read_text(encoding="utf-8"))
PROCESSED = MODEL_DIR / "processed"
OUTPUTS = MODEL_DIR / "outputs"
GFS_DIR = MODEL_DIR / "external_cache" / "gfs"
DYNAMIC_PATH = PROCESSED / "04_多源动态触发逐小时缓存.parquet"
EVENT_TIME = pd.Timestamp(CONFIG["event_time_beijing"])
TARGET_HEIGHT = float(CONFIG["source_target_elevation_m"])


def clean(dataset: h5py.Dataset) -> np.ndarray:
    values = np.asarray(dataset[:], dtype=float)
    fill = dataset.attrs.get("_FillValue")
    if fill is not None:
        values[np.isclose(values, float(np.asarray(fill).ravel()[0]))] = np.nan
    values[np.abs(values) > 1e20] = np.nan
    return values


def vertical_temperature_c(heights: np.ndarray, temperatures: np.ndarray) -> float:
    valid = np.isfinite(heights) & np.isfinite(temperatures)
    if valid.sum() < 2:
        return np.nan
    z = heights[valid]
    t = temperatures[valid]
    order = np.argsort(z)
    z, t = z[order], t[order]
    if TARGET_HEIGHT < z.min() or TARGET_HEIGHT > z.max():
        return np.nan
    return float(np.interp(TARGET_HEIGHT, z, t) - 273.15)


def read_cycle(path: Path) -> pd.DataFrame:
    with h5py.File(path, "r") as h5:
        lat = clean(h5["latitude"])
        lon = clean(h5["longitude"])
        ref = CONFIG["source_reference_point_wgs84"]
        iy = int(np.abs(lat - ref["latitude"]).argmin())
        ix = int(np.abs(lon - ref["longitude"]).argmin())
        leads = clean(h5["forecast_period"]).astype(int)
        valid_utc = pd.to_datetime(clean(h5["valid_time"]), unit="s", utc=True)
        valid_bjt = valid_utc.tz_convert("Asia/Shanghai").tz_localize(None)
        reference_utc = pd.to_datetime(float(np.asarray(h5["forecast_reference_time"])), unit="s", utc=True)
        reference_bjt = reference_utc.tz_convert("Asia/Shanghai").tz_localize(None)

        height = clean(h5["geopotential_height_extended"])[:, :, iy, ix]
        temperature = clean(h5["air_temperature_extended"])[:, :, iy, ix]
        source_temp = [vertical_temperature_c(height[i], temperature[i]) for i in range(len(leads))]
        temp2m = clean(h5["temperature_2m"])[:, iy, ix] - 273.15
        precip = clean(h5["precipitation_3h"])[:, iy, ix]
        shortwave = clean(h5["shortwave_mean_3h_qc"])[:, iy, ix]
        freezing = clean(h5["freezing_level_geopotential_height"])[:, iy, ix]
        return pd.DataFrame(
            {
                "cycle_beijing": reference_bjt,
                "valid_time_beijing": valid_bjt,
                "lead_hours": leads,
                "gfs_source_temperature_c": source_temp,
                "gfs_grid_2m_temperature_c": temp2m,
                "gfs_precipitation_3h_mm": precip,
                "gfs_shortwave_mean_3h_w_m2": shortwave,
                "gfs_freezing_level_m": freezing,
                "gfs_grid_lat": float(lat[iy]),
                "gfs_grid_lon": float(lon[ix]),
                "source_file": path.name,
            }
        )


def lead_band(hours: pd.Series) -> pd.Categorical:
    return pd.cut(hours, bins=[-0.1, 24, 48, 72, 120], labels=["0-24h", "27-48h", "51-72h", "75-120h"])


def main() -> None:
    paths = sorted(GFS_DIR.glob("gfs_*_interval_aware.nc"))
    if not paths:
        raise FileNotFoundError(GFS_DIR)
    forecast = pd.concat([read_cycle(path) for path in paths], ignore_index=True)
    forecast = forecast.loc[forecast["cycle_beijing"] <= EVENT_TIME].copy()

    era = pd.read_parquet(DYNAMIC_PATH).sort_index()
    era["era_source_temperature_c"] = era["source_temperature"]
    era["era_precipitation_3h_mm"] = era["precipitation_center"].rolling(3, min_periods=3).sum()
    era["era_shortwave_mean_3h_w_m2"] = era["shortwave_w_m2"].rolling(3, min_periods=3).mean()
    era["era_freezing_level_m"] = era["freezing_level"]
    verify = forecast.merge(
        era[
            [
                "era_source_temperature_c",
                "era_precipitation_3h_mm",
                "era_shortwave_mean_3h_w_m2",
                "era_freezing_level_m",
            ]
        ],
        left_on="valid_time_beijing",
        right_index=True,
        how="left",
    )
    verify["temperature_error_c"] = verify["gfs_source_temperature_c"] - verify["era_source_temperature_c"]
    verify["precipitation_error_3h_mm"] = verify["gfs_precipitation_3h_mm"] - verify["era_precipitation_3h_mm"]
    verify["shortwave_error_w_m2"] = verify["gfs_shortwave_mean_3h_w_m2"] - verify["era_shortwave_mean_3h_w_m2"]
    verify["freezing_level_error_m"] = verify["gfs_freezing_level_m"] - verify["era_freezing_level_m"]
    verify["lead_band"] = lead_band(verify["lead_hours"])
    verify.to_parquet(PROCESSED / "10_GFS事件周预报核查.parquet", index=False)

    rows = []
    for band, group in verify.groupby("lead_band", observed=True):
        rows.append(
            {
                "lead_band": str(band),
                "n": len(group),
                "temperature_bias_c": group["temperature_error_c"].mean(),
                "temperature_mae_c": group["temperature_error_c"].abs().mean(),
                "precipitation_bias_3h_mm": group["precipitation_error_3h_mm"].mean(),
                "precipitation_mae_3h_mm": group["precipitation_error_3h_mm"].abs().mean(),
                "shortwave_bias_w_m2": group["shortwave_error_w_m2"].mean(),
                "shortwave_mae_w_m2": group["shortwave_error_w_m2"].abs().mean(),
                "freezing_level_bias_m": group["freezing_level_error_m"].mean(),
                "freezing_level_mae_m": group["freezing_level_error_m"].abs().mean(),
            }
        )
    skill = pd.DataFrame(rows)
    skill.to_csv(OUTPUTS / "27_GFS分时效个例核查.csv", index=False, encoding="utf-8-sig")

    available_valid_times = pd.DatetimeIndex(verify["valid_time_beijing"].dropna().unique())
    event_valid = available_valid_times[np.argmin(np.abs(available_valid_times - EVENT_TIME))]
    event_forecasts = verify.loc[verify["valid_time_beijing"].eq(event_valid)].sort_values("cycle_beijing")
    event_forecasts.to_csv(OUTPUTS / "28_GFS事件时刻多起报预报.csv", index=False, encoding="utf-8-sig")

    report = f"""# GFS 预报可靠性个例核查

- 纳入起报：{forecast['cycle_beijing'].nunique()} 次，均早于 2026-08-26 10:52。
- 网格中心：({forecast.iloc[0]['gfs_grid_lon']:.2f}, {forecast.iloc[0]['gfs_grid_lat']:.2f})，分辨率明显粗于源区尺度。
- 源区气温不是直接使用 GFS 2 m 格点温度，而是由压力层温度—位势高度廓线插值至 {TARGET_HEIGHT:.0f} m。
- 预报核查参照为 ERA5-Land 源区重建；这是事件周个例一致性，不等同于多年预报技巧评分。
- 事件时刻按最邻近的 3 小时预报有效时次 {event_valid} 核查。

## 分时效误差

{skill.to_string(index=False)}

GFS 的合理用途是把已经建立的动态触发指标向未来 1—3 天外推；静态空间易发性不随每次预报改变。由于高山地形和 0.25° 网格尺度差异，论文应同时报告预报时效误差，而不能把 GFS 当作源区真值。
"""
    (OUTPUTS / "29_GFS预报可靠性说明.md").write_text(report, encoding="utf-8")
    print(skill.to_string(index=False))
    print("\nEvent-valid forecasts:")
    print(event_forecasts[["cycle_beijing", "lead_hours", "gfs_source_temperature_c", "gfs_precipitation_3h_mm", "gfs_shortwave_mean_3h_w_m2"]].to_string(index=False))


if __name__ == "__main__":
    main()
