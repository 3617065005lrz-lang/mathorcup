from __future__ import annotations

"""Extract CHIRPS v3 daily rainfall at and around the reconstructed source area.

The global monthly files are large, so this script reads only the nearest grid
cell and its 3 x 3 neighbourhood.  The point value is the primary source-area
rainfall series; the neighbourhood mean is retained for spatial sensitivity.
"""

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent
CONFIG = json.loads((MODEL_DIR / "config" / "model_config.json").read_text(encoding="utf-8"))
PROCESSED = MODEL_DIR / "processed"
OUTPUTS = MODEL_DIR / "outputs"

CHIRPS_ROOT = Path(r"D:\MathorCup_Data\External_Gyirong\P1-1_CHIRPS_v3\official_monthly_netcdf")
PATTERN = re.compile(r"chirps-v3\.0\.(?P<year>\d{4})\.(?P<month>\d{2})\.days_p05\.nc$")
EVENT_YEAR = pd.Timestamp(CONFIG["event_time_beijing"]).year
YEARS = set(CONFIG["background_years"] + [EVENT_YEAR])


def nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.abs(np.asarray(values, dtype=float) - target).argmin())


def extract_file(path: Path) -> pd.DataFrame:
    with Dataset(path) as ds:
        lats = np.asarray(ds.variables["latitude"][:], dtype=float)
        lons = np.asarray(ds.variables["longitude"][:], dtype=float)
        ref = CONFIG["source_reference_point_wgs84"]
        iy = nearest_index(lats, ref["latitude"])
        ix = nearest_index(lons, ref["longitude"])
        y0, y1 = max(0, iy - 1), min(len(lats), iy + 2)
        x0, x1 = max(0, ix - 1), min(len(lons), ix + 2)

        precip = ds.variables["precip"]
        point = np.ma.filled(precip[:, iy, ix], np.nan).astype(float)
        neighbourhood = np.ma.filled(precip[:, y0:y1, x0:x1], np.nan).astype(float)
        neighbourhood[neighbourhood < 0] = np.nan
        point[point < 0] = np.nan
        regional_mean = np.nanmean(neighbourhood, axis=(1, 2))
        regional_sd = np.nanstd(neighbourhood, axis=(1, 2), ddof=0)

        time_var = ds.variables["time"]
        dates = pd.to_datetime(
            [
                str(value)[:10]
                for value in num2date(
                    time_var[:],
                    units=time_var.units,
                    calendar=getattr(time_var, "calendar", "standard"),
                    only_use_cftime_datetimes=False,
                )
            ]
        )
        return pd.DataFrame(
            {
                "date": dates,
                "chirps_source_mm": point,
                "chirps_neighbourhood_mean_mm": regional_mean,
                "chirps_neighbourhood_sd_mm": regional_sd,
                "chirps_grid_lon": float(lons[ix]),
                "chirps_grid_lat": float(lats[iy]),
                "chirps_neighbourhood_n": int((y1 - y0) * (x1 - x0)),
                "source_file": path.name,
            }
        )


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for path in sorted(CHIRPS_ROOT.glob("*.nc")):
        match = PATTERN.fullmatch(path.name)
        if match and int(match["year"]) in YEARS:
            paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No target CHIRPS files found under {CHIRPS_ROOT}")

    frames = []
    for index, path in enumerate(paths, start=1):
        print(f"[{index:02d}/{len(paths):02d}] {path.name}", flush=True)
        frames.append(extract_file(path))
    daily = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    if daily["date"].duplicated().any():
        raise RuntimeError("Duplicate CHIRPS dates found")

    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    daily["is_background"] = daily["year"].isin(CONFIG["background_years"])
    daily["is_event_year"] = daily["year"].eq(EVENT_YEAR)
    daily.to_parquet(PROCESSED / "03_CHIRPS源区逐日缓存.parquet", index=False)
    daily.to_csv(OUTPUTS / "04_CHIRPS源区逐日摘要.csv", index=False, encoding="utf-8-sig")

    annual = (
        daily.groupby("year", as_index=False)
        .agg(
            days=("date", "size"),
            valid_days=("chirps_source_mm", "count"),
            period_rain_mm=("chirps_source_mm", "sum"),
            max_daily_mm=("chirps_source_mm", "max"),
            neighbourhood_mean_mm=("chirps_neighbourhood_mean_mm", "mean"),
        )
    )
    annual.to_csv(OUTPUTS / "05_CHIRPS年度质量汇总.csv", index=False, encoding="utf-8-sig")

    event_date = pd.Timestamp(CONFIG["event_time_beijing"]).normalize()
    event_row = daily.loc[daily["date"].eq(event_date)]
    event_text = "not found"
    if not event_row.empty:
        event_text = f"{event_row.iloc[0]['chirps_source_mm']:.3f} mm/day"
    report = f"""# CHIRPS 源区降水缓存报告

- 产品：CHIRPS v3.0，0.05°逐日降水。
- 纳入年份：{sorted(YEARS)}；2025 年按研究设计排除。
- 文件数：{len(paths)}；逐日记录数：{len(daily)}；缺失数：{int(daily['chirps_source_mm'].isna().sum())}。
- 源区参考点：({CONFIG['source_reference_point_wgs84']['longitude']}, {CONFIG['source_reference_point_wgs84']['latitude']})。
- 最近网格中心：({daily.iloc[0]['chirps_grid_lon']:.3f}, {daily.iloc[0]['chirps_grid_lat']:.3f})。
- 2026-08-26 源区网格逐日降水：{event_text}。
- 主值采用最近网格；3×3 邻域均值和标准差仅用于空间敏感性检验。
- CHIRPS 用于 15/30 天前期降水背景；短时触发强度由 IMERG 和 ERA5-Land 交叉刻画。
"""
    (OUTPUTS / "06_CHIRPS源区缓存报告.md").write_text(report, encoding="utf-8")
    print(annual.to_string(index=False))
    print(f"Saved {len(daily)} daily rows")


if __name__ == "__main__":
    main()
