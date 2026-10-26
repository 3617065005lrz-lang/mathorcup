from __future__ import annotations

"""Build a source-area MODIS snow-cover and snow-albedo cache.

Only the reconstructed source AOI is read from each 2400 x 2400 HDF4 tile.
The cache keeps Terra and Aqua diagnostics separately and then fuses them at
the daily level.  No post-event observation is used by this script to compute
an operational trigger variable; that lag is imposed in the next stage.
"""

from datetime import datetime, timedelta
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from pyhdf.SD import SD, SDC
from rasterio.warp import transform


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent
CONFIG = json.loads((MODEL_DIR / "config" / "model_config.json").read_text(encoding="utf-8"))
PROCESSED = MODEL_DIR / "processed"
OUTPUTS = MODEL_DIR / "outputs"

MODIS_ROOT = Path(r"D:\MathorCup_Data\External_Gyirong\P1-2_MODIS_Snow")
INPUT_DIRS = [MODIS_ROOT / "data_historical_2021_2024", MODIS_ROOT / "data"]

SINUSOIDAL = "+proj=sinu +R=6371007.181 +nadgrids=@null +wktext"
ULX = 7783653.637667
ULY = 3335851.559000
PIXEL = 463.312716527778
NROWS = 2400
NCOLS = 2400
FILENAME = re.compile(r"(?P<sensor>MOD10A1|MYD10A1)\.A(?P<year>\d{4})(?P<doy>\d{3})\..*\.hdf$")


def aoi_window_and_mask() -> tuple[int, int, int, int, np.ndarray]:
    aoi = CONFIG["source_aoi_wgs84"]
    lon = [aoi["west"], aoi["west"], aoi["east"], aoi["east"]]
    lat = [aoi["south"], aoi["north"], aoi["south"], aoi["north"]]
    xs, ys = transform("EPSG:4326", SINUSOIDAL, lon, lat)
    cols = [(x - ULX) / PIXEL for x in xs]
    rows = [(ULY - y) / PIXEL for y in ys]
    col0 = max(0, int(np.floor(min(cols))) - 1)
    col1 = min(NCOLS, int(np.ceil(max(cols))) + 2)
    row0 = max(0, int(np.floor(min(rows))) - 1)
    row1 = min(NROWS, int(np.ceil(max(rows))) + 2)

    rr, cc = np.meshgrid(np.arange(row0, row1), np.arange(col0, col1), indexing="ij")
    x_center = ULX + (cc.ravel() + 0.5) * PIXEL
    y_center = ULY - (rr.ravel() + 0.5) * PIXEL
    pixel_lon, pixel_lat = transform(SINUSOIDAL, "EPSG:4326", x_center.tolist(), y_center.tolist())
    mask = (
        (np.asarray(pixel_lon) >= aoi["west"])
        & (np.asarray(pixel_lon) <= aoi["east"])
        & (np.asarray(pixel_lat) >= aoi["south"])
        & (np.asarray(pixel_lat) <= aoi["north"])
    ).reshape(rr.shape)
    if not mask.any():
        raise RuntimeError("MODIS source AOI contains no 500 m pixel centres")
    return row0, row1, col0, col1, mask


def parse_date(path: Path) -> tuple[str, pd.Timestamp]:
    match = FILENAME.fullmatch(path.name)
    if not match:
        raise ValueError(f"Unexpected MODIS filename: {path.name}")
    date = datetime(int(match["year"]), 1, 1) + timedelta(days=int(match["doy"]) - 1)
    return match["sensor"], pd.Timestamp(date)


def read_sds(hdf: SD, name: str, row0: int, row1: int, col0: int, col1: int) -> np.ndarray:
    return np.asarray(hdf.select(name)[row0:row1, col0:col1])


def summarize_file(path: Path, window: tuple[int, int, int, int, np.ndarray]) -> dict[str, object]:
    sensor, date = parse_date(path)
    row0, row1, col0, col1, mask = window
    hdf = SD(str(path), SDC.READ)
    try:
        ndsi = read_sds(hdf, "NDSI_Snow_Cover", row0, row1, col0, col1).astype(float)
        qa = read_sds(hdf, "NDSI_Snow_Cover_Basic_QA", row0, row1, col0, col1).astype(float)
        albedo = read_sds(hdf, "Snow_Albedo_Daily_Tile", row0, row1, col0, col1).astype(float)
    finally:
        hdf.end()

    ndsi = ndsi[mask]
    qa = qa[mask]
    albedo = albedo[mask]
    qa_max = int(CONFIG["modis"]["strict_basic_qa_max"])
    ndsi_threshold = float(CONFIG["modis"]["snow_ndsi_threshold_percent"])
    valid = (qa >= 0) & (qa <= qa_max) & (ndsi >= 0) & (ndsi <= 100)
    snow = valid & (ndsi >= ndsi_threshold)
    valid_albedo = snow & (albedo >= 0) & (albedo <= 100)
    total = int(mask.sum())
    return {
        "date": date,
        "sensor": "Terra" if sensor == "MOD10A1" else "Aqua",
        "granule": path.name,
        "aoi_pixel_count": total,
        "qa_valid_pixel_count": int(valid.sum()),
        "qa_valid_fraction": float(valid.sum() / total),
        "snow_pixel_count": int(snow.sum()),
        "snow_fraction_40": float(snow.sum() / valid.sum()) if valid.any() else np.nan,
        "ndsi_snow_cover_mean_percent": float(np.mean(ndsi[valid])) if valid.any() else np.nan,
        "snow_albedo_pixel_count": int(valid_albedo.sum()),
        "snow_albedo_mean": float(np.mean(albedo[valid_albedo]) / 100.0) if valid_albedo.any() else np.nan,
    }


def weighted(values: pd.Series, weights: pd.Series) -> float:
    usable = values.notna() & weights.gt(0)
    if not usable.any():
        return np.nan
    return float(np.average(values[usable], weights=weights[usable]))


def fuse_daily(per_sensor: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date, group in per_sensor.groupby("date", sort=True):
        rows.append(
            {
                "date": date,
                "sensor_count": int(group["sensor"].nunique()),
                "qa_valid_fraction": weighted(group["qa_valid_fraction"], group["aoi_pixel_count"]),
                "snow_fraction_40": weighted(group["snow_fraction_40"], group["qa_valid_pixel_count"]),
                "ndsi_snow_cover_mean_percent": weighted(
                    group["ndsi_snow_cover_mean_percent"], group["qa_valid_pixel_count"]
                ),
                "snow_albedo_mean": weighted(group["snow_albedo_mean"], group["snow_albedo_pixel_count"]),
                "qa_valid_pixel_observations": int(group["qa_valid_pixel_count"].sum()),
                "snow_albedo_pixel_observations": int(group["snow_albedo_pixel_count"].sum()),
            }
        )
    daily = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    daily["year"] = daily["date"].dt.year
    daily["day_of_year"] = daily["date"].dt.dayofyear
    return daily


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    paths = sorted(path for directory in INPUT_DIRS for path in directory.glob("*.hdf"))
    if len(paths) != 669:
        raise RuntimeError(f"MODIS input count mismatch: {len(paths)}/669")
    window = aoi_window_and_mask()
    records = []
    for index, path in enumerate(paths, 1):
        records.append(summarize_file(path, window))
        if index % 50 == 0 or index == len(paths):
            print(f"{index}/{len(paths)} MODIS granules processed", flush=True)
    per_sensor = pd.DataFrame(records).sort_values(["date", "sensor"]).reset_index(drop=True)
    daily = fuse_daily(per_sensor)
    per_sensor.to_parquet(PROCESSED / "01_MODIS源区逐星每日缓存.parquet", index=False)
    daily.to_parquet(PROCESSED / "02_MODIS源区融合每日缓存.parquet", index=False)
    daily.to_csv(OUTPUTS / "01_MODIS源区融合每日摘要.csv", index=False, encoding="utf-8-sig")

    year_summary = (
        daily.groupby("year")
        .agg(
            天数=("date", "size"),
            雪盖有效天数=("snow_fraction_40", "count"),
            反照率有效天数=("snow_albedo_mean", "count"),
            平均QA有效比例=("qa_valid_fraction", "mean"),
        )
        .reset_index()
    )
    year_summary.to_csv(OUTPUTS / "02_MODIS年度质量汇总.csv", index=False, encoding="utf-8-sig")
    row0, row1, col0, col1, mask = window
    report = f"""# MODIS源区积雪与反照率缓存报告

- 原始文件：{len(paths)}景（Terra/Aqua，2021—2024与2026）；
- 源区AOI内500 m像元中心数：{int(mask.sum())}；
- 仅读取原始2400×2400瓦片中行`{row0}:{row1}`、列`{col0}:{col1}`的小窗口；
- 严格质量控制：`NDSI_Snow_Cover_Basic_QA <= {CONFIG['modis']['strict_basic_qa_max']}`；
- 雪像元定义：`NDSI_Snow_Cover >= {CONFIG['modis']['snow_ndsi_threshold_percent']}%`；
- 反照率仅在满足质量控制的雪像元上求均值；
- Terra与Aqua按有效像元数加权融合，未使用灾害标签；
- 当日过境可能晚于事件发生时刻，因此下一阶段构造事前指标时必须使用至少滞后1天的反照率，禁止用2026-08-26灾后过境信息。
"""
    (OUTPUTS / "03_MODIS源区缓存报告.md").write_text(report, encoding="utf-8")
    print(year_summary.to_string(index=False))


if __name__ == "__main__":
    main()
