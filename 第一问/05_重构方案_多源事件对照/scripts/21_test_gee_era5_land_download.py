from __future__ import annotations

"""Download a one-day ERA5-Land sample from Google Earth Engine."""

import json
from pathlib import Path
import time

import ee
import requests


PROJECT = "era5-512804"
COLLECTION = "ECMWF/ERA5_LAND/HOURLY"
START = "2021-08-20"
END = "2021-08-21"
REGION = [85.050001, 27.850001, 85.949999, 28.749999]
CRS_TRANSFORM = [0.1, 0, -180.05, 0, -0.1, 90.05]
OUT = Path(r"D:\MathorCup_Data\ERA5_Land_Gyirong\earth_engine_test")
BANDS = [
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


def main() -> None:
    started = time.perf_counter()
    ee.Initialize(project=PROJECT)
    collection = (
        ee.ImageCollection(COLLECTION)
        .filterDate(START, END)
        .filterBounds(ee.Geometry.Rectangle(REGION))
        .select(BANDS)
        .sort("system:time_start")
    )
    image_count = int(collection.size().getInfo())
    if image_count != 24:
        raise RuntimeError(f"Expected 24 hourly images, got {image_count}")

    stacked = collection.toBands().toFloat()
    band_names = stacked.bandNames().getInfo()
    url = stacked.getDownloadURL(
        {
            "name": "era5_land_gyirong_20210820_snow_deepsoil",
            "region": ee.Geometry.Rectangle(REGION),
            "crs": "EPSG:4326",
            "crs_transform": CRS_TRANSFORM,
            "format": "GEO_TIFF",
        }
    )
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "era5_land_gyirong_20210820_snow_deepsoil_nativegrid.tif"
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    target.write_bytes(response.content)
    elapsed = time.perf_counter() - started
    metadata = {
        "project": PROJECT,
        "collection": COLLECTION,
        "start_utc": START,
        "end_utc_exclusive": END,
        "region_wsen": REGION,
        "crs_transform": CRS_TRANSFORM,
        "hourly_image_count": image_count,
        "band_count": len(band_names),
        "band_names": band_names,
        "output": str(target),
        "bytes": target.stat().st_size,
        "elapsed_seconds": elapsed,
    }
    (OUT / "sample_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in metadata.items() if k != "band_names"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
