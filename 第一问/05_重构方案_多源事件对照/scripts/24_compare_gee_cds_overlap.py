from __future__ import annotations

"""Compare an aligned Earth Engine day against the CDS GRIB source."""

import csv
import json
from pathlib import Path

import numpy as np
import rasterio


GEE = Path(r"D:\MathorCup_Data\ERA5_Land_Gyirong\earth_engine_test\era5_land_gyirong_20210820_snow_deepsoil_nativegrid.tif")
CDS = Path(r"D:\MathorCup_Data\ERA5_Land_Gyirong\raw_official\q1_history_snow_deepsoil\era5_land_gyirong_2021_08_snow_deepsoil_supplement.grib")
OUT = Path(__file__).resolve().parents[1] / "results" / "gee_era5_quality"
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
CDS_VARIABLES = [
    "snow_cover",
    "snow_depth",
    "snow_depth_water_equivalent",
    "snowfall",
    "snowmelt",
    "soil_temperature_level_2",
    "soil_temperature_level_3",
    "soil_temperature_level_4",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
]


def metrics(left: np.ndarray, right: np.ndarray) -> dict:
    mask = np.isfinite(left) & np.isfinite(right)
    x = left[mask].astype("f8")
    y = right[mask].astype("f8")
    delta = x - y
    if x.size > 1 and np.std(x) > 0 and np.std(y) > 0:
        correlation = float(np.corrcoef(x, y)[0, 1])
    else:
        correlation = None
    return {
        "cells": int(x.size),
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "max_abs_error": float(np.max(np.abs(delta))),
        "correlation": correlation,
        "gee_mean": float(np.mean(x)),
        "cds_mean": float(np.mean(y)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with rasterio.open(GEE) as gee, rasterio.open(CDS) as cds:
        if gee.width != cds.width or gee.height != cds.height:
            raise RuntimeError("spatial dimensions do not match")
        for cds_index, variable in enumerate(CDS_VARIABLES):
            gee_index = GEE_BANDS.index(variable)
            gee_values = []
            cds_values = []
            for hour in range(24):
                gee_band = hour * len(GEE_BANDS) + gee_index + 1
                cds_band = ((20 - 1) * 24 + hour) * len(CDS_VARIABLES) + cds_index + 1
                left = gee.read(gee_band).astype("f8")
                right = cds.read(cds_band).astype("f8")
                # The GDAL GRIB comment labels these bands as C, but the stored
                # ERA5 values are Kelvin in both files; compare the values directly.
                gee_values.append(left)
                cds_values.append(right)
            row = {"variable": variable, **metrics(np.stack(gee_values), np.stack(cds_values))}
            rows.append(row)
    with (OUT / "gee_vs_cds_20210820.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "gee_vs_cds_20210820.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
