from __future__ import annotations

"""Validate the complete Earth Engine ERA5-Land delivery."""

import csv
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import rasterio


ROOT = Path(r"D:\MathorCup_Data\ERA5_Land_Gyirong\earth_engine_nativegrid_2021_2026")
REPORT_DIR = Path(__file__).resolve().parents[1] / "results" / "gee_era5_quality"
YEARS = range(2021, 2027)
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
PATTERN = re.compile(r"era5_land_gyirong_(\d{8})_(\d{8})_snow_deepsoil\.tif$")


def expected_dates() -> list[date]:
    values = []
    for year in YEARS:
        cursor = date(year, 6, 1)
        stop = date(year, 10, 1)
        while cursor < stop:
            values.append(cursor)
            cursor += timedelta(days=1)
    return values


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(ROOT.glob("20??/*.tif"))
    expected = expected_dates()
    date_to_path: dict[date, Path] = {}
    filename_errors = []
    for path in paths:
        match = PATTERN.fullmatch(path.name)
        if not match:
            filename_errors.append(str(path))
            continue
        start = date.fromisoformat(f"{match.group(1)[:4]}-{match.group(1)[4:6]}-{match.group(1)[6:]}")
        end = date.fromisoformat(f"{match.group(2)[:4]}-{match.group(2)[4:6]}-{match.group(2)[6:]}")
        if start != end or start.year != int(path.parent.name):
            filename_errors.append(str(path))
        date_to_path[start] = path

    missing_dates = [value.isoformat() for value in expected if value not in date_to_path]
    extra_dates = [value.isoformat() for value in date_to_path if value not in set(expected)]
    reference_profile = None
    profile_mismatches = []
    unreadable = []
    total_cells = {name: 0 for name in BANDS}
    finite_cells = {name: 0 for name in BANDS}
    minima = {name: np.inf for name in BANDS}
    maxima = {name: -np.inf for name in BANDS}
    sums = {name: 0.0 for name in BANDS}
    negatives = {name: 0 for name in BANDS}
    inventory_rows = []

    for index, path in enumerate(paths, 1):
        try:
            with rasterio.open(path) as dataset:
                profile = {
                    "width": dataset.width,
                    "height": dataset.height,
                    "count": dataset.count,
                    "crs": str(dataset.crs),
                    "transform": tuple(dataset.transform),
                    "bounds": tuple(dataset.bounds),
                    "dtype": dataset.dtypes[0],
                    "nodata": dataset.nodata,
                }
                if reference_profile is None:
                    reference_profile = profile
                elif profile != reference_profile:
                    profile_mismatches.append({"path": str(path), "profile": profile})
                if dataset.count != 24 * len(BANDS):
                    raise RuntimeError(f"unexpected band count {dataset.count}")
                array = dataset.read()
                for variable_index, name in enumerate(BANDS):
                    values = array[variable_index :: len(BANDS)].astype("f8", copy=False)
                    mask = np.isfinite(values)
                    finite = values[mask]
                    total_cells[name] += int(values.size)
                    finite_cells[name] += int(finite.size)
                    if finite.size:
                        minima[name] = min(minima[name], float(finite.min()))
                        maxima[name] = max(maxima[name], float(finite.max()))
                        sums[name] += float(finite.sum())
                        negatives[name] += int((finite < 0).sum())
            inventory_rows.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "status": "ok",
                }
            )
        except Exception as exc:  # noqa: BLE001
            unreadable.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
        if index % 100 == 0:
            print(f"validated {index}/{len(paths)}", flush=True)

    stats = []
    for name in BANDS:
        count = finite_cells[name]
        stats.append(
            {
                "variable": name,
                "total_cells": total_cells[name],
                "finite_cells": count,
                "missing_cells": total_cells[name] - count,
                "missing_fraction": (total_cells[name] - count) / total_cells[name] if total_cells[name] else None,
                "min": minima[name] if count else None,
                "max": maxima[name] if count else None,
                "mean": sums[name] / count if count else None,
                "negative_cells": negatives[name],
            }
        )

    range_rules = {
        # ERA5-Land snow cover is delivered as percent in this collection.
        "snow_cover": (0.0, 100.0),
        # Tiny negative values (about 1e-23) are floating-point packing noise.
        "snow_depth": (-1e-10, 50.0),
        "snow_depth_water_equivalent": (-1e-10, 20.0),
        "snowfall": (-1e-5, 20.0),
        "snowmelt": (-1e-5, 20.0),
        "snowfall_hourly": (-1e-5, 5.0),
        "snowmelt_hourly": (-1e-5, 5.0),
        "soil_temperature_level_2": (180.0, 340.0),
        "soil_temperature_level_3": (180.0, 340.0),
        "soil_temperature_level_4": (180.0, 340.0),
        "volumetric_soil_water_layer_2": (0.0, 1.0),
        "volumetric_soil_water_layer_3": (0.0, 1.0),
        "volumetric_soil_water_layer_4": (0.0, 1.0),
    }
    range_violations = []
    for row in stats:
        low, high = range_rules[row["variable"]]
        if row["min"] is None or row["min"] < low or row["max"] > high:
            range_violations.append({**row, "expected_min": low, "expected_max": high})

    schema_rows = []
    for hour in range(24):
        for variable_index, name in enumerate(BANDS):
            schema_rows.append(
                {
                    "band_1based": hour * len(BANDS) + variable_index + 1,
                    "hour_utc": f"{hour:02d}:00",
                    "variable": name,
                }
            )
    with (REPORT_DIR / "band_schema.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(schema_rows[0]))
        writer.writeheader()
        writer.writerows(schema_rows)
    with (REPORT_DIR / "file_inventory_sha256.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory_rows[0]))
        writer.writeheader()
        writer.writerows(inventory_rows)
    with (REPORT_DIR / "variable_statistics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stats[0]))
        writer.writeheader()
        writer.writerows(stats)

    report = {
        "root": str(ROOT),
        "expected_files": len(expected),
        "actual_files": len(paths),
        "valid_files": len(inventory_rows),
        "missing_dates": missing_dates,
        "extra_dates": extra_dates,
        "filename_errors": filename_errors,
        "unreadable": unreadable,
        "reference_profile": reference_profile,
        "profile_mismatches": profile_mismatches,
        "variables": BANDS,
        "variable_statistics": stats,
        "range_violations": range_violations,
        "band_descriptions_embedded": False,
        "band_schema_file": str(REPORT_DIR / "band_schema.csv"),
        "quality_pass": not any(
            [missing_dates, extra_dates, filename_errors, unreadable, profile_mismatches, range_violations]
        ),
    }
    (REPORT_DIR / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "variable_statistics"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
