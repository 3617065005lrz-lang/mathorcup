from __future__ import annotations

"""Download complete Jun-Sep ERA5-Land snow/deep-soil data via Earth Engine.

The download is split into daily GeoTIFF stacks. Existing non-empty TIFFs
are treated as cache, so rerunning safely resumes an interrupted download.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
import json
from pathlib import Path
import threading
import time

import ee
import requests


PROJECT = "era5-512804"
COLLECTION = "ECMWF/ERA5_LAND/HOURLY"
YEARS = tuple(range(2021, 2027))
START_MONTH_DAY = (6, 1)
END_MONTH_DAY_EXCLUSIVE = (10, 1)
CHUNK_DAYS = 1
WORKERS = 5
MAX_ATTEMPTS = 6
# ERA5-Land native 0.1 degree grid.  The region is inset by 1e-6 degree so
# Earth Engine returns exactly the 9x9 CDS grid (centres 85.1..85.9 E and
# 27.9..28.7 N) instead of adding a boundary pixel.
REGION = [85.050001, 27.850001, 85.949999, 28.749999]
CRS_TRANSFORM = [0.1, 0.0, -180.05, 0.0, -0.1, 90.05]
OUT = Path(r"D:\MathorCup_Data\ERA5_Land_Gyirong\earth_engine_nativegrid_2021_2026")
MANIFEST = OUT / "download_manifest.json"
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

_write_lock = threading.Lock()


def build_jobs() -> list[dict]:
    jobs: list[dict] = []
    for year in YEARS:
        cursor = date(year, *START_MONTH_DAY)
        season_end = date(year, *END_MONTH_DAY_EXCLUSIVE)
        while cursor < season_end:
            stop = min(cursor + timedelta(days=CHUNK_DAYS), season_end)
            target_dir = OUT / str(year)
            target = target_dir / f"era5_land_gyirong_{cursor:%Y%m%d}_{stop - timedelta(days=1):%Y%m%d}_snow_deepsoil.tif"
            jobs.append(
                {
                    "year": year,
                    "start": cursor.isoformat(),
                    "end": stop.isoformat(),
                    "days": (stop - cursor).days,
                    "target": target,
                }
            )
            cursor = stop
    return jobs


def valid_tiff(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1024:
        return False
    with path.open("rb") as handle:
        return handle.read(4) in {b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"}


def save_manifest(results: list[dict]) -> None:
    with _write_lock:
        MANIFEST.write_text(
            json.dumps(
                sorted(results, key=lambda row: (row["start"], row["end"])),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def download_job(job: dict) -> dict:
    target: Path = job["target"]
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_images = job["days"] * 24
    expected_bands = expected_images * len(BANDS)
    if valid_tiff(target):
        return {
            **{k: v for k, v in job.items() if k != "target"},
            "status": "cached",
            "target": str(target),
            "bytes": target.stat().st_size,
            "hourly_images": expected_images,
            "bands": expected_bands,
            "attempts": 0,
        }

    partial = target.with_suffix(target.suffix + ".part")
    last_error = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            collection = (
                ee.ImageCollection(COLLECTION)
                .filterDate(job["start"], job["end"])
                .filterBounds(ee.Geometry.Rectangle(REGION))
                .select(BANDS)
                .sort("system:time_start")
            )
            actual_images = int(collection.size().getInfo())
            if actual_images != expected_images:
                raise RuntimeError(
                    f"catalogue incomplete: {actual_images}/{expected_images} hourly images"
                )
            stacked = collection.toBands().toFloat()
            actual_bands = int(stacked.bandNames().size().getInfo())
            if actual_bands != expected_bands:
                raise RuntimeError(f"band count mismatch: {actual_bands}/{expected_bands}")
            url = stacked.getDownloadURL(
                {
                    "name": target.stem,
                    "region": ee.Geometry.Rectangle(REGION),
                    "crs": "EPSG:4326",
                    "crs_transform": CRS_TRANSFORM,
                    "format": "GEO_TIFF",
                }
            )
            response = requests.get(url, timeout=600)
            response.raise_for_status()
            partial.write_bytes(response.content)
            if not valid_tiff(partial):
                raise RuntimeError("downloaded payload is not a valid non-empty TIFF")
            partial.replace(target)
            return {
                **{k: v for k, v in job.items() if k != "target"},
                "status": "downloaded",
                "target": str(target),
                "bytes": target.stat().st_size,
                "hourly_images": actual_images,
                "bands": actual_bands,
                "attempts": attempt,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            partial.unlink(missing_ok=True)
            if attempt < MAX_ATTEMPTS:
                time.sleep(min(120, 10 * 2 ** (attempt - 1)))
    return {
        **{k: v for k, v in job.items() if k != "target"},
        "status": "failed",
        "target": str(target),
        "bytes": 0,
        "hourly_images": 0,
        "bands": 0,
        "attempts": MAX_ATTEMPTS,
        "error": last_error,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ee.Initialize(project=PROJECT)
    jobs = build_jobs()
    results: list[dict] = []
    print(
        f"Earth Engine jobs={len(jobs)}, years={YEARS[0]}-{YEARS[-1]}, "
        f"season=Jun-Sep, workers={WORKERS}, variables={len(BANDS)}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(download_job, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            save_manifest(results)
            print(
                f"GEE {index}/{len(jobs)} {result['start']}..{result['end']} "
                f"{result['status']} {result['bytes'] / 1048576:.2f} MiB "
                f"attempt={result['attempts']}",
                flush=True,
            )
    failed = [row for row in results if row["status"] == "failed"]
    print(f"Earth Engine complete={len(results)-len(failed)}/{len(results)}, failed={len(failed)}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
