from __future__ import annotations

"""Download official CHIRPS v3 Final daily rainfall COG subsets for Gyirong.

The source consists of global 0.05-degree daily COGs.  This script uses HTTP
range reads and stores only the declared study box, while retaining the source
URL and date in each GeoTIFF.  It therefore avoids downloading roughly 90 GB of
global annual files that would never enter the analysis.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import datetime as dt
from pathlib import Path
import time

import rasterio
from rasterio.windows import from_bounds


BASE = "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/rnl/cogs"
OUT = Path(r"D:\MathorCup_Data\External_Gyirong\P1-1_CHIRPS_v3\official_aoi_subset")
MANIFEST = OUT.parent / "download_manifest.csv"
BOUNDS = (85.1, 27.9, 85.9, 28.7)  # west, south, east, north
YEARS = [*range(2001, 2025), 2026]  # 2025 intentionally excluded from baseline
MONTHS = {6, 7, 8, 9}


def dates() -> list[dt.date]:
    result: list[dt.date] = []
    for year in YEARS:
        day = dt.date(year, 6, 1)
        end = dt.date(year, 9, 30)
        while day <= end:
            if day.month in MONTHS:
                result.append(day)
            day += dt.timedelta(days=1)
    return result


def download_one(day: dt.date) -> dict[str, str | int]:
    year_dir = OUT / str(day.year)
    year_dir.mkdir(parents=True, exist_ok=True)
    stem = f"chirps-v3.0.rnl.{day:%Y.%m.%d}"
    target = year_dir / f"{stem}_gyirong_aoi.tif"
    source = f"{BASE}/{day.year}/{stem}.cog"
    if target.exists() and target.stat().st_size > 0:
        return {"date": day.isoformat(), "status": "cached", "bytes": target.stat().st_size, "source_url": source, "target": str(target)}

    last_error = ""
    for attempt in range(1, 6):
        try:
            with rasterio.open(source) as src:
                window = from_bounds(*BOUNDS, transform=src.transform).round_offsets().round_lengths()
                data = src.read(1, window=window)
                transform = src.window_transform(window)
                profile = src.profile.copy()
                profile.update(
                    driver="GTiff",
                    height=data.shape[0],
                    width=data.shape[1],
                    count=1,
                    transform=transform,
                    compress="DEFLATE",
                    predictor=3,
                )
                partial = target.with_suffix(".tif.part")
                with rasterio.open(partial, "w", **profile) as dst:
                    dst.write(data, 1)
                    dst.update_tags(
                        source_url=source,
                        product="CHIRPS v3 Final daily rnl",
                        date=day.isoformat(),
                        aoi_wsen=",".join(map(str, BOUNDS)),
                        subset_note="Official COG HTTP range subset; source global file not locally duplicated",
                    )
                partial.replace(target)
            time.sleep(1)
            return {"date": day.isoformat(), "status": "downloaded", "bytes": target.stat().st_size, "source_url": source, "target": str(target)}
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            target.with_suffix(".tif.part").unlink(missing_ok=True)
            if attempt < 5:
                time.sleep(30 * attempt)
    return {"date": day.isoformat(), "status": "failed", "bytes": 0, "source_url": source, "target": str(target), "error": last_error}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = dates()
    fields = ["date", "status", "bytes", "source_url", "target", "error"]
    done = 0
    failed = 0
    with MANIFEST.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        # The CHC server rate-limits concurrent COG range reads.  A single
        # worker is intentionally used so a long run remains reproducible.
        with ThreadPoolExecutor(max_workers=1) as pool:
            futures = {pool.submit(download_one, day): day for day in jobs}
            for future in as_completed(futures):
                row = future.result()
                writer.writerow({field: row.get(field, "") for field in fields})
                fh.flush()
                done += 1
                failed += row["status"] == "failed"
                if done % 25 == 0 or done == len(jobs):
                    print(f"CHIRPS {done}/{len(jobs)} failed={failed}", flush=True)
    print(f"CHIRPS complete={done-failed}/{len(jobs)} failed={failed} manifest={MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
