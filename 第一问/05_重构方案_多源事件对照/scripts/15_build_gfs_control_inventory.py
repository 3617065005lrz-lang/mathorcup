#!/usr/bin/env python3
"""Build an NCEI GFS inventory for calendar-matched 2021-2024 controls.

This creates only the official-file catalogue.  The existing verified P0-5
downloader then retrieves selected GRIB byte ranges and crops the Gyirong AOI.
"""

from __future__ import annotations

import csv
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


YEARS = (2021, 2022, 2023, 2024)
DAYS = range(22, 27)
CYCLES = (0, 6, 12, 18)
LEADS = range(0, 121, 3)
OUTPUT = Path(
    r"D:\MathorCup_Data\External_Gyirong\P0-5_GFS_historical_controls"
    r"\original_toolkit\帮助_交接包\P0-5_GFS\catalog\gfs_official_inventory.csv"
)


def make_row(year: int, day: int, cycle: int, lead: int) -> dict[str, object]:
    ymd = f"{year}08{day:02d}"
    filename = f"gfs_4_{ymd}_{cycle:02d}00_{lead:03d}.grb2"
    base = (
        "https://www.ncei.noaa.gov/oa/prod-model/global-forecast-system/"
        f"access/grid-004-0.5-degree/forecast/{year}08/{ymd}/{filename}"
    )
    ref = datetime(year, 8, day, cycle, tzinfo=timezone.utc)
    return {
        "forecast_reference_time_utc": ref.isoformat().replace("+00:00", "Z"),
        "lead_hours": lead,
        "valid_time_utc": "",
        "filename": filename,
        "full_grib_size_bytes": 0,
        "grib_url": base,
        "inventory_url": base + ".inv",
        "available": False,
        "planned_output": f"gfs_{ymd}{cycle:02d}_f{lead:03d}_gyirong.nc",
    }


def probe(row: dict[str, object]) -> dict[str, object]:
    req = urllib.request.Request(str(row["grib_url"]), method="HEAD")
    req.add_header("User-Agent", "MathorCup-GFS-control-inventory/1.0")
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            row["full_grib_size_bytes"] = int(response.headers["Content-Length"])
            row["available"] = response.status == 200
    except Exception:
        pass
    return row


def main() -> None:
    rows = [
        make_row(year, day, cycle, lead)
        for year in YEARS
        for day in DAYS
        for cycle in CYCLES
        for lead in LEADS
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    done: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(probe, row) for row in rows]
        for index, future in enumerate(as_completed(futures), 1):
            done.append(future.result())
            if index % 100 == 0:
                print(f"inventory {index}/{len(rows)}", flush=True)
    done.sort(key=lambda r: (str(r["forecast_reference_time_utc"]), int(r["lead_hours"])))
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(done[0]))
        writer.writeheader()
        writer.writerows(done)
    available = sum(bool(row["available"]) for row in done)
    print(f"saved {OUTPUT}; available={available}/{len(done)}", flush=True)


if __name__ == "__main__":
    main()
