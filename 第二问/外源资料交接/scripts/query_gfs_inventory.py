#!/usr/bin/env python3
"""Inventory NCEI GFS grid-004 forecasts for 2026-08-22..26.

The NOAA object-store listing is queried without downloading the global GRIB2
payloads. The companion downloader uses each .inv file and HTTP byte ranges to
retrieve only requested fields, then crops them to the Gyirong background AOI.
"""

from __future__ import annotations

import csv
import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, timedelta, datetime, timezone
from pathlib import Path


DEST = Path(__file__).resolve().parents[1] / "P0-5_GFS" / "catalog"
S3_LIST = "https://www.ncei.noaa.gov/oa/prod-model"
OBJECT_BASE = "https://www.ncei.noaa.gov/oa/prod-model/"
PREFIX_ROOT = "global-forecast-system/access/grid-004-0.5-degree/forecast"
START = date(2026, 8, 22)
END = date(2026, 8, 26)
CYCLES = (0, 6, 12, 18)
LEADS = tuple(range(0, 121, 3))


def list_prefix(prefix: str) -> dict[str, int]:
    params = urllib.parse.urlencode({"list-type": "2", "prefix": prefix})
    request = urllib.request.Request(
        f"{S3_LIST}?{params}", headers={"User-Agent": "Codex-P0-456/1.0"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        root = ET.fromstring(response.read())
    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    found = {}
    for item in root.findall("s3:Contents", namespace):
        key = item.findtext("s3:Key", default="", namespaces=namespace)
        size = int(item.findtext("s3:Size", default="0", namespaces=namespace))
        found[key] = size
    return found


def day_range(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    objects: dict[str, int] = {}
    for day in day_range(START, END):
        day_text = day.strftime("%Y%m%d")
        prefix = f"{PREFIX_ROOT}/{day:%Y%m}/{day_text}/"
        objects.update(list_prefix(prefix))

    rows = []
    for day in day_range(START, END):
        day_text = day.strftime("%Y%m%d")
        for cycle in CYCLES:
            for lead in LEADS:
                name = f"gfs_4_{day_text}_{cycle:02d}00_{lead:03d}.grb2"
                key = f"{PREFIX_ROOT}/{day:%Y%m}/{day_text}/{name}"
                inv_key = f"{key}.inv"
                rows.append(
                    {
                        "forecast_reference_time_utc": f"{day_text}T{cycle:02d}:00:00Z",
                        "lead_hours": lead,
                        "valid_time_utc": (
                            datetime(day.year, day.month, day.day, cycle, tzinfo=timezone.utc)
                            + timedelta(hours=lead)
                        ).isoformat().replace("+00:00", "Z"),
                        "filename": name,
                        "full_grib_size_bytes": objects.get(key, 0),
                        "grib_url": OBJECT_BASE + key,
                        "inventory_url": OBJECT_BASE + inv_key,
                        "available": key in objects and inv_key in objects,
                        "planned_output": f"gfs_{day_text}{cycle:02d}_f{lead:03d}_gyirong.nc",
                    }
                )

    csv_path = DEST / "gfs_official_inventory.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "query_time_utc": datetime.now(timezone.utc).isoformat(),
        "source": "NOAA NCEI NESDIS object store",
        "grid": "004 (0.5 degree)",
        "date_start": START.isoformat(),
        "date_end": END.isoformat(),
        "cycles_utc": CYCLES,
        "forecast_leads_hours": [min(LEADS), max(LEADS), 3],
        "expected_file_count": len(rows),
        "available_file_count": sum(row["available"] for row in rows),
        "full_global_volume_gib": round(
            sum(row["full_grib_size_bytes"] for row in rows) / 1024**3, 3
        ),
        "download_plan": (
            "Use .inv indexes plus HTTP Range for the requested fields only; "
            "decode and crop to the background AOI instead of downloading global GRIB2 files."
        ),
    }
    (DEST / "gfs_inventory_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
