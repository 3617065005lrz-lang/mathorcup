from __future__ import annotations

"""下载ERA5压力层温度和位势高度，用于动态递减率与0摄氏度层。"""

import argparse
from pathlib import Path
import time

import cdsapi


ROOT = Path(r"D:\MathorCup_Data\ERA5_Land_Gyirong\raw_official\q1_pressure_levels")
YEARS = (2021, 2022, 2023, 2024, 2026)
MONTHS = ("06", "07", "08", "09")
LEVELS = ("400", "450", "500", "550", "600", "650", "700")
HOURS = [f"{hour:02d}:00" for hour in range(24)]
DAYS = [f"{day:02d}" for day in range(1, 32)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-attempts", type=int, default=4)
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client(quiet=True)
    total = len(YEARS) * len(MONTHS)
    done = 0
    for year in YEARS:
        for month in MONTHS:
            target = ROOT / f"era5_pressure_gyirong_{year}_{month}_t_z_400-700hpa.grib"
            if target.exists() and target.stat().st_size > 0:
                done += 1
                print(f"pressure {done}/{total} cached {year}-{month}", flush=True)
                continue
            partial = target.with_suffix(target.suffix + ".part")
            for attempt in range(1, args.max_attempts + 1):
                try:
                    partial.unlink(missing_ok=True)
                    client.retrieve(
                        "reanalysis-era5-pressure-levels",
                        {
                            "product_type": ["reanalysis"],
                            "variable": ["temperature", "geopotential"],
                            "pressure_level": list(LEVELS),
                            "year": str(year),
                            "month": [month],
                            "day": DAYS,
                            "time": HOURS,
                            "area": [28.7, 85.1, 27.9, 85.9],
                            "data_format": "grib",
                            "download_format": "unarchived"
                        },
                        str(partial),
                    )
                    if not partial.exists() or partial.stat().st_size == 0:
                        raise RuntimeError("CDS未生成有效文件")
                    partial.replace(target)
                    done += 1
                    print(f"pressure {done}/{total} downloaded {year}-{month} {target.stat().st_size/1048576:.1f} MiB", flush=True)
                    break
                except Exception as exc:
                    print(f"pressure retry {year}-{month} attempt={attempt}: {type(exc).__name__}: {exc}", flush=True)
                    if attempt == args.max_attempts:
                        raise
                    time.sleep(30 * attempt)


if __name__ == "__main__":
    main()
