from __future__ import annotations

"""Resume-capable official CHIRPS v3 monthly NetCDF downloader.

This replaces thousands of COG range requests with 100 sequential monthly files.
Files are kept in their original official form and can be cropped locally later.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import threading
import time

import requests


BASE = "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/rnl/netcdf/byMonth"
OUT = Path(r"D:\MathorCup_Data\External_Gyirong\P1-1_CHIRPS_v3\official_monthly_netcdf")
MANIFEST = OUT / "download_manifest.json"
YEARS = [2026, 2024, 2023, 2022, 2021, *range(2001, 2021)]
MONTHS = [8, 7, 6, 9]
LOCK = threading.Lock()


def jobs() -> list[tuple[int, int, str, Path]]:
    result = []
    for year in YEARS:
        for month in MONTHS:
            name = f"chirps-v3.0.{year}.{month:02d}.days_p05.nc"
            result.append((year, month, f"{BASE}/{name}", OUT / name))
    return result


def remote_size(session: requests.Session, url: str) -> int:
    response = session.head(url, timeout=60, allow_redirects=True)
    response.raise_for_status()
    return int(response.headers["Content-Length"])


def download_one(job: tuple[int, int, str, Path]) -> dict:
    year, month, url, target = job
    partial = target.with_suffix(target.suffix + ".part")
    error = ""
    for attempt in range(1, 6):
        try:
            with requests.Session() as session:
                expected = remote_size(session, url)
                if target.exists() and target.stat().st_size == expected:
                    return {"year": year, "month": month, "status": "cached", "bytes": expected, "url": url, "target": str(target)}
                command = [
                    "curl.exe", "--silent", "--show-error", "--location",
                    "--continue-at", "-", "--retry", "8", "--retry-all-errors",
                    "--retry-delay", "15", "--output", str(partial), url,
                ]
                subprocess.run(command, check=True)
                if partial.stat().st_size != expected:
                    raise RuntimeError(f"size mismatch {partial.stat().st_size} != {expected}")
                partial.replace(target)
                return {"year": year, "month": month, "status": "downloaded", "bytes": expected, "url": url, "target": str(target)}
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            if attempt < 5:
                time.sleep(30 * attempt)
    return {"year": year, "month": month, "status": "failed", "bytes": 0, "url": url, "target": str(target), "error": error}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    work = jobs()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(download_one, job): job for job in work}
        for future in as_completed(futures):
            row = future.result()
            with LOCK:
                results.append(row)
                MANIFEST.write_text(
                    json.dumps(sorted(results, key=lambda x: (x["year"], x["month"])), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                done = sum(x["status"] in {"cached", "downloaded"} for x in results)
                failed = sum(x["status"] == "failed" for x in results)
            print(
                f"CHIRPS monthly {done}/{len(work)} failed={failed} "
                f"latest={row['year']}-{row['month']:02d} {row['status']} {row.get('bytes', 0) / 1024**2:.1f} MB",
                flush=True,
            )


if __name__ == "__main__":
    main()
