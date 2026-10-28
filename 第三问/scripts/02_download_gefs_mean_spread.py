"""Download a compact GEFS event-week archive using ensemble mean and spread.

The script retrieves selected GRIB messages by HTTP byte range instead of downloading
global files.  It keeps five issuance times and 0--168 h lead times every 6 h, matching
the five analysis windows used in Questions 1--3.  Ensemble mean and spread are enough
to construct a transparent Gaussian/error scenario ensemble without storing all 21 members.
"""

from __future__ import annotations

import concurrent.futures
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "外源资料" / "02_GEFS集合预报"
OUT.mkdir(parents=True, exist_ok=True)

BASE = "https://noaa-gefs-pds.s3.amazonaws.com"
ISSUES = [
    # 00Z cycles nearest to the nominal 168 h and 120 h warning horizons.
    # The physical onset is 2026-08-26 02:52 UTC, so their exact lead times
    # are about 170.9 h and 122.9 h; results retain the exact lead time.
    ("20260819", "00"),
    ("20260821", "00"),
    ("20260823", "00"),
    ("20260824", "00"),
    ("20260825", "00"),
    ("20260825", "12"),
    ("20260826", "00"),
]
PRODUCTS = ("geavg", "gespr")
# f174 is retained because the 00Z cycle nearest to the nominal 168 h issue
# precedes the 02:52 UTC onset by 170.9 h.  Interpolation between f168 and
# f174 therefore covers the physical onset without extrapolation.
LEADS = range(0, 175, 6)

# Variables are deliberately restricted to interpretable drivers used by Q2.
WANTED = (
    ("TMP", "2 m above ground"),
    ("SOILW", "0-0.1 m below ground"),
    ("APCP", "surface"),
    ("DSWRF", "surface"),
)


@dataclass(frozen=True)
class Job:
    date: str
    cycle: str
    product: str
    lead: int


def urls(job: Job) -> tuple[str, str]:
    name = f"{job.product}.t{job.cycle}z.pgrb2s.0p25.f{job.lead:03d}"
    base = f"{BASE}/gefs.{job.date}/{job.cycle}/atmos/pgrb2sp25/{name}"
    return base, base + ".idx"


def wanted_line(line: str) -> bool:
    fields = line.rstrip().split(":")
    if len(fields) < 5:
        return False
    var, level = fields[3], fields[4]
    return any(var == v and level == lev for v, lev in WANTED)


def download_job(job: Job) -> dict[str, object]:
    run_dir = OUT / f"gefs_{job.date}_{job.cycle}z"
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / f"{job.product}_f{job.lead:03d}_selected.grib2"
    if target.exists() and target.stat().st_size > 0:
        return {"file": str(target), "status": "existing", "bytes": target.stat().st_size}

    data_url, idx_url = urls(job)
    idx_response = requests.get(idx_url, timeout=60)
    idx_response.raise_for_status()
    lines = [x for x in idx_response.text.splitlines() if x.strip()]
    parsed: list[tuple[int, str]] = []
    for line in lines:
        match = re.match(r"^\d+:(\d+):", line)
        if match:
            parsed.append((int(match.group(1)), line))

    ranges: list[tuple[int, int | None, str]] = []
    for i, (start, line) in enumerate(parsed):
        if wanted_line(line):
            end = parsed[i + 1][0] - 1 if i + 1 < len(parsed) else None
            ranges.append((start, end, line))

    if not ranges:
        return {"file": str(target), "status": "no_matching_records", "bytes": 0}

    tmp = target.with_suffix(target.suffix + ".part")
    total = 0
    with tmp.open("wb") as stream:
        for start, end, _ in ranges:
            range_value = f"bytes={start}-{'' if end is None else end}"
            response = requests.get(data_url, headers={"Range": range_value}, timeout=120)
            response.raise_for_status()
            stream.write(response.content)
            total += len(response.content)
    tmp.replace(target)
    return {"file": str(target), "status": "downloaded", "bytes": total}


def main() -> None:
    jobs = [Job(d, c, p, h) for d, c in ISSUES for p in PRODUCTS for h in LEADS]
    records: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        future_map = {pool.submit(download_job, job): job for job in jobs}
        for number, future in enumerate(concurrent.futures.as_completed(future_map), 1):
            job = future_map[future]
            try:
                row = download_job(job) if False else future.result()
            except Exception as exc:  # retain a complete failure manifest
                row = {"file": "", "status": f"error: {exc}", "bytes": 0}
            row.update(date=job.date, cycle=job.cycle, product=job.product, lead_h=job.lead)
            records.append(row)
            print(f"[{number}/{len(jobs)}] {job} -> {row['status']}", flush=True)

    manifest = OUT / "download_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("date", "cycle", "product", "lead_h", "status", "bytes", "file"),
        )
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda r: (r["date"], r["cycle"], r["product"], r["lead_h"])))


if __name__ == "__main__":
    main()

