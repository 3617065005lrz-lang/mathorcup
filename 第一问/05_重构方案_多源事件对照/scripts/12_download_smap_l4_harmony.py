from __future__ import annotations

"""Submit, resume and download SMAP L4 V8 Gyirong subsets via NASA Harmony."""

import calendar
import datetime as dt
import json
from pathlib import Path
import time

import earthaccess
from harmony import BBox, Client, Collection, Request


COLLECTION = "C3480440870-NSIDC_CPRD"  # SPL4SMGP.008
OUT = Path(r"D:\MathorCup_Data\External_Gyirong\P1-8_SMAP_L4_V8")
STATE = OUT / "harmony_jobs.json"
YEARS = [2021, 2022, 2023, 2024, 2026]  # 2025 excluded from normal baseline
MONTHS = [8, 6, 7, 9]  # event-month counterparts first
ACTIVE_LIMIT = 2
VARIABLES = [
    "cell_lat",
    "cell_lon",
    "Geophysical_Data/sm_surface",
    "Geophysical_Data/sm_surface_wetness",
    "Geophysical_Data/sm_rootzone",
    "Geophysical_Data/sm_rootzone_pctl",
    "Geophysical_Data/sm_rootzone_wetness",
    "Geophysical_Data/soil_temp_layer1",
]


def save(state: dict) -> None:
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def periods() -> list[tuple[int, int]]:
    return [(year, month) for month in MONTHS for year in YEARS]


def request_for(year: int, month: int) -> Request:
    start = dt.datetime(year, month, 1)
    stop = dt.datetime(year, month, calendar.monthrange(year, month)[1], 23, 59, 59)
    return Request(
        Collection(COLLECTION),
        spatial=BBox(85.1, 27.9, 85.9, 28.7),
        temporal={"start": start, "stop": stop},
        variables=VARIABLES,
        format="application/netcdf",
        max_results=300,
        skip_preview=True,
        labels=["mathorcup-gyirong-smap"],
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    auth = earthaccess.login(strategy="netrc")
    if not auth.authenticated:
        raise RuntimeError("Earthdata authentication failed")
    client = Client(token=auth.token["access_token"], check_interval=5)
    wanted = [f"{y}-{m:02d}" for y, m in periods()]

    while True:
        active = 0
        completed = 0
        failed = 0
        for key in wanted:
            entry = state.get(key)
            if not entry:
                continue
            status = client.status(entry["job_id"])
            entry["status"] = status["status"]
            entry["progress"] = status.get("progress")
            entry["message"] = status.get("message", "")
            # Harmony may finish a large monthly request with a small number of
            # failed granules.  The successful outputs are still downloadable;
            # retain them and record that the month needs a later gap audit.
            if status["status"] in {"successful", "complete_with_errors"} and not entry.get("downloaded"):
                target_dir = OUT / key
                target_dir.mkdir(parents=True, exist_ok=True)
                try:
                    futures = client.download_all(entry["job_id"], directory=str(target_dir), overwrite=False)
                    files = [str(future.result()) for future in futures]
                    entry["downloaded"] = bool(files)
                    entry["files"] = files
                    entry["file_count"] = len(files)
                    entry["needs_gap_audit"] = status["status"] == "complete_with_errors"
                except Exception as exc:  # keep other months moving overnight
                    entry["download_error"] = f"{type(exc).__name__}: {exc}"
            if entry.get("downloaded"):
                completed += 1
            elif status["status"] in {"running", "accepted", "paused"}:
                active += 1
            elif status["status"] in {"failed", "canceled", "complete_with_errors"}:
                retry_count = int(entry.get("retry_count", 0))
                if retry_count < 2:
                    year, month = map(int, key.split("-"))
                    previous_jobs = list(entry.get("previous_jobs", [])) + [entry["job_id"]]
                    job_id = client.submit(request_for(year, month))
                    state[key] = {
                        "job_id": job_id,
                        "status": "resubmitted",
                        "downloaded": False,
                        "retry_count": retry_count + 1,
                        "previous_jobs": previous_jobs,
                    }
                    active += 1
                    print(f"SMAP resubmitted {key} retry={retry_count + 1} job={job_id}", flush=True)
                else:
                    failed += 1
        save(state)

        while active < ACTIVE_LIMIT:
            missing = next((key for key in wanted if key not in state), None)
            if missing is None:
                break
            year, month = map(int, missing.split("-"))
            job_id = client.submit(request_for(year, month))
            state[missing] = {"job_id": job_id, "status": "submitted", "downloaded": False}
            save(state)
            active += 1
            print(f"SMAP submitted {missing} job={job_id}", flush=True)

        print(f"SMAP completed={completed}/{len(wanted)} active={active} failed={failed}", flush=True)
        if completed + failed == len(wanted):
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
