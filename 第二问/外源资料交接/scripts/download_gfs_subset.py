#!/usr/bin/env python3
"""Download and crop the required NOAA NCEI GFS historical forecasts.

The script never downloads a complete global GRIB2. It reads the official
inventory, requests only the byte ranges for selected GRIB messages, decodes
those messages with ecCodes, and writes one compact NetCDF file per model cycle.

Runtime dependencies:
    pip install eccodes netCDF4 numpy
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path


WORK_DEPS = Path(__file__).resolve().parents[1] / "python_deps"
if WORK_DEPS.exists():
    sys.path.insert(0, str(WORK_DEPS))

import numpy as np
from eccodes import (
    codes_get,
    codes_get_array,
    codes_new_from_message,
    codes_release,
)
from netCDF4 import Dataset, date2num


ROOT = Path(__file__).resolve().parents[1] / "P0-5_GFS"
DATA_DIR = ROOT / "data"
CATALOG_DIR = ROOT / "catalog"
LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
STAGING_DIR = Path(__file__).resolve().parents[1] / ".work" / "gfs_staging"
INVENTORY_CSV = CATALOG_DIR / "gfs_official_inventory.csv"

REQUESTED_AOI = (85.1, 27.9, 85.9, 28.7)
# Half-cell padded selection, then constrained to actual 0.5-degree grid centres.
GRID_BOUNDS = (84.75, 27.65, 86.25, 28.95)
PRESSURE_LEVELS = (500, 600, 700, 850)
SOIL_LAYERS = ("0-0.1", "0.1-0.4", "0.4-1", "1-2")
SOIL_TOP = np.array([0.0, 0.1, 0.4, 1.0], dtype="f4")
SOIL_BOTTOM = np.array([0.1, 0.4, 1.0, 2.0], dtype="f4")
USER_AGENT = "Codex-P0-456-GFS-Subset/1.0"


def url_read(url: str, byte_range: tuple[int, int] | None = None, retries: int = 6) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    last_error = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                if byte_range is not None:
                    start, end = byte_range
                    if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
                        raise RuntimeError("Server did not honor the requested GRIB byte range")
                    payload = response.read(end - start + 2)
                    if len(payload) != end - start + 1:
                        raise RuntimeError("Unexpected byte-range response length")
                    return payload
                return response.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            last_error = error
            time.sleep(min(20, 1.5**attempt))
    raise RuntimeError(f"Failed after {retries} attempts: {url}") from last_error


def parse_inventory(text: str, grib_size: int) -> list[dict]:
    records = []
    for line in text.splitlines():
        parts = line.split(":")
        if len(parts) < 6:
            continue
        records.append(
            {
                "record": int(parts[0]),
                "start": int(parts[1]),
                "reference": parts[2],
                "short": parts[3],
                "level": parts[4],
                "description": ":".join(parts[5:]).strip(),
                "inventory_line": line,
            }
        )
    for index, record in enumerate(records):
        record["end"] = (
            records[index + 1]["start"] - 1 if index + 1 < len(records) else grib_size - 1
        )
    return records


def desired_key(record: dict) -> str | None:
    short = record["short"]
    level = record["level"]
    if short == "PRES" and level == "surface":
        return "surface_pressure"
    if short == "TMP" and level == "2 m above ground":
        return "temperature_2m"
    if short == "DPT" and level == "2 m above ground":
        return "dewpoint_temperature_2m"
    if short == "RH" and level == "2 m above ground":
        return "relative_humidity_2m"
    if short == "UGRD" and level == "10 m above ground":
        return "u_component_of_wind_10m"
    if short == "VGRD" and level == "10 m above ground":
        return "v_component_of_wind_10m"
    if short == "APCP" and level == "surface":
        return "total_precipitation_accumulation"
    if short == "DSWRF" and level == "surface":
        return "downward_shortwave_radiation_flux"
    soil_match = re.fullmatch(r"([0-9.]+-[0-9.]+) m below ground", level)
    if soil_match and soil_match.group(1) in SOIL_LAYERS:
        if short == "TSOIL":
            return f"soil_temperature|{soil_match.group(1)}"
        if short == "SOILW":
            return f"volumetric_soil_moisture|{soil_match.group(1)}"
    pressure_match = re.fullmatch(r"(\d+) mb", level)
    if pressure_match and int(pressure_match.group(1)) in PRESSURE_LEVELS:
        if short == "TMP":
            return f"air_temperature_isobaric|{pressure_match.group(1)}"
        if short == "HGT":
            return f"geopotential_height_isobaric|{pressure_match.group(1)}"
    return None


def choose_records(records: list[dict]) -> dict[str, dict]:
    chosen = {}
    for record in records:
        key = desired_key(record)
        # Some forecast inventories expose duplicate APCP records. Keep the first
        # occurrence consistently and retain the exact inventory line in provenance.
        if key is not None and key not in chosen:
            chosen[key] = record
    return chosen


def fetch_record(url: str, key: str, record: dict) -> tuple[str, dict, bytes]:
    message = url_read(url, (record["start"], record["end"]))
    if not message.startswith(b"GRIB") or not message.endswith(b"7777"):
        raise RuntimeError(f"Invalid GRIB message for {key}: {record['inventory_line']}")
    return key, record, message


def extract_values(message: bytes, grid_cache: dict) -> tuple[np.ndarray, dict]:
    gid = codes_new_from_message(message)
    try:
        ni = int(codes_get(gid, "Ni"))
        nj = int(codes_get(gid, "Nj"))
        grid_key = (ni, nj)
        if grid_key not in grid_cache:
            lats = np.asarray(codes_get_array(gid, "latitudes"), dtype="f8")
            lons = np.asarray(codes_get_array(gid, "longitudes"), dtype="f8")
            west, south, east, north = GRID_BOUNDS
            mask = (lons >= west) & (lons <= east) & (lats >= south) & (lats <= north)
            indices = np.flatnonzero(mask)
            selected_lats = np.unique(lats[indices])[::-1]
            selected_lons = np.unique(lons[indices])
            index_lookup = []
            for latitude in selected_lats:
                for longitude in selected_lons:
                    matches = indices[(lats[indices] == latitude) & (lons[indices] == longitude)]
                    if matches.size != 1:
                        raise RuntimeError("Unexpected GFS grid geometry in AOI")
                    index_lookup.append(int(matches[0]))
            grid_cache[grid_key] = {
                "indices": np.asarray(index_lookup, dtype="i8"),
                "latitudes": selected_lats.astype("f4"),
                "longitudes": selected_lons.astype("f4"),
            }
        grid = grid_cache[grid_key]
        values = np.asarray(codes_get_array(gid, "values"), dtype="f4")[grid["indices"]]
        values = values.reshape(len(grid["latitudes"]), len(grid["longitudes"]))
        metadata = {
            "short_name": str(codes_get(gid, "shortName")),
            "name": str(codes_get(gid, "name")),
            "units": str(codes_get(gid, "units")),
            "type_of_level": str(codes_get(gid, "typeOfLevel")),
            "level": float(codes_get(gid, "level")),
        }
        return values, metadata
    finally:
        codes_release(gid)


def init_arrays(n_time: int, n_lat: int, n_lon: int) -> dict[str, np.ndarray]:
    fill = np.float32(np.nan)
    arrays = {
        name: np.full((n_time, n_lat, n_lon), fill, dtype="f4")
        for name in (
            "surface_pressure",
            "temperature_2m",
            "dewpoint_temperature_2m",
            "relative_humidity_2m",
            "u_component_of_wind_10m",
            "v_component_of_wind_10m",
            "total_precipitation_accumulation",
            "downward_shortwave_radiation_flux",
        )
    }
    arrays["soil_temperature"] = np.full(
        (n_time, len(SOIL_LAYERS), n_lat, n_lon), fill, dtype="f4"
    )
    arrays["volumetric_soil_moisture"] = np.full_like(arrays["soil_temperature"], fill)
    arrays["air_temperature_isobaric"] = np.full(
        (n_time, len(PRESSURE_LEVELS), n_lat, n_lon), fill, dtype="f4"
    )
    arrays["geopotential_height_isobaric"] = np.full_like(
        arrays["air_temperature_isobaric"], fill
    )
    return arrays


def assign_array(arrays: dict, key: str, time_index: int, values: np.ndarray) -> None:
    if "|" not in key:
        arrays[key][time_index] = values
        return
    variable, qualifier = key.split("|", 1)
    if qualifier in SOIL_LAYERS:
        arrays[variable][time_index, SOIL_LAYERS.index(qualifier)] = values
    else:
        arrays[variable][time_index, PRESSURE_LEVELS.index(int(qualifier))] = values


def write_cycle(
    output: Path,
    cycle_rows: list[dict],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    arrays: dict[str, np.ndarray],
    provenance_rows: list[dict],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    temp = STAGING_DIR / (output.name + ".part")
    reference = datetime.strptime(
        cycle_rows[0]["forecast_reference_time_utc"], "%Y%m%dT%H:%M:%SZ"
    ).replace(tzinfo=timezone.utc)
    valid_times = [
        datetime.strptime(row["valid_time_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
        for row in cycle_rows
    ]
    nc = Dataset("regional.nc", "w", format="NETCDF4", memory=4 * 1024 * 1024)
    try:
        nc.title = "NOAA NCEI GFS grid-004 regional subset for Gyirong"
        nc.source = "NOAA National Centers for Environmental Information GFS archive"
        nc.source_url = cycle_rows[0]["grib_url"].rsplit("/", 1)[0] + "/"
        nc.history = f"Created {datetime.now(timezone.utc).isoformat()} by byte-range field extraction"
        nc.model_grid = "GFS grid 004, 0.5 degree"
        nc.requested_aoi = "west=85.1, south=27.9, east=85.9, north=28.7"
        nc.actual_grid_centres = (
            f"lon={longitudes.tolist()}, lat={latitudes.tolist()} (half-cell padded selection)"
        )
        nc.forecast_reference_time_utc = reference.isoformat()
        nc.processing_note = (
            "Only selected GRIB messages were transferred; global files were not downloaded. "
            "Post-event initialization cycles must not be used to predict earlier valid times."
        )
        nc.createDimension("forecast_period", len(cycle_rows))
        nc.createDimension("latitude", len(latitudes))
        nc.createDimension("longitude", len(longitudes))
        nc.createDimension("soil_layer", len(SOIL_LAYERS))
        nc.createDimension("pressure_level", len(PRESSURE_LEVELS))

        lat_var = nc.createVariable("latitude", "f4", ("latitude",))
        lat_var[:] = latitudes
        lat_var.units = "degrees_north"
        lon_var = nc.createVariable("longitude", "f4", ("longitude",))
        lon_var[:] = longitudes
        lon_var.units = "degrees_east"
        lead_var = nc.createVariable("forecast_period", "i4", ("forecast_period",))
        lead_var[:] = [int(row["lead_hours"]) for row in cycle_rows]
        lead_var.units = "hours"
        valid_var = nc.createVariable("valid_time", "i8", ("forecast_period",))
        valid_var.units = "seconds since 1970-01-01 00:00:00 UTC"
        valid_var.calendar = "standard"
        valid_var[:] = date2num(valid_times, valid_var.units, valid_var.calendar)
        ref_var = nc.createVariable("forecast_reference_time", "i8")
        ref_var.units = valid_var.units
        ref_var.calendar = "standard"
        ref_var.assignValue(date2num(reference, ref_var.units, ref_var.calendar))
        p_var = nc.createVariable("pressure_level", "i4", ("pressure_level",))
        p_var[:] = PRESSURE_LEVELS
        p_var.units = "hPa"
        p_var.positive = "down"
        top_var = nc.createVariable("soil_depth_top", "f4", ("soil_layer",))
        bottom_var = nc.createVariable("soil_depth_bottom", "f4", ("soil_layer",))
        top_var[:] = SOIL_TOP
        bottom_var[:] = SOIL_BOTTOM
        top_var.units = bottom_var.units = "m"
        top_var.positive = bottom_var.positive = "down"

        definitions = {
            "surface_pressure": ("surface air pressure", "Pa", ("forecast_period", "latitude", "longitude")),
            "temperature_2m": ("2 metre temperature", "K", ("forecast_period", "latitude", "longitude")),
            "dewpoint_temperature_2m": ("2 metre dewpoint temperature", "K", ("forecast_period", "latitude", "longitude")),
            "relative_humidity_2m": ("2 metre relative humidity", "%", ("forecast_period", "latitude", "longitude")),
            "u_component_of_wind_10m": ("10 metre U wind component", "m s-1", ("forecast_period", "latitude", "longitude")),
            "v_component_of_wind_10m": ("10 metre V wind component", "m s-1", ("forecast_period", "latitude", "longitude")),
            "total_precipitation_accumulation": ("accumulated precipitation over the forecast interval", "kg m-2", ("forecast_period", "latitude", "longitude")),
            "downward_shortwave_radiation_flux": ("downward short-wave radiation flux averaged over the forecast interval", "W m-2", ("forecast_period", "latitude", "longitude")),
            "soil_temperature": ("soil temperature", "K", ("forecast_period", "soil_layer", "latitude", "longitude")),
            "volumetric_soil_moisture": ("volumetric soil moisture content", "fraction", ("forecast_period", "soil_layer", "latitude", "longitude")),
            "air_temperature_isobaric": ("air temperature on isobaric levels", "K", ("forecast_period", "pressure_level", "latitude", "longitude")),
            "geopotential_height_isobaric": ("geopotential height on isobaric levels", "gpm", ("forecast_period", "pressure_level", "latitude", "longitude")),
        }
        for name, (long_name, units, dimensions) in definitions.items():
            variable = nc.createVariable(
                name, "f4", dimensions, zlib=True, complevel=4, shuffle=True, fill_value=np.float32(9.96921e36)
            )
            variable[:] = np.ma.masked_invalid(arrays[name])
            variable.long_name = long_name
            variable.units = units
            if name in {"total_precipitation_accumulation", "downward_shortwave_radiation_flux"}:
                variable.note = "Undefined at forecast lead +000; values begin at +003 h."
    finally:
        memory_image = nc.close()
    temp.write_bytes(memory_image)
    shutil.copy2(temp, output)
    temp.unlink()
    prov_path = output.with_name(output.stem + "_provenance.csv")
    with prov_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(provenance_rows[0].keys()))
        writer.writeheader()
        writer.writerows(provenance_rows)


def process_cycle(cycle_rows: list[dict], workers: int, force: bool) -> dict:
    cycle_key = datetime.strptime(
        cycle_rows[0]["forecast_reference_time_utc"], "%Y%m%dT%H:%M:%SZ"
    ).strftime("%Y%m%d%H")
    output = DATA_DIR / f"gfs_{cycle_key}_f000-f120_gyirong.nc"
    if output.exists() and not force:
        return {"cycle": cycle_key, "status": "skipped_existing", "output": str(output)}

    print(f"START {cycle_key} ({len(cycle_rows)} forecast leads)", flush=True)
    grid_cache = {}
    arrays = None
    latitudes = longitudes = None
    provenance_rows = []
    failures = []
    expected_keys = set()
    for name in (
        "surface_pressure",
        "temperature_2m",
        "dewpoint_temperature_2m",
        "relative_humidity_2m",
        "u_component_of_wind_10m",
        "v_component_of_wind_10m",
        "soil_temperature",
        "volumetric_soil_moisture",
        "air_temperature_isobaric",
        "geopotential_height_isobaric",
    ):
        expected_keys.add(name)

    for time_index, row in enumerate(cycle_rows):
        grib_size = int(row["full_grib_size_bytes"])
        inventory_text = url_read(row["inventory_url"]).decode("utf-8")
        records = parse_inventory(inventory_text, grib_size)
        chosen = choose_records(records)
        messages = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(fetch_record, row["grib_url"], key, record)
                for key, record in chosen.items()
            ]
            for future in as_completed(futures):
                try:
                    messages.append(future.result())
                except Exception as error:
                    failures.append(
                        {
                            "forecast_reference_time_utc": row["forecast_reference_time_utc"],
                            "lead_hours": row["lead_hours"],
                            "error": repr(error),
                        }
                    )
        for key, record, message in messages:
            values, metadata = extract_values(message, grid_cache)
            if latitudes is None:
                only_grid = next(iter(grid_cache.values()))
                latitudes = only_grid["latitudes"]
                longitudes = only_grid["longitudes"]
                arrays = init_arrays(len(cycle_rows), len(latitudes), len(longitudes))
            assign_array(arrays, key, time_index, values)
            provenance_rows.append(
                {
                    "forecast_reference_time_utc": row["forecast_reference_time_utc"],
                    "lead_hours": row["lead_hours"],
                    "valid_time_utc": row["valid_time_utc"],
                    "output_variable": key,
                    "source_grib_url": row["grib_url"],
                    "inventory_line": record["inventory_line"],
                    "byte_start": record["start"],
                    "byte_end": record["end"],
                    "bytes_transferred": record["end"] - record["start"] + 1,
                    "grib_name": metadata["name"],
                    "grib_units": metadata["units"],
                    "grib_level_type": metadata["type_of_level"],
                    "grib_level": metadata["level"],
                }
            )
        if time_index % 5 == 0 or time_index == len(cycle_rows) - 1:
            transferred = sum(int(item["bytes_transferred"]) for item in provenance_rows)
            print(
                f"PROGRESS {cycle_key} lead={int(row['lead_hours']):03d} "
                f"fields={len(messages)} transferred_MiB={transferred / 1024**2:.1f}",
                flush=True,
            )

    if arrays is None or latitudes is None or longitudes is None:
        raise RuntimeError(f"No GFS fields decoded for {cycle_key}")
    if failures:
        failure_path = LOG_DIR / f"gfs_{cycle_key}_failures.json"
        failure_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError(f"{len(failures)} field downloads failed for {cycle_key}")
    write_cycle(output, cycle_rows, latitudes, longitudes, arrays, provenance_rows)
    transferred = sum(int(item["bytes_transferred"]) for item in provenance_rows)
    result = {
        "cycle": cycle_key,
        "status": "downloaded",
        "output": str(output),
        "file_size_bytes": output.stat().st_size,
        "selected_grib_messages": len(provenance_rows),
        "source_bytes_transferred": transferred,
        "grid_latitudes": latitudes.tolist(),
        "grid_longitudes": longitudes.tolist(),
    }
    print("DONE " + json.dumps(result, ensure_ascii=False), flush=True)
    return result


def load_inventory() -> list[dict]:
    with INVENTORY_CSV.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if any(row["available"].lower() != "true" for row in rows):
        raise RuntimeError("GFS inventory contains unavailable target files")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--cycle-start", type=int, default=0)
    parser.add_argument("--max-cycles", type=int, default=0, help="0 means all cycles")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_inventory()
    by_cycle = {}
    for row in rows:
        by_cycle.setdefault(row["forecast_reference_time_utc"], []).append(row)
    cycles = []
    for key in sorted(by_cycle):
        cycle_rows = sorted(by_cycle[key], key=lambda item: int(item["lead_hours"]))
        cycles.append(cycle_rows)
    cycles = cycles[args.cycle_start :]
    if args.max_cycles:
        cycles = cycles[: args.max_cycles]
    results = []
    for cycle_rows in cycles:
        results.append(process_cycle(cycle_rows, args.workers, args.force))
    first_cycle = results[0]["cycle"] if results else "none"
    last_cycle = results[-1]["cycle"] if results else "none"
    summary_path = CATALOG_DIR / f"gfs_download_results_{first_cycle}_{last_cycle}.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
