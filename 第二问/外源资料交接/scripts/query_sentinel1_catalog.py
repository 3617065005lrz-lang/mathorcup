#!/usr/bin/env python3
"""Query the official Copernicus Data Space catalogue for Sentinel-1 candidates.

The script is intentionally metadata-only. Sentinel-1 product downloads require a
CDSE account token; credentials are never read or stored here.
"""

from __future__ import annotations

import csv
import json
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


DEST = Path(__file__).resolve().parents[1] / "P0-4_Sentinel-1" / "catalog"
BASE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
START = "2026-06-01T00:00:00.000Z"
END = "2026-09-15T23:59:59.999Z"
BBOX = (85.19, 28.25, 85.69, 28.56)
EVENT = datetime(2026, 8, 26, tzinfo=timezone.utc)


def fetch(product_type: str) -> list[dict]:
    west, south, east, north = BBOX
    polygon = (
        f"POLYGON(({west} {south},{east} {south},{east} {north},"
        f"{west} {north},{west} {south}))"
    )
    product_filter = (
        "Collection/Name eq 'SENTINEL-1' and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{polygon}') and "
        f"ContentDate/Start ge {START} and ContentDate/Start le {END} and "
        "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq "
        f"'productType' and att/OData.CSC.StringAttribute/Value eq '{product_type}')"
    )
    query = urllib.parse.urlencode(
        {
            "$filter": product_filter,
            "$top": 1000,
            "$orderby": "ContentDate/Start asc",
            "$expand": "Attributes",
        }
    )
    request = urllib.request.Request(
        f"{BASE}?{query}", headers={"User-Agent": "Codex-P0-456/1.0"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    return payload["value"]


def fetch_precise_orbits() -> list[dict]:
    orbit_filter = (
        "Collection/Name eq 'SENTINEL-1' and "
        "ContentDate/Start ge 2026-05-15T00:00:00.000Z and "
        "ContentDate/Start le 2026-10-25T23:59:59.999Z and "
        "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq "
        "'productType' and att/OData.CSC.StringAttribute/Value eq 'AUX_POEORB')"
    )
    query = urllib.parse.urlencode(
        {"$filter": orbit_filter, "$top": 1000, "$orderby": "ContentDate/Start asc", "$expand": "Attributes"}
    )
    request = urllib.request.Request(
        f"{BASE}?{query}", headers={"User-Agent": "Codex-P0-456/1.0"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)["value"]


def flatten_orbits(products: list[dict]) -> list[dict]:
    import re

    rows = []
    pattern = re.compile(r"^(S1[A-Z]).*_V(\d{8}T\d{6})_(\d{8}T\d{6})\.EOF$")
    for product in products:
        match = pattern.match(product["Name"])
        if not match:
            continue
        rows.append(
            {
                "product_id": product["Id"],
                "name": product["Name"],
                "platform": match.group(1),
                "validity_start_utc": datetime.strptime(match.group(2), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc).isoformat(),
                "validity_end_utc": datetime.strptime(match.group(3), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc).isoformat(),
                "publication_time_utc": product["ContentDate"]["Start"],
                "size_bytes": product.get("ContentLength", ""),
                "online": product.get("Online", ""),
                "s3_path": product.get("S3Path", ""),
                "odata_url": f"{BASE}({product['Id']})/$value",
            }
        )
    return rows


def annotate_orbit_availability(rows: list[dict], orbits: list[dict]) -> None:
    for row in rows:
        platform = row["name"][:3]
        acquisition = datetime.fromisoformat(row["acquisition_start_utc"].replace("Z", "+00:00"))
        matches = []
        for orbit in orbits:
            start = datetime.fromisoformat(orbit["validity_start_utc"])
            end = datetime.fromisoformat(orbit["validity_end_utc"])
            if orbit["platform"] == platform and start <= acquisition <= end:
                matches.append(orbit["name"])
        row["precise_orbit_available"] = bool(matches)
        row["precise_orbit_file"] = ";".join(matches)


def flatten(product: dict) -> dict:
    attrs = {item["Name"]: item.get("Value") for item in product.get("Attributes", [])}
    footprint = product.get("GeoFootprint")
    start = product["ContentDate"]["Start"]
    date = datetime.fromisoformat(start.replace("Z", "+00:00"))
    phase = "pre_event" if date < EVENT else "post_event_validation_only"
    return {
        "product_id": product["Id"],
        "name": product["Name"],
        "acquisition_start_utc": start,
        "acquisition_end_utc": product["ContentDate"]["End"],
        "product_type": attrs.get("productType", ""),
        "processing_level": attrs.get("processingLevel", ""),
        "operational_mode": attrs.get("operationalMode", ""),
        "relative_orbit": attrs.get("relativeOrbitNumber", ""),
        "absolute_orbit": attrs.get("orbitNumber", ""),
        "orbit_direction": attrs.get("orbitDirection", ""),
        "polarization": attrs.get("polarisationChannels", ""),
        "swath": attrs.get("swathIdentifier", ""),
        "slice_number": attrs.get("sliceNumber", ""),
        "processor_version": attrs.get("processorVersion", ""),
        "online": product.get("Online", ""),
        "size_bytes": product.get("ContentLength", ""),
        "size_gib": round(product.get("ContentLength", 0) / 1024**3, 3),
        "s3_path": product.get("S3Path", ""),
        "odata_url": f"{BASE}({product['Id']})/$value",
        "phase": phase,
        "aoi_bbox": ",".join(map(str, BBOX)),
        "footprint_geojson": json.dumps(footprint, ensure_ascii=False, separators=(",", ":")),
    }


def pair_candidates(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if row["product_type"] != "IW_SLC__1S":
            continue
        key = (
            row["relative_orbit"],
            row["orbit_direction"],
            row["polarization"],
            row["swath"],
            row["slice_number"],
        )
        groups[key].append(row)

    pairs = []
    for key, items in groups.items():
        items.sort(key=lambda x: x["acquisition_start_utc"])
        dates = [datetime.fromisoformat(x["acquisition_start_utc"].replace("Z", "+00:00")) for x in items]
        before = [i for i, dt in enumerate(dates) if dt < EVENT]
        after = [i for i, dt in enumerate(dates) if dt >= EVENT]
        selected: list[tuple[str, int, int]] = []
        if len(before) >= 2:
            selected.append(("pre_event_continuous", before[-2], before[-1]))
        if before and after:
            selected.append(("cross_event", before[-1], after[0]))
        if len(after) >= 2:
            selected.append(("post_event_validation", after[0], after[1]))
        for kind, left, right in selected:
            a, b = items[left], items[right]
            dt_a = dates[left]
            dt_b = dates[right]
            pairs.append(
                {
                    "pair_role": kind,
                    "master_product_id": a["product_id"],
                    "master_name": a["name"],
                    "master_time_utc": a["acquisition_start_utc"],
                    "slave_product_id": b["product_id"],
                    "slave_name": b["name"],
                    "slave_time_utc": b["acquisition_start_utc"],
                    "temporal_baseline_days": round((dt_b - dt_a).total_seconds() / 86400, 3),
                    "relative_orbit": key[0],
                    "orbit_direction": key[1],
                    "polarization": key[2],
                    "swath": key[3],
                    "slice_number": key[4],
                    "pair_size_gib": round(a["size_gib"] + b["size_gib"], 3),
                    "download_recommendation": "catalog_only_over_10_GiB",
                    "post_event_validation_only": kind in {"cross_event", "post_event_validation"},
                }
            )
    return pairs


def delivery_state(row: dict) -> tuple[str, str]:
    name = row['name'] + '.zip'
    if (DEST.parent / 'GRD' / name).exists():
        return 'downloaded', 'local_GRD_directory'
    evidence = DEST / 'sentinel1_grd_verified.csv'
    if evidence.exists():
        with evidence.open(encoding='utf-8-sig', newline='') as handle:
            for verified in csv.DictReader(handle):
                if verified['file'] == name and verified.get('ok', '').lower() == 'true':
                    return 'downloaded', 'historically_verified_external_parts_check_with_verify_delivery'
    return 'requires_login', 'not_downloaded'


def grd_first_batch(rows: list[dict]) -> list[dict]:
    recommendations = []
    targets = [
        (121, "DESCENDING", "balanced nearest cross-event pair"),
        (85, "ASCENDING", "ascending-view pair with the earliest post-event acquisition"),
    ]
    for relative_orbit, direction, rationale in targets:
        candidates = [
            row
            for row in rows
            if row["product_type"] == "IW_GRDH_1S"
            and int(row["relative_orbit"]) == relative_orbit
            and row["orbit_direction"] == direction
        ]
        candidates.sort(key=lambda item: item["acquisition_start_utc"])
        pre = [r for r in candidates if datetime.fromisoformat(r["acquisition_start_utc"].replace("Z", "+00:00")) < EVENT]
        post = [r for r in candidates if datetime.fromisoformat(r["acquisition_start_utc"].replace("Z", "+00:00")) >= EVENT]
        if not pre or not post:
            continue
        for role, row in (("pre_event", pre[-1]), ("post_event_validation_only", post[0])):
            recommendations.append(
                {
                    "pair_relative_orbit": relative_orbit,
                    "pair_orbit_direction": direction,
                    "scene_role": role,
                    "product_id": row["product_id"],
                    "name": row["name"],
                    "acquisition_start_utc": row["acquisition_start_utc"],
                    "size_gib": row["size_gib"],
                    "odata_url": row["odata_url"],
                    "precise_orbit_available": row["precise_orbit_available"],
                    "precise_orbit_file": row["precise_orbit_file"],
                    "status": delivery_state(row)[0],
                    "storage_location": delivery_state(row)[1],
                    "post_event_validation_only": role == "post_event_validation_only",
                    "rationale": rationale,
                }
            )
    return recommendations


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    raw = fetch("IW_GRDH_1S") + fetch("IW_SLC__1S")
    rows = [flatten(item) for item in raw]
    rows.sort(key=lambda x: (x["acquisition_start_utc"], x["product_type"], x["name"]))
    orbit_raw = fetch_precise_orbits()
    orbit_rows = flatten_orbits(orbit_raw)
    annotate_orbit_availability(rows, orbit_rows)
    pairs = pair_candidates(rows)
    grd_batch = grd_first_batch(rows)
    write_csv(DEST / "sentinel1_catalog.csv", rows)
    write_csv(DEST / "sentinel1_insar_candidate_pairs.csv", pairs)
    write_csv(DEST / "sentinel1_precise_orbit_catalog.csv", orbit_rows)
    write_csv(DEST / "sentinel1_grd_first_batch.csv", grd_batch)
    (DEST / "sentinel1_catalog_raw.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "query_time_utc": datetime.now(timezone.utc).isoformat(),
        "aoi_bbox": BBOX,
        "time_start": START,
        "time_end": END,
        "event_time_reference": "2026-08-26 (date only; source initiation time not assumed)",
        "product_counts": {
            "IW_GRDH_1S": sum(r["product_type"] == "IW_GRDH_1S" for r in rows),
            "IW_SLC__1S": sum(r["product_type"] == "IW_SLC__1S" for r in rows),
        },
        "total_catalog_size_gib": round(sum(r["size_gib"] for r in rows), 3),
        "candidate_pair_count": len(pairs),
        "precise_orbit_file_count": len(orbit_rows),
        "all_scene_precise_orbits_available": all(r["precise_orbit_available"] for r in rows),
        "recommended_grd_first_batch_scene_count": len(grd_batch),
        "recommended_grd_first_batch_size_gib": round(sum(float(r["size_gib"]) for r in grd_batch), 3),
        "recommended_grd_first_batch_status": "downloaded" if all(r['status'] == 'downloaded' for r in grd_batch) else "partial",
        "download_note": "SLC pairs are catalog-only because a minimum pair exceeds 10 GiB.",
    }
    (DEST / "sentinel1_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
