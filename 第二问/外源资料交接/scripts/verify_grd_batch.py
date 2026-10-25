#!/usr/bin/env python
"""Verify the downloaded Sentinel-1 GRD SAFE archives and emit a catalog.

For every ``*.SAFE.zip`` in the GRD folder this script

* re-reads the whole archive so that every member CRC is checked,
* confirms the expected Sentinel-1 SAFE layout (``<name>.SAFE/manifest.safe``),
* extracts the acquisition parameters from ``manifest.safe`` (mission, product
  type, mode, class, polarisation, start/stop time, absolute and relative orbit,
  orbit direction, processing baseline),
* compares the byte count against the CDSE catalogue ``ContentLength``,
* records MD5 and SHA-256.

Outputs
-------
``catalog/sentinel1_grd_verified.csv``  one row per scene
``catalog/sentinel1_grd_verified.json`` summary + per-scene detail
"""

import csv
import hashlib
import json
import os
import re
import ssl
import sys
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRD_DIR = os.path.join(ROOT, "P0-4_Sentinel-1", "GRD")
CAT_DIR = os.path.join(ROOT, "P0-4_Sentinel-1", "catalog")
BATCH_CSV = os.path.join(CAT_DIR, "sentinel1_grd_first_batch.csv")
OUT_CSV = os.path.join(CAT_DIR, "sentinel1_grd_verified.csv")
OUT_JSON = os.path.join(CAT_DIR, "sentinel1_grd_verified.json")

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) sentinel1-grd-verify"


def digest(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def catalogue_length(pid, ctx):
    url = ("https://catalogue.dataspace.copernicus.eu/odata/v1/Products(%s)"
           "?$select=Name,ContentLength,Online,ContentDate" % pid)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
        return json.loads(r.read().decode("utf-8"))


def manifest_fields(blob):
    """Pull the key acquisition parameters out of the SAFE manifest.safe XML.

    SAFE uses several XML namespaces (safe:, s1:, s1sarl1:), so the parser
    strips namespace URIs and keeps the first occurrence of each local name;
    that ordering matches the manifest (start before stop, acquisition period
    before downlink period).
    """
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(blob)
    except Exception:
        return {}

    flat = {}
    polarisations = []
    for el in root.iter():
        name = el.tag.rsplit("}", 1)[-1]
        text = (el.text or "").strip()
        if not text or not name:
            continue
        if name == "transmitterReceiverPolarisation":
            if text not in polarisations:
                polarisations.append(text)
        else:
            flat.setdefault(name, text)

    return {
        "mission": flat.get("missionId", ""),
        "product_type": flat.get("productType", ""),
        "mode": flat.get("mode", "") or flat.get("sensorMode", ""),
        "product_class": flat.get("productClass", ""),
        "polarisation": "+".join(polarisations),
        "orbit_direction": flat.get("pass", ""),
        "relative_orbit": flat.get("relativeOrbitNumber", ""),
        "absolute_orbit": flat.get("orbitNumber", ""),
        "start_time": flat.get("startTime", ""),
        "stop_time": flat.get("stopTime", ""),
        "processing_level": flat.get("processingLevel", ""),
        "processor_version": flat.get("softwareVersion", ""),
    }


def verify_scene(path, expected, ctx):
    name = os.path.basename(path)
    rec = {"file": name, "sha256": digest(path, "sha256"), "md5": digest(path, "md5")}
    rec["bytes"] = os.path.getsize(path)
    rec["catalogue_bytes"] = expected
    rec["size_match"] = bool(expected) and rec["bytes"] == expected
    problems = []

    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()          # reads every member, checks CRC
            members = zf.namelist()
            rec["member_count"] = len(members)
            rec["crc_ok"] = bad is None
            if bad is not None:
                problems.append("bad CRC in %s" % bad)
            root = name[: -len(".zip")]
            manifest_member = next((m for m in members if m == root + "/manifest.safe"), None)
            if manifest_member is None:
                problems.append("manifest.safe missing under %s/" % root)
            else:
                rec.update(manifest_fields(zf.read(manifest_member)))
            rec["safe_root"] = root
            rec["has_measurement"] = sum(1 for m in members if "/measurement/" in m)
            rec["has_annotation"] = sum(1 for m in members if "/annotation/" in m)
    except Exception as exc:
        rec["crc_ok"] = False
        problems.append("%s: %s" % (type(exc).__name__, exc))

    if not rec["size_match"]:
        problems.append("size %d != catalogue %d" % (rec["bytes"], expected))
    if rec.get("product_type") and "GRD" not in rec["product_type"]:
        problems.append("unexpected product type %s" % rec["product_type"])
    rec["problems"] = problems
    rec["ok"] = not problems
    return rec


def main():
    ctx = ssl.create_default_context()
    with open(BATCH_CSV, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))

    scene_by_name = {}
    for row in rows:
        scene_by_name[row["name"].strip() + ".zip"] = row

    files = sorted(f for f in os.listdir(GRD_DIR) if f.endswith(".SAFE.zip"))
    records = []
    for fname in files:
        path = os.path.join(GRD_DIR, fname)
        row = scene_by_name.get(fname, {})
        pid = row.get("product_id", "").strip()
        expected = 0
        meta = {}
        if pid:
            meta = catalogue_length(pid, ctx)
            expected = int(meta.get("ContentLength") or 0)
        print("verifying", fname, flush=True)
        rec = verify_scene(path, expected, ctx)
        rec["product_id"] = pid
        rec["role"] = row.get("scene_role", "")
        rec["relative_orbit_planned"] = row.get("pair_relative_orbit", "")
        rec["orbit_direction_planned"] = row.get("pair_orbit_direction", "")
        rec["precise_orbit_available"] = row.get("precise_orbit_available", "")
        rec["online"] = meta.get("Online")
        records.append(rec)
        print("   ok=%s crc=%s members=%s type=%s rel_orbit=%s"
              % (rec["ok"], rec.get("crc_ok"), rec.get("member_count"),
                 rec.get("product_type"), rec.get("relative_orbit")), flush=True)

    cols = ["file", "product_id", "role", "bytes", "catalogue_bytes", "size_match",
            "member_count", "crc_ok", "has_measurement", "has_annotation",
            "product_type", "polarisation", "mode", "relative_orbit", "orbit_direction",
            "absolute_orbit", "start_time", "stop_time", "processing_level",
            "processor_version", "md5", "sha256", "problems", "ok"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            out = dict(rec)
            out["problems"] = "; ".join(rec["problems"])
            writer.writerow(out)

    summary = {
        "scenes": len(records),
        "all_ok": all(r["ok"] for r in records),
        "total_bytes": sum(r["bytes"] for r in records),
        "total_gib": round(sum(r["bytes"] for r in records) / 2 ** 30, 3),
        "files": records,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print("\nSUMMARY: %d scenes, all_ok=%s, %.3f GiB"
          % (summary["scenes"], summary["all_ok"], summary["total_gib"]))
    print("wrote", OUT_CSV)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()
