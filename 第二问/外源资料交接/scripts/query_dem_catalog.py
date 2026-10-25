#!/usr/bin/env python3
"""Query official CDSE metadata for the GLO-30 N28E085 tile."""

from __future__ import annotations

import csv
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


DEST = Path(__file__).resolve().parents[1] / "P0-6_Copernicus_DEM_GLO-30" / "catalog"
BASE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"


def main() -> None:
    bbox = "POLYGON((85.19 28.25,85.69 28.25,85.69 28.56,85.19 28.56,85.19 28.25))"
    product_filter = (
        f"OData.CSC.Intersects(area=geography'SRID=4326;{bbox}') and "
        "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
        "and att/OData.CSC.StringAttribute/Value eq 'SAR_DGE_30_A4AD')"
    )
    query = urllib.parse.urlencode({"$filter": product_filter, "$top": 100, "$expand": "Attributes"})
    request = urllib.request.Request(f"{BASE}?{query}", headers={"User-Agent": "Codex-P0-456/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        products = json.load(response)["value"]
    rows = []
    for product in products:
        attrs = {item["Name"]: item.get("Value") for item in product.get("Attributes", [])}
        if attrs.get("gridId") != "N28_E085":
            continue
        rows.append(
            {
                "product_id": product["Id"],
                "name": product["Name"],
                "dataset_version": attrs.get("dataset", ""),
                "grid_id": attrs.get("gridId", ""),
                "product_type": attrs.get("productType", ""),
                "time_start": attrs.get("beginningDateTime", ""),
                "time_end": attrs.get("endingDateTime", ""),
                "content_length_bytes": product.get("ContentLength", ""),
                "online": product.get("Online", ""),
                "s3_path": product.get("S3Path", ""),
                "odata_url": f"{BASE}({product['Id']})/$value",
                "footprint_geojson": json.dumps(product.get("GeoFootprint"), separators=(",", ":")),
            }
        )
    rows.sort(key=lambda item: item["dataset_version"])
    DEST.mkdir(parents=True, exist_ok=True)
    with (DEST / "cop_dem_glo30_cdse_catalog.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "query_time_utc": datetime.now(timezone.utc).isoformat(),
        "tile": "N28_E085",
        "candidate_count": len(rows),
        "latest_catalog_entry": rows[-1] if rows else None,
        "downloaded_public_cog_tile": "Copernicus_DSM_COG_10_N28_00_E085_00_DEM",
        "note": "The public COG distribution was downloaded; CDSE native packages require authentication.",
    }
    (DEST / "cop_dem_catalog_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
