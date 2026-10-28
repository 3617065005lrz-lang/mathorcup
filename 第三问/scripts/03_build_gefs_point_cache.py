from __future__ import annotations

"""Extract a tiny, reusable source-point cache from the downloaded GEFS GRIB files."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


MODEL_DIR = Path(__file__).resolve().parents[1]
CONFIG = json.loads((MODEL_DIR / "config" / "model_config.json").read_text(encoding="utf-8"))
GRIB_ROOT = MODEL_DIR / "外源资料" / "02_GEFS集合预报"
CACHE = MODEL_DIR / "cache"
OUTPUTS = MODEL_DIR / "outputs"
CACHE.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

LON = float(CONFIG["source_point_wgs84"]["longitude"])
LAT = float(CONFIG["source_point_wgs84"]["latitude"])
NAME_RE = re.compile(r"(?P<product>geavg|gespr)_f(?P<lead>\d{3})_selected\.grib2$")
RUN_RE = re.compile(r"gefs_(?P<date>\d{8})_(?P<cycle>\d{2})z$")

VARIABLES = {
    "TMP": "temperature_2m_c",
    "SOILW": "soil_water_0_10m",
    "APCP": "precipitation_6h_mm",
    "APCP06": "precipitation_6h_mm",
    "DSWRF": "shortwave_down_w_m2",
}


def extract_file(path: Path) -> list[dict[str, object]]:
    m = NAME_RE.search(path.name)
    r = RUN_RE.search(path.parent.name)
    if not m or not r:
        return []
    rows: list[dict[str, object]] = []
    with rasterio.open(path) as ds:
        sample = next(ds.sample([(LON, LAT)]))
        for band, value in enumerate(sample, 1):
            tags = ds.tags(band)
            element = tags.get("GRIB_ELEMENT", "")
            if element not in VARIABLES:
                continue
            valid_epoch = int(tags.get("GRIB_VALID_TIME", 0))
            valid_utc = pd.to_datetime(valid_epoch, unit="s", utc=True)
            rows.append(
                {
                    "date": r.group("date"),
                    "cycle": r.group("cycle"),
                    "product": m.group("product"),
                    "lead_h": int(m.group("lead")),
                    "valid_time_utc": valid_utc,
                    "valid_time_beijing": valid_utc.tz_convert("Asia/Shanghai").tz_localize(None),
                    "variable": VARIABLES[element],
                    "value": float(value),
                    "grib_element": element,
                    "unit": tags.get("GRIB_UNIT", ""),
                    "source_file": str(path),
                }
            )
    return rows


def main() -> None:
    records: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    for run_dir in sorted(GRIB_ROOT.glob("gefs_*_*z")):
        for path in sorted(run_dir.glob("*_selected.grib2")):
            try:
                rows = extract_file(path)
                records.extend(rows)
                audit.append({"file": str(path), "bytes": path.stat().st_size, "variables": len(rows), "status": "ok"})
            except Exception as exc:
                audit.append({"file": str(path), "bytes": path.stat().st_size, "variables": 0, "status": f"error: {exc}"})

    if not records:
        raise RuntimeError("No readable GEFS records were found")
    frame = pd.DataFrame(records).sort_values(["date", "cycle", "product", "lead_h", "variable"])
    frame.to_parquet(CACHE / "01_GEFS源区点预报缓存.parquet", index=False)
    pd.DataFrame(audit).to_csv(OUTPUTS / "01_GEFS文件质量审计.csv", index=False, encoding="utf-8-sig")

    expected = set(VARIABLES.values())
    summary = []
    for (date, cycle, product), group in frame.groupby(["date", "cycle", "product"]):
        present_leads = sorted(group.lead_h.unique())
        variables = set(group.variable.unique())
        summary.append(
            {
                "date": date,
                "cycle": cycle,
                "product": product,
                "lead_min_h": min(present_leads),
                "lead_max_h": max(present_leads),
                "lead_count": len(present_leads),
                "variables": ";".join(sorted(variables)),
                "all_four_variables": expected.issubset(variables),
                "covers_f174": max(present_leads) >= 174,
            }
        )
    summary_df = pd.DataFrame(summary).sort_values(["date", "cycle", "product"])
    summary_df.to_csv(OUTPUTS / "02_GEFS起报完整性审计.csv", index=False, encoding="utf-8-sig")
    print(f"Saved {len(frame):,} source-point records from {len(audit):,} GRIB files")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
