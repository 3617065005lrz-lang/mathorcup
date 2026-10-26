from __future__ import annotations

"""Extract the nearest 9 km SMAP L4 source-area soil-moisture cell."""

import json
from pathlib import Path
import re

import h5py
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent
CONFIG = json.loads((MODEL_DIR / "config" / "model_config.json").read_text(encoding="utf-8"))
PROCESSED = MODEL_DIR / "processed"
OUTPUTS = MODEL_DIR / "outputs"

SMAP_ROOT = Path(r"D:\MathorCup_Data\External_Gyirong\P1-8_SMAP_L4_V8")
TARGET_YEARS = set(CONFIG["background_years"] + [pd.Timestamp(CONFIG["event_time_beijing"]).year])
DIR_PATTERN = re.compile(r"(?P<year>\d{4})-(?P<month>\d{2})$")
TIME_PATTERN = re.compile(r"_(?P<stamp>\d{8}T\d{6})_")
VARIABLES = [
    "sm_surface",
    "sm_surface_wetness",
    "sm_rootzone",
    "sm_rootzone_wetness",
    "sm_rootzone_pctl",
    "soil_temp_layer1",
]


def nearest_cell(lat: np.ndarray, lon: np.ndarray, target_lat: float, target_lon: float) -> tuple[int, int]:
    distance2 = (lat - target_lat) ** 2 + ((lon - target_lon) * np.cos(np.deg2rad(target_lat))) ** 2
    return tuple(int(v) for v in np.unravel_index(np.nanargmin(distance2), distance2.shape))


def parse_time(path: Path) -> pd.Timestamp:
    match = TIME_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"SMAP timestamp missing from {path.name}")
    # Product filenames are UTC; all model timestamps are naive Beijing time.
    return pd.to_datetime(match["stamp"], format="%Y%m%dT%H%M%S") + pd.Timedelta(hours=8)


def extract(path: Path) -> dict[str, object]:
    ref = CONFIG["source_reference_point_wgs84"]
    with h5py.File(path, "r") as h5:
        lat = np.asarray(h5["cell_lat"][:], dtype=float)
        lon = np.asarray(h5["cell_lon"][:], dtype=float)
        row, col = nearest_cell(lat, lon, ref["latitude"], ref["longitude"])
        row0, row1 = max(0, row - 1), min(lat.shape[0], row + 2)
        col0, col1 = max(0, col - 1), min(lat.shape[1], col + 2)
        result: dict[str, object] = {
            "timestamp_beijing": parse_time(path),
            "smap_cell_lat": float(lat[row, col]),
            "smap_cell_lon": float(lon[row, col]),
            "source_file": path.name,
        }
        for name in VARIABLES:
            data = np.asarray(h5[f"Geophysical_Data/{name}"][:], dtype=float)
            valid = np.where(data <= -9000, np.nan, data)
            result[f"smap_{name}"] = float(valid[row, col])
            result[f"smap_{name}_3x3_mean"] = float(np.nanmean(valid[row0:row1, col0:col1]))
        return result


def main() -> None:
    paths: list[Path] = []
    for directory in sorted(SMAP_ROOT.iterdir()):
        match = DIR_PATTERN.fullmatch(directory.name) if directory.is_dir() else None
        if match and int(match["year"]) in TARGET_YEARS:
            paths.extend(sorted(directory.glob("*.nc4")))
    if not paths:
        raise FileNotFoundError(f"No SMAP files found under {SMAP_ROOT}")

    rows = []
    for index, path in enumerate(paths, start=1):
        if index == 1 or index % 500 == 0 or index == len(paths):
            print(f"[{index}/{len(paths)}] {path.name}", flush=True)
        rows.append(extract(path))
    smap = pd.DataFrame(rows).sort_values("timestamp_beijing")
    # Duplicates can occur at month boundaries; their values should be identical.
    smap = smap.drop_duplicates("timestamp_beijing", keep="first").set_index("timestamp_beijing")
    smap["year"] = smap.index.year
    smap.index.name = "timestamp_beijing"
    smap.to_parquet(PROCESSED / "04_SMAP源区三小时土壤水分缓存.parquet")

    annual = (
        smap.groupby("year", as_index=False)
        .agg(
            records=("smap_sm_surface", "size"),
            valid_surface=("smap_sm_surface", "count"),
            valid_rootzone=("smap_sm_rootzone", "count"),
            surface_mean=("smap_sm_surface", "mean"),
            rootzone_mean=("smap_sm_rootzone", "mean"),
        )
    )
    annual.to_csv(OUTPUTS / "10_SMAP年度质量汇总.csv", index=False, encoding="utf-8-sig")
    report = f"""# SMAP 源区土壤水分缓存报告

- 产品：SMAP L4 9 km、3 小时土壤状态。
- 目标年份：{sorted(TARGET_YEARS)}；2025 年排除。
- 原始文件：{len(paths)}；去重记录：{len(smap)}。
- 最近网格中心：({smap.iloc[0]['smap_cell_lon']:.4f}, {smap.iloc[0]['smap_cell_lat']:.4f})。
- 主变量：0–5 cm 表层土壤水分、0–100 cm 根区土壤水分及其湿润度/百分位。
- 9 km 分辨率不能描述沟道局地含水量，仅用于独立验证区域湿润背景；ERA5-Land 土壤水分仍保留为高时间分辨率对照。
"""
    (OUTPUTS / "11_SMAP源区缓存报告.md").write_text(report, encoding="utf-8")
    print(annual.to_string(index=False))


if __name__ == "__main__":
    main()
