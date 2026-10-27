from __future__ import annotations

"""水文连通性、土地覆盖和冰湖邻近性补充实验。"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import shapefile
from rasterio.windows import from_bounds


MODEL_DIR = Path(__file__).resolve().parent.parent
OUTPUTS = MODEL_DIR / "outputs"
EXTERNAL = Path(r"D:\MathorCup_Data\External_Gyirong\Q2_core_completion")
COMPUTER3 = Path(r"D:\cs\WeChat Files\wxid_moov9hgwckde22\FileStorage\File\2026-10\电脑3外源资料")
CONFIG = json.loads((MODEL_DIR / "config" / "model_config.json").read_text(encoding="utf-8"))
AOI = CONFIG["source_aoi_wgs84"]
POINT = CONFIG["source_reference_point_wgs84"]


def read_aoi(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        win = from_bounds(AOI["west"], AOI["south"], AOI["east"], AOI["north"], ds.transform).round_offsets().round_lengths()
        x = ds.read(window=win).astype(float)
        for i, nodata in enumerate(ds.nodatavals):
            if nodata is not None:
                x[i][x[i] == nodata] = np.nan
        return x


def haversine_km(lon1: float, lat1: float, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def glacier_lakes() -> pd.DataFrame:
    folder = EXTERNAL / "P1_Glacier_Lake_Inventories" / "17948783"
    rows = []
    for path in sorted(folder.glob("*.shp")):
        reader = shapefile.Reader(str(path))
        fields = [f[0] for f in reader.fields[1:]]
        records = pd.DataFrame([list(r) for r in reader.records()], columns=fields)
        lon = pd.to_numeric(records["Longitude"], errors="coerce").to_numpy()
        lat = pd.to_numeric(records["Latitude"], errors="coerce").to_numpy()
        area = pd.to_numeric(records["Area"], errors="coerce").to_numpy()
        dist = haversine_km(POINT["longitude"], POINT["latitude"], lon, lat)
        good = np.isfinite(dist)
        nearest = int(np.nanargmin(dist))
        rows.append(
            {
                "inventory": path.stem,
                "n_global": int(good.sum()),
                "nearest_distance_km": float(dist[nearest]),
                "nearest_lake_area_original_unit": float(area[nearest]),
                "nearest_lake_elevation_m": float(pd.to_numeric(records.iloc[nearest]["Lake_Elev"], errors="coerce")),
                "count_within_5km": int(np.sum(dist <= 5)),
                "count_within_10km": int(np.sum(dist <= 10)),
                "count_within_20km": int(np.sum(dist <= 20)),
                "total_area_within_10km_original_unit": float(np.nansum(area[dist <= 10])),
            }
        )
    return pd.DataFrame(rows)


def merit_jrc() -> pd.DataFrame:
    merit_path = EXTERNAL / "P1_MERIT_Hydro" / "MERIT_Hydro_v1_0_1_AOI_90m.tif"
    jrc_path = EXTERNAL / "P1_JRC_Global_Surface_Water" / "JRC_GSW1_4_AOI_60m.tif"
    m = read_aoi(merit_path)
    j = read_aoi(jrc_path)
    merit_names = ["elevation_m", "flow_direction_code", "upstream_area_km2", "channel_width_m", "water_fraction", "height_above_nearest_drainage_m"]
    rows = []
    for name, x in zip(merit_names, m):
        rows.append({"dataset": "MERIT_Hydro", "indicator": name, "source_aoi_mean": float(np.nanmean(x)), "source_aoi_median": float(np.nanmedian(x)), "nonzero_fraction": float(np.nanmean(x > 0))})
    jrc_names = ["occurrence_percent", "change_abs", "seasonality_months", "recurrence_percent", "transition", "max_extent"]
    for name, x in zip(jrc_names, j):
        rows.append({"dataset": "JRC_GSW", "indicator": name, "source_aoi_mean": float(np.nanmean(x)), "source_aoi_median": float(np.nanmedian(x)), "nonzero_fraction": float(np.nanmean(x > 0))})
    return pd.DataFrame(rows)


def worldcover() -> pd.DataFrame:
    folder = COMPUTER3 / "data" / "worldcover"
    paths = sorted(folder.glob("ESA_WorldCover_10m_*_N27E084_Map.tif"))
    class_names = {10: "tree", 20: "shrub", 30: "grass", 40: "cropland", 50: "built", 60: "bare_sparse", 70: "snow_ice", 80: "permanent_water", 90: "wetland", 95: "mangrove", 100: "moss_lichen"}
    rows = []
    for path in paths:
        year_match = next((y for y in [2020, 2021] if str(y) in path.name), None)
        x = read_aoi(path)[0]
        valid = np.isfinite(x) & (x > 0)
        total = int(valid.sum())
        for code, name in class_names.items():
            rows.append({"year": year_match, "class_code": code, "class_name": name, "pixel_count": int(np.sum(valid & (x == code))), "source_aoi_fraction": float(np.sum(valid & (x == code)) / total) if total else np.nan})
    return pd.DataFrame(rows)


def md_table(frame: pd.DataFrame) -> str:
    x = frame.copy()
    for col in x.select_dtypes(include=["float"]).columns:
        x[col] = x[col].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    lines = ["| " + " | ".join(x.columns) + " |", "|" + "|".join(["---"] * len(x.columns)) + "|"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in x.itertuples(index=False, name=None))
    return "\n".join(lines)


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    lakes = glacier_lakes()
    hydro = merit_jrc()
    cover = worldcover()
    lakes.to_csv(OUTPUTS / "43_更新冰湖源区邻近性.csv", index=False, encoding="utf-8-sig")
    hydro.to_csv(OUTPUTS / "44_MERIT_JRC源区水文核查.csv", index=False, encoding="utf-8-sig")
    cover.to_csv(OUTPUTS / "45_WorldCover源区土地覆盖.csv", index=False, encoding="utf-8-sig")
    report = f"""# 第二问静态背景补充实验

## 冰湖邻近性

{md_table(lakes)}

这些指标只表示源区周边的冰湖背景和可能的物源/水源条件，不能仅凭距离判定某一冰湖是本次灾害触发源。

## MERIT Hydro 与 JRC 水体

{md_table(hydro)}

MERIT 用作对附件 DEM 流域连通性的独立旁证；JRC 是长期地表水产品，无水不等于事件时不存在短时融水或临时堰塞水体。

## ESA WorldCover

{md_table(cover[cover.source_aoi_fraction > 0])}

土地覆盖只进入空间背景解释，不作为小时级动态触发量。
"""
    (OUTPUTS / "46_第二问静态背景补充实验报告.md").write_text(report, encoding="utf-8")
    print(lakes.to_string(index=False))
    print(hydro.to_string(index=False))
    print(cover[cover.source_aoi_fraction > 0].to_string(index=False))


if __name__ == "__main__":
    main()
