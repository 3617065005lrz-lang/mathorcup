from __future__ import annotations

"""Audit historical glacial-lake context near the reconstructed source area."""

import json
import math
from pathlib import Path
import re

import pandas as pd
import shapefile


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent
CONFIG = json.loads((MODEL_DIR / "config" / "model_config.json").read_text(encoding="utf-8"))
OUTPUTS = MODEL_DIR / "outputs"
HMA_ROOT = Path(r"D:\MathorCup_Data\External_Gyirong\P1-6_HMA_GLI\data")
PERIOD = re.compile(r".*_(?P<period>\d{4}_\d{2}|\d{4}_\d{4})$")


def distance_km(lon: float, lat: float, lon0: float, lat0: float) -> float:
    return math.sqrt(
        ((lon - lon0) * 111.32 * math.cos(math.radians(lat0))) ** 2
        + ((lat - lat0) * 111.32) ** 2
    )


def main() -> None:
    ref = CONFIG["source_reference_point_wgs84"]
    rows = []
    nearest_rows = []
    for path in sorted(HMA_ROOT.glob("*.shp")):
        match = PERIOD.fullmatch(path.stem)
        period = match["period"].replace("_", "-") if match else path.stem
        reader = shapefile.Reader(str(path))
        local = []
        for record in reader.iterRecords(fields=["CENTROID_X", "CENTROID_Y", "Area_m2"]):
            lon, lat, area = float(record[0]), float(record[1]), float(record[2])
            if 84.0 < lon < 87.0 and 27.0 < lat < 30.0:
                local.append((distance_km(lon, lat, ref["longitude"], ref["latitude"]), lon, lat, area))
        local.sort()
        nearest = local[0]
        for radius in [5, 10, 20, 30]:
            selected = [item for item in local if item[0] <= radius]
            rows.append(
                {
                    "period": period,
                    "radius_km": radius,
                    "lake_count": len(selected),
                    "total_lake_area_m2": sum(item[3] for item in selected),
                    "largest_lake_area_m2": max((item[3] for item in selected), default=0.0),
                }
            )
        nearest_rows.append(
            {
                "period": period,
                "nearest_lake_distance_km": nearest[0],
                "nearest_lake_lon": nearest[1],
                "nearest_lake_lat": nearest[2],
                "nearest_lake_area_m2": nearest[3],
            }
        )

    summary = pd.DataFrame(rows)
    nearest = pd.DataFrame(nearest_rows)
    summary.to_csv(OUTPUTS / "30_HMA冰湖邻域统计.csv", index=False, encoding="utf-8-sig")
    nearest.to_csv(OUTPUTS / "31_HMA最近冰湖历史变化.csv", index=False, encoding="utf-8-sig")

    first, last = nearest.iloc[0], nearest.iloc[-1]
    change = (last["nearest_lake_area_m2"] / first["nearest_lake_area_m2"] - 1) * 100
    report = f"""# 源区附近冰湖背景审计

- 数据：HMA GLI 五期冰湖目录（1990s 至 2015—2018），不是 2026 年实时冰湖边界。
- 最近目录冰湖距参考源点约 {last['nearest_lake_distance_km']:.2f} km。
- 最近冰湖面积由 {first['nearest_lake_area_m2']:.0f} m² 变为 {last['nearest_lake_area_m2']:.0f} m²，目录变化率 {change:.1f}%（不等同于严格同一湖泊的测量误差已校正趋势）。
- 2015—2018 期 20 km 内冰湖数为 {int(summary.loc[(summary['period'].eq(last['period'])) & (summary['radius_km'].eq(20)), 'lake_count'].iloc[0])} 个。

该目录可以说明区域冰湖背景，但不能证明最近冰湖与 2026 源区水文连通，也不能替代 2026 事件前的光学影像。因此冰湖邻近性不进入主触发模型，只作为灾害链排他性讨论：现有事件资料更支持冰/岩崩塌—碎屑流链，而不是把事件直接写成冰湖溃决。
"""
    (OUTPUTS / "32_冰湖背景使用边界.md").write_text(report, encoding="utf-8")
    print(nearest.to_string(index=False))
    print("\n", summary.loc[summary["radius_km"].eq(20)].to_string(index=False))


if __name__ == "__main__":
    main()
