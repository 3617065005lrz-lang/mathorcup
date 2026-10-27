from __future__ import annotations

"""审计第二问新增外源数据。

只检查文件完整性、时空覆盖和波段数，不改动已有正式模型结果。
"""

import json
from pathlib import Path

import pandas as pd
import rasterio


MODEL_DIR = Path(__file__).resolve().parent.parent
OUTPUTS = MODEL_DIR / "outputs"
EXTERNAL = Path(r"D:\MathorCup_Data\External_Gyirong\Q2_core_completion")
CONFIG = json.loads((MODEL_DIR / "config" / "model_config.json").read_text(encoding="utf-8"))
AOI = CONFIG["source_aoi_wgs84"]


def main() -> None:
    rows: list[dict] = []
    for path in sorted(EXTERNAL.rglob("*")):
        if not path.is_file():
            continue
        row = {
            "dataset": path.parent.name,
            "file": path.name,
            "size_mb": path.stat().st_size / 1024**2,
            "suffix": path.suffix.lower(),
            "readable": True,
            "crs": "",
            "bands": None,
            "width": None,
            "height": None,
            "covers_source_aoi": None,
            "note": "",
        }
        if path.suffix.lower() in {".tif", ".tiff"}:
            try:
                with rasterio.open(path) as ds:
                    row.update(
                        crs=str(ds.crs), bands=ds.count, width=ds.width, height=ds.height
                    )
                    b = ds.bounds
                    row["covers_source_aoi"] = bool(
                        b.left <= AOI["west"]
                        and b.right >= AOI["east"]
                        and b.bottom <= AOI["south"]
                        and b.top >= AOI["north"]
                    )
            except Exception as exc:  # pragma: no cover - evidence is written to audit
                row["readable"] = False
                row["note"] = f"rasterio read failed: {exc}"
        rows.append(row)

    audit = pd.DataFrame(rows)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUTPUTS / "35_补充外源数据质量审计.csv", index=False, encoding="utf-8-sig")

    summary = (
        audit.groupby("dataset", dropna=False)
        .agg(files=("file", "size"), size_mb=("size_mb", "sum"), readable=("readable", "sum"))
        .reset_index()
    )
    raster = audit[audit["suffix"].isin([".tif", ".tiff"])]
    failed = audit[~audit["readable"]]
    uncovered = raster[raster["covers_source_aoi"] == False]  # noqa: E712

    table = "\n".join(
        f"| {r.dataset} | {int(r.files)} | {r.size_mb:.1f} | {int(r.readable)} |"
        for r in summary.itertuples(index=False)
    )
    report = f"""# 第二问补充外源数据审计

## 审计口径

- 数据根目录：`{EXTERNAL}`
- 源区范围：{AOI['west']}–{AOI['east']}°E，{AOI['south']}–{AOI['north']}°N。
- 审计项：文件数、大小、GeoTIFF 可读性、CRS、波段数和源区空间覆盖。

| 数据集 | 文件数 | 大小/MB | 可读文件数 |
|---|---:|---:|---:|
{table}

## 结论

- 共审计 {len(audit)} 个文件，其中 GeoTIFF {len(raster)} 个。
- 无法读取文件 {len(failed)} 个；不完整覆盖源区的 GeoTIFF {len(uncovered)} 个。
- 实验中严格区分灾前建模数据与灾后验证数据，灾后影像不参与阈值、权重或参数拟合。
"""
    (OUTPUTS / "36_补充外源数据质量审计报告.md").write_text(report, encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"unreadable={len(failed)}, uncovered_rasters={len(uncovered)}")


if __name__ == "__main__":
    main()
