from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import shapefile
from rasterio.features import rasterize
from rasterio.transform import from_origin


HERE = Path(__file__).resolve()
Q4 = HERE.parents[1]
CONFIG = Q4 / "config" / "model_config.json"


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def read_grid(zip_path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    with zipfile.ZipFile(zip_path) as zf:
        meta = json.loads(zf.read("grid.json"))
        with np.load(io.BytesIO(zf.read("grid.npz")), allow_pickle=False) as src:
            arrays = {name: src[name].copy() for name in src.files}
    return meta, arrays


def choose_crop(meta: dict, arrays: dict[str, np.ndarray], max_chainage: float, margin: int) -> tuple[slice, slice]:
    chain = arrays["upper_chainage_m"]
    source = arrays["source_mask"]
    selected = (np.isfinite(chain) & (chain >= 0) & (chain <= max_chainage)) | source
    rows, cols = np.where(selected)
    if not rows.size:
        raise RuntimeError("上游里程与源区掩膜未形成有效计算域。")
    r0 = max(0, int(rows.min()) - margin)
    r1 = min(selected.shape[0], int(rows.max()) + margin + 1)
    c0 = max(0, int(cols.min()) - margin)
    c1 = min(selected.shape[1], int(cols.max()) + margin + 1)
    return slice(r0, r1), slice(c0, c1)


def shape_geometries(path: Path) -> list[dict]:
    reader = shapefile.Reader(str(path))
    return [item.__geo_interface__ for item in reader.shapes()]


def prepare(resolution: int, force: bool = False, mode: str = "corridor") -> Path:
    cfg = load_config()
    zip_path = Path(cfg["inputs"][f"grid_{resolution}m_zip"])
    suffix = "" if mode == "corridor" else f"_{mode}"
    out = Q4 / "cache" / f"prepared_{resolution}m{suffix}.npz"
    report_path = Q4 / "outputs" / f"input_audit_{resolution}m{suffix}.json"
    if out.exists() and not force:
        print(f"使用已有缓存：{out}")
        return out

    meta, arrays = read_grid(zip_path)
    cell = float(meta["cell_m"])
    rs, cs = choose_crop(
        meta,
        arrays,
        float(cfg["domain"]["max_upper_chainage_m"]),
        int(cfg["domain"]["crop_margin_cells"]),
    )
    z = arrays["z"][rs, cs].astype(np.float32)
    z_raw = arrays["z_raw"][rs, cs].astype(np.float32)
    source = arrays["source_mask"][rs, cs].astype(bool)
    active0 = arrays["active"][rs, cs].astype(bool)
    chain = arrays["upper_chainage_m"][rs, cs].astype(np.float32)
    # 里程用于选择源区—下游裁剪范围，但不能再次充当活动掩膜：若只保留有里程值的
    # 中心带，会在弯道外侧制造人工墙。扩大后的裁剪框内保留原始600 m活动带。
    if mode == "corridor":
        active = active0 | source
    elif mode == "wide":
        # 宽域敏感性：不使用事件走廊活动带限制运动方向，裁剪框内所有有效地形均可计算。
        active = np.isfinite(z) | source
    else:
        raise ValueError(f"未知计算域模式：{mode}")

    xmin, xmax, ymin, ymax = map(float, meta["extent_utm45n"])
    row0, col0 = rs.start, cs.start
    crop_xmin = xmin + col0 * cell
    crop_ymax = ymax - row0 * cell
    transform = from_origin(crop_xmin, crop_ymax, cell, cell)

    footprint_path = Path(cfg["inputs"]["unosat_shapefile"])
    observed = rasterize(
        [(geom, 1) for geom in shape_geometries(footprint_path)],
        out_shape=z.shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)
    observed &= active

    stations = meta["stations"]
    port = stations["gyirong_cctv_arrival"]
    port_row = int(port["row"]) - row0
    port_col = int(port["col"]) - col0
    if not (0 <= port_row < z.shape[0] and 0 <= port_col < z.shape[1]):
        raise RuntimeError("口岸观测点不在裁剪后的计算域内。")

    source_volume = float(cfg["event"]["source_volume_center_m3"])
    source_thickness = source_volume / (max(1, int(source.sum())) * cell * cell)
    slope_y, slope_x = np.gradient(z.astype(np.float64), cell, cell)
    slope = np.hypot(slope_x, slope_y).astype(np.float32)

    # 校准区：源区至口岸；留出区：口岸下游至30 km。空间指标仅在有卫星覆盖的活动带计算。
    port_chainage = float(chain[port_row, port_col])
    calibration = observed & np.isfinite(chain) & (chain <= port_chainage)
    holdout = observed & np.isfinite(chain) & (chain > port_chainage)

    np.savez_compressed(
        out,
        z=z,
        z_raw=z_raw,
        active=active,
        source_mask=source,
        upper_chainage_m=chain,
        slope=slope,
        observed_footprint=observed,
        calibration_mask=calibration,
        holdout_mask=holdout,
        cell_m=np.float32(cell),
        xmin=np.float64(crop_xmin),
        ymax=np.float64(crop_ymax),
        source_thickness_m=np.float32(source_thickness),
        source_volume_m3=np.float64(source_volume),
        port_row=np.int32(port_row),
        port_col=np.int32(port_col),
        port_chainage_m=np.float32(port_chainage),
        row_offset=np.int32(row0),
        col_offset=np.int32(col0),
    )

    audit = {
        "resolution_m": cell,
        "domain_mode": mode,
        "grid_zip": str(zip_path),
        "grid_shape": list(z.shape),
        "active_cells": int(active.sum()),
        "source_cells": int(source.sum()),
        "source_area_km2": float(source.sum() * cell * cell / 1e6),
        "source_volume_m3": source_volume,
        "derived_source_thickness_m": source_thickness,
        "port_cell": [port_row, port_col],
        "port_chainage_m": port_chainage,
        "observed_cells_in_domain": int(observed.sum()),
        "calibration_observed_cells": int(calibration.sum()),
        "holdout_observed_cells": int(holdout.sum()),
        "crs": meta["crs"],
        "vertical_datum": meta.get("vertical_datum"),
        "simulation_bed_note": meta.get("simulation_bed"),
        "evidence_boundary": "网格地形和观测坐标来自公开重建资产；本项目重新计算模型，不继承其参数和结果。UNOSAT为初步卫星解译范围。",
    }
    report_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="准备第四问60/30米计算域与观测掩膜")
    parser.add_argument("--resolution", type=int, choices=[30, 60], default=60)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--mode", choices=["corridor", "wide"], default="corridor")
    args = parser.parse_args()
    prepare(args.resolution, args.force, args.mode)


if __name__ == "__main__":
    sys.exit(main())
