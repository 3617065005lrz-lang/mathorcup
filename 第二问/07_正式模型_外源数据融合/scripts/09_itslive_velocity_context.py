from __future__ import annotations

"""Summarize historical ITS_LIVE velocity in the source AOI."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rasterio.warp import transform
import xarray as xr


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent
CONFIG = json.loads((MODEL_DIR / "config" / "model_config.json").read_text(encoding="utf-8"))
OUTPUTS = MODEL_DIR / "outputs"
ITS_PATH = Path(r"D:\MathorCup_Data\External_Gyirong\P1-5_ITS_LIVE\aoi_netcdf\itslive_annual_1990_2000_2010_2018_aoi.nc")


def main() -> None:
    with xr.open_dataset(ITS_PATH) as ds:
        proj4 = ds["UTM_Projection"].attrs["spatial_proj4"]
        aoi = CONFIG["source_aoi_wgs84"]
        lon = [aoi["west"], aoi["west"], aoi["east"], aoi["east"]]
        lat = [aoi["south"], aoi["north"], aoi["south"], aoi["north"]]
        xs, ys = transform("EPSG:4326", proj4, lon, lat)
        sub = ds.sel(x=slice(min(xs), max(xs)), y=slice(max(ys), min(ys))).load()

    rows = []
    for year in sub["time"].values:
        data = sub.sel(time=year)
        velocity = data["v"].values.astype(float)
        error = data["v_err"].values.astype(float)
        ice = data["ice"].values == 1
        valid = ice & np.isfinite(velocity)
        rows.append(
            {
                "year": int(year),
                "ice_pixels": int(ice.sum()),
                "valid_ice_velocity_pixels": int(valid.sum()),
                "ice_velocity_median_m_per_year": float(np.nanmedian(velocity[valid])),
                "ice_velocity_q25_m_per_year": float(np.nanquantile(velocity[valid], 0.25)),
                "ice_velocity_q75_m_per_year": float(np.nanquantile(velocity[valid], 0.75)),
                "ice_velocity_error_median_m_per_year": float(np.nanmedian(error[valid])),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUTS / "33_ITS_LIVE源区历史冰速.csv", index=False, encoding="utf-8-sig")
    first, last = summary.iloc[0], summary.iloc[-1]
    report = f"""# ITS_LIVE 源区冰速背景

- 产品年份：{summary['year'].tolist()}，分辨率约 240 m；用于长期背景核查，不代表 2026 年事发前即时速度。
- 源区 AOI 冰像元中位速度由 {first['ice_velocity_median_m_per_year']:.2f} m/a（{int(first['year'])}）变为 {last['ice_velocity_median_m_per_year']:.2f} m/a（{int(last['year'])}）。
- 现有四期结果不支持把“长期持续加速”直接写成已证实事实。
- 主空间模型仍使用赛题附件提供的灾前冰速/冰厚；ITS_LIVE 仅用于说明长期背景与资料边界。
"""
    (OUTPUTS / "34_ITS_LIVE使用边界.md").write_text(report, encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
