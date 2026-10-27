from __future__ import annotations

"""第二问遥感补充实验。

实验原则：
1. 2021—2024 为背景，2025 不进入背景；
2. 事件日及灾后影像不用于灾前参数估计；
3. Sentinel-1 GRD 只解释为后向散射异常，不解释为 InSAR 形变；
4. 所有指数都保留到原始波段的映射。
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds


MODEL_DIR = Path(__file__).resolve().parent.parent
OUTPUTS = MODEL_DIR / "outputs"
PROCESSED = MODEL_DIR / "processed"
EXTERNAL = Path(r"D:\MathorCup_Data\External_Gyirong\Q2_core_completion")
CONFIG = json.loads((MODEL_DIR / "config" / "model_config.json").read_text(encoding="utf-8"))
AOI = CONFIG["source_aoi_wgs84"]
EVENT = pd.Timestamp(CONFIG["event_time_beijing"])


def aoi_cube(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        win = from_bounds(AOI["west"], AOI["south"], AOI["east"], AOI["north"], ds.transform)
        win = win.round_offsets().round_lengths()
        arr = ds.read(window=win).astype("float64")
        for i, nodata in enumerate(ds.nodatavals):
            if nodata is not None:
                if np.isfinite(nodata):
                    arr[i][arr[i] == nodata] = np.nan
                else:
                    arr[i][~np.isfinite(arr[i])] = np.nan
        return arr


def safe_mean(x: np.ndarray) -> float:
    return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan


def empirical_percentile(value: float, background: pd.Series) -> float:
    x = background.dropna().to_numpy(float)
    if not np.isfinite(value) or len(x) == 0:
        return np.nan
    return float(100 * (np.sum(x < value) + 0.5 * np.sum(x == value)) / len(x))


def markdown_table(frame: pd.DataFrame, float_digits: int = 3) -> str:
    """不依赖 tabulate 的简单 Markdown 表。"""
    shown = frame.copy()
    for col in shown.select_dtypes(include=["float", "float64", "float32"]).columns:
        shown[col] = shown[col].map(lambda v: "" if pd.isna(v) else f"{v:.{float_digits}f}")
    headers = [str(c) for c in shown.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in shown.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def extract_modis() -> pd.DataFrame:
    folder = EXTERNAL / "P1_MODIS_Albedo_LST_GEE_clipped"
    rows = []
    for year in [2021, 2022, 2023, 2024, 2026]:
        dates = pd.date_range(f"{year}-06-01", f"{year}-09-30" if year < 2026 else f"{year}-08-26")
        a3 = aoi_cube(folder / f"MCD43A3_{year}_warm_season_stack_500m.tif")
        a2 = aoi_cube(folder / f"MCD43A2_{year}_warm_season_QA_stack_500m.tif")
        terra = aoi_cube(folder / f"MOD11A1_{year}_warm_season_stack_1km.tif")
        aqua = aoi_cube(folder / f"MYD11A1_{year}_warm_season_stack_1km.tif")
        assert a3.shape[0] == len(dates) * 3
        assert a2.shape[0] == len(dates) * 3
        assert terra.shape[0] == len(dates) * 4
        assert aqua.shape[0] == len(dates) * 4
        for i, date in enumerate(dates):
            bsa, wsa, qa = a3[i * 3 : i * 3 + 3]
            valid_albedo = (qa <= 1) & (bsa > 0) & (bsa < 1000) & (wsa > 0) & (wsa < 1000)
            albedo = np.where(valid_albedo, (bsa + wsa) * 0.0005, np.nan)
            snow, snow_qa, uncertainty = a2[i * 3 : i * 3 + 3]
            # MCD43A2 Snow_BRDF_Albedo is a binary snow flag (0/1); 255 is fill.
            # The following bands describe Band-1 BRDF quality/uncertainty and are
            # not a quality mask for the binary snow flag itself.
            valid_snow = np.isin(snow, [0, 1])

            def lst_metrics(cube: np.ndarray) -> tuple[float, float]:
                day, qd, night, qn = cube[i * 4 : i * 4 + 4]
                qd_int = np.nan_to_num(qd, nan=255).astype("int64")
                qn_int = np.nan_to_num(qn, nan=255).astype("int64")
                valid_day = (day > 0) & ((qd_int & 3) <= 1)
                valid_night = (night > 0) & ((qn_int & 3) <= 1)
                return (
                    safe_mean(np.where(valid_day, day * 0.02 - 273.15, np.nan)),
                    safe_mean(np.where(valid_night, night * 0.02 - 273.15, np.nan)),
                )

            td, tn = lst_metrics(terra)
            ad, an = lst_metrics(aqua)
            rows.append(
                {
                    "date": date,
                    "year": year,
                    "albedo": safe_mean(albedo),
                    "albedo_valid_fraction": float(np.mean(valid_albedo)),
                    "mcd43_snow_fraction": safe_mean(np.where(valid_snow, snow, np.nan)),
                    "terra_day_lst_c": td,
                    "terra_night_lst_c": tn,
                    "aqua_day_lst_c": ad,
                    "aqua_night_lst_c": an,
                }
            )
    return pd.DataFrame(rows)


def extract_sentinel2() -> pd.DataFrame:
    rows = []
    folder = EXTERNAL / "P0-3_Sentinel2_GEE_clipped"
    for path in sorted(folder.glob("*.tif")):
        x = aoi_cube(path)
        blue, green, red, nir, swir1, swir2 = x
        valid = np.all(np.isfinite(x), axis=0) & np.all(x > 0, axis=0)
        ndsi = np.where(valid, (green - swir1) / (green + swir1 + 1e-9), np.nan)
        ndvi = np.where(valid, (nir - red) / (nir + red + 1e-9), np.nan)
        bsi = np.where(valid, ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + 1e-9), np.nan)
        name = path.stem
        m = re.search(r"(20\d{2})_(\d{2})", name)
        phase = "monthly"
        if "pre_early" in name:
            phase = "2026_pre_early"
        elif "pre_late" in name:
            phase = "2026_pre_late"
        elif "post" in name:
            phase = "post_validation_only"
        rows.append(
            {
                "file": path.name,
                "year": int(m.group(1)) if m else 2026,
                "month": int(m.group(2)) if m else 8,
                "phase": phase,
                "valid_fraction": float(np.mean(valid)),
                "ndsi_mean": safe_mean(ndsi),
                "snow_fraction_ndsi_ge_0p4": safe_mean(np.where(valid, ndsi >= 0.4, np.nan)),
                "ndvi_mean": safe_mean(ndvi),
                "bare_soil_index_mean": safe_mean(bsi),
            }
        )
    return pd.DataFrame(rows)


def extract_landsat() -> pd.DataFrame:
    rows = []
    folder = EXTERNAL / "P1-3_Landsat_GEE_clipped"
    for path in sorted(folder.glob("*.tif")):
        x = aoi_cube(path)
        sr = x[:6] * 0.0000275 - 0.2
        blue, green, red, nir, swir1, swir2 = sr
        valid = np.all(np.isfinite(sr), axis=0) & np.all(x[:6] > 0, axis=0) & np.all((sr > 0) & (sr < 1), axis=0)
        ndsi_den = green + swir1
        ndvi_den = nir + red
        bsi_den = (swir1 + red) + (nir + blue)
        ndsi = np.where(valid & (np.abs(ndsi_den) > 0.02), (green - swir1) / ndsi_den, np.nan)
        ndvi = np.where(valid & (np.abs(ndvi_den) > 0.02), (nir - red) / ndvi_den, np.nan)
        bsi = np.where(valid & (np.abs(bsi_den) > 0.02), ((swir1 + red) - (nir + blue)) / bsi_den, np.nan)
        m = re.search(r"(20\d{2})", path.name)
        phase = "warm_season"
        if "pre" in path.stem:
            phase = "2026_pre"
        if "post" in path.stem:
            phase = "post_validation_only"
        rows.append(
            {
                "file": path.name,
                "year": int(m.group(1)),
                "phase": phase,
                "valid_fraction": float(np.mean(valid)),
                "ndsi_mean": safe_mean(ndsi),
                "snow_fraction_ndsi_ge_0p4": safe_mean(np.where(valid, ndsi >= 0.4, np.nan)),
                "ndvi_mean": safe_mean(ndvi),
                "bare_soil_index_mean": safe_mean(bsi),
            }
        )
    return pd.DataFrame(rows)


def extract_sentinel1() -> pd.DataFrame:
    folder = EXTERNAL / "P0-4_Sentinel1_GRD_GEE_clipped"
    local_catalog = pd.read_csv(folder / "scene_catalog.csv")
    external_catalog_path = Path(
        r"D:\MathorCup_Data\External_Gyirong\P0-5_GFS_historical_controls\original_toolkit\帮助_交接包\P0-4_Sentinel-1\catalog\sentinel1_catalog.csv"
    )
    external_catalog = pd.read_csv(external_catalog_path) if external_catalog_path.exists() else pd.DataFrame()
    rows = []
    for path in sorted(folder.glob("*.tif")):
        stem = path.stem.removesuffix("_30m")
        x = aoi_cube(path)
        vv, vh, angle = x
        valid = np.isfinite(vv) & np.isfinite(vh) & np.isfinite(angle)
        local = local_catalog[local_catalog["system_index"] == stem]
        time = pd.to_datetime(local.iloc[0]["time_utc"], utc=True) if len(local) else pd.NaT
        match = external_catalog[
            external_catalog["name"].astype(str).str.startswith(stem)
        ] if len(external_catalog) else pd.DataFrame()
        rel_orbit = int(match.iloc[0]["relative_orbit"]) if len(match) else np.nan
        direction = str(match.iloc[0]["orbit_direction"]) if len(match) else "unknown"
        rows.append(
            {
                "file": path.name,
                "time_utc": time,
                "relative_orbit": rel_orbit,
                "orbit_direction": direction,
                "post_event_validation_only": bool(time > EVENT.tz_localize("Asia/Shanghai").tz_convert("UTC")),
                "valid_fraction": float(np.mean(valid)),
                "vv_db_median": float(np.nanmedian(np.where(valid, vv, np.nan))),
                "vh_db_median": float(np.nanmedian(np.where(valid, vh, np.nan))),
                "vv_minus_vh_db": float(np.nanmedian(np.where(valid, vv - vh, np.nan))),
                "incidence_angle_mean_deg": safe_mean(np.where(valid, angle, np.nan)),
            }
        )
    out = pd.DataFrame(rows).sort_values("time_utc")
    out["track"] = out["relative_orbit"].astype("Int64").astype(str) + "_" + out["orbit_direction"]
    for col in ["vv_db_median", "vh_db_median", "vv_minus_vh_db"]:
        zname = col.replace("_median", "") + "_robust_z_pre"
        out[zname] = np.nan
        for track, idx in out.groupby("track").groups.items():
            pre = out.loc[idx][~out.loc[idx, "post_event_validation_only"]]
            med = pre[col].median()
            mad = (pre[col] - med).abs().median()
            if mad > 0:
                out.loc[idx, zname] = 0.6745 * (out.loc[idx, col] - med) / mad
    return out


def update_eti_with_mcd43(modis: pd.DataFrame) -> pd.DataFrame:
    hourly_path = PROCESSED / "04_多源动态触发逐小时缓存.parquet"
    hourly = pd.read_parquet(hourly_path).sort_index()
    work = modis[["date", "albedo"]].dropna().copy()
    work["available_date"] = work["date"] + pd.Timedelta(days=1)
    daily = pd.DataFrame({"day": pd.to_datetime(hourly.index.normalize().unique())})
    mapped = pd.merge_asof(
        daily.sort_values("day"), work.sort_values("available_date"),
        left_on="day", right_on="available_date", direction="backward",
        tolerance=pd.Timedelta(days=7),
    ).set_index("day")
    calc = hourly.copy()
    calc["day"] = calc.index.normalize()
    calc = calc.join(mapped[["albedo", "date"]].rename(columns={"date": "mcd43_observation_date"}), on="day")
    calc["albedo_mcd43_lagged"] = calc["albedo"].where(calc["albedo"].between(0.05, 0.95), calc["albedo_operational"])
    eti = CONFIG["eti_model"]
    shortwave = calc["sw_down_center"].clip(lower=0) / 3600.0
    texcess = (calc["source_temperature"] - eti["temperature_threshold_c"]).clip(lower=0)
    absorbed = (1 - calc["albedo_mcd43_lagged"]) * shortwave
    calc["eti_mcd43_mm_h"] = np.where(
        calc["source_temperature"] > eti["temperature_threshold_c"],
        eti["temperature_factor_mm_h_c"] * texcess + eti["shortwave_factor_mm_h_wm2"] * absorbed,
        0.0,
    )
    calc["eti_mcd43_72h_mm"] = calc["eti_mcd43_mm_h"].rolling(72, min_periods=72).sum()
    calc["eti_original_72h_mm"] = calc["eti_melt_mm_h"].rolling(72, min_periods=72).sum()
    event_end = pd.Timestamp(CONFIG["event_last_complete_hour_beijing"])
    event_row = calc.loc[event_end]
    event_doy = event_end.dayofyear
    background = calc[
        calc.index.year.isin(CONFIG["background_years"])
        & (calc.index.hour == event_end.hour)
        & (np.abs(calc.index.dayofyear - event_doy) <= 15)
    ]
    return pd.DataFrame(
        [
            {
                "metric": "original_ERA5_fallback_ETI_72h_mm",
                "event_value": event_row["eti_original_72h_mm"],
                "background_august_percentile": empirical_percentile(event_row["eti_original_72h_mm"], background["eti_original_72h_mm"]),
            },
            {
                "metric": "MCD43_lagged_albedo_ETI_72h_mm",
                "event_value": event_row["eti_mcd43_72h_mm"],
                "background_august_percentile": empirical_percentile(event_row["eti_mcd43_72h_mm"], background["eti_mcd43_72h_mm"]),
            },
            {
                "metric": "MCD43_lagged_albedo_used",
                "event_value": event_row["albedo_mcd43_lagged"],
                "background_august_percentile": empirical_percentile(event_row["albedo_mcd43_lagged"], background["albedo_mcd43_lagged"]),
            },
        ]
    )


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    modis = extract_modis()
    s2 = extract_sentinel2()
    landsat = extract_landsat()
    s1 = extract_sentinel1()
    eti = update_eti_with_mcd43(modis)

    modis.to_csv(OUTPUTS / "37_MODIS反照率与地表温度源区时序.csv", index=False, encoding="utf-8-sig")
    eti.to_csv(OUTPUTS / "38_MCD43反照率替换ETI敏感性.csv", index=False, encoding="utf-8-sig")
    s2.to_csv(OUTPUTS / "39_Sentinel2雪冰覆盖核查.csv", index=False, encoding="utf-8-sig")
    landsat.to_csv(OUTPUTS / "40_Landsat长期背景核查.csv", index=False, encoding="utf-8-sig")
    s1.to_csv(OUTPUTS / "41_Sentinel1后向散射核查.csv", index=False, encoding="utf-8-sig")

    event_modis = modis[(modis.year == 2026) & (modis.date.between("2026-08-19", "2026-08-25"))]
    bg_modis = modis[(modis.year.isin([2021, 2022, 2023, 2024])) & (modis.date.dt.month == 8)]
    late = s2[s2.phase == "2026_pre_late"].iloc[0]
    bg_aug = s2[(s2.year.isin([2021, 2022, 2023, 2024])) & (s2.month == 8) & (s2.phase == "monthly")]
    pre_s1 = s1[~s1.post_event_validation_only]
    last_by_track = pre_s1.sort_values("time_utc").groupby("track", as_index=False).tail(1)

    eti_md = markdown_table(eti)
    s1_md = markdown_table(last_by_track[['time_utc','track','vv_db_median','vh_db_median','vv_db_robust_z_pre','vh_db_robust_z_pre']])
    report = f"""# 第二问遥感补充实验报告

## 1. 实验边界

- 背景年仅使用 2021—2024，2025 不参与建模。
- 2026-08-26 当日与灾后数据只作独立验证，不参与灾前阈值或权重估计。
- Sentinel-1 产品是 GRD，因此只报告后向散射变化，不声称得到了毫米级形变。

## 2. 真实反照率对 ETI 的影响

{eti_md}

解释原则：若替换前后事件百分位和量级接近，说明热力结论对反照率数据源稳健；若差异较大，应以 MCD43 版为主，并将原结果降为敏感性对照。

## 3. MODIS 灾前 7 日源区背景

- MCD43 平均反照率：{event_modis.albedo.mean():.3f}；历史 8 月日序列中位数：{bg_modis.albedo.median():.3f}。
- MCD43 雪覆盖指标均值：{event_modis.mcd43_snow_fraction.mean():.3f}；历史 8 月中位数：{bg_modis.mcd43_snow_fraction.median():.3f}。
- 该结果用于补强雪冰表面状态解释，不单独生成“泥石流发生概率”。

## 4. Sentinel-2 灾前雪冰覆盖

- 2026 年8月13—25日源区 NDSI 均值：{late.ndsi_mean:.3f}，NDSI≥0.4 像元比例：{late.snow_fraction_ndsi_ge_0p4:.3f}。
- 2021—2024 年8月合成的 NDSI 均值中位数：{bg_aug.ndsi_mean.median():.3f}，雪冰像元比例中位数：{bg_aug.snow_fraction_ndsi_ge_0p4.median():.3f}。
- 同期只有 4 个背景年，因此定位为物理旁证而非独立统计定律。

## 5. Sentinel-1 GRD 灾前后向散射异常

{s1_md}

仅当同轨道灾前最后一景的稳健 z 分数明显偏离时，才可作为表面含水、粗糙度或雪冰状态变化的遥感旁证；不能据此直接判定坡体形变。

## 6. Landsat 和灾后影像的用法

- Landsat 序列用于检查 2026 灾前暖季的雪冰/植被/裸地状态是否偏离 2013—2024 长期背景。
- Sentinel-2 和 Landsat 灾后数据仅用于验证灾区表面确实发生改变，不反过来调整灾前模型。
"""
    (OUTPUTS / "42_第二问遥感补充实验报告.md").write_text(report, encoding="utf-8")
    print(eti.to_string(index=False))
    print(f"MODIS rows={len(modis)}, S2={len(s2)}, Landsat={len(landsat)}, S1={len(s1)}")


if __name__ == "__main__":
    main()
