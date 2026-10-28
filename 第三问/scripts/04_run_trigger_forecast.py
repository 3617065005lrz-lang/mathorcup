from __future__ import annotations

"""Mechanism-constrained trigger-time hindcast for Question 3.

This is deliberately not a fitted event/no-event classifier.  Each weather
scenario is passed through the Q2 mechanism score and its first persistent
crossing of a historical background threshold is recorded.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_DIR = Path(__file__).resolve().parents[1]
ROOT = MODEL_DIR.parent
Q2 = ROOT / "第二问" / "07_正式模型_外源数据融合"
CONFIG = json.loads((MODEL_DIR / "config" / "model_config.json").read_text(encoding="utf-8"))
CACHE = MODEL_DIR / "cache"
PROCESSED = MODEL_DIR / "processed"
OUTPUTS = MODEL_DIR / "outputs"
PROCESSED.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

FC_PATH = CACHE / "01_GEFS源区点预报缓存.parquet"
Q2_DYNAMIC = Q2 / "processed" / "04_多源动态触发逐小时缓存.parquet"
MCD43 = Q2 / "outputs" / "37_MODIS反照率与地表温度源区时序.csv"
SPATIAL_GRID = Q2 / "processed" / "13_MCD43升级时空风险对照格网.parquet"
GFS_RELIABILITY = Q2 / "outputs" / "27_GFS分时效个例核查.csv"

WINDOWS = [int(x) for x in CONFIG["antecedent_windows_h"]]
N_DRAWS = int(CONFIG["scenario_draws"])
RNG = np.random.default_rng(int(CONFIG["random_seed"]))
EVENT_CATALOG_UTC = pd.Timestamp(CONFIG["event_catalog_time_utc"])
EVENT_TRIGGER_UTC = EVENT_CATALOG_UTC + pd.Timedelta(seconds=float(CONFIG["event_first_detachment_offset_s"]))
EVENT_TRIGGER_BJ = EVENT_TRIGGER_UTC.tz_convert("Asia/Shanghai").tz_localize(None)
FORECAST_NATIVE_STEP_H = 6.0
RESOLUTION_HALF_STEP_H = FORECAST_NATIVE_STEP_H / 2


def cdf(reference: np.ndarray | pd.Series, values: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(reference, dtype=float))
    ref = ref[np.isfinite(ref)]
    x = np.asarray(values, dtype=float)
    out = np.full(x.shape, np.nan, dtype=float)
    ok = np.isfinite(x)
    out[ok] = np.searchsorted(ref, x[ok], side="right") / len(ref)
    return out


def rolling_sum(a: np.ndarray, window: int) -> np.ndarray:
    values = np.nan_to_num(a, nan=0.0)
    valid = np.isfinite(a).astype(np.int16)
    cs = np.concatenate([np.zeros((a.shape[0], 1)), np.cumsum(values, axis=1)], axis=1)
    cc = np.concatenate([np.zeros((a.shape[0], 1)), np.cumsum(valid, axis=1)], axis=1)
    out = np.full_like(a, np.nan, dtype=float)
    sums = cs[:, window:] - cs[:, :-window]
    counts = cc[:, window:] - cc[:, :-window]
    out[:, window - 1 :] = np.where(counts == window, sums, np.nan)
    return out


def rolling_mean(a: np.ndarray, window: int) -> np.ndarray:
    return rolling_sum(a, window) / window


def attach_mcd43(frame: pd.DataFrame) -> pd.DataFrame:
    modis = pd.read_csv(MCD43, parse_dates=["date"])
    available = modis[["date", "albedo"]].dropna().sort_values("date")
    available["available_date"] = available["date"] + pd.Timedelta(days=1)
    days = pd.DataFrame({"day": pd.to_datetime(frame.index.normalize().unique())})
    mapping = pd.merge_asof(
        days.sort_values("day"),
        available.sort_values("available_date"),
        left_on="day",
        right_on="available_date",
        direction="backward",
        tolerance=pd.Timedelta(days=7),
    ).set_index("day")
    out = frame.copy()
    out["day"] = out.index.normalize()
    out = out.join(mapping[["albedo"]], on="day")
    out["albedo_mcd43"] = out["albedo"].where(out["albedo"].between(0.05, 0.95), out["albedo_operational"])
    return out


def add_historical_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    p = CONFIG["eti_model"]
    temp = out["source_temperature"].to_numpy(float)
    sw = out["shortwave_w_m2"].to_numpy(float)
    alb = out["albedo_mcd43"].to_numpy(float)
    hourly_melt = np.where(
        temp > p["central_temperature_threshold_c"],
        p["central_temperature_factor"] * np.maximum(temp - p["central_temperature_threshold_c"], 0)
        + p["central_shortwave_factor"] * (1 - alb) * np.maximum(sw, 0),
        0.0,
    )
    out["era5_rain_360h_mm"] = np.nan
    pieces = []
    for _, group in out.groupby(out.index.year):
        g = group.copy()
        loc = out.index.get_indexer(g.index)
        melt = pd.Series(hourly_melt[loc], index=g.index)
        for w in WINDOWS:
            g[f"eti_mcd43_{w}h_mm"] = melt.rolling(w, min_periods=w).sum()
        g["era5_rain_360h_mm"] = g["precipitation_center"].rolling(360, min_periods=360).sum()
        pieces.append(g)
    return pd.concat(pieces).sort_index()


def background_table(frame: pd.DataFrame) -> pd.DataFrame:
    event_hour = 10
    event_doy = pd.Timestamp("2026-08-26").dayofyear
    return frame.loc[
        frame.index.year.isin([2021, 2022, 2023, 2024])
        & (frame.index.hour == event_hour)
        & (np.abs(frame.index.dayofyear - event_doy) <= 15)
    ].copy()


def build_references(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, float], float, float]:
    bg = background_table(frame)
    refs: dict[str, np.ndarray] = {
        "p15": bg["chirps_rain_prev_15d_mm"].dropna().to_numpy(float),
    }
    era360 = bg["era5_rain_360h_mm"].to_numpy(float)
    ratio = bg["chirps_rain_prev_15d_mm"].to_numpy(float) / np.where(era360 > 1e-6, era360, np.nan)
    rain_scale = float(np.nanmedian(ratio))
    rain_scale = float(np.clip(rain_scale, 0.25, 4.0))

    spatial = pd.read_parquet(SPATIAL_GRID)
    static = spatial["static_mechanism_score"].rank(pct=True).to_numpy(float)
    source = spatial["event_source_aoi_label"].astype(bool).to_numpy()
    # A robust hotspot represents the part of the mapped source area capable of
    # initiating failure; using the maximum would be overly sensitive to one cell.
    source_static_hotspot = float(np.quantile(static[source], 0.90))
    thresholds: dict[str, float] = {}
    for w in WINDOWS:
        refs[f"thermal_{w}"] = bg[f"eti_mcd43_{w}h_mm"].dropna().to_numpy(float)
        refs[f"rain_{w}"] = bg[f"era5_rain_{w}h_mm"].dropna().to_numpy(float)
        soil_col = f"smap_rootzone_mean_{w}h"
        refs[f"soil_{w}"] = bg[soil_col].dropna().to_numpy(float)
        thermal = cdf(refs[f"thermal_{w}"], bg[f"eti_mcd43_{w}h_mm"].to_numpy(float))
        rain = cdf(refs[f"rain_{w}"], bg[f"era5_rain_{w}h_mm"].to_numpy(float))
        ant = cdf(refs["p15"], bg["chirps_rain_prev_15d_mm"].to_numpy(float))
        soil = cdf(refs[f"soil_{w}"], bg[soil_col].to_numpy(float))
        moisture = np.nanmedian(np.vstack([ant, soil]), axis=0)
        dynamic = 1 - (1 - thermal) * (1 - np.sqrt(np.maximum(rain, 0) * np.maximum(moisture, 0)))
        joint_reference = np.sqrt(static[:, None] * dynamic[None, :])
        thresholds[f"uniform_{w}"] = float(np.nanquantile(joint_reference, CONFIG["trigger_reference_quantile"]))

    # Main mixed-window model: ETI72 + rain72 + root-zone moisture120.
    thermal = cdf(refs["thermal_72"], bg["eti_mcd43_72h_mm"].to_numpy(float))
    rain = cdf(refs["rain_72"], bg["era5_rain_72h_mm"].to_numpy(float))
    ant = cdf(refs["p15"], bg["chirps_rain_prev_15d_mm"].to_numpy(float))
    soil = cdf(refs["soil_120"], bg["smap_rootzone_mean_120h"].to_numpy(float))
    moisture = np.nanmedian(np.vstack([ant, soil]), axis=0)
    dynamic = 1 - (1 - thermal) * (1 - np.sqrt(np.maximum(rain, 0) * np.maximum(moisture, 0)))
    joint_reference = np.sqrt(static[:, None] * dynamic[None, :])
    thresholds["main"] = float(np.nanquantile(joint_reference, CONFIG["trigger_reference_quantile"]))
    bg_scores = pd.DataFrame(
        {"thermal": thermal, "short_rain": rain, "antecedent_moisture": moisture, "dynamic": dynamic},
        index=bg.index,
    )
    return bg_scores, refs, thresholds, rain_scale, source_static_hotspot


def forecast_run(cache: pd.DataFrame, issue_utc: pd.Timestamp) -> pd.DataFrame:
    date = issue_utc.strftime("%Y%m%d")
    cycle = issue_utc.strftime("%H")
    g = cache.loc[(cache.date.astype(str) == date) & (cache.cycle.astype(str).str.zfill(2) == cycle)].copy()
    if g.empty:
        raise FileNotFoundError(f"Missing GEFS run {date} {cycle}Z")
    pivot = g.pivot_table(index=["lead_h", "valid_time_beijing"], columns=["product", "variable"], values="value")
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    return pivot.reset_index().sort_values("lead_h")


def ar1_errors(n: int, steps: int, rho: float) -> np.ndarray:
    corr = np.array(
        [[1.0, 0.50, -0.25, -0.20], [0.50, 1.0, -0.35, -0.10], [-0.25, -0.35, 1.0, 0.20], [-0.20, -0.10, 0.20, 1.0]]
    )
    chol = np.linalg.cholesky(corr)
    innovations = RNG.standard_normal((n, steps, 4)) @ chol.T
    out = np.zeros_like(innovations)
    out[:, 0] = innovations[:, 0]
    scale = np.sqrt(1 - rho**2)
    for j in range(1, steps):
        out[:, j] = rho * out[:, j - 1] + scale * innovations[:, j]
    return out


def lead_error_floors(leads: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map Q2 lead-band MAE to GEFS structural-error floors.

    The audit is from the related GFS system, so its bias is not transferred to
    the GEFS mean.  Only MAE is used as an additional representativeness-error
    floor.  Two 3-hour precipitation errors are combined by root-sum-square for
    the downloaded six-hour GEFS accumulation.
    """
    audit = pd.read_csv(GFS_RELIABILITY).set_index("lead_band")
    band = np.select(
        [leads <= 24, leads <= 48, leads <= 72, leads <= 120],
        ["0-24h", "27-48h", "51-72h", "75-120h"],
        default="75-120h",
    )
    t = np.array([audit.loc[x, "temperature_mae_c"] for x in band], dtype=float)
    rain3 = np.array([audit.loc[x, "precipitation_mae_3h_mm"] for x in band], dtype=float)
    sw = np.array([audit.loc[x, "shortwave_mae_w_m2"] for x in band], dtype=float)
    return t, np.sqrt(2.0) * rain3, sw


def interp_rows(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    return np.vstack([np.interp(x, xp, row) for row in fp])


def first_persistent_crossing(times: pd.DatetimeIndex, risk: np.ndarray, threshold: float, persistence: int) -> np.ndarray:
    crossed = risk >= threshold
    out = np.full(risk.shape[0], np.datetime64("NaT"), dtype="datetime64[ns]")
    t_ns = times.to_numpy(dtype="datetime64[ns]").astype("int64")
    for i in range(risk.shape[0]):
        run = np.convolve(crossed[i].astype(int), np.ones(persistence, dtype=int), mode="valid")
        hits = np.flatnonzero(run == persistence)
        if not len(hits):
            continue
        k = int(hits[0])
        if k == 0 or not np.isfinite(risk[i, k - 1]) or risk[i, k] == risk[i, k - 1]:
            out[i] = np.datetime64(times[k])
        else:
            frac = np.clip((threshold - risk[i, k - 1]) / (risk[i, k] - risk[i, k - 1]), 0, 1)
            out[i] = np.datetime64(int(t_ns[k - 1] + frac * (t_ns[k] - t_ns[k - 1])), "ns")
    return out


def persistent_peak_time(times: pd.DatetimeIndex, risk: np.ndarray, threshold: float, persistence: int) -> np.ndarray:
    """Return the end of the highest trailing persistent-risk window.

    The window represents antecedent conditions accumulated through time t.
    Returning its centre would create a systematic half-window early bias.
    """
    out = np.full(risk.shape[0], np.datetime64("NaT"), dtype="datetime64[ns]")
    kernel = np.ones(persistence, dtype=float) / persistence
    for i in range(risk.shape[0]):
        smooth = np.convolve(risk[i], kernel, mode="valid")
        if not np.isfinite(smooth).any():
            continue
        k = int(np.nanargmax(smooth))
        if smooth[k] < threshold:
            continue
        window_end = min(k + persistence - 1, len(times) - 1)
        out[i] = np.datetime64(times[window_end])
    return out


def simulate_issue(
    issue_utc: pd.Timestamp,
    nominal_h: int,
    fc: pd.DataFrame,
    history: pd.DataFrame,
    refs: dict[str, np.ndarray],
    thresholds: dict[str, float],
    rain_scale: float,
    source_static_hotspot: float,
    n: int,
    stochastic: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    issue_bj = issue_utc.tz_convert("Asia/Shanghai").tz_localize(None)
    max_valid = pd.Timestamp(fc.valid_time_beijing.max())
    timeline = pd.date_range(issue_bj - pd.Timedelta(hours=359), max_valid, freq="h")
    hist = history.reindex(timeline)
    numeric_cols = hist.select_dtypes(include=[np.number, "bool"]).columns
    hist.loc[:, numeric_cols] = hist.loc[:, numeric_cols].interpolate(method="time").ffill().bfill()
    other_cols = hist.columns.difference(numeric_cols)
    if len(other_cols):
        hist.loc[:, other_cols] = hist.loc[:, other_cols].ffill().bfill()
    ntime = len(timeline)
    future_mask = timeline > issue_bj
    future_h = ((timeline[future_mask] - issue_bj) / pd.Timedelta(hours=1)).to_numpy(float)
    leads = fc.lead_h.to_numpy(float)

    temp = np.tile(hist.source_temperature.to_numpy(float), (n, 1))
    sw = np.tile(hist.shortwave_w_m2.to_numpy(float), (n, 1))
    rain = np.tile(hist.precipitation_center.to_numpy(float), (n, 1))
    soil_init_series = hist.smap_sm_rootzone.interpolate().ffill().bfill().to_numpy(float)
    soil = np.tile(soil_init_series, (n, 1))
    albedo_base = hist.albedo_mcd43.interpolate().ffill().bfill().to_numpy(float)
    albedo = np.tile(albedo_base, (n, 1))

    def col(name: str, default: float = 0.0) -> np.ndarray:
        if name not in fc:
            return np.full(len(fc), default, dtype=float)
        x = fc[name].to_numpy(float)
        if np.isfinite(x).sum() < 2:
            return np.nan_to_num(x, nan=default)
        return pd.Series(x).interpolate(limit_direction="both").to_numpy(float)

    tmean_grid = col("geavg_temperature_2m_c")
    smean_grid = col("geavg_soil_water_0_10m")
    swmean_grid = np.maximum(col("geavg_shortwave_down_w_m2"), 0)
    rainmean_grid = np.maximum(col("geavg_precipitation_6h_mm"), 0)
    tspread = np.maximum(col("gespr_temperature_2m_c"), 0.15)
    sspread = np.maximum(col("gespr_soil_water_0_10m"), 0.002)
    swspread = np.maximum(col("gespr_shortwave_down_w_m2"), 5.0)
    rainspread = np.maximum(col("gespr_precipitation_6h_mm"), 0.05)

    if stochastic:
        temp_floor, rain6_floor, sw_floor = lead_error_floors(leads)
        tspread = np.hypot(tspread, temp_floor)
        rainspread = np.hypot(rainspread, rain6_floor)
        swspread = np.hypot(swspread, sw_floor)

    init_row = history.loc[:issue_bj].iloc[-1]
    init_src_temp = float(init_row.source_temperature)
    init_soil = float(init_row.smap_sm_rootzone) if np.isfinite(init_row.smap_sm_rootzone) else float(init_row.soil_water_1_center)
    src_mean_grid = init_src_temp + (tmean_grid - tmean_grid[0])
    root_mean_grid = init_soil + (smean_grid - smean_grid[0])

    if stochastic:
        z = ar1_errors(n, len(leads), float(CONFIG["weather_ar1"]))
    else:
        z = np.zeros((n, len(leads), 4))
    src_grid = src_mean_grid[None, :] + z[:, :, 0] * tspread[None, :]
    sw_grid = np.maximum(swmean_grid[None, :] + z[:, :, 1] * swspread[None, :], 0)
    rain_grid = np.maximum(rainmean_grid[None, :] + z[:, :, 2] * rainspread[None, :], 0)
    soil_grid = np.clip(root_mean_grid[None, :] + z[:, :, 3] * sspread[None, :], 0.02, 0.70)

    temp[:, future_mask] = interp_rows(future_h, leads, src_grid)
    sw[:, future_mask] = interp_rows(future_h, leads, sw_grid)
    soil[:, future_mask] = interp_rows(future_h, leads, soil_grid)
    # Each f006/f012... precipitation value describes its preceding six-hour interval.
    future_positions = np.flatnonzero(future_mask)
    for pos, hour in zip(future_positions, future_h):
        k = int(np.searchsorted(leads, hour, side="left"))
        k = min(max(k, 1), len(leads) - 1)
        rain[:, pos] = rain_grid[:, k] / 6.0

    p = CONFIG["eti_model"]
    if stochastic:
        tf = RNG.uniform(*p["temperature_factor_range"], size=(n, 1))
        srf = RNG.uniform(*p["shortwave_factor_range"], size=(n, 1))
        t0 = RNG.uniform(*p["temperature_threshold_range_c"], size=(n, 1))
        alb_delta = RNG.normal(0, p["albedo_sd"], size=(n, 1))
    else:
        tf = np.full((n, 1), p["central_temperature_factor"])
        srf = np.full((n, 1), p["central_shortwave_factor"])
        t0 = np.full((n, 1), p["central_temperature_threshold_c"])
        alb_delta = np.zeros((n, 1))
    albedo = np.clip(albedo + alb_delta, 0.05, 0.95)
    melt = np.where(temp > t0, tf * np.maximum(temp - t0, 0) + srf * (1 - albedo) * np.maximum(sw, 0), 0)

    eti_roll = {w: rolling_sum(melt, w) for w in WINDOWS}
    rain_roll = {w: rolling_sum(rain, w) for w in WINDOWS}
    soil_roll = {w: rolling_mean(soil, w) for w in WINDOWS}
    p15 = rolling_sum(rain, 360) * rain_scale

    # Retrospective event forecast window: from issue time through 24 h after
    # the verified onset, bounded by the available forecast horizon.  Without
    # this target window, a 24 h issue would be allowed to select a risk peak a
    # week after the event and its interval would have no operational meaning.
    eval_times = timeline[(timeline >= issue_bj) & (timeline <= EVENT_TRIGGER_BJ + pd.Timedelta(hours=24))]
    eval_pos = timeline.get_indexer(eval_times)
    if np.any(eval_pos < 0):
        raise RuntimeError("Forecast evaluation time not found on hourly timeline")

    score_rows: list[pd.DataFrame] = []
    trigger_rows: list[pd.DataFrame] = []
    models = [("main", 72, 72, 120)] + [(f"uniform_{w}", w, w, w) for w in WINDOWS]
    for model, thermal_w, rain_w, soil_w in models:
        thermal = cdf(refs[f"thermal_{thermal_w}"], eti_roll[thermal_w][:, eval_pos])
        short_rain = cdf(refs[f"rain_{rain_w}"], rain_roll[rain_w][:, eval_pos])
        antecedent_rain = cdf(refs["p15"], p15[:, eval_pos])
        rootzone = cdf(refs[f"soil_{soil_w}"], soil_roll[soil_w][:, eval_pos])
        moisture = np.nanmedian(np.stack([antecedent_rain, rootzone]), axis=0)
        rain_moisture = np.sqrt(np.maximum(short_rain, 0) * np.maximum(moisture, 0))
        dynamic = 1 - (1 - thermal) * (1 - rain_moisture)
        joint_risk = np.sqrt(source_static_hotspot * dynamic)
        threshold = thresholds[model]
        persistence_hours = int(CONFIG["persistence_steps_6h"]) * 6
        warning_onset = first_persistent_crossing(eval_times, joint_risk, threshold, persistence_hours)
        trigger = persistent_peak_time(eval_times, joint_risk, threshold, persistence_hours)
        trigger_rows.append(
            pd.DataFrame(
                {
                    "nominal_issue_h": nominal_h,
                    "issue_time_beijing": issue_bj,
                    "model": model,
                    "scenario": np.arange(n),
                    "warning_onset_beijing": pd.to_datetime(warning_onset),
                    "trigger_time_beijing": pd.to_datetime(trigger),
                    "threshold": threshold,
                    "stochastic": stochastic,
                }
            )
        )
        if model == "main":
            score_rows.append(
                pd.DataFrame(
                    {
                        "nominal_issue_h": nominal_h,
                        "issue_time_beijing": issue_bj,
                        "scenario": np.repeat(np.arange(n), len(eval_times)),
                        "valid_time_beijing": np.tile(eval_times, n),
                        "thermal_score": thermal.ravel(),
                        "short_rain_score": short_rain.ravel(),
                        "antecedent_moisture_score": moisture.ravel(),
                        "rain_moisture_score": rain_moisture.ravel(),
                        "dynamic_score": dynamic.ravel(),
                        "source_joint_risk": joint_risk.ravel(),
                        "threshold": threshold,
                        "stochastic": stochastic,
                    }
                )
            )
    return pd.concat(trigger_rows, ignore_index=True), pd.concat(score_rows, ignore_index=True), pd.DataFrame(
        {"nominal_issue_h": [nominal_h], "issue_time_beijing": [issue_bj], "actual_lead_h": [(EVENT_TRIGGER_BJ - issue_bj) / pd.Timedelta(hours=1)]}
    )


def summarize_triggers(draws: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in draws.groupby(["nominal_issue_h", "issue_time_beijing", "model", "stochastic", "threshold"]):
        nominal, issue, model, stochastic, threshold = keys
        valid = group.trigger_time_beijing.dropna().sort_values()
        row = {
            "nominal_issue_h": nominal,
            "issue_time_beijing": issue,
            "actual_lead_h": (EVENT_TRIGGER_BJ - issue) / pd.Timedelta(hours=1),
            "model": model,
            "stochastic": stochastic,
            "threshold": threshold,
            "scenario_n": len(group),
            "trigger_support": len(valid) / len(group),
            "warning_at_issue_share": float(
                (group.warning_onset_beijing == pd.Timestamp(issue)).fillna(False).mean()
            ),
            "trigger_q025": valid.quantile(0.025) if len(valid) else pd.NaT,
            "trigger_median": valid.quantile(0.5) if len(valid) else pd.NaT,
            "trigger_q975": valid.quantile(0.975) if len(valid) else pd.NaT,
        }
        row["interval_width_h"] = (row["trigger_q975"] - row["trigger_q025"]) / pd.Timedelta(hours=1) if len(valid) else np.nan
        row["covers_true_trigger"] = bool(row["trigger_q025"] <= EVENT_TRIGGER_BJ <= row["trigger_q975"]) if len(valid) else False
        # Hourly interpolation smooths the mechanism score, but the downloaded
        # GEFS fields contain no information inside their native six-hour step.
        # Preserve the raw scenario interval and also report a resolution-aware
        # operational interval expanded by half a native step at each end.
        row["native_resolution_h"] = FORECAST_NATIVE_STEP_H
        row["operational_q025"] = row["trigger_q025"] - pd.Timedelta(hours=RESOLUTION_HALF_STEP_H) if len(valid) else pd.NaT
        row["operational_q975"] = row["trigger_q975"] + pd.Timedelta(hours=RESOLUTION_HALF_STEP_H) if len(valid) else pd.NaT
        row["operational_width_h"] = row["interval_width_h"] + FORECAST_NATIVE_STEP_H if len(valid) else np.nan
        row["covers_true_operational"] = bool(
            row["operational_q025"] <= EVENT_TRIGGER_BJ <= row["operational_q975"]
        ) if len(valid) else False
        row["median_warning_lead_h"] = (row["trigger_median"] - issue) / pd.Timedelta(hours=1) if len(valid) else np.nan
        row["median_abs_error_h"] = abs((row["trigger_median"] - EVENT_TRIGGER_BJ) / pd.Timedelta(hours=1)) if len(valid) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "stochastic", "nominal_issue_h"], ascending=[True, True, False])


def main() -> None:
    for path in [FC_PATH, Q2_DYNAMIC, MCD43, SPATIAL_GRID, GFS_RELIABILITY]:
        if not path.exists():
            raise FileNotFoundError(path)
    history = add_historical_features(attach_mcd43(pd.read_parquet(Q2_DYNAMIC).sort_index()))
    if 2025 in set(history.index.year):
        raise RuntimeError("2025 leaked into the formal background")
    bg_scores, refs, thresholds, rain_scale, source_static_hotspot = build_references(history)
    bg_scores.to_parquet(PROCESSED / "02_历史同期触发评分参照.parquet")
    pd.DataFrame([{"model": k, "q80_threshold": v} for k, v in thresholds.items()]).to_csv(
        OUTPUTS / "03_触发阈值.csv", index=False, encoding="utf-8-sig"
    )

    cache = pd.read_parquet(FC_PATH)
    all_triggers = []
    all_scores = []
    issue_audit = []
    for nominal_text, issue_text in CONFIG["nominal_issue_hours"].items():
        nominal = int(nominal_text)
        issue_utc = pd.Timestamp(issue_text)
        fc = forecast_run(cache, issue_utc)
        for stochastic, n in [(False, 1), (True, N_DRAWS)]:
            triggers, scores, audit = simulate_issue(
                issue_utc, nominal, fc, history, refs, thresholds, rain_scale, source_static_hotspot, n, stochastic
            )
            all_triggers.append(triggers)
            all_scores.append(scores)
            issue_audit.append(audit.assign(stochastic=stochastic, forecast_end_beijing=fc.valid_time_beijing.max()))

    trigger_draws = pd.concat(all_triggers, ignore_index=True)
    score_draws = pd.concat(all_scores, ignore_index=True)
    trigger_draws.to_parquet(PROCESSED / "03_触发时间情景样本.parquet", index=False)
    score_draws.to_parquet(PROCESSED / "04_逐时风险情景样本.parquet", index=False)
    summary = summarize_triggers(trigger_draws)
    summary.to_csv(OUTPUTS / "04_触发时间区间与覆盖.csv", index=False, encoding="utf-8-sig")

    risk_summary = (
        score_draws.loc[score_draws.stochastic]
        .groupby(["nominal_issue_h", "issue_time_beijing", "valid_time_beijing"])
        .source_joint_risk.agg(
            risk_q025=lambda x: x.quantile(0.025),
            risk_median="median",
            risk_q975=lambda x: x.quantile(0.975),
            risk_mean="mean",
        )
        .reset_index()
    )
    risk_summary.to_csv(OUTPUTS / "05_集合风险演化.csv", index=False, encoding="utf-8-sig")
    pd.concat(issue_audit, ignore_index=True).drop_duplicates().to_csv(OUTPUTS / "06_五起报时效审计.csv", index=False, encoding="utf-8-sig")

    main_summary = summary.loc[summary.model.eq("main")].copy()
    report = f"""# 第三问触发时间预报结果

- 方法：第二问机理评分持续越阈决定预警起点，未来持续风险峰值决定触发窗口中心；不使用普通 Logistic 分类。
- 背景：2021—2024；2025未参与背景、阈值或参数选择。
- 主模型：MCD43-ETI 72 h、短时降水72 h、根区湿润状态120 h和15日前期降水。
- 空间门控：源区AOI内部静态易发性第90百分位（{source_static_hotspot:.3f}），避免用单一最大像元。
- 阈值：2021—2024历史时空联合风险的第{int(CONFIG['trigger_reference_quantile']*100)}百分位，且连续{CONFIG['persistence_steps_6h']*6}小时越阈。
- 真实启动：{EVENT_TRIGGER_BJ}（目录时刻后{CONFIG['event_first_detachment_offset_s']} s）；10:30初报未被当作真值。
- 95%区间：天气集合、ETI参数和反照率不确定性共同形成的情景分位区间；不是多事件频率学保证。
- 结构误差下限：第二问GFS分时效核查得到的温度、降水和短波MAE以均方根方式并入GEFS离散度；只扩充不确定性，不把GFS偏差硬套到GEFS均值。
- 操作区间：在原始情景95%区间两端各增加3小时，仅反映GEFS原始6小时分辨率的半步不确定性；原始区间与操作区间同时保留，不能只报告较宽者。
- 若起报时刻已经越阈，首次越阈只表示“应立即预警”，不会被误写为真实物理触发；触发时间另由未来风险峰值给出。

## 主模型汇总

{main_summary.to_string(index=False)}

## 解释边界

GEFS 2 m气温只提供未来变化和集合离散度，源区温度中心从起报时刻的源区订正温度锚定；未把粗网格2 m气温直接当成4991 m源区绝对温度。VPD与位温梯度因当前集合预报变量不足，保留为起报时刻监测协变量，不被虚构为未来可知量。
"""
    (OUTPUTS / "07_触发模型结果说明.md").write_text(report, encoding="utf-8")
    print(main_summary.to_string(index=False))


if __name__ == "__main__":
    main()
