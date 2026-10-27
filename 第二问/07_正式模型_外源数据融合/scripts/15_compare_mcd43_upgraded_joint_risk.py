from __future__ import annotations

"""将 MCD43 反照率的 ETI 实验结果传递到统一时空风险，与旧回退版对照。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_DIR = Path(__file__).resolve().parent.parent
OUTPUTS = MODEL_DIR / "outputs"
PROCESSED = MODEL_DIR / "processed"
CONFIG = json.loads((MODEL_DIR / "config" / "model_config.json").read_text(encoding="utf-8"))
EVENT = pd.Timestamp(CONFIG["event_last_complete_hour_beijing"])


def cdf(reference: pd.Series, value: pd.Series | float) -> np.ndarray | float:
    ref = np.sort(reference.dropna().to_numpy(float))
    if np.isscalar(value):
        return float(np.searchsorted(ref, float(value), side="right") / len(ref))
    x = pd.Series(value).to_numpy(float)
    out = np.full(x.shape, np.nan)
    valid = np.isfinite(x)
    out[valid] = np.searchsorted(ref, x[valid], side="right") / len(ref)
    return out


def attach_mcd43_eti(frame: pd.DataFrame) -> pd.DataFrame:
    modis = pd.read_csv(OUTPUTS / "37_MODIS反照率与地表温度源区时序.csv", parse_dates=["date"])
    available = modis[["date", "albedo"]].dropna().sort_values("date")
    available["available_date"] = available.date + pd.Timedelta(days=1)
    days = pd.DataFrame({"day": pd.to_datetime(frame.index.normalize().unique())})
    mapping = pd.merge_asof(days.sort_values("day"), available.sort_values("available_date"), left_on="day", right_on="available_date", direction="backward", tolerance=pd.Timedelta(days=7)).set_index("day")
    out = frame.copy()
    out["day"] = out.index.normalize()
    out = out.join(mapping[["albedo"]], on="day")
    out["albedo_mcd43"] = out.albedo.where(out.albedo.between(.05, .95), out.albedo_operational)
    p = CONFIG["eti_model"]
    sw = out.sw_down_center.clip(lower=0) / 3600
    te = (out.source_temperature - p["temperature_threshold_c"]).clip(lower=0)
    hourly = np.where(out.source_temperature > p["temperature_threshold_c"], p["temperature_factor_mm_h_c"] * te + p["shortwave_factor_mm_h_wm2"] * (1 - out.albedo_mcd43) * sw, 0)
    out["eti_mcd43_72h_mm"] = pd.Series(hourly, index=out.index).groupby(out.index.year).rolling(72, min_periods=72).sum().reset_index(level=0, drop=True)
    return out


def seasonal(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame.index.year.isin(CONFIG["background_years"]) & (frame.index.hour == EVENT.hour) & (np.abs(frame.index.dayofyear - EVENT.dayofyear) <= 15)].copy()


def score(frame: pd.DataFrame, thermal_feature: str) -> tuple[pd.DataFrame, dict[str, float]]:
    bg = seasonal(frame)
    event = frame.loc[EVENT]
    features = {"thermal": thermal_feature, "era_rain": "era5_rain_72h_mm", "imerg_rain": "imerg_rain_72h_mm", "antecedent_rain": "chirps_rain_prev_15d_mm", "rootzone_moisture": "smap_rootzone_mean_72h"}
    ranked = pd.DataFrame(index=bg.index)
    e = {}
    for key, col in features.items():
        ranked[key] = cdf(bg[col], bg[col])
        e[key] = cdf(bg[col], float(event[col]))
    ranked["short_rainfall"] = ranked[["era_rain", "imerg_rain"]].median(axis=1)
    ranked["antecedent_moisture"] = ranked[["antecedent_rain", "rootzone_moisture"]].median(axis=1)
    ranked["rain_moisture"] = np.sqrt(ranked.short_rainfall * ranked.antecedent_moisture)
    ranked["dynamic"] = 1 - (1 - ranked.thermal) * (1 - ranked.rain_moisture)
    e["short_rainfall"] = float(np.median([e["era_rain"], e["imerg_rain"]]))
    e["antecedent_moisture"] = float(np.median([e["antecedent_rain"], e["rootzone_moisture"]]))
    e["rain_moisture"] = float(np.sqrt(e["short_rainfall"] * e["antecedent_moisture"]))
    e["dynamic"] = float(1 - (1 - e["thermal"]) * (1 - e["rain_moisture"]))
    return ranked, e


def thresholds(static: np.ndarray, dynamic: np.ndarray) -> dict[str, float]:
    ref = np.concatenate([np.sqrt(static * d) for d in dynamic[np.isfinite(dynamic)]])
    return {f"q{int(q*100)}": float(np.quantile(ref, q)) for q in [.5, .8, .9, .95]}


def classify(values: np.ndarray, t: dict[str, float]) -> np.ndarray:
    return np.select([values >= t["q95"], values >= t["q80"], values >= t["q50"]], ["极高", "高", "中"], default="低")


def summarize(name: str, spatial: pd.DataFrame, risk: np.ndarray, levels: np.ndarray, e: dict[str, float], t: dict[str, float]) -> dict:
    source = spatial.event_source_aoi_label.astype(bool).to_numpy()
    slevels = pd.Series(levels[source]).value_counts(normalize=True)
    return {"model": name, "thermal_score": e["thermal"], "rain_moisture_score": e["rain_moisture"], "dynamic_score": e["dynamic"], "q80_threshold": t["q80"], "q95_threshold": t["q95"], "source_mean_risk": float(np.mean(risk[source])), "source_high_share": float(np.mean(np.isin(levels[source], ["高", "极高"]))), "source_extreme_share": float(slevels.get("极高", 0))}


def main() -> None:
    frame = attach_mcd43_eti(pd.read_parquet(PROCESSED / "04_多源动态触发逐小时缓存.parquet").sort_index())
    spatial = pd.read_parquet(PROCESSED / "07_空间易发性评分格网.parquet").copy()
    spatial["static_pct"] = spatial.static_mechanism_score.rank(pct=True)
    s = spatial.static_pct.to_numpy(float)
    old_bg, old_e = score(frame, "eti_melt_72h_mm")
    new_bg, new_e = score(frame, "eti_mcd43_72h_mm")
    old_t, new_t = thresholds(s, old_bg.dynamic.to_numpy()), thresholds(s, new_bg.dynamic.to_numpy())
    old_r, new_r = np.sqrt(s * old_e["dynamic"]), np.sqrt(s * new_e["dynamic"])
    old_l, new_l = classify(old_r, old_t), classify(new_r, new_t)
    spatial["risk_era5_albedo_control"] = old_r
    spatial["level_era5_albedo_control"] = old_l
    spatial["risk_mcd43_albedo_upgraded"] = new_r
    spatial["level_mcd43_albedo_upgraded"] = new_l
    spatial.to_parquet(PROCESSED / "13_MCD43升级时空风险对照格网.parquet", index=False)
    summary = pd.DataFrame([summarize("ERA5_albedo_control", spatial, old_r, old_l, old_e, old_t), summarize("MCD43_albedo_upgraded", spatial, new_r, new_l, new_e, new_t)])
    summary.to_csv(OUTPUTS / "52_MCD43升级统一风险对照.csv", index=False, encoding="utf-8-sig")

    mc = pd.read_parquet(PROCESSED / "12_MCD43_ETI参数蒙特卡洛.parquet")
    mc["dynamic"] = 1 - (1 - mc.event_risk_percentile) * (1 - new_e["rain_moisture"])
    source_median_s = float(spatial.loc[spatial.event_source_aoi_label.eq(1), "static_pct"].median())
    mc["risk"] = np.sqrt(source_median_s * mc.dynamic)
    mc["level"] = classify(mc.risk.to_numpy(), new_t)
    freq = mc.level.value_counts(normalize=True).rename_axis("risk_level").reset_index(name="share")
    freq.to_csv(OUTPUTS / "53_MCD43升级综合风险蒙特卡洛等级.csv", index=False, encoding="utf-8-sig")
    report = f"""# MCD43 升级后的统一时空风险对照

{summary.to_string(index=False)}

MCD43 不仅提高 ETI 中心值，也同步重建了 2021—2024 同期动态参照分布，因此这里的等级与连续风险仍共用同一套判断规则。

参数 Monte Carlo 下源区中位空间位置的等级频率：{dict(zip(freq.risk_level, freq.share))}。
"""
    (OUTPUTS / "54_MCD43升级统一风险报告.md").write_text(report, encoding="utf-8")
    print(summary.to_string(index=False))
    print(freq.to_string(index=False))


if __name__ == "__main__":
    main()
