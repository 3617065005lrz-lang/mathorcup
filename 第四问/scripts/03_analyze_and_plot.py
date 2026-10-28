from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from rasterio.features import shapes
from rasterio.transform import from_origin
from scipy.stats import spearmanr


HERE = Path(__file__).resolve()
Q4 = HERE.parents[1]
OUT = Q4 / "outputs"
FIG = Q4 / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RASTER = OUT / "rasters"
RASTER.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    f60 = pd.read_csv(OUT / "scenario_metrics_60m.csv")
    all30 = OUT / "scenario_metrics_30m_all.csv"
    f30 = pd.read_csv(all30 if all30.exists() else OUT / "scenario_metrics_30m.csv")
    weights = pd.read_csv(OUT / "M4_GLUE_weights.csv")
    with np.load(Q4 / "cache" / "prepared_30m.npz") as src:
        prepared = {name: src[name].copy() for name in src.files}
    cfg = json.loads((Q4 / "config" / "model_config.json").read_text(encoding="utf-8"))
    return f60, f30, weights, prepared, cfg


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) / weights.sum()
    return float(np.interp(q, cumulative, values))


def scenario_arrays(scenario_id: str) -> dict[str, np.ndarray]:
    with np.load(OUT / "scenarios_30m" / f"{scenario_id}.npz") as src:
        return {name: src[name].copy() for name in src.files}


def write_geotiff(path: Path, array: np.ndarray, prepared: dict, nodata: float | None = np.nan) -> None:
    transform = from_origin(
        float(prepared["xmin"]),
        float(prepared["ymax"]),
        float(prepared["cell_m"]),
        float(prepared["cell_m"]),
    )
    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32645",
        "transform": transform,
        "compress": "deflate",
        "predictor": 2,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(np.float32), 1)


def write_runout_geojson(path: Path, mask: np.ndarray, prepared: dict) -> None:
    transform = from_origin(
        float(prepared["xmin"]),
        float(prepared["ymax"]),
        float(prepared["cell_m"]),
        float(prepared["cell_m"]),
    )
    features = []
    for geom, value in shapes(mask.astype(np.uint8), mask=mask, transform=transform):
        if int(value) == 1:
            features.append({"type": "Feature", "properties": {"depth_threshold_m": 0.1}, "geometry": geom})
    value = {
        "type": "FeatureCollection",
        "name": "best_M3_runout_depth_ge_0p1m",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32645"}},
        "features": features,
    }
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def write_zone_geojson(path: Path, zones: np.ndarray, prepared: dict) -> None:
    transform = from_origin(
        float(prepared["xmin"]), float(prepared["ymax"]), float(prepared["cell_m"]), float(prepared["cell_m"])
    )
    labels = {1: "外包络区_p_ge_0.025", 2: "主要可能区_p_ge_0.5", 3: "核心区_p_ge_0.975"}
    features = []
    for geom, value in shapes(zones.astype(np.uint8), mask=zones > 0, transform=transform):
        code = int(value)
        features.append({"type": "Feature", "properties": {"zone_code": code, "zone_name": labels[code]}, "geometry": geom})
    collection = {
        "type": "FeatureCollection",
        "name": "M4_conditional_hazard_zones",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32645"}},
        "features": features,
    }
    path.write_text(json.dumps(collection, ensure_ascii=False), encoding="utf-8")


def parameter_sensitivity(frame60: pd.DataFrame) -> pd.DataFrame:
    part = frame60[frame60["model"].isin(["M2_zoned_hypermobile", "M3_zoned_entrainment"])].copy()
    parameters = [
        "mu_source",
        "xi_source",
        "mu_valley",
        "xi_valley",
        "erosion_k",
        "erosion_ucrit",
        "erodible_depth",
        "volume_scale",
        "release_duration_s",
        "release_speed_m_s",
    ]
    targets = ["score", "arrival_time_s", "calibration_csi", "holdout_csi"]
    rows = []
    for parameter in parameters:
        for target in targets:
            valid = part[[parameter, target]].dropna()
            if valid[parameter].nunique() < 3 or len(valid) < 8:
                rho, p = np.nan, np.nan
            else:
                rho, p = spearmanr(valid[parameter], valid[target])
            rows.append({"parameter": parameter, "target": target, "spearman_rho": rho, "p_value": p, "n": len(valid)})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "parameter_sensitivity_spearman.csv", index=False, encoding="utf-8-sig")
    return result


def summaries(f60: pd.DataFrame, f30: pd.DataFrame, weights: pd.DataFrame, prepared: dict, cfg: dict) -> tuple[str, dict]:
    best_cal = f30.sort_values("score", ascending=False).iloc[0]
    behavior_ids = set(weights["scenario_id"].astype(str))
    candidates = f30[f30["scenario_id"].astype(str).isin(behavior_ids) & f30["model"].str.startswith("M3")].copy()
    candidates["robust_selection_score"] = 0.70 * candidates["score"] + 0.30 * candidates["holdout_csi"]
    best_m3 = candidates.sort_values("robust_selection_score", ascending=False).iloc[0]
    best_hold = f30.sort_values("holdout_csi", ascending=False).iloc[0]
    selected = str(best_m3["scenario_id"])
    arrays = scenario_arrays(selected)
    r, c = int(prepared["port_row"]), int(prepared["port_col"])
    weights_by_id = weights.set_index("scenario_id")["glue_weight"]
    port_rows = []
    for _, row in f30.iterrows():
        sid = str(row["scenario_id"])
        data = scenario_arrays(sid)
        port_rows.append(
            {
                "scenario_id": sid,
                "model": row["model"],
                "arrival_time_s": float(data["arrival_time_s"][r, c]),
                "max_depth_at_port_m": float(data["hmax_m"][r, c]),
                "max_speed_at_port_m_s": float(data["vmax_m_s"][r, c]),
                "max_impact_intensity_at_port_m3_s2": float(data["max_impact_intensity_m3_s2"][r, c]),
                "max_dynamic_pressure_proxy_at_port_mpa": float(data["max_dynamic_pressure_pa"][r, c] / 1.0e6),
                "erosion_depth_at_port_m": float(data["erosion_depth_m"][r, c]),
                "glue_weight": float(weights_by_id.get(sid, 0.0)),
            }
        )
    port = pd.DataFrame(port_rows)
    port.to_csv(OUT / "port_impact_by_scenario.csv", index=False, encoding="utf-8-sig")
    grid = f30[["scenario_id", "arrival_time_s", "calibration_csi", "holdout_csi"]].merge(
        f60[["scenario_id", "arrival_time_s", "calibration_csi", "holdout_csi"]],
        on="scenario_id",
        suffixes=("_30m", "_60m"),
    )
    grid["arrival_delta_30m_minus_60m_s"] = grid["arrival_time_s_30m"] - grid["arrival_time_s_60m"]
    grid["abs_arrival_delta_s"] = grid["arrival_delta_30m_minus_60m_s"].abs()
    grid.to_csv(OUT / "grid_30m_60m_sensitivity.csv", index=False, encoding="utf-8-sig")
    w = port["glue_weight"].to_numpy()
    arr = port["arrival_time_s"].to_numpy()
    dep = port["max_depth_at_port_m"].to_numpy()
    vel = port["max_speed_at_port_m_s"].to_numpy()
    impact = port["max_impact_intensity_at_port_m3_s2"].to_numpy()
    pressure = port["max_dynamic_pressure_proxy_at_port_mpa"].to_numpy()
    event_uncertainty = float(cfg["event"]["arrival_clock_uncertainty_s"])
    uncertainty = {
        "selected_main_scenario": selected,
        "selection_reason": "参数搜索与GLUE权重只使用到时和上游校准区；在固定行为集合后，以0.70×校准得分+0.30×下游留出CSI选择主情景，兼顾拟合与空间迁移。",
        "robust_selection_score": float(best_m3["robust_selection_score"]),
        "best_calibration_scenario": str(best_cal["scenario_id"]),
        "best_holdout_scenario": str(best_hold["scenario_id"]),
        "ensemble_arrival_weighted_mean_s": float(np.sum(w * arr)),
        "ensemble_arrival_p10_s": weighted_quantile(arr, w, 0.10),
        "ensemble_arrival_p90_s": weighted_quantile(arr, w, 0.90),
        "observation_aware_arrival_interval_s": [
            float(weighted_quantile(arr, w, 0.10) - event_uncertainty),
            float(weighted_quantile(arr, w, 0.90) + event_uncertainty),
        ],
        "ensemble_port_depth_weighted_mean_m": float(np.sum(w * dep)),
        "ensemble_port_speed_weighted_mean_m_s": float(np.sum(w * vel)),
        "ensemble_port_impact_intensity_weighted_mean_m3_s2": float(np.sum(w * impact)),
        "ensemble_port_dynamic_pressure_proxy_weighted_mean_mpa": float(np.sum(w * pressure)),
        "target_travel_time_s": float(cfg["event"]["target_travel_time_s"]),
        "grid_30m_60m_arrival_delta_range_s": [
            float(grid["arrival_delta_30m_minus_60m_s"].min()),
            float(grid["arrival_delta_30m_minus_60m_s"].max()),
        ],
        "median_absolute_grid_arrival_delta_s": float(grid["abs_arrival_delta_s"].median()),
        "grid_note": "30/60 m同时更换了地形离散网格，因此该差值是分辨率—地形离散联合敏感性，不能只归因于像元大小。",
        "evidence_note": "集合离散范围仅表示保留情景差异；观测感知区间另叠加CCTV时钟约±60 s，不是严格贝叶斯可信区间。",
    }
    (OUT / "uncertainty_summary.json").write_text(json.dumps(uncertainty, ensure_ascii=False, indent=2), encoding="utf-8")

    # M0-M4对比表：M0没有流深和到时，不能与动力学模型假装同一分数。
    grouped = f60.groupby("model").agg(
        scenarios=("scenario_id", "count"),
        arrived=("arrival_time_s", lambda x: int(x.notna().sum())),
        best_score=("score", "max"),
        median_score=("score", "median"),
        best_calibration_csi=("calibration_csi", "max"),
        best_holdout_csi=("holdout_csi", "max"),
    ).reset_index()
    model_table = pd.concat(
        [
            pd.DataFrame(
                [{
                    "model": "M0_path_only",
                    "scenarios": 1,
                    "arrived": 0,
                    "best_score": np.nan,
                    "median_score": np.nan,
                    "best_calibration_csi": np.nan,
                    "best_holdout_csi": np.nan,
                    "role": "只检验沟谷路径连通，不输出流深、速度或到时",
                }]
            ),
            grouped.assign(role="60 m参数筛选"),
            pd.DataFrame(
                [{
                    "model": "M4_GLUE_ensemble",
                    "scenarios": len(weights),
                    "arrived": len(weights),
                    "best_score": float(f30["score"].max()),
                    "median_score": float(f30["score"].median()),
                    "best_calibration_csi": float(f30["calibration_csi"].max()),
                    "best_holdout_csi": float(f30["holdout_csi"].max()),
                    "role": "30 m有效情景集合与不确定性表达",
                }]
            ),
        ],
        ignore_index=True,
    )
    model_table.to_csv(OUT / "M0_M4_model_comparison.csv", index=False, encoding="utf-8-sig")

    # 第三问回接：不把条件仿真强行转成“真实发生概率”。
    back = pd.DataFrame(
        [{
            "scenario_id": selected,
            "scenario_role": "第四问主情景（M3）",
            "scenario_weight": float(weights_by_id.get(selected, 0.0)),
            "mu_source": float(best_m3["mu_source"]),
            "xi_source_m_s2": float(best_m3["xi_source"]),
            "mu_valley": float(best_m3["mu_valley"]),
            "xi_valley_m_s2": float(best_m3["xi_valley"]),
            "release_volume_m3": float(best_m3["initial_volume_m3"]),
            "entrained_volume_m3": float(best_m3["entrained_volume_m3"]),
            "port_arrival_time_s": float(best_m3["arrival_time_s"]),
            "calibration_csi": float(best_m3["calibration_csi"]),
            "holdout_csi": float(best_m3["holdout_csi"]),
            "max_depth_raster": "rasters/best_M3_max_depth_m.tif",
            "max_velocity_raster": "rasters/best_M3_max_velocity_m_s.tif",
            "arrival_time_raster": "rasters/best_M3_arrival_time_s.tif",
            "runout_polygon": "rasters/best_M3_runout_depth_ge_0p1m.geojson",
            "impact_intensity_raster": "rasters/best_M3_max_impact_intensity_m3_s2.tif",
            "dynamic_pressure_proxy_raster": "rasters/best_M3_max_dynamic_pressure_pa.tif",
            "erosion_depth_raster": "rasters/best_M3_erosion_depth_m.tif",
            "probability_raster_note": "M4栅格为条件情景超阈概率，不是泥石流发生概率。",
        }]
    )
    back.to_csv(OUT / "Q3_backconnect_summary.csv", index=False, encoding="utf-8-sig")
    return selected, arrays


def export_spatial(selected: str, arrays: dict, prepared: dict) -> dict:
    with np.load(OUT / "M4_ensemble_30m.npz", allow_pickle=True) as src:
        ensemble = {name: src[name].copy() for name in src.files if src[name].ndim == 2}
    active = prepared["active"].astype(bool)
    h = np.where(active, arrays["hmax_m"], np.nan)
    v = np.where(active, arrays["vmax_m_s"], np.nan)
    t = np.where(active, arrays["arrival_time_s"], np.nan)
    write_geotiff(RASTER / "best_M3_max_depth_m.tif", h, prepared)
    write_geotiff(RASTER / "best_M3_max_velocity_m_s.tif", v, prepared)
    write_geotiff(RASTER / "best_M3_arrival_time_s.tif", t, prepared)
    write_geotiff(RASTER / "best_M3_max_impact_intensity_m3_s2.tif", np.where(active, arrays["max_impact_intensity_m3_s2"], np.nan), prepared)
    write_geotiff(RASTER / "best_M3_max_dynamic_pressure_pa.tif", np.where(active, arrays["max_dynamic_pressure_pa"], np.nan), prepared)
    write_geotiff(RASTER / "best_M3_erosion_depth_m.tif", np.where(active, arrays["erosion_depth_m"], np.nan), prepared)
    write_geotiff(RASTER / "M4_probability_depth_ge_0p1m.tif", np.where(active, ensemble["probability_depth_ge_0p1m"], np.nan), prepared)
    write_geotiff(RASTER / "M4_probability_depth_ge_0p5m.tif", np.where(active, ensemble["probability_depth_ge_0p5m"], np.nan), prepared)
    write_geotiff(RASTER / "M4_probability_speed_ge_5m_s.tif", np.where(active, ensemble["probability_speed_ge_5m_s"], np.nan), prepared)
    for key, filename in [
        ("arrival_median_s", "M4_arrival_median_s.tif"),
        ("arrival_p10_s", "M4_arrival_p10_s.tif"),
        ("arrival_p90_s", "M4_arrival_p90_s.tif"),
    ]:
        write_geotiff(RASTER / filename, np.where(active, ensemble[key], np.nan), prepared)
    write_runout_geojson(RASTER / "best_M3_runout_depth_ge_0p1m.geojson", arrays["hmax_m"] >= 0.1, prepared)
    p = ensemble["probability_depth_ge_0p5m"]
    zones = np.zeros(p.shape, np.uint8)
    zones[(p >= 0.025) & active] = 1
    zones[(p >= 0.50) & active] = 2
    zones[(p >= 0.975) & active] = 3
    write_geotiff(RASTER / "M4_conditional_hazard_zones.tif", np.where(active, zones, np.nan), prepared)
    write_zone_geojson(RASTER / "M4_conditional_hazard_zones.geojson", zones, prepared)
    cell_area = float(prepared["cell_m"]) ** 2 / 1e6
    area = pd.DataFrame(
        [
            {"zone": "核心区_p_ge_0.975", "area_km2": float(np.count_nonzero(zones == 3) * cell_area)},
            {"zone": "主要可能区_p_ge_0.5", "area_km2": float(np.count_nonzero(zones >= 2) * cell_area)},
            {"zone": "外包络区_p_ge_0.025", "area_km2": float(np.count_nonzero(zones >= 1) * cell_area)},
        ]
    )
    area.to_csv(OUT / "M4_conditional_zone_areas.csv", index=False, encoding="utf-8-sig")
    return ensemble


def backconnect_to_q3() -> None:
    target = Q4.parent / "第三问" / "outputs" / "q4_backconnect"
    target.mkdir(parents=True, exist_ok=True)
    sources = [
        OUT / "Q3_backconnect_summary.csv",
        OUT / "M4_GLUE_weights.csv",
        OUT / "M4_conditional_zone_areas.csv",
        OUT / "uncertainty_summary.json",
        OUT / "grid_30m_60m_sensitivity.csv",
        RASTER / "best_M3_max_depth_m.tif",
        RASTER / "best_M3_max_velocity_m_s.tif",
        RASTER / "best_M3_arrival_time_s.tif",
        RASTER / "best_M3_runout_depth_ge_0p1m.geojson",
        RASTER / "best_M3_max_impact_intensity_m3_s2.tif",
        RASTER / "best_M3_max_dynamic_pressure_pa.tif",
        RASTER / "best_M3_erosion_depth_m.tif",
        RASTER / "M4_probability_depth_ge_0p5m.tif",
        RASTER / "M4_conditional_hazard_zones.tif",
        RASTER / "M4_conditional_hazard_zones.geojson",
        RASTER / "M4_arrival_median_s.tif",
        RASTER / "M4_arrival_p10_s.tif",
        RASTER / "M4_arrival_p90_s.tif",
    ]
    for source in sources:
        shutil.copy2(source, target / source.name)


def plot_all(f60: pd.DataFrame, f30: pd.DataFrame, weights: pd.DataFrame, prepared: dict, selected: str, arrays: dict, ensemble: dict, sensitivity: pd.DataFrame) -> None:
    # 1 模型级比较
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    order = ["M1_uniform_literature", "M2_zoned_hypermobile", "M3_zoned_entrainment"]
    labels = ["M1 均一基线", "M2 分区摩阻", "M3 分区+卷入"]
    data = [f60.loc[f60.model == name, "score"].dropna() for name in order]
    axes[0].boxplot(data, tick_labels=labels, showfliers=False)
    axes[0].set_ylabel("综合校准得分")
    axes[0].set_title("60 m 参数筛选：模型升级效果")
    arrival = f30.sort_values("arrival_time_s")
    axes[1].barh(arrival["scenario_id"], arrival["arrival_time_s"] / 60, color=["#e68653" if m.startswith("M3") else "#4f86c6" for m in arrival.model])
    axes[1].axvline(463 / 60, color="black", linestyle="--", label="观测约束 7.72 min")
    axes[1].set_xlabel("源区至口岸到达时间（min）")
    axes[1].set_title("30 m 有效情景到时")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "01_M1_M3模型比较.png", bbox_inches="tight")
    plt.close(fig)

    # 2 主情景三联图
    extent = [float(prepared["xmin"]) / 1000, (float(prepared["xmin"]) + arrays["hmax_m"].shape[1] * float(prepared["cell_m"])) / 1000,
              (float(prepared["ymax"]) - arrays["hmax_m"].shape[0] * float(prepared["cell_m"])) / 1000, float(prepared["ymax"]) / 1000]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    layers = [
        (np.where(arrays["hmax_m"] >= 0.1, arrays["hmax_m"], np.nan), "viridis", "最大流深（m）", 0, 60),
        (np.where(arrays["vmax_m_s"] >= 0.5, arrays["vmax_m_s"], np.nan), "magma", "最大速度（m/s）", 0, 120),
        (arrays["arrival_time_s"] / 60, "turbo", "到达时间（min）", 0, 18),
    ]
    for ax, (layer, cmap, title, vmin, vmax) in zip(axes, layers):
        im = ax.imshow(layer, extent=extent, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("UTM Easting（km）")
        fig.colorbar(im, ax=ax, shrink=0.78)
    axes[0].set_ylabel("UTM Northing（km）")
    fig.suptitle(f"M3主情景 {selected}：源区—口岸动力学结果", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "02_M3主情景流深速度到时.png", bbox_inches="tight")
    plt.close(fig)

    # 3 观测范围与模拟范围
    obs = prepared["observed_footprint"].astype(bool)
    sim = arrays["hmax_m"] >= 0.1
    cls = np.zeros(obs.shape, np.uint8)
    cls[obs & ~sim] = 1
    cls[~obs & sim] = 2
    cls[obs & sim] = 3
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = ListedColormap(["white", "#4f86c6", "#e68653", "#4c956c"])
    ax.imshow(cls, extent=extent, origin="upper", cmap=cmap, vmin=0, vmax=3)
    patches = [plt.Rectangle((0, 0), 1, 1, color=color, label=label) for color, label in [("#4f86c6", "仅卫星解译"), ("#e68653", "仅模型"), ("#4c956c", "重合")]]
    ax.legend(handles=patches, loc="lower left", frameon=True)
    ax.set_title("M3模拟淹没范围与UNOSAT初步解译对照")
    ax.set_xlabel("UTM Easting（km）")
    ax.set_ylabel("UTM Northing（km）")
    fig.tight_layout()
    fig.savefig(FIG / "03_模拟与卫星范围对照.png", bbox_inches="tight")
    plt.close(fig)

    # 4 条件超阈概率
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, key, title in [
        (axes[0], "probability_depth_ge_0p1m", "流深≥0.1 m 条件情景比例"),
        (axes[1], "probability_speed_ge_5m_s", "速度≥5 m/s 条件情景比例"),
    ]:
        im = ax.imshow(np.where(prepared["active"], ensemble[key], np.nan), extent=extent, origin="upper", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xlabel("UTM Easting（km）")
        fig.colorbar(im, ax=ax, shrink=0.8)
    axes[0].set_ylabel("UTM Northing（km）")
    fig.suptitle("M4有效情景集合（注意：不是灾害发生概率）")
    fig.tight_layout()
    fig.savefig(FIG / "04_M4条件超阈概率.png", bbox_inches="tight")
    plt.close(fig)

    # 5 参数敏感性
    sens = sensitivity[sensitivity.target == "score"].dropna().copy()
    sens = sens.reindex(sens.spearman_rho.abs().sort_values().index)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = np.where(sens.spearman_rho >= 0, "#4c956c", "#c8553d")
    ax.barh(sens.parameter, sens.spearman_rho, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Spearman ρ（对综合得分）")
    ax.set_title("第四问参数敏感性：方向而非因果贡献")
    fig.tight_layout()
    fig.savefig(FIG / "05_参数敏感性.png", bbox_inches="tight")
    plt.close(fig)

    # 6 沿程结果
    chain = prepared["upper_chainage_m"] / 1000
    bins = np.arange(0, 30.5, 0.5)
    centers = (bins[:-1] + bins[1:]) / 2
    depth90, speed90, arrival50 = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = np.isfinite(chain) & (chain >= lo) & (chain < hi) & (arrays["hmax_m"] >= 0.1)
        depth90.append(np.nanpercentile(arrays["hmax_m"][mask], 90) if mask.any() else np.nan)
        speed90.append(np.nanpercentile(arrays["vmax_m_s"][mask], 90) if mask.any() else np.nan)
        tt = arrays["arrival_time_s"][mask]
        arrival50.append(np.nanmedian(tt) / 60 if np.isfinite(tt).any() else np.nan)
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(centers, depth90, color="#2a9d8f"); axes[0].set_ylabel("P90流深（m）")
    axes[1].plot(centers, speed90, color="#e76f51"); axes[1].set_ylabel("P90速度（m/s）")
    axes[2].plot(centers, arrival50, color="#457b9d"); axes[2].set_ylabel("中位到时（min）"); axes[2].set_xlabel("源区起算沿程距离（km）")
    for ax in axes:
        ax.axvline(float(prepared["port_chainage_m"]) / 1000, color="black", linestyle="--", alpha=0.7)
        ax.grid(alpha=0.2)
    axes[0].set_title("M3主情景沿程动力学演化（虚线为口岸）")
    fig.tight_layout()
    fig.savefig(FIG / "06_沿程流深速度到时.png", bbox_inches="tight")
    plt.close(fig)

    # 7 动态过程快照
    dynamic_path = OUT / "extensions" / f"{selected}_dynamic.npz"
    if dynamic_path.exists():
        with np.load(dynamic_path) as src:
            depth_keys = sorted([k for k in src.files if k.startswith("depth_t")])
            fig, axes = plt.subplots(1, len(depth_keys), figsize=(3.4 * len(depth_keys), 4.5), sharex=True, sharey=True)
            if len(depth_keys) == 1:
                axes = [axes]
            for ax, key in zip(axes, depth_keys):
                layer = src[key]
                im = ax.imshow(np.where(layer >= 0.1, layer, np.nan), extent=extent, origin="upper", cmap="inferno", vmin=0, vmax=60)
                seconds = int(key.rsplit("_t", 1)[1].rstrip("s"))
                ax.set_title(f"t={seconds} s")
                ax.set_xlabel("Easting（km）")
            axes[0].set_ylabel("Northing（km）")
            fig.colorbar(im, ax=axes, shrink=0.72, label="流深（m）")
            fig.suptitle(f"{selected}泥石流传播过程快照")
            fig.savefig(FIG / "07_动态传播过程快照.png", bbox_inches="tight")
            plt.close(fig)

    # 8 冲击强度与侵蚀厚度
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    extension_layers = [
        (arrays["max_impact_intensity_m3_s2"], "magma", "最大冲击强度 h·v²（m³/s²）"),
        (arrays["max_dynamic_pressure_pa"] / 1e6, "plasma", "动力压力代理 0.5ρv²（MPa）"),
        (arrays["erosion_depth_m"], "YlOrBr", "最大侵蚀厚度（m）"),
    ]
    for ax, (layer, cmap, title) in zip(axes, extension_layers):
        positive = layer[layer > 0]
        vmax = float(np.percentile(positive, 99)) if positive.size else 1.0
        im = ax.imshow(np.where(layer > 0, layer, np.nan), extent=extent, origin="upper", cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Easting（km）")
        fig.colorbar(im, ax=ax, shrink=0.75)
    axes[0].set_ylabel("Northing（km）")
    fig.suptitle("主情景冲击与侵蚀空间分布（压力为代理量，非结构设计荷载）")
    fig.tight_layout()
    fig.savefig(FIG / "08_冲击强度与侵蚀厚度.png", bbox_inches="tight")
    plt.close(fig)

    # 9 宽域稳健性检查
    wide_path = OUT / "extensions" / f"{selected}_wide.npz"
    if wide_path.exists():
        with np.load(wide_path) as src:
            wide_h = src["hmax_m"]
            wide_extent = [float(src["xmin"]) / 1000, (float(src["xmin"]) + wide_h.shape[1] * float(src["cell_m"])) / 1000,
                           (float(src["ymax"]) - wide_h.shape[0] * float(src["cell_m"])) / 1000, float(src["ymax"]) / 1000]
            fig, axes = plt.subplots(1, 2, figsize=(11, 5))
            axes[0].imshow(np.where(arrays["hmax_m"] >= 0.1, arrays["hmax_m"], np.nan), extent=extent, origin="upper", cmap="viridis", vmin=0, vmax=60)
            axes[0].set_title("600 m活动带")
            im = axes[1].imshow(np.where(wide_h >= 0.1, wide_h, np.nan), extent=wide_extent, origin="upper", cmap="viridis", vmin=0, vmax=60)
            axes[1].set_title("完整宽域（无走廊约束）")
            for ax in axes:
                ax.set_xlabel("Easting（km）")
            axes[0].set_ylabel("Northing（km）")
            fig.colorbar(im, ax=axes, shrink=0.78, label="最大流深（m）")
            fig.suptitle("活动带约束敏感性：路径是否由预设走廊决定")
            fig.savefig(FIG / "09_活动带与宽域敏感性.png", bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    f60, f30, weights, prepared, cfg = load()
    sensitivity = parameter_sensitivity(f60)
    selected, arrays = summaries(f60, f30, weights, prepared, cfg)
    ensemble = export_spatial(selected, arrays, prepared)
    plot_all(f60, f30, weights, prepared, selected, arrays, ensemble, sensitivity)
    backconnect_to_q3()
    print(f"分析、栅格和PNG图件已生成；主情景：{selected}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        main()
