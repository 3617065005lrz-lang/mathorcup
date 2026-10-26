from __future__ import annotations

"""Build a transparent spatial susceptibility model for Question 2.

Two complementary outputs are produced:
1) an auxiliary presence/background logistic baseline using all 84 historical
   polygon centroids and county-wide DEM samples;
2) a mechanism-constrained source-area model using terrain, glacier state and
   channel connectivity.  The 2026 source AOI label is never used to fit either
   model; it is only used for an external plausibility check.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent
MODELING_ROOT = MODEL_DIR.parents[1]
CONFIG = json.loads((MODEL_DIR / "config" / "model_config.json").read_text(encoding="utf-8"))
PROCESSED = MODEL_DIR / "processed"
OUTPUTS = MODEL_DIR / "outputs"

Q2_FEATURE_DIR = MODELING_ROOT / "第二问" / "02_变量构造"
GRID_PATH = Q2_FEATURE_DIR / "processed" / "11_附件4静态候选变量及外部源区标签.parquet"
CENTROIDS_PATH = Q2_FEATURE_DIR / "outputs" / "14_历史灾点多边形质心.csv"
COUNTY_DEM = Path(r"D:\MathorCup_Data\External_Gyirong\P0-1_TPDC\data\terrain_data\吉隆县DEM.tif")
RNG = np.random.default_rng(20261026)

LOGISTIC_FEATURES = ["elevation_m", "slope_deg", "relief_330m", "roughness_330m", "abs_curvature_1pm"]


def percentile_rank(series: pd.Series, high_risk_is_high: bool = True) -> pd.Series:
    rank = series.rank(method="average", pct=True, na_option="keep")
    return rank if high_risk_is_high else 1.0 - rank


def patch_features(src: rasterio.DatasetReader, lon: float, lat: float, radius: int = 6) -> dict[str, float]:
    row, col = src.index(lon, lat)
    data = src.read(
        1,
        window=Window(col - radius, row - radius, 2 * radius + 1, 2 * radius + 1),
        boundless=True,
        fill_value=np.nan,
    ).astype(float)
    if src.nodata is not None:
        data[np.isclose(data, src.nodata)] = np.nan
    center = radius
    if not np.isfinite(data[center, center]):
        return {name: np.nan for name in LOGISTIC_FEATURES}
    dx = abs(src.transform.a) * 111_320.0 * np.cos(np.deg2rad(lat))
    dy = abs(src.transform.e) * 111_320.0
    dzdy, dzdx = np.gradient(data, dy, dx)
    slope = np.degrees(np.arctan(np.hypot(dzdx[center, center], dzdy[center, center])))
    inner = data[1:-1, 1:-1]
    laplacian = (
        (data[center, center - 1] - 2 * data[center, center] + data[center, center + 1]) / dx**2
        + (data[center - 1, center] - 2 * data[center, center] + data[center + 1, center]) / dy**2
    )
    return {
        "elevation_m": float(data[center, center]),
        "slope_deg": float(slope),
        "relief_330m": float(np.nanmax(inner) - np.nanmin(inner)),
        "roughness_330m": float(np.nanstd(inner)),
        "abs_curvature_1pm": float(abs(laplacian)),
    }


def build_logistic_training() -> pd.DataFrame:
    centroids = pd.read_csv(CENTROIDS_PATH)
    with rasterio.open(COUNTY_DEM) as src:
        present = []
        for row in centroids.itertuples(index=False):
            features = patch_features(src, row.centroid_lon_cgcs2000, row.centroid_lat_cgcs2000)
            present.append({"lon": row.centroid_lon_cgcs2000, "lat": row.centroid_lat_cgcs2000, "label": 1, **features})

        valid_presence = pd.DataFrame(present).dropna(subset=LOGISTIC_FEATURES)
        if len(valid_presence) != 84:
            raise RuntimeError(f"Expected 84 valid historical centroids, got {len(valid_presence)}")

        background = []
        target_n = len(valid_presence) * 10
        presence_lon = valid_presence["lon"].to_numpy()
        presence_lat = valid_presence["lat"].to_numpy()
        attempts = 0
        while len(background) < target_n and attempts < target_n * 50:
            attempts += 1
            row = int(RNG.integers(7, src.height - 7))
            col = int(RNG.integers(7, src.width - 7))
            lon, lat = src.xy(row, col)
            distance_km = np.sqrt(
                ((presence_lon - lon) * 111.32 * np.cos(np.deg2rad(lat))) ** 2
                + ((presence_lat - lat) * 111.32) ** 2
            )
            if np.nanmin(distance_km) < 2.0:
                continue
            features = patch_features(src, lon, lat)
            if all(np.isfinite(features[name]) for name in LOGISTIC_FEATURES):
                background.append({"lon": lon, "lat": lat, "label": 0, **features})
        if len(background) < target_n:
            raise RuntimeError("Could not draw enough valid background points")
    return pd.concat([valid_presence, pd.DataFrame(background)], ignore_index=True)


def spatial_groups(frame: pd.DataFrame) -> np.ndarray:
    lon_bin = pd.cut(frame["lon"], bins=10, labels=False, include_lowest=True)
    lat_bin = pd.cut(frame["lat"], bins=10, labels=False, include_lowest=True)
    return (lon_bin.astype(int) * 10 + lat_bin.astype(int)).to_numpy()


def fit_logistic_baseline(training: pd.DataFrame, grid: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    x = training[LOGISTIC_FEATURES].to_numpy()
    y = training["label"].to_numpy()
    groups = spatial_groups(training)
    cv_rows = []
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261026)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups), start=1):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
        model.fit(x[train_idx], y[train_idx])
        probability = model.predict_proba(x[test_idx])[:, 1]
        auc = roc_auc_score(y[test_idx], probability) if len(np.unique(y[test_idx])) == 2 else np.nan
        cv_rows.append(
            {
                "fold": fold,
                "test_n": len(test_idx),
                "test_positive_n": int(y[test_idx].sum()),
                "spatial_block_auc": auc,
            }
        )

    final = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
    final.fit(x, y)
    grid_x = np.column_stack(
        [
            grid["dem_elevation_m"],
            grid["dem_slope_deg"],
            grid["dem_local_relief_approx330m_m"],
            grid["dem_roughness_approx330m_m"],
            grid["dem_laplacian_curvature_1pm"].abs(),
        ]
    )
    probability = final.predict_proba(grid_x)[:, 1]
    scaler = final.named_steps["standardscaler"]
    logistic = final.named_steps["logisticregression"]
    coefficient = pd.DataFrame(
        {
            "feature": LOGISTIC_FEATURES,
            "coefficient_per_sd": logistic.coef_[0],
            "odds_ratio_per_sd": np.exp(logistic.coef_[0]),
            "training_mean": scaler.mean_,
            "training_sd": scaler.scale_,
        }
    )
    return probability, pd.DataFrame(cv_rows), coefficient


def mechanism_model(grid: pd.DataFrame) -> pd.DataFrame:
    out = grid.copy()
    out["risk_slope"] = percentile_rank(out["dem_slope_deg"])
    out["risk_relief"] = percentile_rank(out["dem_local_relief_approx330m_m"])
    out["risk_roughness"] = percentile_rank(out["dem_roughness_approx330m_m"])
    out["risk_stream_power"] = percentile_rank(out["stream_power_index_log1p_raw_dem"])
    out["risk_channel_proximity"] = percentile_rank(out["distance_to_channel_0p5km2_m"], high_risk_is_high=False)
    out["risk_terrain"] = np.sqrt(out["risk_slope"] * out["risk_relief"])
    out["risk_rock_instability"] = np.sqrt(out["risk_terrain"] * out["risk_roughness"])

    glacier = out["is_glacier"].astype(bool)
    out["risk_ice_thickness"] = 0.0
    out["risk_ice_velocity"] = 0.0
    out["risk_ice_thinning"] = 0.0
    out.loc[glacier, "risk_ice_thickness"] = percentile_rank(out.loc[glacier, "thickness"]).fillna(0.5)
    out.loc[glacier, "risk_ice_velocity"] = percentile_rank(out.loc[glacier, "v"]).fillna(0.5)
    out.loc[glacier, "risk_ice_thinning"] = percentile_rank(
        -out.loc[glacier, "recent_thinning_rate_2015_2020_mpy"]
    ).fillna(0.5)
    out["risk_cryosphere"] = np.cbrt(
        out["risk_ice_thickness"] * out["risk_ice_velocity"] * out["risk_ice_thinning"]
    )
    out["risk_source_material"] = np.maximum(out["risk_cryosphere"], out["risk_rock_instability"])
    out["risk_connectivity"] = np.sqrt(out["risk_stream_power"] * out["risk_channel_proximity"])
    out["static_mechanism_score"] = np.cbrt(
        out["risk_terrain"] * out["risk_source_material"] * out["risk_connectivity"]
    )

    # Equal-indicator baseline, retained only for transparent comparison.
    baseline_columns = [
        "risk_slope",
        "risk_relief",
        "risk_roughness",
        "risk_stream_power",
        "risk_channel_proximity",
        "risk_ice_thickness",
        "risk_ice_velocity",
        "risk_ice_thinning",
    ]
    out["static_equal_indicator_baseline"] = out[baseline_columns].mean(axis=1)
    return out


def plausibility_metrics(frame: pd.DataFrame, score: str) -> pd.DataFrame:
    rows = []
    label = frame["event_source_aoi_label"].astype(bool)
    base_rate = label.mean()
    for top_share in [0.10, 0.15, 0.20, 0.25, 0.30]:
        threshold = frame[score].quantile(1 - top_share)
        high = frame[score] >= threshold
        recall = float((high & label).sum() / label.sum())
        precision = float((high & label).sum() / high.sum())
        rows.append(
            {
                "score": score,
                "top_share": top_share,
                "threshold": threshold,
                "source_aoi_coverage": recall,
                "high_zone_in_source_aoi": precision,
                "enrichment_over_aoi_base_rate": precision / base_rate,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    for path in [GRID_PATH, CENTROIDS_PATH, COUNTY_DEM]:
        if not path.exists():
            raise FileNotFoundError(path)
    grid = pd.read_parquet(GRID_PATH)
    training = build_logistic_training()
    logistic_probability, cv, coefficient = fit_logistic_baseline(training, grid)
    spatial = mechanism_model(grid)
    spatial["historical_logistic_probability"] = logistic_probability

    training.to_parquet(PROCESSED / "06_历史灾点PresenceBackground训练样本.parquet", index=False)
    keep = [
        "row",
        "col",
        "longitude_wgs84",
        "latitude_wgs84",
        "event_source_aoi_label",
        "is_glacier",
        "risk_terrain",
        "risk_cryosphere",
        "risk_source_material",
        "risk_connectivity",
        "static_mechanism_score",
        "static_equal_indicator_baseline",
        "historical_logistic_probability",
    ]
    spatial[keep].to_parquet(PROCESSED / "07_空间易发性评分格网.parquet", index=False)
    cv.to_csv(OUTPUTS / "16_历史灾点空间基线CV.csv", index=False, encoding="utf-8-sig")
    coefficient.to_csv(OUTPUTS / "17_历史灾点空间基线系数.csv", index=False, encoding="utf-8-sig")

    metrics = pd.concat(
        [
            plausibility_metrics(spatial, "static_equal_indicator_baseline"),
            plausibility_metrics(spatial, "static_mechanism_score"),
            plausibility_metrics(spatial, "historical_logistic_probability"),
        ],
        ignore_index=True,
    )
    metrics.to_csv(OUTPUTS / "18_空间模型源区合理性检验.csv", index=False, encoding="utf-8-sig")

    module_summary = (
        spatial.groupby("event_source_aoi_label")
        [["risk_terrain", "risk_cryosphere", "risk_source_material", "risk_connectivity", "static_mechanism_score", "historical_logistic_probability"]]
        .agg(["median", "mean"])
    )
    module_summary.to_csv(OUTPUTS / "19_源区内外空间模块对照.csv", encoding="utf-8-sig")

    cv_mean = cv["spatial_block_auc"].mean()
    mech20 = metrics.loc[
        metrics["score"].eq("static_mechanism_score") & metrics["top_share"].eq(0.20)
    ].iloc[0]
    log20 = metrics.loc[
        metrics["score"].eq("historical_logistic_probability") & metrics["top_share"].eq(0.20)
    ].iloc[0]
    report = f"""# 空间易发性模型结果

## 历史灾点辅助基线

- 84 条历史 PolygonZ 仅作为 presence 空间记录，不含发生时间，不被当作 84 次带时刻事件。
- 背景样本为县域 DEM 中距历史质心至少 2 km 的 840 个位置。
- 模型为标准化 L2 Logistic，仅含高程、坡度、局地起伏、粗糙度和曲率绝对值。
- 五折空间分块 CV 平均 AUC：{cv_mean:.3f}。该 AUC 只衡量历史空间区分能力，不衡量 2026 年事件预测能力。
- 由于 2026 源区位于尼泊尔侧且超出历史图层东界，历史模型只能作为辅助参照；其前 20% 高值区对源区 AOI 覆盖率为 {log20['source_aoi_coverage']:.3f}。

## 机制约束主模型

- 地形模块：坡度与 330 m 局地起伏的几何聚合。
- 物源模块：冰厚、冰速、近期减薄构成冰冻圈不稳定度，并与陡峭粗糙岩壁代理取较大值。
- 连通性模块：河流功率指数与距 0.5 km² 汇水沟道距离。
- 三模块以几何平均耦合，避免把多个同类变量重复计权。
- `event_source_aoi_label` 未参与变量变换、模型拟合或阈值选择，只用于事后空间合理性检查。
- 主模型前 20% 高值区对源区 AOI 覆盖率为 {mech20['source_aoi_coverage']:.3f}，高值区位于源区 AOI 的比例为 {mech20['high_zone_in_source_aoi']:.3f}。

主模型优先用于 2026 冰—岩崩塌源区解释；历史 Logistic 基线用于说明传统历史灾点模型在跨境外推时的局限。
"""
    (OUTPUTS / "20_空间易发性结果说明.md").write_text(report, encoding="utf-8")
    print(cv.to_string(index=False))
    print("\n", coefficient.to_string(index=False))
    print("\n", metrics.to_string(index=False))


if __name__ == "__main__":
    main()
