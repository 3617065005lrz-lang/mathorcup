from __future__ import annotations

"""Interpretable nonlinear benchmark and spatial robustness experiments.

The 2026 source AOI is never used for fitting or hyper-parameter selection.  It
is only evaluated after all models and weight scenarios have been fixed.
"""

from pathlib import Path
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


MODEL_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MODEL_DIR.parents[1]
PROCESSED = MODEL_DIR / "processed"
OUTPUTS = MODEL_DIR / "outputs"
Q2_FEATURE_DIR = PROJECT_ROOT / "第二问" / "02_变量构造"
RAW_GRID = Q2_FEATURE_DIR / "processed" / "11_附件4静态候选变量及外部源区标签.parquet"
TRAINING = PROCESSED / "06_历史灾点PresenceBackground训练样本.parquet"
SPATIAL_GRID = PROCESSED / "07_空间易发性评分格网.parquet"

FEATURES = ["elevation_m", "slope_deg", "relief_330m", "roughness_330m", "abs_curvature_1pm"]
GRID_FEATURES = [
    "dem_elevation_m",
    "dem_slope_deg",
    "dem_local_relief_approx330m_m",
    "dem_roughness_approx330m_m",
    "dem_laplacian_curvature_1pm",
]
FEATURE_CN = {
    "elevation_m": "高程",
    "slope_deg": "坡度",
    "relief_330m": "330 m局地起伏",
    "roughness_330m": "330 m地形粗糙度",
    "abs_curvature_1pm": "曲率绝对值",
}
BASE_SEED = 20261028

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)


def spatial_groups(frame: pd.DataFrame) -> np.ndarray:
    lon_bin = pd.cut(frame.lon, bins=10, labels=False, include_lowest=True)
    lat_bin = pd.cut(frame.lat, bins=10, labels=False, include_lowest=True)
    return (lon_bin.astype(int) * 10 + lat_bin.astype(int)).to_numpy()


def logistic_model() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=BASE_SEED),
    )


def lightgbm_model(seed: int = BASE_SEED, n_estimators: int = 180) -> lgb.LGBMClassifier:
    # Fixed a priori: shallow trees and strong regularisation avoid a broad
    # hyper-parameter search on only 84 presence locations.
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=7,
        max_depth=3,
        min_child_samples=24,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.5,
        reg_lambda=2.0,
        class_weight="balanced",
        importance_type="gain",
        random_state=seed,
        n_jobs=1,
        verbosity=-1,
    )


def evaluate_score(score: np.ndarray, aoi: np.ndarray, top_share: float = 0.20) -> dict[str, float]:
    threshold = float(np.nanquantile(score, 1 - top_share))
    high = score >= threshold
    overlap = high & aoi
    coverage = float(overlap.sum() / aoi.sum())
    precision = float(overlap.sum() / high.sum())
    base_rate = float(aoi.mean())
    return {
        "top_share": top_share,
        "threshold": threshold,
        "source_aoi_coverage": coverage,
        "high_zone_in_source_aoi": precision,
        "enrichment_over_aoi_base_rate": precision / base_rate,
    }


def repeated_spatial_cv(training: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = training[FEATURES].to_numpy(float)
    y = training.label.to_numpy(int)
    groups = spatial_groups(training)
    rows: list[dict[str, float | int | str]] = []
    shap_parts: list[np.ndarray] = []
    shap_x_parts: list[np.ndarray] = []
    permutation_parts: list[np.ndarray] = []

    for repeat, seed in enumerate(range(BASE_SEED, BASE_SEED + 10), start=1):
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups), start=1):
            if len(np.unique(y[test_idx])) < 2:
                continue
            for name, model in [("Logistic", logistic_model()), ("LightGBM", lightgbm_model(seed + fold))]:
                model.fit(x[train_idx], y[train_idx])
                p = model.predict_proba(x[test_idx])[:, 1]
                rows.append(
                    {
                        "repeat": repeat,
                        "seed": seed,
                        "fold": fold,
                        "model": name,
                        "train_n": len(train_idx),
                        "test_n": len(test_idx),
                        "test_positive_n": int(y[test_idx].sum()),
                        "spatial_block_auc": roc_auc_score(y[test_idx], p),
                        "average_precision": average_precision_score(y[test_idx], p),
                    }
                )
                if name == "LightGBM" and repeat == 1:
                    contrib = model.booster_.predict(x[test_idx], pred_contrib=True)[:, :-1]
                    shap_parts.append(contrib)
                    shap_x_parts.append(x[test_idx])
                    perm = permutation_importance(
                        model,
                        x[test_idx],
                        y[test_idx],
                        scoring="roc_auc",
                        n_repeats=20,
                        random_state=seed + fold,
                        n_jobs=1,
                    )
                    permutation_parts.append(perm.importances)

    cv = pd.DataFrame(rows)
    final = lightgbm_model()
    final.fit(x, y)
    gain = final.feature_importances_.astype(float)
    gain_share = gain / gain.sum() if gain.sum() else np.zeros_like(gain)
    shap_values = np.vstack(shap_parts)
    shap_x = np.vstack(shap_x_parts)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    shap_share = mean_abs_shap / mean_abs_shap.sum()
    permutations = np.hstack(permutation_parts)
    explanation_rows = []
    for j, feature in enumerate(FEATURES):
        direction = spearmanr(shap_x[:, j], shap_values[:, j], nan_policy="omit").statistic
        explanation_rows.append(
            {
                "feature": feature,
                "feature_cn": FEATURE_CN[feature],
                "gain_importance": gain[j],
                "gain_share": gain_share[j],
                "oof_mean_abs_shap": mean_abs_shap[j],
                "oof_shap_share": shap_share[j],
                "shap_direction_spearman": direction,
                "oof_permutation_auc_drop_mean": float(np.mean(permutations[j])),
                "oof_permutation_auc_drop_sd": float(np.std(permutations[j], ddof=1)),
            }
        )
    return cv, pd.DataFrame(explanation_rows)


def final_model_grid(training: pd.DataFrame, raw_grid: pd.DataFrame, spatial: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = training[FEATURES].to_numpy(float)
    y = training.label.to_numpy(int)
    grid_x = raw_grid[GRID_FEATURES].to_numpy(float)
    model = lightgbm_model()
    model.fit(x, y)
    lightgbm_probability = model.predict_proba(grid_x)[:, 1]

    out = spatial[["row", "col", "longitude_wgs84", "latitude_wgs84", "event_source_aoi_label"]].copy()
    out["historical_lightgbm_probability"] = lightgbm_probability
    out["historical_logistic_probability"] = spatial.historical_logistic_probability.to_numpy(float)
    out["static_mechanism_score"] = spatial.static_mechanism_score.to_numpy(float)

    aoi = out.event_source_aoi_label.astype(bool).to_numpy()
    rows = []
    for name, column in [
        ("historical_logistic", "historical_logistic_probability"),
        ("historical_lightgbm", "historical_lightgbm_probability"),
        ("mechanism_main", "static_mechanism_score"),
    ]:
        for share in [0.10, 0.15, 0.20, 0.25, 0.30]:
            rows.append({"model": name, **evaluate_score(out[column].to_numpy(float), aoi, share)})
    return out, pd.DataFrame(rows)


def mechanism_weight_sensitivity(spatial: pd.DataFrame) -> pd.DataFrame:
    terrain = spatial.risk_terrain.to_numpy(float)
    material = spatial.risk_source_material.to_numpy(float)
    connectivity = spatial.risk_connectivity.to_numpy(float)
    aoi = spatial.event_source_aoi_label.astype(bool).to_numpy()
    main = spatial.static_mechanism_score.to_numpy(float)
    scenarios: list[tuple[str, float, float, float]] = [("equal_main", 1 / 3, 1 / 3, 1 / 3)]
    candidates = np.round(np.arange(0.15, 0.701, 0.05), 2)
    for wt in candidates:
        for wm in candidates:
            wc = round(1.0 - wt - wm, 2)
            if 0.15 <= wc <= 0.70:
                scenarios.append((f"grid_{wt:.2f}_{wm:.2f}_{wc:.2f}", wt, wm, wc))

    rows = []
    eps = 1e-6
    seen: set[tuple[float, float, float]] = set()
    for scenario, wt, wm, wc in scenarios:
        key = (round(wt, 6), round(wm, 6), round(wc, 6))
        if key in seen:
            continue
        seen.add(key)
        score = np.exp(
            wt * np.log(np.clip(terrain, eps, 1))
            + wm * np.log(np.clip(material, eps, 1))
            + wc * np.log(np.clip(connectivity, eps, 1))
        )
        rho = float(spearmanr(main, score, nan_policy="omit").statistic)
        source_pct = pd.Series(score).rank(pct=True).to_numpy()[aoi]
        for share in [0.10, 0.15, 0.20, 0.25, 0.30]:
            rows.append(
                {
                    "scenario": scenario,
                    "terrain_weight": wt,
                    "source_material_weight": wm,
                    "connectivity_weight": wc,
                    "top_share": share,
                    "rank_spearman_vs_equal_main": rho,
                    "source_aoi_median_percentile": float(np.median(source_pct)),
                    **{k: v for k, v in evaluate_score(score, aoi, share).items() if k != "top_share"},
                }
            )
    return pd.DataFrame(rows)


def background_sampling_sensitivity(
    training: pd.DataFrame, raw_grid: pd.DataFrame, spatial: pd.DataFrame, repeats: int = 100
) -> pd.DataFrame:
    positive = training.loc[training.label.eq(1)].copy()
    background = training.loc[training.label.eq(0)].copy()
    grid_x = raw_grid[GRID_FEATURES].to_numpy(float)
    aoi = spatial.event_source_aoi_label.astype(bool).to_numpy()
    full_reference = {
        "Logistic": spatial.historical_logistic_probability.to_numpy(float),
    }
    rows = []
    for repeat in range(repeats):
        seed = BASE_SEED + repeat
        sampled_background = background.sample(n=len(positive) * 5, replace=False, random_state=seed)
        sample = pd.concat([positive, sampled_background], ignore_index=True)
        x = sample[FEATURES].to_numpy(float)
        y = sample.label.to_numpy(int)
        for name, model in [("Logistic", logistic_model()), ("LightGBM", lightgbm_model(seed, n_estimators=120))]:
            model.fit(x, y)
            score = model.predict_proba(grid_x)[:, 1]
            if name not in full_reference:
                full_reference[name] = score if repeat == 0 else full_reference.get(name, score)
            metrics = evaluate_score(score, aoi, 0.20)
            rows.append(
                {
                    "repeat": repeat + 1,
                    "seed": seed,
                    "model": name,
                    "positive_n": len(positive),
                    "background_n": len(sampled_background),
                    "rank_spearman_vs_reference": float(spearmanr(score, full_reference[name]).statistic),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def q(values: pd.Series, p: float) -> float:
    return float(values.quantile(p))


def main() -> None:
    for path in [TRAINING, RAW_GRID, SPATIAL_GRID]:
        if not path.exists():
            raise FileNotFoundError(path)
    training = pd.read_parquet(TRAINING)
    raw_grid = pd.read_parquet(RAW_GRID).sort_values(["row", "col"]).reset_index(drop=True)
    spatial = pd.read_parquet(SPATIAL_GRID).sort_values(["row", "col"]).reset_index(drop=True)
    if not np.array_equal(raw_grid[["row", "col"]].to_numpy(), spatial[["row", "col"]].to_numpy()):
        raise RuntimeError("Raw and processed grids are not aligned")

    cv, explanation = repeated_spatial_cv(training)
    enhanced_grid, model_comparison = final_model_grid(training, raw_grid, spatial)
    weights = mechanism_weight_sensitivity(spatial)
    background = background_sampling_sensitivity(training, raw_grid, spatial)

    cv.to_csv(OUTPUTS / "55_非线性空间基线重复分块CV.csv", index=False, encoding="utf-8-sig")
    explanation.to_csv(OUTPUTS / "56_LightGBM空间模型解释.csv", index=False, encoding="utf-8-sig")
    model_comparison.to_csv(OUTPUTS / "57_空间增强模型源区对照.csv", index=False, encoding="utf-8-sig")
    weights.to_csv(OUTPUTS / "58_静态三模块权重敏感性.csv", index=False, encoding="utf-8-sig")
    background.to_csv(OUTPUTS / "59_背景抽样稳健性.csv", index=False, encoding="utf-8-sig")
    enhanced_grid.to_parquet(PROCESSED / "14_空间非线性增强格网.parquet", index=False)

    cv_summary = cv.groupby("model").spatial_block_auc.agg(["mean", "median", "std", "min", "max"])
    cv_paired = cv.pivot(index=["repeat", "fold"], columns="model", values="spatial_block_auc")
    cv_delta = cv_paired["LightGBM"] - cv_paired["Logistic"]
    cv_wins = int((cv_delta > 0).sum())
    cv_losses = int((cv_delta < 0).sum())
    compare20 = model_comparison.loc[np.isclose(model_comparison.top_share, 0.20)].copy()
    weight20 = weights.loc[np.isclose(weights.top_share, 0.20)]
    grid_weight20 = weight20.loc[weight20.scenario.ne("equal_main")]
    equal20 = weight20.loc[weight20.scenario.eq("equal_main")].iloc[0]
    background_summary = background.groupby("model").agg(
        coverage_median=("source_aoi_coverage", "median"),
        coverage_q025=("source_aoi_coverage", lambda x: q(x, 0.025)),
        coverage_q975=("source_aoi_coverage", lambda x: q(x, 0.975)),
        rank_median=("rank_spearman_vs_reference", "median"),
        rank_q025=("rank_spearman_vs_reference", lambda x: q(x, 0.025)),
    )
    report = f"""# 第二问空间模型增强实验报告

## 1. 实验定位

本实验增加LightGBM非线性空间辅助基线、重复空间分块验证、LightGBM贡献解释、静态三模块权重敏感性和背景样本抽样稳健性。2026源区AOI不参与训练、特征选择、超参数选择或主权重选择，只在模型冻结后作空间合理性检查。

LightGBM采用固定浅树和正则化参数，不进行面向2026源区覆盖率的网格调参。它是历史灾点辅助模型，不替代可跨境解释的地形—物源—沟道连通性主模型。

## 2. 重复空间分块结果

10组空间分块种子×5折，比较同一批84个历史空间记录和840个背景点：

{cv_summary.to_string()}

在完全相同的50个“重复—折”组合上，LightGBM相对Logistic的AUC差值均值为{cv_delta.mean():.3f}，中位数为{cv_delta.median():.3f}；LightGBM仅在{cv_wins}/50折中更高，在{cv_losses}/50折中更低。重复折并非相互独立样本，因此这里只作配对描述，不把它包装为独立显著性检验。

这些AUC只表示中国侧历史灾点的空间区分能力，不是2026事件发生概率或跨境预测精度。

## 3. 2026源区事后空间合理性检查

{compare20.to_string(index=False)}

本次LightGBM平均空间分块AUC低于Logistic，说明在84个空间记录和5个地形变量下，增加非线性并未改善历史空间泛化。LightGBM对2026源区覆盖虽略高于Logistic，但仍明显低于机制主模型；因此不能按单个源区覆盖结果反向选择复杂模型。

## 4. LightGBM解释

`oof_mean_abs_shap`来自LightGBM原生`pred_contrib`的空间折外贡献绝对值；`shap_direction_spearman`只表示特征值与贡献方向的单调关系，不等于因果效应。

{explanation.sort_values('oof_shap_share', ascending=False).to_string(index=False)}

## 5. 静态模块权重敏感性

主模型继续使用地形、物源、连通性各1/3的几何耦合。另枚举每个模块0.15—0.70、权重和为1的预设方案，不根据2026源区表现挑选最优权重。

- 等权主模型前20%高值区源区覆盖率：{equal20.source_aoi_coverage:.3f}；
- 预设权重网格覆盖率中位数：{grid_weight20.source_aoi_coverage.median():.3f}；
- 预设权重网格95%范围：{q(grid_weight20.source_aoi_coverage, 0.025):.3f}—{q(grid_weight20.source_aoi_coverage, 0.975):.3f}；
- 与等权主模型空间排序的Spearman中位数：{grid_weight20.rank_spearman_vs_equal_main.median():.3f}。

如果覆盖率范围较宽，应把权重选择列为模型结构不确定性；不能只引用表现最好的权重。

## 6. 背景样本抽样稳健性

固定全部84个历史记录，每次从840个候选背景中随机抽取420个，重复100次：

{background_summary.to_string()}

该实验反映现有候选池内的背景抽样不确定性，不能替代尼泊尔侧同机理灾害样本。

## 7. 最终使用建议

1. 机制约束模型仍为第二问空间主模型；
2. Logistic保留为最透明的历史统计基线；
3. LightGBM作为非线性增强对照，并报告空间分块AUC与贡献解释；
4. 不因某个辅助模型AUC较高就把其输出称为2026源区发生概率；
5. 不按2026源区覆盖率选择权重或超参数。
"""
    report = "\n".join(line.rstrip() for line in report.splitlines()) + "\n"
    (OUTPUTS / "60_第二问空间增强实验报告.md").write_text(report, encoding="utf-8")
    print(cv_summary.to_string())
    print(compare20.to_string(index=False))
    print(background_summary.to_string())


if __name__ == "__main__":
    main()
