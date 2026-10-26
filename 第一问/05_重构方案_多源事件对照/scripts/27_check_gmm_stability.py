from __future__ import annotations

"""Check GMM weather-type selection and seed stability."""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler


SCRIPT_DIR = Path(__file__).resolve().parent
WORK_DIR = SCRIPT_DIR.parent
CACHE_DIR = WORK_DIR / "cache"
OUTPUT_DIR = WORK_DIR / "outputs"
CONTROL_YEARS = [2021, 2022, 2023, 2024]
EVENT = pd.Timestamp("2026-08-26 10:00:00")
FEATURES = [
    "sta_tmean_24h",
    "sta_precip_sum_24h",
    "sta_RH_mean_24h",
    "sta_pressure_mean_24h",
    "sta_wind10_mean_24h",
    "sta_positive_netrad_24h",
]
SEEDS = list(range(2026, 2036))


def pairwise_mean(values: list[np.ndarray], metric) -> float:
    scores = [metric(values[i], values[j]) for i, j in combinations(range(len(values)), 2)]
    return float(np.mean(scores))


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


def evaluate(mode: str, components: int) -> dict[str, object]:
    frame = pd.read_parquet(CACHE_DIR / f"station_daily_10h_features_{mode}.parquet")
    coverage = frame[FEATURES].notna().sum(axis=1)
    usable = coverage >= 4
    train_mask = frame["Timestamp"].dt.year.isin(CONTROL_YEARS) & usable
    train = frame.loc[train_mask, FEATURES].copy()
    medians = train.median()
    all_values = frame.loc[usable, FEATURES].fillna(medians)
    train = train.fillna(medians)
    precip_index = FEATURES.index("sta_precip_sum_24h")
    train.iloc[:, precip_index] = np.log1p(train.iloc[:, precip_index].clip(lower=0))
    all_values.iloc[:, precip_index] = np.log1p(all_values.iloc[:, precip_index].clip(lower=0))
    scaler = RobustScaler().fit(train)
    z_train = scaler.transform(train)
    z_all = scaler.transform(all_values)
    event_position = int(np.flatnonzero(frame.loc[usable, "Timestamp"].eq(EVENT).to_numpy())[0])

    labels = []
    event_memberships = []
    bics = []
    sizes = []
    for seed in SEEDS:
        model = GaussianMixture(
            n_components=components,
            covariance_type="full",
            n_init=20,
            reg_covar=1e-5,
            random_state=seed,
        ).fit(z_train)
        train_labels = model.predict(z_train)
        all_labels = model.predict(z_all)
        event_label = all_labels[event_position]
        labels.append(train_labels)
        membership = train_labels == event_label
        event_memberships.append(membership)
        sizes.append(int(membership.sum()))
        bics.append(float(model.bic(z_train)))
    return {
        "清洗版本": mode,
        "天气型数": components,
        "BIC均值": float(np.mean(bics)),
        "BIC标准差": float(np.std(bics)),
        "聚类ARI均值": pairwise_mean(labels, adjusted_rand_score),
        "事件同型样本Jaccard均值": pairwise_mean(event_memberships, jaccard),
        "事件天气型历史样本数中位数": float(np.median(sizes)),
        "随机初值数": len(SEEDS),
    }


def main() -> None:
    rows = [evaluate(mode, k) for mode in ["strict", "expanded"] for k in range(2, 8)]
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_DIR / "27_GMM随机初值稳定性.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# GMM天气型选择与随机初值稳定性",
        "",
        "每个清洗版本、每个天气型数使用10个随机种子；每次模型内部仍采用20次初始化。",
        "ARI衡量整体聚类一致性，事件同型样本Jaccard衡量用于条件对照的历史样本集合是否稳定。",
        "",
        "| 清洗版本 | 类别数 | BIC均值 | BIC标准差 | ARI均值 | 事件同型Jaccard | 事件同型历史样本中位数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in result.iterrows():
        lines.append(
            f"| {row['清洗版本']} | {int(row['天气型数'])} | {row['BIC均值']:.1f} | "
            f"{row['BIC标准差']:.2f} | {row['聚类ARI均值']:.3f} | "
            f"{row['事件同型样本Jaccard均值']:.3f} | {row['事件天气型历史样本数中位数']:.0f} |"
        )
    (OUTPUT_DIR / "28_GMM稳定性报告.md").write_text("\n".join(lines), encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
