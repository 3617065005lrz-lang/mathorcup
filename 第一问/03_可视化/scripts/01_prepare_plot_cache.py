from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
VIS_DIR = SCRIPT_DIR.parent
QUESTION_DIR = VIS_DIR.parent
ANALYSIS_DIR = QUESTION_DIR / "02_特征构造与主分析"
ANALYSIS_OUTPUT = ANALYSIS_DIR / "outputs"
CACHE_DIR = VIS_DIR / "plot_cache"
MAIN_SCRIPT = ANALYSIS_DIR / "scripts" / "01_run_2026_main_analysis.py"

CACHE_MODE = "strict"
TOP_FACTOR_COUNT = 5

FEATURE_METADATA = {
    "vwc_delta_24h": {
        "公式": "五层VWC逐时均值(t) - 五层VWC逐时均值(t-24h)",
        "单位": "m³/m³",
        "含义": "土壤剖面含水状态的24小时变化",
    },
    "temp_mean_24h": {
        "公式": "mean[T_a,2m(t-23h:t)]",
        "单位": "°C",
        "含义": "近24小时区域热量背景",
    },
    "temp_mean_72h": {
        "公式": "mean[T_a,2m(t-71h:t)]",
        "单位": "°C",
        "含义": "近72小时区域热量背景",
    },
    "pdd_72h": {
        "公式": "Σmax(T_a,2m,0)/24",
        "单位": "°C·d",
        "含义": "近72小时正积温",
    },
    "positive_netrad_24h": {
        "公式": "Σmax(Rsd-Rsu+Rld-Rlu,0)/1000",
        "单位": "kWh/m²",
        "含义": "近24小时地表可获得的正净辐射能量",
    },
    "precip_sum_72h": {
        "公式": "ΣPrecip(t-71h:t)",
        "单位": "mm",
        "含义": "近72小时累计降水输入",
    },
    "rh_mean_24h": {
        "公式": "mean[RH_2m(t-23h:t)]",
        "单位": "%",
        "含义": "近24小时空气相对湿度背景",
    },
    "vpd_ambient_median_24h": {
        "公式": "median[(amb_e_sat-amb_e)×10]",
        "单位": "hPa",
        "含义": "近24小时环境水汽压亏缺",
    },
}


def load_analysis_module():
    spec = importlib.util.spec_from_file_location("main_analysis", MAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_top_factors(stability: pd.DataFrame, scenarios: pd.DataFrame) -> pd.DataFrame:
    top = stability.head(TOP_FACTOR_COUNT).copy()
    rows = []
    for _, factor_row in top.iterrows():
        factor = factor_row["物理因素"]
        display = factor_row["最常入选代表指标"]
        matched = scenarios[
            (scenarios["物理因素"] == factor)
            & (scenarios["指标中文名"] == display)
        ]
        if matched.empty:
            raise ValueError(f"无法为因素“{factor}”匹配代表衍生指标“{display}”。")
        row = matched.iloc[0]
        feature = row["衍生指标"]
        if feature not in FEATURE_METADATA:
            raise KeyError(
                f"代表指标 {feature} 尚未在FEATURE_METADATA中登记公式和单位，请先补充。"
            )
        rows.append(
            {
                "稳定性顺序": len(rows) + 1,
                "物理因素": factor,
                "衍生指标": feature,
                "指标中文名": display,
                "对应原始指标": row["对应原始指标"],
                "前五入选率": factor_row["进入前五比例"],
                "可用场景比例": factor_row["可用场景比例"],
                "主导机制状态": factor_row["主导机制状态"],
                **FEATURE_METADATA[feature],
            }
        )
    return pd.DataFrame(rows)


def matched_trajectory_cache(
    features: pd.DataFrame,
    top_factors: pd.DataFrame,
    event_cutoff: pd.Timestamp,
    control_years: list[int],
) -> pd.DataFrame:
    indexed = features.set_index("Timestamp")
    event_times = pd.date_range(
        event_cutoff - pd.Timedelta(hours=167), event_cutoff, freq="h"
    )
    rows = []
    for _, meta in top_factors.iterrows():
        feature = meta["衍生指标"]
        for relative_hour, event_time in zip(range(-167, 1), event_times):
            event_value = indexed[feature].get(event_time, np.nan)
            controls = []
            for year in control_years:
                control_time = event_time.replace(year=year)
                value = indexed[feature].get(control_time, np.nan)
                if pd.notna(value):
                    controls.append(float(value))
            if controls:
                q1, median, q3 = np.quantile(controls, [0.25, 0.50, 0.75])
                scale = (q3 - q1) / 1.349
                robust_z = (
                    (float(event_value) - median) / scale
                    if pd.notna(event_value) and scale > 1e-12
                    else np.nan
                )
                control_min = min(controls)
                control_max = max(controls)
            else:
                q1 = median = q3 = control_min = control_max = robust_z = np.nan
            rows.append(
                {
                    "稳定性顺序": meta["稳定性顺序"],
                    "物理因素": meta["物理因素"],
                    "衍生指标": feature,
                    "指标中文名": meta["指标中文名"],
                    "对应原始指标": meta["对应原始指标"],
                    "单位": meta["单位"],
                    "相对小时": relative_hour,
                    "相对天": relative_hour / 24.0,
                    "事件时间": event_time,
                    "事件值": event_value,
                    "历史Q1": q1,
                    "历史中位数": median,
                    "历史Q3": q3,
                    "历史最小值": control_min,
                    "历史最大值": control_max,
                    "稳健标准分": robust_z,
                    "有效历史年数": len(controls),
                }
            )
    return pd.DataFrame(rows)


def distribution_cache(
    features: pd.DataFrame,
    top_factors: pd.DataFrame,
    event_cutoff: pd.Timestamp,
    control_years: list[int],
) -> pd.DataFrame:
    rows = []
    for _, meta in top_factors.iterrows():
        feature = meta["衍生指标"]
        for year in control_years + [event_cutoff.year]:
            end = event_cutoff if year == event_cutoff.year else event_cutoff.replace(year=year)
            mask = (
                (features["Timestamp"] > end - pd.Timedelta(hours=24))
                & (features["Timestamp"] <= end)
            )
            values = features.loc[mask, ["Timestamp", feature]].dropna()
            for _, value_row in values.iterrows():
                rows.append(
                    {
                        "物理因素": meta["物理因素"],
                        "衍生指标": feature,
                        "指标中文名": meta["指标中文名"],
                        "单位": meta["单位"],
                        "年份": year,
                        "样本组": "2026事件窗口" if year == 2026 else "2021—2024历史同期",
                        "Timestamp": value_row["Timestamp"],
                        "数值": value_row[feature],
                    }
                )
    return pd.DataFrame(rows)


def sensitivity_grid_cache(
    scenarios: pd.DataFrame,
    factors: list[str],
) -> pd.DataFrame:
    rows = []
    expected_replicates = (
        scenarios[["清洗版本", "FLUX复核点处理"]].drop_duplicates().shape[0]
    )
    for factor in factors:
        for hours in sorted(scenarios["评价窗口小时"].unique()):
            for band in sorted(scenarios["同期日期带天数"].unique()):
                part = scenarios[
                    (scenarios["物理因素"] == factor)
                    & (scenarios["评价窗口小时"] == hours)
                    & (scenarios["同期日期带天数"] == band)
                ]
                rows.append(
                    {
                        "物理因素": factor,
                        "评价窗口小时": hours,
                        "同期日期带天数": band,
                        "预期重复数": expected_replicates,
                        "可用重复数": len(part),
                        "可用比例": len(part) / expected_replicates,
                        "进入前五次数": int((part["因素排名"] <= 5).sum()),
                        "进入前五比例": float(
                            (part["因素排名"] <= 5).sum() / expected_replicates
                        ),
                        "因素排名中位数": part["因素排名"].median(),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    source_paths = {
        "base_ranking": ANALYSIS_OUTPUT / "03_物理因素排序.csv",
        "event_trajectory": ANALYSIS_OUTPUT / "05_2026灾前7天衍生指标轨迹.parquet",
        "sensitivity_scenarios": ANALYSIS_OUTPUT / "06_敏感性检验全部场景.csv",
        "stability_summary": ANALYSIS_OUTPUT / "07_物理因素稳定性汇总.csv",
        "analysis_script": MAIN_SCRIPT,
    }
    missing = [str(path) for path in source_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少绘图缓存来源文件：\n" + "\n".join(missing))

    module = load_analysis_module()
    stability = pd.read_csv(source_paths["stability_summary"])
    scenarios = pd.read_csv(source_paths["sensitivity_scenarios"])
    base_ranking = pd.read_csv(source_paths["base_ranking"])
    top_factors = select_top_factors(stability, scenarios)

    raw = module.load_mode(CACHE_MODE, exclude_flux_review=False)
    features = module.build_features(raw)
    if 2025 in set(features["Timestamp"].dt.year):
        raise RuntimeError("缓存构造发现2025年数据，已停止以避免验证信息泄漏。")

    trajectory = matched_trajectory_cache(
        features, top_factors, module.EVENT_CUTOFF, module.CONTROL_YEARS
    )
    distributions = distribution_cache(
        features, top_factors, module.EVENT_CUTOFF, module.CONTROL_YEARS
    )
    grid = sensitivity_grid_cache(scenarios, top_factors["物理因素"].tolist())

    base_ranking.to_csv(CACHE_DIR / "base_factor_ranking.csv", index=False, encoding="utf-8-sig")
    stability.to_csv(CACHE_DIR / "factor_stability.csv", index=False, encoding="utf-8-sig")
    top_factors.to_csv(CACHE_DIR / "top5_factor_metadata.csv", index=False, encoding="utf-8-sig")
    grid.to_csv(CACHE_DIR / "top5_sensitivity_grid.csv", index=False, encoding="utf-8-sig")
    trajectory.to_parquet(CACHE_DIR / "top5_matched_trajectory.parquet", index=False)
    distributions.to_parquet(CACHE_DIR / "top5_event_control_distribution.parquet", index=False)

    manifest = {
        "cache_version": 1,
        "cache_mode": CACHE_MODE,
        "event_time": str(module.EVENT_TIME),
        "event_cutoff": str(module.EVENT_CUTOFF),
        "control_years": module.CONTROL_YEARS,
        "excluded_validation_year": 2025,
        "top_factor_count": TOP_FACTOR_COUNT,
        "source_files": {
            name: {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for name, path in source_paths.items()
        },
        "cache_files": [
            "base_factor_ranking.csv",
            "factor_stability.csv",
            "top5_factor_metadata.csv",
            "top5_sensitivity_grid.csv",
            "top5_matched_trajectory.parquet",
            "top5_event_control_distribution.parquet",
        ],
    }
    (CACHE_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"绘图缓存已更新：{CACHE_DIR}")


if __name__ == "__main__":
    main()

