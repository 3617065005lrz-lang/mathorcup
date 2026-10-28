from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc


HERE = Path(__file__).resolve()
Q4 = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))

from q4_solver import Scenario, metrics_dict, run_scenario  # noqa: E402


def config() -> dict:
    return json.loads((Q4 / "config" / "model_config.json").read_text(encoding="utf-8"))


def ensure_prepared(resolution: int, mode: str = "corridor") -> Path:
    suffix = "" if mode == "corridor" else f"_{mode}"
    path = Q4 / "cache" / f"prepared_{resolution}m{suffix}.npz"
    if path.exists():
        return path
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("prepare_q4", HERE.parent / "01_prepare_inputs.py")
    module = module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.prepare(resolution, mode=mode)


def lhs(n: int, ranges: dict[str, list[float]], seed: int) -> list[dict[str, float]]:
    keys = list(ranges)
    sampler = qmc.LatinHypercube(d=len(keys), seed=seed)
    sample = sampler.random(n)
    lower = np.array([ranges[k][0] for k in keys], float)
    upper = np.array([ranges[k][1] for k in keys], float)
    scaled = qmc.scale(sample, lower, upper)
    return [{key: float(row[i]) for i, key in enumerate(keys)} for row in scaled]


def scenario_rows(cfg: dict) -> list[Scenario]:
    seed = int(cfg["search"]["seed"])
    ranges = cfg["search"]["ranges"]
    hyper = cfg["search"]["hypermobile_ranges"]
    scenarios: list[Scenario] = []

    # M1：文献常用量级的均一Voellmy基线。它用于回答“普通泥石流参数是否足够”。
    m1_ranges = {
        "mu_uniform": ranges["mu_uniform"],
        "xi_uniform_m_s2": ranges["xi_uniform_m_s2"],
        "volume_scale": ranges["volume_scale"],
    }
    for i, p in enumerate(lhs(int(cfg["search"]["lhs_samples_m1"]), m1_ranges, seed + 1), 1):
        scenarios.append(
            Scenario(
                f"M1_{i:03d}",
                "M1_uniform_literature",
                p["mu_uniform"],
                p["xi_uniform_m_s2"],
                p["mu_uniform"],
                p["xi_uniform_m_s2"],
                volume_scale=p["volume_scale"],
            )
        )

    # M2：分区Voellmy。参数范围扩展到冰岩崩塌—高含水长程运动的有效摩阻；
    # 是否需要这一扩展由到时和留出区表现共同判断，而不是先验宣称。
    m2_ranges = {
        "mu_source": hyper["mu_source"],
        "xi_source_m_s2": hyper["xi_source_m_s2"],
        "mu_valley": hyper["mu_valley"],
        "xi_valley_m_s2": hyper["xi_valley_m_s2"],
        "volume_scale": hyper["volume_scale"],
        "release_duration_s": hyper["release_duration_s"],
        "release_speed_m_s": hyper["release_speed_m_s"],
    }
    for i, p in enumerate(lhs(int(cfg["search"]["lhs_samples_m2"]), m2_ranges, seed + 2), 1):
        scenarios.append(
            Scenario(
                f"M2_{i:03d}",
                "M2_zoned_hypermobile",
                p["mu_source"],
                p["xi_source_m_s2"],
                p["mu_valley"],
                p["xi_valley_m_s2"],
                volume_scale=p["volume_scale"],
                release_duration_s=p["release_duration_s"],
                release_speed_m_s=p["release_speed_m_s"],
            )
        )

    m3_ranges = dict(m2_ranges)
    m3_ranges.update(
        {
            "erosion_k": hyper["erosion_k"],
            "erosion_ucrit_m_s": hyper["erosion_ucrit_m_s"],
            "erodible_depth_m": hyper["erodible_depth_m"],
        }
    )
    for i, p in enumerate(lhs(int(cfg["search"]["lhs_samples_m3"]), m3_ranges, seed + 3), 1):
        scenarios.append(
            Scenario(
                f"M3_{i:03d}",
                "M3_zoned_entrainment",
                p["mu_source"],
                p["xi_source_m_s2"],
                p["mu_valley"],
                p["xi_valley_m_s2"],
                erosion_k=p["erosion_k"],
                erosion_ucrit=p["erosion_ucrit_m_s"],
                erodible_depth=p["erodible_depth_m"],
                volume_scale=p["volume_scale"],
                release_duration_s=p["release_duration_s"],
                release_speed_m_s=p["release_speed_m_s"],
            )
        )
    return scenarios


def run_60m(cfg: dict, force: bool) -> tuple[pd.DataFrame, dict[str, Scenario]]:
    out_csv = Q4 / "outputs" / "scenario_metrics_60m.csv"
    scenario_json = Q4 / "outputs" / "scenario_parameters.json"
    scenarios = scenario_rows(cfg)
    scenario_map = {s.scenario_id: s for s in scenarios}
    if out_csv.exists() and scenario_json.exists() and not force:
        return pd.read_csv(out_csv), scenario_map

    prepared = ensure_prepared(60)
    rows = []
    total = len(scenarios)
    for j, scenario in enumerate(scenarios, 1):
        metrics = run_scenario(prepared, scenario, cfg)
        row = {**asdict(scenario), **metrics_dict(metrics)}
        rows.append(row)
        print(
            f"[{j:02d}/{total}] {scenario.scenario_id} {metrics.status} "
            f"arrival={metrics.arrival_time_s:.1f}s score={metrics.score:.3f}"
        )
    frame = pd.DataFrame(rows).sort_values("score", ascending=False)
    frame.to_csv(out_csv, index=False, encoding="utf-8-sig")
    scenario_json.write_text(
        json.dumps({key: asdict(value) for key, value in scenario_map.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return frame, scenario_map


def accepted(frame: pd.DataFrame) -> pd.DataFrame:
    ok = frame[
        (frame["status"] == "ok")
        & frame["arrival_time_s"].notna()
        & (frame["max_speed_m_s"] <= 250.0)
        & (frame["speed_p99_m_s"] <= 145.0)
        & (frame["numerical_volume_error_fraction"].abs() <= 0.03)
        & (frame["entrained_volume_m3"] <= 1.5 * frame["initial_volume_m3"])
    ].copy()
    return ok.sort_values("score", ascending=False)


def rerun_30m(cfg: dict, frame60: pd.DataFrame, scenarios: dict[str, Scenario], force: bool) -> pd.DataFrame:
    out_csv = Q4 / "outputs" / "scenario_metrics_30m.csv"
    if out_csv.exists() and not force:
        return pd.read_csv(out_csv)
    shortlist = accepted(frame60)
    shortlist = shortlist[shortlist["model"].isin(["M2_zoned_hypermobile", "M3_zoned_entrainment"])]
    shortlist = shortlist.head(int(cfg["search"]["keep_for_30m"]))
    if shortlist.empty:
        raise RuntimeError("60 m筛选后没有可供30 m复算的物理可接受情景。")
    prepared = ensure_prepared(30)
    rows = []
    for scenario_id in shortlist["scenario_id"]:
        scenario = scenarios[str(scenario_id)]
        arrays = Q4 / "outputs" / "scenarios_30m" / f"{scenario.scenario_id}.npz"
        metrics = run_scenario(prepared, scenario, cfg, arrays)
        rows.append({**asdict(scenario), **metrics_dict(metrics)})
        print(f"[30m] {scenario.scenario_id} arrival={metrics.arrival_time_s:.1f}s holdout={metrics.holdout_csi:.3f}")
    frame = pd.DataFrame(rows).sort_values("score", ascending=False)
    frame.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return frame


def refined_ranges(cfg: dict, frame60: pd.DataFrame) -> dict[str, list[float]]:
    source = frame60[(frame60["model"] == "M3_zoned_entrainment") & (frame60["status"] == "ok")]
    source = source.sort_values("score", ascending=False).head(10)
    limits = cfg["search"]["hypermobile_ranges"]
    mapping = {
        "mu_source": "mu_source",
        "xi_source_m_s2": "xi_source",
        "mu_valley": "mu_valley",
        "xi_valley_m_s2": "xi_valley",
        "erosion_k": "erosion_k",
        "erosion_ucrit_m_s": "erosion_ucrit",
        "erodible_depth_m": "erodible_depth",
        "volume_scale": "volume_scale",
        "release_duration_s": "release_duration_s",
        "release_speed_m_s": "release_speed_m_s",
    }
    result = {}
    for range_key, column in mapping.items():
        lo0, hi0 = map(float, limits[range_key])
        lo, hi = float(source[column].min()), float(source[column].max())
        pad = max((hi - lo) * 0.15, (hi0 - lo0) * 0.03)
        result[range_key] = [max(lo0, lo - pad), min(hi0, hi + pad)]
    return result


def run_refined_30m(cfg: dict, frame60: pd.DataFrame, force: bool) -> pd.DataFrame:
    out_csv = Q4 / "outputs" / "scenario_metrics_30m_refined.csv"
    if out_csv.exists() and not force:
        return pd.read_csv(out_csv)
    ranges = refined_ranges(cfg, frame60)
    (Q4 / "outputs" / "refined_parameter_ranges.json").write_text(
        json.dumps(ranges, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    samples = lhs(24, ranges, int(cfg["search"]["seed"]) + 44)
    prepared = ensure_prepared(30)
    rows = []
    for i, p in enumerate(samples, 1):
        scenario = Scenario(
            f"R3_{i:03d}",
            "M3_refined_entrainment",
            p["mu_source"],
            p["xi_source_m_s2"],
            p["mu_valley"],
            p["xi_valley_m_s2"],
            erosion_k=p["erosion_k"],
            erosion_ucrit=p["erosion_ucrit_m_s"],
            erodible_depth=p["erodible_depth_m"],
            volume_scale=p["volume_scale"],
            release_duration_s=p["release_duration_s"],
            release_speed_m_s=p["release_speed_m_s"],
        )
        arrays = Q4 / "outputs" / "scenarios_30m" / f"{scenario.scenario_id}.npz"
        metrics = run_scenario(prepared, scenario, cfg, arrays)
        rows.append({**asdict(scenario), **metrics_dict(metrics)})
        print(
            f"[refine30 {i:02d}/24] {scenario.scenario_id} {metrics.status} "
            f"arrival={metrics.arrival_time_s:.1f}s cal={metrics.calibration_csi:.3f} hold={metrics.holdout_csi:.3f}"
        )
    frame = pd.DataFrame(rows).sort_values("score", ascending=False)
    frame.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return frame


def behavioral(frame30: pd.DataFrame) -> pd.DataFrame:
    valid = accepted(frame30)
    strict = valid[(valid["arrival_error_s"].abs() <= 60.0) & (valid["calibration_csi"] >= 0.55)]
    chosen = strict.sort_values("score", ascending=False).head(20)
    if len(chosen) < 20:
        fill = valid[~valid["scenario_id"].isin(chosen["scenario_id"])]
        chosen = pd.concat([chosen, fill.sort_values("score", ascending=False).head(20 - len(chosen))])
    return chosen.sort_values("score", ascending=False)


def ensemble(frame30: pd.DataFrame) -> None:
    valid = behavioral(frame30)
    if valid.empty:
        raise RuntimeError("30 m复算没有物理可接受情景。")
    # GLUE式似然权重：以综合适合度相对最优值构造；不声称是后验概率。
    weights = np.exp(8.0 * (valid["score"].to_numpy() - valid["score"].max()))
    weights /= weights.sum()
    stack_h, stack_v, stack_t = [], [], []
    for scenario_id in valid["scenario_id"]:
        with np.load(Q4 / "outputs" / "scenarios_30m" / f"{scenario_id}.npz") as src:
            stack_h.append(src["hmax_m"].astype(np.float32))
            stack_v.append(src["vmax_m_s"].astype(np.float32))
            stack_t.append(src["arrival_time_s"].astype(np.float32))
    h = np.stack(stack_h)
    v = np.stack(stack_v)
    t = np.stack(stack_t)
    w = weights[:, None, None]
    prob_h01 = np.sum(w * (h >= 0.1), axis=0)
    prob_h05 = np.sum(w * (h >= 0.5), axis=0)
    prob_h10 = np.sum(w * (h >= 1.0), axis=0)
    prob_v5 = np.sum(w * (v >= 5.0), axis=0)
    h_mean = np.sum(w * h, axis=0)
    v_mean = np.sum(w * v, axis=0)
    # 到达时间只对到达过的情景取中位数；未到达不伪装为迟到。
    with np.errstate(all="ignore"):
        t_median = np.nanmedian(t, axis=0)
        t_p10 = np.nanpercentile(t, 10, axis=0)
        t_p90 = np.nanpercentile(t, 90, axis=0)
    np.savez_compressed(
        Q4 / "outputs" / "M4_ensemble_30m.npz",
        probability_depth_ge_0p1m=prob_h01.astype(np.float32),
        probability_depth_ge_0p5m=prob_h05.astype(np.float32),
        probability_depth_ge_1m=prob_h10.astype(np.float32),
        probability_speed_ge_5m_s=prob_v5.astype(np.float32),
        weighted_mean_max_depth_m=h_mean.astype(np.float32),
        weighted_mean_max_speed_m_s=v_mean.astype(np.float32),
        arrival_median_s=t_median.astype(np.float32),
        arrival_p10_s=t_p10.astype(np.float32),
        arrival_p90_s=t_p90.astype(np.float32),
        scenario_ids=valid["scenario_id"].astype(str).to_numpy(),
        weights=weights.astype(np.float64),
    )
    weights_frame = valid[["scenario_id", "model", "score", "arrival_time_s", "holdout_csi"]].copy()
    weights_frame["glue_weight"] = weights
    weights_frame.to_csv(Q4 / "outputs" / "M4_GLUE_weights.csv", index=False, encoding="utf-8-sig")
    valid.to_csv(Q4 / "outputs" / "M4_behavioral_scenarios.csv", index=False, encoding="utf-8-sig")


def write_summary(frame60: pd.DataFrame, frame30: pd.DataFrame) -> None:
    by_model = (
        frame60.groupby("model")
        .agg(
            scenarios=("scenario_id", "count"),
            accepted=("status", lambda s: int((s == "ok").sum())),
            best_score=("score", "max"),
            median_score=("score", "median"),
            best_arrival_error_s=("arrival_error_s", lambda s: float(np.nanmin(np.abs(s))) if s.notna().any() else np.nan),
        )
        .reset_index()
    )
    best30 = frame30.sort_values("score", ascending=False).iloc[0].to_dict()
    summary = {
        "model_comparison_60m": by_model.to_dict(orient="records"),
        "best_30m": {key: (value.item() if isinstance(value, np.generic) else value) for key, value in best30.items()},
        "interpretation_boundary": [
            "M1若无法再现到时，不等于Voellmy方程无效，只说明普通均一泥石流参数不足以描述本次高位冰岩崩塌链。",
            "M2/M3的mu与xi是栅格尺度下的有效参数，不应解释为可直接移植到其他沟谷的材料常数。",
            "30 m留出区表现用于检验空间迁移；UNOSAT为初步卫星解译，不能等同现场逐点实测。",
        ],
    }
    (Q4 / "outputs" / "model_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="运行第四问M1-M4模型链")
    parser.add_argument("--force", action="store_true", help="忽略已有CSV并全部重算")
    parser.add_argument("--force-30", action="store_true", help="保留60 m筛选，仅重算30 m与集合")
    parser.add_argument("--force-refine", action="store_true", help="保留粗筛与5组复算，仅重算30 m局部加密")
    args = parser.parse_args()
    cfg = config()
    frame60, scenarios = run_60m(cfg, args.force)
    frame30 = rerun_30m(cfg, frame60, scenarios, args.force or args.force_30)
    refined30 = run_refined_30m(cfg, frame60, args.force or args.force_30 or args.force_refine)
    all30 = pd.concat([frame30, refined30], ignore_index=True).sort_values("score", ascending=False)
    all30.to_csv(Q4 / "outputs" / "scenario_metrics_30m_all.csv", index=False, encoding="utf-8-sig")
    ensemble(all30)
    write_summary(frame60, all30)
    print("第四问数值实验完成。")


if __name__ == "__main__":
    main()
