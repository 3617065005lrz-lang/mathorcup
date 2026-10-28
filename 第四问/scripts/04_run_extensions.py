from __future__ import annotations

import json
import sys
from dataclasses import asdict
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve()
Q4 = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))

from q4_solver import Scenario, metrics_dict, run_scenario  # noqa: E402


def load_cfg() -> dict:
    return json.loads((Q4 / "config" / "model_config.json").read_text(encoding="utf-8"))


def prepare(resolution: int, mode: str) -> Path:
    suffix = "" if mode == "corridor" else f"_{mode}"
    path = Q4 / "cache" / f"prepared_{resolution}m{suffix}.npz"
    if path.exists():
        return path
    spec = spec_from_file_location("prepare_q4", HERE.parent / "01_prepare_inputs.py")
    module = module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.prepare(resolution, force=True, mode=mode)


def scenario_from_row(row: pd.Series, scenario_id: str | None = None, background: float | None = None) -> Scenario:
    return Scenario(
        scenario_id or str(row["scenario_id"]),
        str(row["model"]),
        float(row["mu_source"]),
        float(row["xi_source"]),
        float(row["mu_valley"]),
        float(row["xi_valley"]),
        erosion_k=float(row["erosion_k"]),
        erosion_ucrit=float(row["erosion_ucrit"]),
        erodible_depth=float(row["erodible_depth"]),
        volume_scale=float(row["volume_scale"]),
        release_duration_s=float(row["release_duration_s"]),
        release_speed_m_s=float(row["release_speed_m_s"]),
        background_water_depth_m=float(background if background is not None else row.get("background_water_depth_m", 0.0)),
    )


def main() -> None:
    cfg = load_cfg()
    behavior = pd.read_csv(Q4 / "outputs" / "M4_behavioral_scenarios.csv")
    m3 = behavior[behavior["model"].str.startswith("M3")].copy()
    m3["robust_selection_score"] = 0.70 * m3["score"] + 0.30 * m3["holdout_csi"]
    m3 = m3.sort_values("robust_selection_score", ascending=False)
    if m3.empty:
        raise RuntimeError("M4集合中没有M3情景。")
    row = m3.iloc[0]
    base = scenario_from_row(row)
    corridor = prepare(30, "corridor")

    snapshot_times = [30.0, 90.0, 180.0, float(row["arrival_time_s"]), 600.0]
    dynamic_path = Q4 / "outputs" / "extensions" / f"{base.scenario_id}_dynamic.npz"
    metrics = run_scenario(corridor, base, cfg, dynamic_path, snapshot_times=snapshot_times)
    (Q4 / "outputs" / "extensions").mkdir(parents=True, exist_ok=True)
    (Q4 / "outputs" / "extensions" / "main_dynamic_metrics.json").write_text(
        json.dumps({"scenario": asdict(base), "metrics": metrics_dict(metrics), "snapshot_times_s": snapshot_times}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[dynamic] {base.scenario_id} snapshots={snapshot_times}")

    wide = prepare(30, "wide")
    wide_scenario = scenario_from_row(row, scenario_id=f"{base.scenario_id}_wide")
    wide_metrics = run_scenario(wide, wide_scenario, cfg, Q4 / "outputs" / "extensions" / f"{base.scenario_id}_wide.npz")
    pd.DataFrame([{**asdict(wide_scenario), **metrics_dict(wide_metrics)}]).to_csv(
        Q4 / "outputs" / "extensions" / "wide_domain_sensitivity.csv", index=False, encoding="utf-8-sig"
    )
    print(f"[wide] global_CSI={wide_metrics.global_csi:.3f} arrival={wide_metrics.arrival_time_s:.1f}s")

    rejection = {
        "status": "not_run_in_final_pipeline",
        "rejected_extension": "uniform_initial_channel_water_depth_scenarios",
        "reason": "在陡峻河床上铺设统一静水深会自行产生重力调整波，不能代表稳态基流或溃决洪水；真实扩展必须使用有来源的上游边界流量过程线。",
        "decision": "不把0.5/2.0/5.0 m任意水深情景写入论文结果，也不用于模型评分。",
    }
    (Q4 / "outputs" / "extensions" / "rejected_background_water_extension.json").write_text(
        json.dumps(rejection, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
