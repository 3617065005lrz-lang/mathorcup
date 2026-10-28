from __future__ import annotations

"""Propagate trigger-time scenarios through the reconstructed A-B-C-D corridor."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_DIR = Path(__file__).resolve().parents[1]
CONFIG = json.loads((MODEL_DIR / "config" / "model_config.json").read_text(encoding="utf-8"))
PROCESSED = MODEL_DIR / "processed"
OUTPUTS = MODEL_DIR / "outputs"
TRIGGERS = PROCESSED / "03_触发时间情景样本.parquet"
RNG = np.random.default_rng(int(CONFIG["random_seed"]) + 17)

CATALOG_UTC = pd.Timestamp(CONFIG["event_catalog_time_utc"])
TRUE_TRIGGER = (CATALOG_UTC + pd.Timedelta(seconds=float(CONFIG["event_first_detachment_offset_s"]))).tz_convert("Asia/Shanghai").tz_localize(None)
TRUE_PORT_FRONT = (CATALOG_UTC + pd.Timedelta(seconds=float(CONFIG["port_front_offset_s"]))).tz_convert("Asia/Shanghai").tz_localize(None)


def sample_speeds(n: int, stochastic: bool) -> tuple[np.ndarray, np.ndarray]:
    segments = CONFIG["propagation_segments"]
    distances_m = np.array([x["ground_distance_km"] for x in segments], dtype=float) * 1000
    if stochastic:
        speeds = np.column_stack(
            [RNG.triangular(x["speed_low_m_s"], x["speed_mode_m_s"], x["speed_high_m_s"], size=n) for x in segments]
        )
    else:
        speeds = np.tile(np.array([x["speed_mode_m_s"] for x in segments], dtype=float), (n, 1))
    return distances_m, speeds


def distance_at_elapsed(elapsed_s: np.ndarray, distances_m: np.ndarray, speeds: np.ndarray) -> np.ndarray:
    out = np.zeros(len(elapsed_s), dtype=float)
    remaining = np.maximum(elapsed_s, 0).copy()
    for j, length in enumerate(distances_m):
        time_needed = length / speeds[:, j]
        travelled = np.minimum(remaining, time_needed) * speeds[:, j]
        out += travelled
        remaining = np.maximum(remaining - time_needed, 0)
    return out / 1000


def main() -> None:
    if not TRIGGERS.exists():
        raise FileNotFoundError(TRIGGERS)
    draws = pd.read_parquet(TRIGGERS)
    draws = draws.loc[draws.model.eq("main")].copy()
    all_arrivals = []
    all_distances = []
    for (nominal, issue, stochastic), group in draws.groupby(["nominal_issue_h", "issue_time_beijing", "stochastic"]):
        valid = group.loc[group.trigger_time_beijing.notna()].copy()
        if valid.empty:
            continue
        d_m, speeds = sample_speeds(len(valid), bool(stochastic))
        segment_seconds = d_m[None, :] / speeds
        travel_seconds = segment_seconds.sum(axis=1)
        valid["travel_time_s"] = travel_seconds
        valid["arrival_front_beijing"] = valid.trigger_time_beijing + pd.to_timedelta(travel_seconds, unit="s")
        for j, seg in enumerate(CONFIG["propagation_segments"]):
            valid[f"speed_{seg['name']}_m_s"] = speeds[:, j]
            valid[f"travel_{seg['name']}_s"] = segment_seconds[:, j]
        all_arrivals.append(valid)

        for minute in [1, 3, 5, 7, 10]:
            distance = distance_at_elapsed(np.full(len(valid), minute * 60.0), d_m, speeds)
            all_distances.append(
                pd.DataFrame(
                    {
                        "nominal_issue_h": nominal,
                        "issue_time_beijing": issue,
                        "stochastic": stochastic,
                        "elapsed_min": minute,
                        "scenario": valid.scenario.to_numpy(),
                        "front_ground_distance_km": distance,
                    }
                )
            )

    arrivals = pd.concat(all_arrivals, ignore_index=True)
    distances = pd.concat(all_distances, ignore_index=True)
    arrivals.to_parquet(PROCESSED / "05_口岸到达时间情景样本.parquet", index=False)
    distances.to_parquet(PROCESSED / "06_传播距离情景样本.parquet", index=False)

    rows = []
    for keys, group in arrivals.groupby(["nominal_issue_h", "issue_time_beijing", "stochastic"]):
        nominal, issue, stochastic = keys
        arrival_q025 = group.arrival_front_beijing.quantile(0.025)
        arrival_q975 = group.arrival_front_beijing.quantile(0.975)
        row = {
                "nominal_issue_h": nominal,
                "issue_time_beijing": issue,
                "stochastic": stochastic,
                "trigger_support": len(group) / len(draws.loc[(draws.nominal_issue_h == nominal) & (draws.stochastic == stochastic)]),
                "travel_time_median_min": group.travel_time_s.median() / 60,
                "travel_time_q025_min": group.travel_time_s.quantile(0.025) / 60,
                "travel_time_q975_min": group.travel_time_s.quantile(0.975) / 60,
                "arrival_q025": arrival_q025,
                "arrival_median": group.arrival_front_beijing.quantile(0.5),
                "arrival_q975": arrival_q975,
                "covers_true_port_front": bool(arrival_q025 <= TRUE_PORT_FRONT <= arrival_q975),
                "actual_warning_lead_to_port_h": (TRUE_PORT_FRONT - issue) / pd.Timedelta(hours=1),
            }
        row["native_resolution_h"] = 6.0
        row["operational_arrival_q025"] = arrival_q025 - pd.Timedelta(hours=3)
        row["operational_arrival_q975"] = arrival_q975 + pd.Timedelta(hours=3)
        row["covers_true_port_front_operational"] = bool(
            row["operational_arrival_q025"] <= TRUE_PORT_FRONT <= row["operational_arrival_q975"]
        )
        rows.append(row)
    arrival_summary = pd.DataFrame(rows).sort_values(["stochastic", "nominal_issue_h"], ascending=[True, False])
    arrival_summary.to_csv(OUTPUTS / "08_口岸到达时间区间.csv", index=False, encoding="utf-8-sig")

    distance_summary = (
        distances.loc[distances.stochastic]
        .groupby(["nominal_issue_h", "elapsed_min"])
        .front_ground_distance_km.agg(
            distance_q025_km=lambda x: x.quantile(0.025),
            distance_median_km="median",
            distance_q975_km=lambda x: x.quantile(0.975),
        )
        .reset_index()
    )
    distance_summary.to_csv(OUTPUTS / "09_触发后前缘距离区间.csv", index=False, encoding="utf-8-sig")

    param_rows = []
    for seg in CONFIG["propagation_segments"]:
        param_rows.append(seg)
    pd.DataFrame(param_rows).to_csv(OUTPUTS / "10_传播参数表.csv", index=False, encoding="utf-8-sig")

    port_ground_km = sum(float(x["ground_distance_km"]) for x in CONFIG["propagation_segments"])
    report = f"""# 第三问传播与空间距离结果

- 路径：A—B—C—D分段沟道模型；速度范围来自近场地震追踪材料。
- 源区至口岸D点：水平中心线约{CONFIG['transport_path_horizontal_to_port_km']:.3f} km，地形修正路径约{port_ground_km:.3f} km。
- 灾后识别的完整运输走廊中心线最长约{CONFIG['observed_transport_corridor_max_km']:.3f} km，只作为回放验证上界，不泄漏进预报传播距离。
- 真实源区第一阶段脱离：{TRUE_TRIGGER}；口岸前缘候选真值：{TRUE_PORT_FRONT}。
- 传播时间的不确定性由三个路段速度范围蒙特卡洛传播得到。
- 操作区间在原始情景区间两端各扩展3小时，用于承认GEFS原始6小时步长；两类区间同时报告。

## 口岸到达区间

{arrival_summary.to_string(index=False)}

## 使用边界

本模型可靠回答“触发后何时到达关键口岸、在给定分钟内沿沟道推进多远”。由于没有独立的事前源量与完整RAMMS工程，28.828 km灾后走廊长度不被包装成独立预测出的最大冲程；最大最终淹没范围应由第四问高分辨率动力模拟补充。
"""
    (OUTPUTS / "11_传播模型结果说明.md").write_text(report, encoding="utf-8")
    print(arrival_summary.to_string(index=False))


if __name__ == "__main__":
    main()
