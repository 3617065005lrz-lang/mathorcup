from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from numba import njit


G = 9.81
H_DRY = 1.0e-4


@dataclass
class Scenario:
    scenario_id: str
    model: str
    mu_source: float
    xi_source: float
    mu_valley: float
    xi_valley: float
    erosion_k: float = 0.0
    erosion_ucrit: float = 8.0
    erodible_depth: float = 0.0
    volume_scale: float = 1.0
    release_duration_s: float = 90.0
    release_speed_m_s: float = 45.0
    background_water_depth_m: float = 0.0


@dataclass
class Metrics:
    scenario_id: str
    model: str
    resolution_m: float
    status: str
    arrival_time_s: float
    arrival_error_s: float
    max_speed_m_s: float
    speed_p99_m_s: float
    max_depth_m: float
    max_impact_intensity_m3_s2: float
    max_dynamic_pressure_pa: float
    initial_volume_m3: float
    background_water_volume_m3: float
    entrained_volume_m3: float
    final_volume_m3: float
    numerical_volume_error_fraction: float
    calibration_csi: float
    calibration_recall: float
    calibration_precision: float
    holdout_csi: float
    holdout_recall: float
    holdout_precision: float
    global_csi: float
    global_recall: float
    global_precision: float
    score: float


def _build_sparse(active: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = active.shape
    aidx = np.full((rows, cols), -1, np.int32)
    rr, cc = np.where(active)
    aidx[rr, cc] = np.arange(rr.size, dtype=np.int32)
    fx_l, fx_r, fy_l, fy_r = [], [], [], []
    for r, c in zip(rr.tolist(), cc.tolist()):
        left = int(aidx[r, c])
        if c + 1 < cols and aidx[r, c + 1] >= 0:
            fx_l.append(left)
            fx_r.append(int(aidx[r, c + 1]))
        if r + 1 < rows and aidx[r + 1, c] >= 0:
            fy_l.append(left)
            fy_r.append(int(aidx[r + 1, c]))
    return aidx, np.asarray(fx_l, np.int32), np.asarray(fx_r, np.int32), np.asarray(fy_l, np.int32), np.asarray(fy_r, np.int32)


@njit(cache=True)
def _hll_flux(h_l, h_r, un_l, un_r, ut_l, ut_r):
    c_l = math.sqrt(G * h_l)
    c_r = math.sqrt(G * h_r)
    s_l = min(un_l - c_l, un_r - c_r)
    s_r = max(un_l + c_l, un_r + c_r)
    f_l0 = h_l * un_l
    f_r0 = h_r * un_r
    f_l1 = h_l * un_l * un_l + 0.5 * G * h_l * h_l
    f_r1 = h_r * un_r * un_r + 0.5 * G * h_r * h_r
    if s_l >= 0.0:
        f0, f1 = f_l0, f_l1
    elif s_r <= 0.0:
        f0, f1 = f_r0, f_r1
    else:
        den = max(1.0e-12, s_r - s_l)
        f0 = (s_r * f_l0 - s_l * f_r0 + s_l * s_r * (h_r - h_l)) / den
        f1 = (
            s_r * f_l1
            - s_l * f_r1
            + s_l * s_r * (h_r * un_r - h_l * un_l)
        ) / den
    f2 = f0 * (ut_l if f0 >= 0.0 else ut_r)
    return f0, f1, f2


@njit(cache=True)
def _advance(
    h,
    qx,
    qy,
    z,
    fx_l,
    fx_r,
    fy_l,
    fy_r,
    mu,
    xi,
    cos_theta,
    erodible,
    erosion_k,
    erosion_ucrit,
    dt,
    dx,
):
    n = h.size
    dh = np.zeros(n, np.float64)
    dqx = np.zeros(n, np.float64)
    dqy = np.zeros(n, np.float64)

    for k in range(fx_l.size):
        l, r = fx_l[k], fx_r[k]
        hl, hr = h[l], h[r]
        zl, zr = z[l], z[r]
        zstar = max(zl, zr)
        hlt = max(0.0, hl + zl - zstar)
        hrt = max(0.0, hr + zr - zstar)
        ul = qx[l] / hl if hl > H_DRY else 0.0
        ur = qx[r] / hr if hr > H_DRY else 0.0
        vl = qy[l] / hl if hl > H_DRY else 0.0
        vr = qy[r] / hr if hr > H_DRY else 0.0
        f0, f1, f2 = _hll_flux(hlt, hrt, ul, ur, vl, vr)
        pl = 0.5 * G * (hl * hl - hlt * hlt)
        pr = 0.5 * G * (hr * hr - hrt * hrt)
        dh[l] -= f0
        dh[r] += f0
        dqx[l] -= f1 + pl
        dqx[r] += f1 + pr
        dqy[l] -= f2
        dqy[r] += f2

    for k in range(fy_l.size):
        l, r = fy_l[k], fy_r[k]
        hl, hr = h[l], h[r]
        zl, zr = z[l], z[r]
        zstar = max(zl, zr)
        hlt = max(0.0, hl + zl - zstar)
        hrt = max(0.0, hr + zr - zstar)
        vl = qy[l] / hl if hl > H_DRY else 0.0
        vr = qy[r] / hr if hr > H_DRY else 0.0
        ul = qx[l] / hl if hl > H_DRY else 0.0
        ur = qx[r] / hr if hr > H_DRY else 0.0
        f0, f1, f2 = _hll_flux(hlt, hrt, vl, vr, ul, ur)
        pl = 0.5 * G * (hl * hl - hlt * hlt)
        pr = 0.5 * G * (hr * hr - hrt * hrt)
        dh[l] -= f0
        dh[r] += f0
        dqy[l] -= f1 + pl
        dqy[r] += f1 + pr
        dqx[l] -= f2
        dqx[r] += f2

    factor = dt / dx
    entrained_depth_sum = 0.0
    for i in range(n):
        hn = h[i] + factor * dh[i]
        qxn = qx[i] + factor * dqx[i]
        qyn = qy[i] + factor * dqy[i]
        if hn <= H_DRY or not math.isfinite(hn):
            h[i], qx[i], qy[i] = max(0.0, hn), 0.0, 0.0
            continue
        if hn < 0.0:
            hn = 0.0
        speed = math.sqrt(qxn * qxn + qyn * qyn) / max(hn, H_DRY)
        # RAMMS/Voellmy-Salm：S = mu*rho*g*h*cos(theta) + rho*g*u^2/xi。
        # 除以单位面积质量rho*h后得到减速度。
        decel = mu[i] * G * cos_theta[i] + G * speed * speed / (max(xi[i], 1.0) * max(hn, H_DRY))
        speed_new = max(0.0, speed - decel * dt)
        # 干湿前缘的厘米级薄层可能产生不代表主体物质的单格速度尖峰。
        # 180 m/s高于由源区有效落差估算的主体速度，作为显式数值安全上限。
        speed_new = min(speed_new, 180.0)
        scale = speed_new / speed if speed > 1.0e-10 else 0.0
        qxn *= scale
        qyn *= scale

        if erosion_k > 0.0 and erodible[i] > 0.0 and speed_new > erosion_ucrit:
            de = min(erodible[i], erosion_k * (speed_new - erosion_ucrit) * dt)
            erodible[i] -= de
            z[i] -= de
            hn += de
            entrained_depth_sum += de

        if hn < 0.02:
            qxn, qyn = 0.0, 0.0
        h[i], qx[i], qy[i] = hn, qxn, qyn
    return entrained_depth_sum


def _binary_metrics(pred: np.ndarray, obs: np.ndarray, domain: np.ndarray) -> tuple[float, float, float]:
    p = pred & domain
    o = obs & domain
    tp = int(np.count_nonzero(p & o))
    fp = int(np.count_nonzero(p & ~o))
    fn = int(np.count_nonzero(~p & o))
    union = tp + fp + fn
    csi = tp / union if union else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    return csi, recall, precision


def _safe(value: float) -> float:
    return 0.0 if not np.isfinite(value) else float(value)


def run_scenario(
    prepared_path: Path,
    scenario: Scenario,
    cfg: dict,
    save_arrays: Path | None = None,
    snapshot_times: list[float] | None = None,
) -> Metrics:
    with np.load(prepared_path, allow_pickle=False) as src:
        data = {name: src[name].copy() for name in src.files}
    active2d = data["active"].astype(bool)
    aidx, fx_l, fx_r, fy_l, fy_r = _build_sparse(active2d)
    flat = np.flatnonzero(active2d.ravel())
    z = data["z"].ravel()[flat].astype(np.float64)
    chain = data["upper_chainage_m"].ravel()[flat].astype(np.float64)
    source = data["source_mask"].ravel()[flat].astype(bool)
    cell = float(data["cell_m"])

    n = flat.size
    h = np.zeros(n, np.float64)
    qx = np.zeros(n, np.float64)
    qy = np.zeros(n, np.float64)
    # 高位崩塌不是浅水流。二维方程从源区物质撞击沟床、转化为沟道入流后开始：
    # 以0--600 m上游里程带作为入流口，将源区体积在有限历时内注入，并按局地最陡下降
    # 方向赋予经能量折减后的初速度。这样避免把高位自由落体错误地当成静水柱。
    release = np.isfinite(chain) & (chain >= 0.0) & (chain <= 600.0)
    if not np.any(release):
        raise RuntimeError("未找到源区—沟道转换带。")
    initial_volume = float(data["source_volume_m3"]) * scenario.volume_scale
    background = np.isfinite(chain) & (chain >= 600.0)
    if scenario.background_water_depth_m > 0.0:
        h[background] = scenario.background_water_depth_m
    background_h = h.copy()
    background_volume = float(h.sum() * cell * cell)
    full_z = data["z"].astype(np.float64)
    grad_r, grad_c = np.gradient(full_z, cell, cell)
    dir_x_full = -grad_c
    dir_y_full = -grad_r
    norm = np.hypot(dir_x_full, dir_y_full)
    dir_x_full = np.divide(dir_x_full, norm, out=np.zeros_like(norm), where=norm > 1.0e-8)
    dir_y_full = np.divide(dir_y_full, norm, out=np.zeros_like(norm), where=norm > 1.0e-8)
    dir_x = dir_x_full.ravel()[flat]
    dir_y = dir_y_full.ravel()[flat]
    hmax = h.copy()
    excess_hmax = np.zeros(n, np.float64)
    vmax = np.zeros(n, np.float64)
    imax = np.zeros(n, np.float64)
    pmax = np.zeros(n, np.float64)
    arrival = np.full(n, np.nan, np.float64)

    source_zone = (~np.isfinite(chain)) | (chain <= float(cfg["domain"]["zone_break_chainage_m"]))
    mu = np.where(source_zone, scenario.mu_source, scenario.mu_valley).astype(np.float64)
    xi = np.where(source_zone, scenario.xi_source, scenario.xi_valley).astype(np.float64)
    slope = data["slope"].ravel()[flat].astype(np.float64)
    cos_theta = 1.0 / np.sqrt(1.0 + slope * slope)
    erodible = np.where(source_zone, 0.4 * scenario.erodible_depth, scenario.erodible_depth).astype(np.float64)
    erodible[source] = 0.0
    erodible_initial = erodible.copy()

    port_r, port_c = int(data["port_row"]), int(data["port_col"])
    port_idx = int(aidx[port_r, port_c])
    if port_idx < 0:
        raise RuntimeError("口岸单元不在活动计算域中。")

    t = 0.0
    injected_volume = 0.0
    aborted_speed = False
    requested_snapshots = sorted(snapshot_times or [])
    snapshots: dict[str, np.ndarray] = {}
    next_snapshot = 0
    t_end = float(cfg["domain"]["t_end_s"])
    threshold = float(cfg["domain"]["arrival_depth_threshold_m"])
    max_dt = float(cfg["domain"]["max_dt_s"])
    cfl = float(cfg["domain"]["cfl"])
    entrained_depth_sum = 0.0
    while t < t_end:
        wet = h > H_DRY
        speed = np.zeros_like(h)
        speed[wet] = np.hypot(qx[wet], qy[wet]) / h[wet]
        wave = speed + np.sqrt(G * np.maximum(h, 0.0))
        dt = min(max_dt, cfl * cell / max(1.0e-6, float(wave.max())), t_end - t)
        # 先加入本时间步的源区水石混合物，再进行通量传播。
        if injected_volume < initial_volume and t < scenario.release_duration_s:
            remaining_time = max(1.0e-6, scenario.release_duration_s - t)
            add_volume = min(initial_volume - injected_volume, initial_volume * dt / scenario.release_duration_s)
            # 最后一步用剩余体积闭合，避免浮点累计少注入。
            if dt >= remaining_time:
                add_volume = initial_volume - injected_volume
            dh_release = add_volume / (float(np.count_nonzero(release)) * cell * cell)
            h_old = h[release].copy()
            h[release] += dh_release
            qx[release] += dh_release * scenario.release_speed_m_s * dir_x[release]
            qy[release] += dh_release * scenario.release_speed_m_s * dir_y[release]
            # 上式等价于新旧动量守恒混合；h_old保留仅用于明确该步不是速度覆盖。
            del h_old
            injected_volume += add_volume

        entrained_depth_sum += _advance(
            h,
            qx,
            qy,
            z,
            fx_l,
            fx_r,
            fy_l,
            fy_r,
            mu,
            xi,
            cos_theta,
            erodible,
            scenario.erosion_k,
            scenario.erosion_ucrit,
            dt,
            cell,
        )
        t += dt
        wet = h > H_DRY
        speed.fill(0.0)
        speed[wet] = np.hypot(qx[wet], qy[wet]) / h[wet]
        hmax = np.maximum(hmax, h)
        excess_hmax = np.maximum(excess_hmax, np.maximum(h - background_h, 0.0))
        vmax = np.maximum(vmax, speed)
        imax = np.maximum(imax, h * speed * speed)
        pmax = np.maximum(pmax, 0.5 * 2000.0 * speed * speed)
        # 极端低摩阻组合会产生明显非物理的超高速并令CFL步长趋近于零。
        # 一旦超过物理筛选上限的缓冲值即停止该情景，而不是浪费计算继续积分。
        if float(speed.max()) > 400.0:
            aborted_speed = True
            break
        just = np.isnan(arrival) & ((h - background_h) >= threshold)
        arrival[just] = t
        while next_snapshot < len(requested_snapshots) and t >= requested_snapshots[next_snapshot]:
            stamp = int(round(requested_snapshots[next_snapshot]))
            snapshots[f"depth_t{stamp:04d}s"] = h.copy()
            snapshots[f"speed_t{stamp:04d}s"] = speed.copy()
            next_snapshot += 1

    shape = active2d.shape
    def full(values: np.ndarray, fill_value=0.0):
        out = np.full(shape[0] * shape[1], fill_value, np.float32)
        out[flat] = values.astype(np.float32)
        return out.reshape(shape)

    hmax2d = full(hmax)
    excess_hmax2d = full(excess_hmax)
    vmax2d = full(vmax)
    arrival2d = full(arrival, np.nan)
    final_h2d = full(h)
    imax2d = full(imax)
    pmax2d = full(pmax)
    erosion2d = full(erodible_initial - erodible)
    predicted = excess_hmax2d >= float(cfg["domain"]["runout_depth_threshold_m"])
    observed = data["observed_footprint"].astype(bool)
    chain2d = data["upper_chainage_m"]
    finite = np.isfinite(chain2d)
    port_chain = float(data["port_chainage_m"])
    evaluation = active2d & finite & (chain2d >= 600.0)
    calibration_domain = evaluation & (chain2d <= port_chain)
    holdout_domain = evaluation & (chain2d > port_chain)
    cal = _binary_metrics(predicted, observed, calibration_domain)
    hold = _binary_metrics(predicted, observed, holdout_domain)
    global_domain = active2d & ~data["source_mask"].astype(bool)
    global_fit = _binary_metrics(predicted, observed, global_domain)

    port_arrival = float(arrival[port_idx]) if np.isfinite(arrival[port_idx]) else float("nan")
    target = float(cfg["event"]["target_travel_time_s"])
    error = port_arrival - target if np.isfinite(port_arrival) else float("nan")
    arrival_skill = math.exp(-abs(error) / float(cfg["scoring"]["arrival_scale_s"])) if np.isfinite(error) else 0.0
    score = (
        float(cfg["scoring"]["arrival_weight"]) * arrival_skill
        + float(cfg["scoring"]["calibration_csi_weight"]) * _safe(cal[0])
        + float(cfg["scoring"]["calibration_recall_weight"]) * _safe(cal[1])
    )
    final_volume = float(h.sum() * cell * cell)
    entrained_volume = float(entrained_depth_sum * cell * cell)
    expected = injected_volume + background_volume + entrained_volume
    numerical_error = (final_volume - expected) / max(expected, 1.0)
    status = "ok"
    if aborted_speed:
        status = "rejected_speed"
    elif not np.isfinite(port_arrival):
        status = "no_port_arrival"
    elif float(vmax.max()) > 250.0:
        status = "rejected_speed"
    elif abs(numerical_error) > 0.03:
        status = "rejected_mass"

    if save_arrays is not None:
        save_arrays.parent.mkdir(parents=True, exist_ok=True)
        extra = {}
        for name, values in snapshots.items():
            extra[name] = full(values)
        np.savez_compressed(
            save_arrays,
            hmax_m=hmax2d,
            max_excess_depth_m=excess_hmax2d,
            vmax_m_s=vmax2d,
            arrival_time_s=arrival2d,
            final_depth_m=final_h2d,
            max_impact_intensity_m3_s2=imax2d,
            max_dynamic_pressure_pa=pmax2d,
            erosion_depth_m=erosion2d,
            observed_footprint=observed,
            active=active2d,
            upper_chainage_m=chain2d,
            z=data["z"],
            cell_m=np.float32(cell),
            xmin=data["xmin"],
            ymax=data["ymax"],
            **extra,
        )

    return Metrics(
        scenario_id=scenario.scenario_id,
        model=scenario.model,
        resolution_m=cell,
        status=status,
        arrival_time_s=port_arrival,
        arrival_error_s=error,
        max_speed_m_s=float(vmax.max()),
        speed_p99_m_s=float(np.percentile(vmax[vmax > 0.0], 99)) if np.any(vmax > 0.0) else 0.0,
        max_depth_m=float(hmax.max()),
        max_impact_intensity_m3_s2=float(imax.max()),
        max_dynamic_pressure_pa=float(pmax.max()),
        initial_volume_m3=initial_volume,
        background_water_volume_m3=background_volume,
        entrained_volume_m3=entrained_volume,
        final_volume_m3=final_volume,
        numerical_volume_error_fraction=numerical_error,
        calibration_csi=cal[0],
        calibration_recall=cal[1],
        calibration_precision=cal[2],
        holdout_csi=hold[0],
        holdout_recall=hold[1],
        holdout_precision=hold[2],
        global_csi=global_fit[0],
        global_recall=global_fit[1],
        global_precision=global_fit[2],
        score=score,
    )


def metrics_dict(metrics: Metrics) -> dict:
    return asdict(metrics)


def scenario_dict(scenario: Scenario) -> dict:
    return asdict(scenario)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
