from __future__ import annotations

"""Independent IMERG precipitation check for the 2026 event window.

IMERG Late Run half-hourly files are used for both 2021--2024 controls and
2026, avoiding a Final/Late product mismatch.  Each rate is stamped at the end
of its 30-minute interval before applying the 10:00 Beijing event cutoff.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import rasterio


SCRIPT_DIR = Path(__file__).resolve().parent
WORK_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = WORK_DIR / "outputs"
ROOT = Path(r"D:\MathorCup_Data\External_Gyirong\P0-2_IMERG")
HISTORY_DIR = ROOT / "data_late_history"
EVENT_DIR = ROOT / "data"
EVENT_ENDPOINT = pd.Timestamp("2026-08-26 10:00:00")
WINDOWS = [24, 48, 72, 120]
SOURCE_LON_INDEX = 2655  # 85.55 E centre on the 0.1 degree global grid
SOURCE_LAT_INDEX = 1182  # 28.25 N centre on the 0.1 degree global grid
PATTERN = re.compile(r"3B-HHR-L\.MS\.MRG\.3IMERG\.(\d{8})-S(\d{6})-")


def read_point(path: Path) -> float:
    with rasterio.open(path) as container:
        candidates = [name for name in container.subdatasets if name.endswith("//Grid/precipitation")]
    if len(candidates) != 1:
        raise RuntimeError(f"IMERG precipitation subdataset missing: {path}")
    with rasterio.open(candidates[0]) as dataset:
        value = float(
            dataset.read(
                1,
                window=((SOURCE_LON_INDEX, SOURCE_LON_INDEX + 1), (SOURCE_LAT_INDEX, SOURCE_LAT_INDEX + 1)),
            )[0, 0]
        )
        nodata = dataset.nodata
    if nodata is not None and np.isclose(value, nodata):
        return np.nan
    return value if value >= 0 else np.nan


def load_half_hourly() -> pd.DataFrame:
    paths = sorted(HISTORY_DIR.glob("*.HDF5")) + sorted(EVENT_DIR.glob("*.HDF5"))
    if len(paths) != 1920:
        raise RuntimeError(f"IMERG input count mismatch: {len(paths)}/1920")
    rows = []
    for index, path in enumerate(paths, 1):
        match = PATTERN.search(path.name)
        if not match:
            raise RuntimeError(f"cannot parse IMERG timestamp: {path.name}")
        start_utc = pd.Timestamp(f"{match.group(1)} {match.group(2)}")
        end_beijing = start_utc + pd.Timedelta(hours=8, minutes=30)
        rate = read_point(path)
        rows.append(
            {
                "Timestamp": end_beijing,
                "precip_rate_mm_h": rate,
                "precip_amount_mm": rate * 0.5 if np.isfinite(rate) else np.nan,
                "file": path.name,
            }
        )
        if index % 240 == 0:
            print(f"IMERG read {index}/{len(paths)}", flush=True)
    frame = pd.DataFrame(rows).sort_values("Timestamp").drop_duplicates("Timestamp")
    return frame.set_index("Timestamp")


def robust_metrics(event_value: float, controls: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(controls, errors="coerce").dropna().astype(float)
    median = float(values.median())
    mad = float(np.median(np.abs(values - median)) * 1.4826)
    less = int((values < event_value).sum())
    equal = int((values == event_value).sum())
    percentile = (less + 0.5 * equal) / len(values)
    p_value = (1 + int((np.abs(values - median) >= abs(event_value - median)).sum())) / (len(values) + 1)
    return {
        "历史样本数": len(values),
        "历史中位数_mm": median,
        "稳健Z": (event_value - median) / mad if mad > 1e-12 else np.nan,
        "历史百分位": percentile,
        "经验双侧P": p_value,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    half_hourly = load_half_hourly()
    half_hourly.to_parquet(WORK_DIR / "cache" / "imerg_source_halfhourly.parquet")
    rows = []
    for hours in WINDOWS:
        amount = half_hourly["precip_amount_mm"]
        rolling_sum = amount.rolling(f"{hours}h", min_periods=hours * 2).sum()
        max1 = amount.rolling("1h", min_periods=2).sum().rolling(f"{hours}h").max()
        max3 = amount.rolling("3h", min_periods=6).sum().rolling(f"{hours}h").max()
        max6 = amount.rolling("6h", min_periods=12).sum().rolling(f"{hours}h").max()
        for metric, series in {
            "累计降水": rolling_sum,
            "最大1小时降水": max1,
            "最大3小时降水": max3,
            "最大6小时降水": max6,
        }.items():
            event_value = float(series.get(EVENT_ENDPOINT, np.nan))
            control_values = []
            for year in [2021, 2022, 2023, 2024]:
                endpoint = EVENT_ENDPOINT.replace(year=year)
                value = series.get(endpoint, np.nan)
                if np.isfinite(value):
                    control_values.append(value)
            metrics = robust_metrics(event_value, pd.Series(control_values))
            rows.append(
                {
                    "产品": "GPM IMERG Late Run V07",
                    "空间支持": "源区最近0.1度格点中心(85.55E,28.25N)",
                    "窗口小时": hours,
                    "指标": metric,
                    "事件值_mm": event_value,
                    "偏离方向": "偏高" if event_value > metrics["历史中位数_mm"] else "偏低",
                    **metrics,
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_DIR / "25_IMERG独立降水核验.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# IMERG源区降水独立核验",
        "",
        "- 历史与事件年均使用IMERG Late Run V07半小时产品，避免版本口径混用。",
        "- 每条降水率乘0.5小时得到雨量，并以半小时时段结束时刻对齐北京时间。",
        "- 资料从8月20日开始，因此只计算24、48、72和120小时；不外推168小时。",
        "- 每个窗口只有4个历史同日样本，百分位仅作产品间方向核验，不单独进入TOPSIS主排序。",
        "",
        "| 窗口 | 指标 | 2026事件值/mm | 历史中位数/mm | 方向 | 历史百分位 |",
        "|---:|---|---:|---:|---|---:|",
    ]
    for _, row in result.iterrows():
        lines.append(
            f"| {int(row['窗口小时'])}h | {row['指标']} | {row['事件值_mm']:.3f} | "
            f"{row['历史中位数_mm']:.3f} | {row['偏离方向']} | {row['历史百分位']:.3f} |"
        )
    (OUTPUT_DIR / "26_IMERG独立降水核验报告.md").write_text("\n".join(lines), encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
