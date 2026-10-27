from __future__ import annotations

"""Quantify how well the Gyirong station represents source-area background.

This is a data-applicability audit.  It does not enter the five-factor ranking
or alter any formal TOPSIS weights.
"""

from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
MODELING_ROOT = ROOT.parents[1]
PREPROCESSED = MODELING_ROOT / "第一问" / "01_数据预处理" / "processed" / "strict"
CACHE = ROOT / "cache"
OUTPUTS = ROOT / "outputs"

EVENT_TIME = pd.Timestamp("2026-08-26 10:00:00")
BACKGROUND_YEARS = (2021, 2022, 2023, 2024)


def read_station() -> pd.DataFrame:
    frames: dict[str, pd.DataFrame] = {}
    for prefix in ("MET", "RADM", "SOIL", "FLUX"):
        path = next(PREPROCESSED.glob(f"{prefix}_Gyirong_2021_2026_strict.parquet"))
        frame = pd.read_parquet(path).set_index("Timestamp").sort_index()
        frames[prefix] = frame

    station = pd.DataFrame(index=frames["MET"].index)
    station["temperature"] = frames["MET"]["Ta_2m(degree C)"]
    station["precipitation"] = frames["MET"]["Precip(mm)"]
    station["net_radiation"] = (
        frames["RADM"]["Rsd(W/m2)"]
        - frames["RADM"]["Rsu(W/m2)"]
        + frames["RADM"]["Rld(W/m2)"]
        - frames["RADM"]["Rlu(W/m2)"]
    )
    station["soil_moisture"] = frames["SOIL"]["VWC_0.1m(m3/m3)"]
    # Recompute VPD from the quality-controlled MET temperature/RH pair.  The
    # legacy FLUX VPD field contains physically impossible extremes and is not
    # the VPD definition used by the formal first-question feature cache.
    temperature = frames["MET"]["Ta_2m(degree C)"]
    relative_humidity = frames["MET"]["RH_2m(%)"].clip(0, 100)
    saturation_kpa = 0.6108 * np.exp(17.27 * temperature / (temperature + 237.3))
    station["vpd"] = saturation_kpa * (1.0 - relative_humidity / 100.0) * 10.0
    return station


def read_source() -> pd.DataFrame:
    source = pd.read_parquet(CACHE / "source_hourly_official_center.parquet").sort_index()
    out = pd.DataFrame(index=source.index)
    out["temperature"] = source["source_temperature"]
    out["precipitation"] = source["precipitation_center"]
    # ERA5-Land accumulated radiation is J m-2 over the hourly interval.
    out["net_radiation"] = source["net_radiation_center"] / 3600.0
    out["soil_moisture"] = source["soil_water_1_center"]
    # Source VPD is stored in kPa; the station FLUX field is hPa.
    out["vpd"] = source["vpd_center"] * 10.0
    return out


def daily_series(hourly: pd.Series, how: str) -> pd.Series:
    groups = hourly.resample("1D")
    if how == "sum":
        value = groups.sum(min_count=18)
    else:
        value = groups.mean()
        value[groups.count() < 18] = np.nan
    return value


def empirical_percentile(sample: pd.Series, value: float) -> float:
    values = sample.dropna().to_numpy(dtype=float)
    if not len(values) or not np.isfinite(value):
        return np.nan
    return float((np.sum(values < value) + 0.5 * np.sum(values == value)) / len(values))


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without the optional tabulate package."""
    show = frame.copy()
    for column in show.select_dtypes(include=["number"]).columns:
        show[column] = show[column].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    headers = [str(column) for column in show.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in show.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def event_feature(series: pd.Series, how: str) -> tuple[float, pd.Series]:
    pieces = []
    for _, group in series.groupby(series.index.year):
        ordered = group.sort_index()
        roller = ordered.rolling(72, min_periods=72)
        if how == "sum":
            value = roller.sum()
        elif how == "positive_sum":
            value = ordered.clip(lower=0).rolling(72, min_periods=72).sum()
        elif how == "p90":
            value = roller.quantile(0.9)
        elif how == "change":
            value = ordered - ordered.shift(71)
            value[roller.count() < 72] = np.nan
        else:
            value = roller.mean()
        pieces.append(value)
    rolled = pd.concat(pieces).sort_index()
    event_value = float(rolled.get(EVENT_TIME, np.nan))
    controls = []
    for year in BACKGROUND_YEARS:
        anchor = EVENT_TIME.replace(year=year)
        mask = (
            (rolled.index.year == year)
            & (rolled.index.hour == EVENT_TIME.hour)
            & (np.abs((rolled.index.normalize() - anchor.normalize()).days) <= 15)
        )
        controls.append(rolled.loc[mask])
    return event_value, pd.concat(controls)


def main() -> None:
    station = read_station()
    source = read_source()

    specs = {
        "temperature": {
            "factor": "空气温度与热量累积",
            "station_field": "Ta_2m(degree C)",
            "source_field": "ERA5-Land邻域温度—高程订正至4991 m",
            "unit": "°C",
            "daily": "mean",
            "event": "positive_sum",
            "risk_direction": "偏高",
            "note": "重点比较同步变化；源区海拔更高，不要求绝对温度与站点相等",
        },
        "precipitation": {
            "factor": "降水条件",
            "station_field": "Precip(mm)",
            "source_field": "ERA5-Land源区格点降水",
            "unit": "mm",
            "daily": "sum",
            "event": "sum",
            "risk_direction": "偏低",
            "note": "高山降水空间差异大；只用于检验能否空间迁移，不替代IMERG独立核验",
        },
        "net_radiation": {
            "factor": "辐射与地表能量",
            "station_field": "Rsd-Rsu+Rld-Rlu",
            "source_field": "ERA5-Land源区净辐射",
            "unit": "W/m²",
            "daily": "mean",
            "event": "mean",
            "risk_direction": "偏高",
            "note": "再分析与地面辐射计空间支持不同，主要比较变化方向",
        },
        "soil_moisture": {
            "factor": "土壤湿润状态",
            "station_field": "VWC_0.1m(m3/m3)",
            "source_field": "ERA5-Land表层土壤水分",
            "unit": "m³/m³",
            "daily": "mean",
            "event": "change",
            "risk_direction": "偏低",
            "note": "二者代表区域陆面湿润背景，不能解释冰体内部或沟道含水量",
        },
        "vpd": {
            "factor": "大气湿度与干湿状态",
            "station_field": "由Ta_2m与RH_2m重算VPD",
            "source_field": "ERA5-Land源区VPD",
            "unit": "hPa",
            "daily": "mean",
            "event": "p90",
            "risk_direction": "偏高",
            "note": "VPD由温湿度共同决定，用于比较暖干背景同步性",
        },
    }

    rows = []
    for key, spec in specs.items():
        station_daily = daily_series(station[key], spec["daily"])
        source_daily = daily_series(source[key], spec["daily"])
        pair = pd.concat(
            [station_daily.rename("station"), source_daily.rename("source")], axis=1
        ).dropna()
        pair = pair.loc[
            pair.index.year.isin(BACKGROUND_YEARS)
            & pair.index.month.isin((6, 7, 8, 9))
        ]

        station_event, station_control = event_feature(station[key], spec["event"])
        source_event, source_control = event_feature(source[key], spec["event"])
        station_pct = empirical_percentile(station_control, station_event)
        source_pct = empirical_percentile(source_control, source_event)
        if spec["risk_direction"] == "偏高":
            direction_match = (station_pct >= 0.5) == (source_pct >= 0.5)
        else:
            direction_match = (station_pct <= 0.5) == (source_pct <= 0.5)

        station_state = pair["station"] >= pair["station"].median()
        source_state = pair["source"] >= pair["source"].median()
        rows.append(
            {
                "物理因素": spec["factor"],
                "站点字段": spec["station_field"],
                "源区参照": spec["source_field"],
                "单位": spec["unit"],
                "日尺度聚合": "日累计" if spec["daily"] == "sum" else "日平均",
                "共同有效日数": len(pair),
                "Pearson相关系数": pair["station"].corr(pair["source"], method="pearson"),
                "Spearman相关系数": pair["station"].corr(pair["source"], method="spearman"),
                "源区减站点平均偏差": (pair["source"] - pair["station"]).mean(),
                "平均绝对误差": (pair["source"] - pair["station"]).abs().mean(),
                "高低状态一致率": float((station_state == source_state).mean()),
                "风险方向": spec["risk_direction"],
                "站点事件前72小时值": station_event,
                "站点事件历史百分位": station_pct,
                "源区事件前72小时值": source_event,
                "源区事件历史百分位": source_pct,
                "事件异常方向一致": "是" if direction_match else "否",
                "解释边界": spec["note"],
            }
        )

    result = pd.DataFrame(rows)
    out_csv = OUTPUTS / "37_站点源区空间迁移可靠性.csv"
    result.to_csv(out_csv, index=False, encoding="utf-8-sig")

    lines = [
        "# 站点—源区空间迁移可靠性",
        "",
        "## 定位",
        "",
        "本分析量化题目观测站与2026灾害源区附近ERA5-Land重建之间的同步性。它属于数据适用性和空间尺度可靠性，不进入第一问五因素排序、TOPSIS权重或显著性门控。",
        "",
        "- 背景期：2021—2024年6—9月共同有效日。",
        "- 时间同步：北京时间。",
        "- 事件核查：2026-08-26 10:00以前72小时，相对于历史各年同期±15天、同为10:00的样本。",
        "- 温度源区值由ERA5-Land 9×9邻域2 m温度—格点高程关系订正至4991 m；不是源区实测。",
        "",
        "## 结果",
        "",
        markdown_table(result),
        "",
        "## 论文解释规则",
        "",
        "1. 相关性较高只说明区域同步变化较强，不表示站点绝对值可以直接替代源区。",
        "2. 降水若相关性较低，应明确写成局地性强，站点降水不能搬移到约65 km外的高山源区。",
        "3. 土壤水分只代表区域陆面背景，不等同于冰体内部、裂隙水或沟道水量。",
        "4. 事件前异常方向一致可作为第一问外源佐证；方向不一致时应分别报告，不能选择性删除。",
        "5. 本表不改变题目附件指标作为第一问主体证据的地位。",
    ]
    (OUTPUTS / "38_站点源区空间迁移可靠性报告.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"Wrote {out_csv.name} and 38 report ({len(result)} rows).")


if __name__ == "__main__":
    main()
