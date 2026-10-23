from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PREPROCESS_DIR = SCRIPT_DIR.parent
WORKSPACE_DIR = SCRIPT_DIR.parents[3]
SOURCE_DIR = WORKSPACE_DIR / "work" / "mathorcup_b" / "a1_hourly"
OUTPUT_DIR = PREPROCESS_DIR / "outputs"

EVENTS = {
    "2025事件": pd.Timestamp("2025-07-08 05:40:00"),
    "2026事件": pd.Timestamp("2026-08-26 10:30:00"),
}

SENTINEL_VALUE = 9999.9

QC_MEANINGS = {
    "0": ("正确", "严格样本和扩展样本均保留"),
    "1": ("可疑", "主分析排除，单独复核或用于敏感性分析"),
    "2": ("错误", "排除并转换为缺失"),
    "4": ("插补数据", "扩展样本保留，严格样本排除，并保留插补标记"),
    "8": ("数据缺失", "转换为缺失"),
    "9": ("未进行质量控制", "主分析排除，单独复核"),
}


def parse_header(column: str) -> tuple[str, str]:
    match = re.match(r"^(.*?)\((.*?)\)$", column)
    if not match:
        return column, ""
    return match.group(1), match.group(2)


def classify_factor(column: str) -> tuple[str, str]:
    base, _ = parse_header(column)
    key = base.lower()
    if key == "timestamp":
        return "时间", "时间索引"
    if key.startswith("qc_"):
        return "质量控制", "用于判断对应原始观测的质量等级，码义待核实"
    if key.startswith("precip"):
        return "降水与水分输入", "累计降水、最大短历时降水、前期降雨指数API"
    if key.startswith("vwc"):
        return "土壤湿润状态", "土壤湿度异常、增量、垂直梯度和高湿持续时间"
    if key.startswith("soilt"):
        return "土壤温度与冻融", "冻融次数、连续正温时长、温度梯度和温度异常"
    if key.startswith("ta_") or key in {"t_sonic", "tstar"}:
        return "空气温度与热量累积", "滚动均温、温度异常、正积温和垂直温度梯度"
    if key.startswith("rsd") or key.startswith("rsu") or key.startswith("rld") or key.startswith("rlu"):
        return "辐射能量", "净短波、净长波、净辐射、累计正净辐射和反照率"
    if key.startswith("rh_") or key in {
        "amb_e", "amb_e_sat", "e", "e_sat", "e_probe", "e_sat_probe",
        "h2o_probe", "h2o_density", "vpd",
    }:
        return "大气湿度与水汽", "湿度异常、水汽压亏缺、高湿持续时间和垂直湿度梯度"
    if key.startswith("pressure") or key == "pa":
        return "气压与天气过程", "气压异常、气压变化率和低压过程持续时间"
    if key.startswith("ws_") or key.startswith("wd_") or key in {
        "ux", "uy", "uz", "tau", "ustar", "tke",
    }:
        return "风场与湍流", "风速切变、风向变化、摩擦速度和湍流强度异常"
    if key in {"le", "h"}:
        return "地表湍流热交换", "感热与潜热累计、异常值和波文比"
    if key.startswith("g_"):
        return "土壤热通量", "土壤热通量累计、异常和不同深度差异"
    if key.startswith("fetch"):
        return "通量足迹范围", "用于通量观测代表性检查，不作为优先致灾因素"
    if key.startswith("fc_mass"):
        return "碳通量", "用于辅助判断生态交换，暂不作为核心候选因素"
    return "待确认", "待结合数据说明核实"


def physical_range(column: str) -> tuple[float | None, float | None]:
    base, _ = parse_header(column)
    key = base.lower()
    if key.startswith("ws_"):
        return 0.0, 100.0
    if key.startswith("wd_"):
        return 0.0, 360.0
    if key.startswith("rh_"):
        return 0.0, 100.0
    if key.startswith("precip"):
        return 0.0, 1000.0
    if key.startswith("vwc"):
        return 0.0, 1.0
    if key.startswith("pressure"):
        return 300.0, 1100.0
    if key.startswith("ta_") or key.startswith("soilt") or key in {"t_sonic", "tstar"}:
        return -80.0, 80.0
    if key.startswith("rsd") or key.startswith("rsu") or key.startswith("rld") or key.startswith("rlu"):
        return -100.0, 2000.0
    return None, None


def is_sentinel(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.Series(np.isclose(numeric, SENTINEL_VALUE), index=series.index)


def profile_file(path: Path):
    group_match = re.match(r"(MET|RADM|SOIL|FLUX)_Gyirong_(\d{4})\.csv$", path.name)
    if not group_match:
        raise ValueError(f"无法识别文件名：{path.name}")
    group, year = group_match.group(1), int(group_match.group(2))
    frame = pd.read_csv(path, low_memory=False)
    timestamps = pd.to_datetime(frame["Timestamp"], errors="coerce")
    observation_columns = [
        column
        for column in frame.columns
        if column != "Timestamp" and not column.lower().startswith("qc_")
    ]
    observation_numeric = frame[observation_columns].apply(pd.to_numeric, errors="coerce")
    observation_sentinel = pd.DataFrame(
        np.isclose(observation_numeric, SENTINEL_VALUE),
        index=observation_numeric.index,
        columns=observation_numeric.columns,
    )
    observation_valid = observation_numeric.notna() & ~observation_sentinel
    qc_lookup = {
        column[3:].lower(): column
        for column in frame.columns
        if column.lower().startswith("qc_")
    }
    strict_valid = observation_valid.copy()
    expanded_valid = observation_valid.copy()
    paired_qc_columns: dict[str, str] = {}
    for column in observation_columns:
        base, _ = parse_header(column)
        qc_column = qc_lookup.get(base.lower())
        if qc_column is None:
            continue
        paired_qc_columns[column] = qc_column
        qc_numeric = pd.to_numeric(frame[qc_column], errors="coerce")
        strict_valid[column] = observation_valid[column] & qc_numeric.eq(0)
        expanded_valid[column] = observation_valid[column] & qc_numeric.isin([0, 4])
    row_any_valid = observation_valid.any(axis=1)
    row_all_valid = observation_valid.all(axis=1)
    row_all_strict = strict_valid.all(axis=1)
    row_all_expanded = expanded_valid.all(axis=1)
    valid_observation_times = timestamps[row_any_valid & timestamps.notna()]
    fully_valid_times = timestamps[row_all_valid & timestamps.notna()]
    valid_ts = timestamps.dropna().sort_values()
    unique_ts = valid_ts.drop_duplicates()

    if len(unique_ts):
        expected = pd.date_range(unique_ts.iloc[0], unique_ts.iloc[-1], freq="h")
        missing_hours = len(expected.difference(pd.DatetimeIndex(unique_ts)))
        non_hourly_rows = int(
            ((valid_ts.dt.minute != 0) | (valid_ts.dt.second != 0)).sum()
        )
        time_steps = unique_ts.diff().dropna()
        irregular_gaps = int((time_steps != pd.Timedelta(hours=1)).sum())
        min_ts = unique_ts.iloc[0]
        max_ts = unique_ts.iloc[-1]
    else:
        missing_hours = 0
        non_hourly_rows = 0
        irregular_gaps = 0
        min_ts = pd.NaT
        max_ts = pd.NaT

    inventory = {
        "数据组": group,
        "年份": year,
        "文件名": path.name,
        "文件大小_字节": path.stat().st_size,
        "行数": len(frame),
        "列数": len(frame.columns),
        "开始时间": min_ts,
        "结束时间": max_ts,
        "开始有效观测时间": valid_observation_times.min() if len(valid_observation_times) else pd.NaT,
        "结束有效观测时间": valid_observation_times.max() if len(valid_observation_times) else pd.NaT,
        "开始全字段有效时间": fully_valid_times.min() if len(fully_valid_times) else pd.NaT,
        "结束全字段有效时间": fully_valid_times.max() if len(fully_valid_times) else pd.NaT,
        "至少一项观测有效行数": int(row_any_valid.sum()),
        "全部原始观测有效行数": int(row_all_valid.sum()),
        "全部观测严格可用行数": int(row_all_strict.sum()),
        "全部观测扩展可用行数": int(row_all_expanded.sum()),
        "全观测缺测占位行数": int((~row_any_valid).sum()),
        "无效时间行数": int(timestamps.isna().sum()),
        "重复时间行数": int(timestamps.duplicated(keep=False).sum()),
        "缺失整点数": missing_hours,
        "非整点记录数": non_hourly_rows,
        "非1小时时间间隔数": irregular_gaps,
    }

    dictionary_rows = []
    quality_rows = []
    qc_rows = []
    for column in frame.columns:
        base, unit = parse_header(column)
        factor, derived = classify_factor(column)
        dictionary_rows.append(
            {
                "数据组": group,
                "字段名": column,
                "基础变量名": base,
                "单位": unit,
                "字段角色": "时间" if column == "Timestamp" else ("质量控制" if column.lower().startswith("qc_") else "原始观测"),
                "对应物理因素": factor,
                "候选衍生指标": derived,
                "首次出现年份": year,
            }
        )

        if column == "Timestamp":
            continue
        series = frame[column]
        numeric = pd.to_numeric(series, errors="coerce")
        sentinel_mask = is_sentinel(series)
        blank_mask = series.isna()
        non_numeric_mask = numeric.isna() & ~blank_mask
        valid_numeric = numeric[~sentinel_mask]
        lower, upper = physical_range(column)
        if lower is None:
            range_flag_count = 0
        else:
            range_flag_count = int(((valid_numeric < lower) | (valid_numeric > upper)).sum())

        qc_column = paired_qc_columns.get(column, "")
        if qc_column:
            qc_numeric_for_observation = pd.to_numeric(frame[qc_column], errors="coerce")
            qc0_count = int(qc_numeric_for_observation.eq(0).sum())
            qc4_count = int(qc_numeric_for_observation.eq(4).sum())
            strict_usable_count = int((numeric.notna() & ~sentinel_mask & qc_numeric_for_observation.eq(0)).sum())
            expanded_usable_count = int(
                (numeric.notna() & ~sentinel_mask & qc_numeric_for_observation.isin([0, 4])).sum()
            )
        else:
            qc0_count = np.nan
            qc4_count = np.nan
            strict_usable_count = int((numeric.notna() & ~sentinel_mask).sum())
            expanded_usable_count = strict_usable_count

        quality_rows.append(
            {
                "数据组": group,
                "年份": year,
                "字段名": column,
                "总行数": len(series),
                "空值数": int(blank_mask.sum()),
                "非数值数": int(non_numeric_mask.sum()),
                "疑似缺测码数": int(sentinel_mask.sum()),
                "暂定有效数": int((numeric.notna() & ~sentinel_mask).sum()),
                "对应QC字段": qc_column,
                "QC0数量": qc0_count,
                "QC4数量": qc4_count,
                "严格可用数": strict_usable_count,
                "扩展可用数": expanded_usable_count,
                "物理范围标记数": range_flag_count,
                "最小值_排除缺测码": valid_numeric.min(skipna=True),
                "最大值_排除缺测码": valid_numeric.max(skipna=True),
            }
        )

        if column.lower().startswith("qc_"):
            counts = Counter(series.dropna().astype(str))
            for code, count in sorted(counts.items()):
                normalized_code = code[:-2] if code.endswith(".0") else code
                meaning, action = QC_MEANINGS.get(
                    normalized_code,
                    ("附件说明未定义", "暂不自动处理"),
                )
                qc_rows.append(
                    {
                        "数据组": group,
                        "年份": year,
                        "质量控制字段": column,
                        "质量控制码": code,
                        "官方含义": meaning,
                        "数量": count,
                        "占非空记录比例": count / max(1, series.notna().sum()),
                        "建议处理": action,
                    }
                )

    event_rows = []
    for event_name, event_time in EVENTS.items():
        if event_time.year != year:
            continue
        index = pd.DatetimeIndex(unique_ts)
        before = index[index <= event_time]
        after = index[index >= event_time]
        event_frame = pd.DataFrame(
            {
                "Timestamp": timestamps,
                "any_valid": row_any_valid,
                "all_valid": row_all_valid,
                "all_strict": row_all_strict,
                "all_expanded": row_all_expanded,
            }
        ).dropna(subset=["Timestamp"])

        def window_count(duration: pd.Timedelta, flag: str) -> int:
            mask = (
                (event_frame["Timestamp"] > event_time - duration)
                & (event_frame["Timestamp"] <= event_time)
                & event_frame[flag]
            )
            return int(mask.sum())

        def cell_coverage(duration: pd.Timedelta, validity: pd.DataFrame) -> float:
            mask = (timestamps > event_time - duration) & (timestamps <= event_time)
            subset = validity.loc[mask]
            if subset.empty:
                return float("nan")
            return float(subset.to_numpy().sum() / subset.size)

        row = {
            "事件": event_name,
            "参考时间": event_time,
            "数据组": group,
            "最近的事前记录": before.max() if len(before) else pd.NaT,
            "最近的事后记录": after.min() if len(after) else pd.NaT,
            "事前24小时记录数": int(((index > event_time - pd.Timedelta(hours=24)) & (index <= event_time)).sum()),
            "事前24小时至少一项有效数": window_count(pd.Timedelta(hours=24), "any_valid"),
            "事前24小时全字段有效数": window_count(pd.Timedelta(hours=24), "all_valid"),
            "事前72小时记录数": int(((index > event_time - pd.Timedelta(hours=72)) & (index <= event_time)).sum()),
            "事前72小时至少一项有效数": window_count(pd.Timedelta(hours=72), "any_valid"),
            "事前72小时全字段有效数": window_count(pd.Timedelta(hours=72), "all_valid"),
            "事前7天记录数": int(((index > event_time - pd.Timedelta(days=7)) & (index <= event_time)).sum()),
            "事前7天至少一项有效数": window_count(pd.Timedelta(days=7), "any_valid"),
            "事前7天全字段有效数": window_count(pd.Timedelta(days=7), "all_valid"),
            "事前7天全字段严格可用数": window_count(pd.Timedelta(days=7), "all_strict"),
            "事前7天全字段扩展可用数": window_count(pd.Timedelta(days=7), "all_expanded"),
            "事前7天严格字段覆盖率": cell_coverage(pd.Timedelta(days=7), strict_valid),
            "事前7天扩展字段覆盖率": cell_coverage(pd.Timedelta(days=7), expanded_valid),
            "事前15天记录数": int(((index > event_time - pd.Timedelta(days=15)) & (index <= event_time)).sum()),
            "事前15天至少一项有效数": window_count(pd.Timedelta(days=15), "any_valid"),
            "事前15天全字段有效数": window_count(pd.Timedelta(days=15), "all_valid"),
            "事前30天记录数": int(((index > event_time - pd.Timedelta(days=30)) & (index <= event_time)).sum()),
            "事后24小时记录数": int(((index > event_time) & (index <= event_time + pd.Timedelta(hours=24))).sum()),
        }
        event_rows.append(row)

    return inventory, dictionary_rows, quality_rows, qc_rows, event_rows


def write_report(
    inventory: pd.DataFrame,
    dictionary: pd.DataFrame,
    quality: pd.DataFrame,
    qc: pd.DataFrame,
    events: pd.DataFrame,
) -> None:
    lines = [
        "# 逐小时数据预处理初检报告",
        "",
        "## 检查范围",
        "",
        f"- 数据目录：`{SOURCE_DIR}`",
        f"- 文件数：{len(inventory)}",
        f"- 数据组：{', '.join(sorted(inventory['数据组'].unique()))}",
        f"- 年份：{inventory['年份'].min()}—{inventory['年份'].max()}",
        "- 本轮仅做审计和标记，不修改原始CSV。",
        "",
        "## 时间覆盖",
        "",
    ]
    for _, row in inventory.sort_values(["数据组", "年份"]).iterrows():
        lines.append(
            f"- {row['数据组']} {row['年份']}：时间戳{row['开始时间']}至{row['结束时间']}；"
            f"实际至少一项有效观测至{row['结束有效观测时间']}；{row['行数']}行，"
            f"全观测缺测占位行{row['全观测缺测占位行数']}行，缺失整点{row['缺失整点数']}个，"
            f"重复时间行{row['重复时间行数']}个。"
        )

    lines.extend(["", "## 事件窗口覆盖", ""])
    for _, row in events.sort_values(["事件", "数据组"]).iterrows():
        lines.append(
            f"- {row['事件']} / {row['数据组']}：最近事前记录{row['最近的事前记录']}，"
            f"事前24小时至少一项有效{row['事前24小时至少一项有效数']}/{row['事前24小时记录数']}条，"
            f"72小时至少一项有效{row['事前72小时至少一项有效数']}/{row['事前72小时记录数']}条，"
            f"7天严格字段覆盖率{row['事前7天严格字段覆盖率']:.2%}，"
            f"扩展字段覆盖率{row['事前7天扩展字段覆盖率']:.2%}。"
        )

    nonzero_missing = quality[
        (quality["空值数"] > 0)
        | (quality["非数值数"] > 0)
        | (quality["疑似缺测码数"] > 0)
        | (quality["物理范围标记数"] > 0)
    ]
    lines.extend(
        [
            "",
            "## 数据质量提示",
            "",
            f"- 出现空值、非数值、疑似缺测码或物理范围标记的字段-年份组合共{len(nonzero_missing)}个。",
            "- 全文件复核后确认数值型缺测占位为9999.9；它与真实的0降水严格区分。",
            "- 文件中的完整年度时间戳可能包含缺测占位行；有效覆盖必须依据观测值而不是仅依据时间戳判断。",
            "- 物理范围检查只用于发现明显异常，不在本轮自动删除数据。",
            "- 官方QC码含义：0正确、1可疑、2错误、4插补、8缺失、9未进行质量控制。",
            "- 后续同时生成严格样本（仅QC=0）和扩展样本（QC=0或4）；二者的排名一致性将作为稳健性检验。",
            "",
            "## 字段与因素映射",
            "",
            f"- 去重后字段数：{dictionary[['数据组', '字段名']].drop_duplicates().shape[0]}。",
            "- 原始字段已经映射到降水、热量、土壤湿度、冻融、辐射、热通量、湿度、气压、风场与湍流等因素组。",
            "- 映射用于后续‘组内选代表指标、组间排前五’，不把同一因素的多个窗口重复计作多个最终因素。",
            "",
            "## 下一步",
            "",
            "1. 形成不覆盖原始数据的严格版和扩展版标准化清洗表。",
            "2. 对无独立QC列的FLUX极端记录生成复核标记，不自动删值。",
            "3. 围绕2026年事件构造多时间尺度衍生指标。",
        ]
    )
    (OUTPUT_DIR / "07_数据预处理初检报告.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(SOURCE_DIR.glob("*_Gyirong_*.csv"))
    if not files:
        raise FileNotFoundError(f"未找到逐小时CSV：{SOURCE_DIR}")

    inventories = []
    dictionaries = []
    qualities = []
    qc_rows = []
    event_rows = []
    for path in files:
        inventory, dictionary, quality, qc, events = profile_file(path)
        inventories.append(inventory)
        dictionaries.extend(dictionary)
        qualities.extend(quality)
        qc_rows.extend(qc)
        event_rows.extend(events)

    inventory_df = pd.DataFrame(inventories).sort_values(["数据组", "年份"])
    dictionary_df = (
        pd.DataFrame(dictionaries)
        .sort_values(["数据组", "字段角色", "字段名", "首次出现年份"])
        .drop_duplicates(["数据组", "字段名"], keep="first")
    )
    quality_df = pd.DataFrame(qualities).sort_values(["数据组", "年份", "字段名"])
    qc_df = pd.DataFrame(qc_rows).sort_values(["数据组", "年份", "质量控制字段", "质量控制码"])
    events_df = pd.DataFrame(event_rows).sort_values(["事件", "数据组"])

    datasets = {
        "01_文件与时间覆盖清单.csv": inventory_df,
        "02_原始字段字典.csv": dictionary_df,
        "03_字段质量汇总.csv": quality_df,
        "04_质量控制码分布.csv": qc_df,
        "05_事件窗口数据覆盖.csv": events_df,
    }
    for filename, frame in datasets.items():
        frame.to_csv(OUTPUT_DIR / filename, index=False, encoding="utf-8-sig")

    issue_df = quality_df[
        (quality_df["空值数"] > 0)
        | (quality_df["非数值数"] > 0)
        | (quality_df["疑似缺测码数"] > 0)
        | (quality_df["物理范围标记数"] > 0)
    ].copy()
    issue_df.to_csv(OUTPUT_DIR / "06_需复核的数据质量问题.csv", index=False, encoding="utf-8-sig")

    write_report(inventory_df, dictionary_df, quality_df, qc_df, events_df)
    print(f"已检查 {len(files)} 个文件，结果保存至：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
