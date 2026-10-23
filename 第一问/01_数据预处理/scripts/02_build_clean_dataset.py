from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PREPROCESS_DIR = SCRIPT_DIR.parent
WORKSPACE_DIR = SCRIPT_DIR.parents[3]
SOURCE_DIR = WORKSPACE_DIR / "work" / "mathorcup_b" / "a1_hourly"
PROCESSED_DIR = PREPROCESS_DIR / "processed"
OUTPUT_DIR = PREPROCESS_DIR / "outputs"

MISSING_SENTINEL = 9999.9
STRICT_QC = {0}
EXPANDED_QC = {0, 4}


def base_name(column: str) -> str:
    return re.sub(r"\(.*\)$", "", column)


def replace_missing_sentinel(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if column == "Timestamp":
            continue
        numeric = pd.to_numeric(result[column], errors="coerce")
        numeric[np.isclose(numeric, MISSING_SENTINEL)] = np.nan
        result[column] = numeric
    return result


def apply_qc(frame: pd.DataFrame, accepted_codes: set[int]) -> tuple[pd.DataFrame, list[dict]]:
    cleaned = frame.copy()
    qc_lookup = {
        column[3:].lower(): column
        for column in cleaned.columns
        if column.lower().startswith("qc_")
    }
    summary = []
    for column in cleaned.columns:
        if column in {"Timestamp", "source_file"} or column.lower().startswith("qc_"):
            continue
        qc_column = qc_lookup.get(base_name(column).lower())
        before = int(cleaned[column].notna().sum())
        if qc_column:
            accepted = cleaned[qc_column].isin(accepted_codes)
            cleaned.loc[~accepted, column] = np.nan
        after = int(cleaned[column].notna().sum())
        summary.append(
            {
                "字段名": column,
                "对应QC字段": qc_column or "无独立QC列",
                "清洗前非缺失数": before,
                "清洗后非缺失数": after,
                "因QC新增缺失数": before - after,
            }
        )
    return cleaned, summary


def flux_review_flags(frame: pd.DataFrame) -> pd.DataFrame:
    flags = []

    def add(mask: pd.Series, rule: str, fields: list[str]) -> None:
        for idx in frame.index[mask.fillna(False)]:
            flags.append(
                {
                    "Timestamp": frame.at[idx, "Timestamp"],
                    "复核规则": rule,
                    "涉及字段": ", ".join(fields),
                    "字段值": "; ".join(f"{field}={frame.at[idx, field]}" for field in fields),
                    "处理方式": "仅标记，不自动删除；稳健统计和敏感性分析中复核",
                }
            )

    add(
        frame["e_sat(kPa)"] > frame["PA(kPa)"],
        "饱和水汽压高于站点气压",
        ["e_sat(kPa)", "PA(kPa)", "VPD(hPa)"],
    )
    add(
        frame["VPD(hPa)"] > frame["PA(kPa)"] * 10,
        "水汽压亏缺高于站点气压",
        ["VPD(hPa)", "PA(kPa)"],
    )
    add(
        frame["LE(W/m2)"].abs() > 1000,
        "潜热通量绝对值超过1000 W/m2",
        ["LE(W/m2)", "H(W/m2)", "FC_mass(mg/m2*s)"],
    )
    add(
        frame["TKE(m2/s2)"] < 0,
        "湍动能为负值",
        ["TKE(m2/s2)"],
    )
    if not flags:
        return pd.DataFrame(columns=["Timestamp", "复核规则", "涉及字段", "字段值", "处理方式"])
    return pd.DataFrame(flags).drop_duplicates()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "strict").mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "expanded").mkdir(parents=True, exist_ok=True)

    all_summary = []
    flux_all = None
    for group in ["MET", "RADM", "SOIL", "FLUX"]:
        group_frames = []
        for path in sorted(SOURCE_DIR.glob(f"{group}_Gyirong_*.csv")):
            frame = pd.read_csv(path, low_memory=False)
            frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], errors="coerce")
            frame = replace_missing_sentinel(frame)
            frame["source_file"] = path.name
            group_frames.append(frame)
        combined = pd.concat(group_frames, ignore_index=True).sort_values("Timestamp")

        strict, strict_summary = apply_qc(combined, STRICT_QC)
        expanded, expanded_summary = apply_qc(combined, EXPANDED_QC)

        strict.to_parquet(
            PROCESSED_DIR / "strict" / f"{group}_Gyirong_2021_2026_strict.parquet",
            index=False,
        )
        expanded.to_parquet(
            PROCESSED_DIR / "expanded" / f"{group}_Gyirong_2021_2026_expanded.parquet",
            index=False,
        )

        for mode, rows in [("严格版_QC0", strict_summary), ("扩展版_QC0或4", expanded_summary)]:
            for row in rows:
                row.update({"数据组": group, "清洗版本": mode})
                all_summary.append(row)

        if group == "FLUX":
            flux_all = combined.copy()

    summary_df = pd.DataFrame(all_summary)
    summary_df.to_csv(OUTPUT_DIR / "08_清洗转换汇总.csv", index=False, encoding="utf-8-sig")

    flags = flux_review_flags(flux_all) if flux_all is not None else pd.DataFrame()
    if not flags.empty:
        flags.insert(1, "年份", pd.to_datetime(flags["Timestamp"]).dt.year)
    flags.to_csv(OUTPUT_DIR / "09_FLUX复核标记.csv", index=False, encoding="utf-8-sig")

    report = [
        "# 标准化清洗数据说明",
        "",
        "## 数据范围",
        "",
        "- 2026年为主分析事件年；2021—2025年仅用于建立历史背景和同期对照。",
        "- 2025年7月事件暂不进入验证分析。",
        "",
        "## 清洗规则",
        "",
        "- 将数值型缺测占位9999.9转换为缺失值。",
        "- 将FLUX文件中的NaN保留为缺失值。",
        "- 严格版只保留QC=0的配对观测。",
        "- 扩展版保留QC=0或QC=4的配对观测。",
        "- 没有独立QC列的降水和FLUX字段不因极端值自动删除。",
        "- FLUX内部物理关系明显异常的记录进入复核标记表，原值仍保留。",
        "",
        "## 输出",
        "",
        "- `processed/strict/`：严格版Parquet文件。",
        "- `processed/expanded/`：扩展版Parquet文件。",
        "- `outputs/08_清洗转换汇总.csv`：各字段清洗前后记录数。",
        "- `outputs/09_FLUX复核标记.csv`：2021—2026年不自动删除的FLUX复核记录。",
    ]
    (OUTPUT_DIR / "10_清洗数据说明.md").write_text("\n".join(report), encoding="utf-8")
    print(f"已生成严格版和扩展版清洗数据：{PROCESSED_DIR}")


if __name__ == "__main__":
    main()
