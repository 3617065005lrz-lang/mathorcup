from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REBUILD_DIR = SCRIPT_DIR.parent
QUESTION_DIR = REBUILD_DIR.parent
PREPROCESS_DIR = QUESTION_DIR / "01_数据预处理"
EXTERNAL_DIR = REBUILD_DIR / "external" / "gyirong-2026"
OUTPUT_DIR = REBUILD_DIR / "outputs"
CONFIG_DIR = REBUILD_DIR / "config"

EVENT_TIME = pd.Timestamp("2026-08-26 10:52:07")
EVENT_CUTOFF = pd.Timestamp("2026-08-26 10:00:00")
WINDOWS = [24, 48, 72, 120, 168]


def audit_station_files() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for mode in ("strict", "expanded"):
        folder = PREPROCESS_DIR / "processed" / mode
        for group in ("MET", "RADM", "SOIL", "FLUX"):
            matches = list(folder.glob(f"{group}_Gyirong_2021_2026_{mode}.parquet"))
            if not matches:
                rows.append(
                    {
                        "清洗版本": mode,
                        "数据组": group,
                        "文件存在": False,
                        "总行数": 0,
                        "起始时间": "",
                        "结束时间": "",
                        "2026事件前最近记录": "",
                        **{f"灾前{w}h记录数": 0 for w in WINDOWS},
                    }
                )
                continue

            frame = pd.read_parquet(matches[0])
            frame["Timestamp"] = pd.to_datetime(frame["Timestamp"])
            value_columns = [
                column
                for column in frame.columns
                if column not in {"Timestamp", "source_file"}
                and not column.lower().startswith("qc_")
            ]
            pre_event = frame.loc[frame["Timestamp"] <= EVENT_CUTOFF, "Timestamp"]
            row: dict[str, object] = {
                "清洗版本": mode,
                "数据组": group,
                "文件存在": True,
                "总行数": len(frame),
                "起始时间": frame["Timestamp"].min(),
                "结束时间": frame["Timestamp"].max(),
                "2026事件前最近记录": pre_event.max() if not pre_event.empty else "",
            }
            for window in WINDOWS:
                mask = (
                    (frame["Timestamp"] > EVENT_CUTOFF - pd.Timedelta(hours=window))
                    & (frame["Timestamp"] <= EVENT_CUTOFF)
                )
                event = frame.loc[mask, value_columns]
                row[f"灾前{window}h记录数"] = int(mask.sum())
                row[f"灾前{window}h至少一项有效数"] = int(event.notna().any(axis=1).sum())
                row[f"灾前{window}h全字段有效数"] = int(event.notna().all(axis=1).sum())
                row[f"灾前{window}h单元格有效率"] = float(event.notna().mean().mean())
            rows.append(row)
    return pd.DataFrame(rows)


def audit_external_products() -> tuple[pd.DataFrame, dict[str, object]]:
    processed = EXTERNAL_DIR / "data" / "processed"
    rows: list[dict[str, object]] = []
    for name in (
        "daily_series.csv",
        "event_window_hourly.csv",
        "climatology.csv",
        "event_hour_diagnostics.json",
        "grid_elevation.nc",
        "polygon_weights.nc",
    ):
        path = processed / name
        row: dict[str, object] = {
            "资源": name,
            "存在": path.exists(),
            "大小字节": path.stat().st_size if path.exists() else 0,
            "起始时间": "",
            "结束时间": "",
            "记录数": "",
        }
        if path.exists() and path.suffix == ".csv":
            frame = pd.read_csv(path)
            row["记录数"] = len(frame)
            time_column = next(
                (column for column in ("date", "valid_time") if column in frame.columns),
                None,
            )
            if time_column:
                timestamps = pd.to_datetime(frame[time_column], errors="coerce")
                row["起始时间"] = timestamps.min()
                row["结束时间"] = timestamps.max()
        rows.append(row)

    diagnostics_path = processed / "event_hour_diagnostics.json"
    diagnostics: dict[str, object] = {}
    if diagnostics_path.exists():
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    return pd.DataFrame(rows), diagnostics


def audit_candidate_dictionary() -> tuple[pd.DataFrame, pd.DataFrame]:
    dictionary = pd.read_csv(CONFIG_DIR / "01_候选变量字典.csv")
    summary = (
        dictionary.groupby(["数据层", "前五资格", "优先级"], dropna=False)
        .size()
        .rename("候选指标数")
        .reset_index()
    )
    factors = (
        dictionary.groupby(["因素组ID", "物理因素"], dropna=False)
        .agg(
            候选指标数=("候选指标ID", "count"),
            可进入前五数=("前五资格", lambda values: int((values == "是").sum())),
            涉及数据层=("数据层", lambda values: "；".join(sorted(set(values)))),
        )
        .reset_index()
    )
    return summary, factors


def write_report(
    station: pd.DataFrame,
    external: pd.DataFrame,
    diagnostics: dict[str, object],
    candidate_summary: pd.DataFrame,
    factor_summary: pd.DataFrame,
) -> None:
    lines = [
        "# 第一问重构输入审计报告",
        "",
        f"- 事件参考时刻：{EVENT_TIME}",
        f"- 小时数据截止：{EVENT_CUTOFF}",
        "- 本报告只核验输入资源和口径，不产生最终前五排序。",
        "",
        "## 题目观测站",
        "",
    ]
    for row in station.to_dict("records"):
        counts = "、".join(
            f"{w}h={row[f'灾前{w}h记录数']}行/"
            f"全字段有效{row[f'灾前{w}h全字段有效数']}行/"
            f"单元格有效率{row[f'灾前{w}h单元格有效率']:.1%}"
            for w in WINDOWS
        )
        lines.append(
            f"- {row['清洗版本']}/{row['数据组']}："
            f"{row['起始时间']} 至 {row['结束时间']}；事件窗口记录数 {counts}。"
        )

    lines.extend(["", "## 外部ERA5-Land复现项目", ""])
    for row in external.to_dict("records"):
        lines.append(
            f"- {row['资源']}：存在={row['存在']}，记录数={row['记录数']}，"
            f"范围={row['起始时间']} 至 {row['结束时间']}。"
        )

    if diagnostics:
        lines.extend(["", "### 外部项目已给出的事件诊断", ""])
        for item in diagnostics.get("precip_windows", []):
            lines.append(
                f"- ERA5-Land降水 {item['hours']}h：{item['total']:.3f} mm，"
                f"外部项目历史百分位={item['percentile']:.1f}%。"
            )
        for item in diagnostics.get("freezing_windows", []):
            lines.append(
                f"- 0℃等温线 {item['hours']}h均值：{item['mean']:.1f} m，"
                f"外部项目历史百分位={item['percentile']:.1f}%。"
            )
        lines.append(
            "- 以上数值仅作独立交叉核验；历史期、空间支持和递减率算法必须在我们的主模型中重新声明。"
        )

    lines.extend(
        [
            "",
            "## 候选变量体系",
            "",
            f"- 当前登记候选指标 {int(candidate_summary['候选指标数'].sum())} 个。",
            f"- 共覆盖 {len(factor_summary)} 个物理因素组。",
            "- 机理支撑层的静态地形、冰川和事后遥感变量不进入气象前兆前五排序。",
            "",
            "## 当前阻塞项",
            "",
            "- 本机未检测到CDS凭证和cdsapi；扩充ERA5-Land变量前需要配置Copernicus CDS个人令牌。",
            "- IMERG 2026年应先核实Final/Late产品档期，再冻结下载版本。",
            "- 外部GitHub项目未发现LICENSE文件，只能核验和引用，不直接复制其实现进入提交代码。",
        ]
    )
    (OUTPUT_DIR / "00_输入资源审计报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    station = audit_station_files()
    external, diagnostics = audit_external_products()
    candidate_summary, factor_summary = audit_candidate_dictionary()

    station.to_csv(OUTPUT_DIR / "01_题目观测站输入审计.csv", index=False, encoding="utf-8-sig")
    external.to_csv(OUTPUT_DIR / "02_外源项目资源审计.csv", index=False, encoding="utf-8-sig")
    candidate_summary.to_csv(OUTPUT_DIR / "03_候选变量分层汇总.csv", index=False, encoding="utf-8-sig")
    factor_summary.to_csv(OUTPUT_DIR / "04_物理因素组汇总.csv", index=False, encoding="utf-8-sig")
    write_report(station, external, diagnostics, candidate_summary, factor_summary)

    print(f"审计完成：{OUTPUT_DIR}")
    print(f"候选指标数：{int(candidate_summary['候选指标数'].sum())}")
    print(f"物理因素组数：{len(factor_summary)}")


if __name__ == "__main__":
    main()

