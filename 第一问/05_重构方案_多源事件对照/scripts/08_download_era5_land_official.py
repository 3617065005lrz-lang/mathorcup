from __future__ import annotations

"""从官方CDS分年、分变量组下载吉隆源区ERA5-Land原始小时数据。

只下载模型实际需要的6—9月和0.8°×0.8°区域，不下载全球全集。
文件非空即视为缓存；中断后重新运行会跳过已完成文件。
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time

import cdsapi


SCRIPT_DIR = Path(__file__).resolve().parent
WORK_DIR = SCRIPT_DIR.parent
CONFIG_PATH = WORK_DIR / "config" / "04_CDS正式下载方案.json"
ALL_DAYS = [f"{day:02d}" for day in range(1, 32)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--group", default="all", help="变量组名；默认all")
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser.parse_args()


def load_plan(config_path: Path) -> dict:
    return json.loads(config_path.read_text(encoding="utf-8"))


def build_jobs(plan: dict, args: argparse.Namespace) -> list[dict]:
    start = args.start_year or int(plan["years"]["start"])
    end = args.end_year or int(plan["years"]["end"])
    if start > end:
        raise ValueError("start-year不能大于end-year")
    groups = plan["variable_groups"]
    if args.group != "all" and args.group not in groups:
        raise ValueError(f"未知变量组{args.group!r}，可选：all, {', '.join(groups)}")
    selected = groups if args.group == "all" else {args.group: groups[args.group]}
    raw_dir = Path(plan["data_root"]) / plan.get("raw_subdir", "raw_official")
    raw_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for year in range(start, end + 1):
        for month in plan["months"]:
            for group_name, variables in selected.items():
                target = raw_dir / f"era5_land_gyirong_{year}_{month}_{group_name}{plan['file_extension']}"
                request = {
                    "variable": variables,
                    "year": str(year),
                    "month": [month],
                    "day": ALL_DAYS,
                    "time": plan["hours"],
                    "area": plan["area_nwse"],
                    "data_format": plan["data_format"],
                    "download_format": plan["download_format"],
                }
                jobs.append(
                    {
                        "dataset": plan["dataset"],
                        "year": year,
                        "month": month,
                        "group": group_name,
                        "request": request,
                        "target": target,
                    }
                )
    return jobs


def retrieve_one(job: dict, max_attempts: int) -> dict:
    target: Path = job["target"]
    if target.exists() and target.stat().st_size > 0:
        return {**job, "status": "cached", "bytes": target.stat().st_size, "attempts": 0}
    partial = target.with_suffix(target.suffix + ".part")
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        try:
            if partial.exists():
                partial.unlink()
            client = cdsapi.Client(quiet=True)
            client.retrieve(job["dataset"], job["request"], str(partial))
            if not partial.exists() or partial.stat().st_size == 0:
                raise RuntimeError("CDS返回后未生成有效文件")
            partial.replace(target)
            return {**job, "status": "downloaded", "bytes": target.stat().st_size, "attempts": attempt}
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            if "cost limits exceeded" in last_error.lower() or "required licences not accepted" in last_error.lower():
                break
            if attempt < max_attempts:
                time.sleep(15 * attempt)
    return {**job, "status": "failed", "bytes": 0, "attempts": attempt, "error": last_error}


def serializable(result: dict) -> dict:
    return {
        "year": result["year"],
        "month": result["month"],
        "group": result["group"],
        "status": result["status"],
        "bytes": result.get("bytes", 0),
        "attempts": result.get("attempts", 0),
        "target": str(result["target"]),
        "error": result.get("error", ""),
    }


def main() -> None:
    args = parse_args()
    plan = load_plan(args.config)
    jobs = build_jobs(plan, args)
    results = []
    manifest_path = Path(plan["data_root"]) / plan.get("raw_subdir", "raw_official") / "download_manifest.json"
    print(f"计划任务数={len(jobs)}，并发={args.workers}（按月分批）", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(retrieve_one, job, args.max_attempts): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            results.append(serializable(result))
            size_mb = result.get("bytes", 0) / 1024**2
            print(f"{result['year']}-{result['month']} {result['group']}: {result['status']} {size_mb:.1f} MB", flush=True)
            manifest_path.write_text(
                json.dumps(sorted(results, key=lambda x: (x["year"], x["month"], x["group"])), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    manifest_path.write_text(
        json.dumps(sorted(results, key=lambda x: (x["year"], x["month"], x["group"])), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    failures = [row for row in results if row["status"] == "failed"]
    print(f"完成={len(results) - len(failures)}，失败={len(failures)}，清单={manifest_path}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
