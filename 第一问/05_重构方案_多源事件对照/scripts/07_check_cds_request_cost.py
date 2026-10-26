from __future__ import annotations

"""校验正式下载方案中每个变量组的CDS单次请求成本。"""

import argparse
import json
from pathlib import Path

import cdsapi


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR.parent / "config" / "04_CDS正式下载方案.json"
ALL_DAYS = [f"{day:02d}" for day in range(1, 32)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = json.loads(args.config.read_text(encoding="utf-8"))
    client = cdsapi.Client(quiet=True)
    failed = False
    for group, variables in plan["variable_groups"].items():
        request = {
            "variable": variables,
            "year": "2026",
            "month": ["08"],
            "day": ALL_DAYS,
            "time": plan["hours"],
            "area": plan["area_nwse"],
            "data_format": plan["data_format"],
            "download_format": plan["download_format"],
        }
        result = client.client.estimate_costs(plan["dataset"], request)
        cost = result.get("cost")
        limit = result.get("limit")
        ok = isinstance(cost, (int, float)) and isinstance(limit, (int, float)) and cost <= limit
        print(f"{group}: variables={len(variables)}, cost={cost}, limit={limit}, ok={ok}")
        failed = failed or not ok
    if failed:
        raise SystemExit("至少一个变量组超过CDS成本限制，停止正式下载。")


if __name__ == "__main__":
    main()
