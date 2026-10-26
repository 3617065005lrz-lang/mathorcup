from __future__ import annotations

"""只读核验已下载ERA5-Land GRIB文件的变量、时间和空间网格。"""

import json
from pathlib import Path

from collections import Counter
from datetime import datetime

from eccodes import codes_get, codes_grib_new_from_file, codes_release


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR.parent / "config" / "04_CDS正式下载方案.json"


def inspect_file(path: Path) -> dict:
    counts: Counter[str] = Counter()
    times: list[datetime] = []
    grids: set[tuple] = set()
    with path.open("rb") as handle:
        while True:
            message = codes_grib_new_from_file(handle)
            if message is None:
                break
            try:
                counts[str(codes_get(message, "shortName"))] += 1
                date = int(codes_get(message, "validityDate"))
                hour_minute = int(codes_get(message, "validityTime"))
                times.append(datetime.strptime(f"{date:08d}{hour_minute:04d}", "%Y%m%d%H%M"))
                grids.add(
                    (
                        int(codes_get(message, "Ni")),
                        int(codes_get(message, "Nj")),
                        float(codes_get(message, "latitudeOfFirstGridPointInDegrees")),
                        float(codes_get(message, "latitudeOfLastGridPointInDegrees")),
                        float(codes_get(message, "longitudeOfFirstGridPointInDegrees")),
                        float(codes_get(message, "longitudeOfLastGridPointInDegrees")),
                    )
                )
            finally:
                codes_release(message)
    return {"counts": counts, "times": times, "grids": grids}


def main() -> None:
    plan = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raw_dir = Path(plan["data_root"]) / "raw_official"
    files = sorted(raw_dir.glob("*.grib"))
    if not files:
        raise SystemExit(f"未找到GRIB文件：{raw_dir}")
    print(f"files={len(files)}")
    for path in files:
        info = inspect_file(path)
        times = info["times"]
        print(f"{path.name}: bytes={path.stat().st_size}, messages={sum(info['counts'].values())}")
        print(f"  variables={dict(sorted(info['counts'].items()))}")
        print(f"  valid_time={min(times)} .. {max(times)}; unique={len(set(times))}")
        print(f"  grids={sorted(info['grids'])}")


if __name__ == "__main__":
    main()
