from __future__ import annotations

"""用NASA Earthaccess下载2021—2024同产品IMERG Late Run半小时对照。"""

from pathlib import Path

import earthaccess


TARGET = Path(r"D:\MathorCup_Data\External_Gyirong\P0-2_IMERG\data_late_history")
YEARS = (2021, 2022, 2023, 2024)


def main() -> None:
    auth = earthaccess.login(strategy="netrc")
    if not auth.authenticated:
        raise RuntimeError("Earthdata authentication failed")
    TARGET.mkdir(parents=True, exist_ok=True)
    for year in YEARS:
        results = earthaccess.search_data(
            short_name="GPM_3IMERGHHL",
            version="07",
            temporal=(f"{year}-08-20T00:00:00Z", f"{year}-08-27T23:59:59Z"),
        )
        if len(results) != 384:
            raise RuntimeError(f"{year} IMERG catalogue incomplete: {len(results)}/384")
        print(f"IMERG {year}: catalogue=384, starting/resuming download", flush=True)
        files = earthaccess.download(results, TARGET, threads=8, show_progress=True)
        valid = sum(path.exists() and path.stat().st_size > 0 for path in files)
        print(f"IMERG {year}: returned={len(files)}, valid={valid}", flush=True)
        if valid != 384:
            raise RuntimeError(f"{year} IMERG download incomplete: {valid}/384")
    print("IMERG Late history 2021-2024 complete", flush=True)


if __name__ == "__main__":
    main()
