from __future__ import annotations

"""用一个时次、一个变量的小请求测试CDS凭证和ERA5-Land许可。"""

from pathlib import Path

import cdsapi


TARGET = Path(r"D:\MathorCup_Data\ERA5_Land_Gyirong\cds_access_test_20210101_00.nc")


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client(quiet=True)
    request = {
        "variable": ["2m_temperature"],
        "year": ["2021"],
        "month": ["01"],
        "day": ["01"],
        "time": ["00:00"],
        "area": [28.7, 85.1, 27.9, 85.9],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    client.retrieve("reanalysis-era5-land", request, str(TARGET))
    if not TARGET.exists() or TARGET.stat().st_size == 0:
        raise RuntimeError("CDS请求完成，但测试文件不存在或为空。")
    print(f"CDS_TEST_OK path={TARGET} bytes={TARGET.stat().st_size}")


if __name__ == "__main__":
    main()
