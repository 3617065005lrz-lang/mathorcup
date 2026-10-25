#!/usr/bin/env python3
"""Reproducibly download the Copernicus GLO-30 public COG tile N28E085."""

from __future__ import annotations

import urllib.request
from pathlib import Path


BASE = "https://copernicus-dem-30m.s3.amazonaws.com/"
PREFIX = "Copernicus_DSM_COG_10_N28_00_E085_00_DEM/"
DEST = Path(__file__).resolve().parents[1] / "P0-6_Copernicus_DEM_GLO-30" / "tiles" / PREFIX.rstrip("/")
FILES = (
    "Copernicus_DSM_COG_10_N28_00_E085_00_DEM.tif",
    "Copernicus_DSM_10_N28_00_E085_00.xml",
    "AUXFILES/Copernicus_DSM_10_N28_00_E085_00_ACM.kml",
    "AUXFILES/Copernicus_DSM_COG_10_N28_00_E085_00_EDM.tif",
    "AUXFILES/Copernicus_DSM_COG_10_N28_00_E085_00_FLM.tif",
    "AUXFILES/Copernicus_DSM_COG_10_N28_00_E085_00_HEM.tif",
    "AUXFILES/Copernicus_DSM_COG_10_N28_00_E085_00_WBM.tif",
    "INFO/eula_F.pdf",
)


def main() -> None:
    for relative in FILES:
        output = DEST / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() and output.stat().st_size > 0:
            print(f"SKIP {output}")
            continue
        request = urllib.request.Request(
            BASE + PREFIX + relative,
            headers={"User-Agent": "Codex-P0-456/1.0"},
        )
        partial = output.with_suffix(output.suffix + ".part")
        with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        partial.replace(output)
        print(f"DOWNLOADED {output} {output.stat().st_size} bytes")


if __name__ == "__main__":
    main()
