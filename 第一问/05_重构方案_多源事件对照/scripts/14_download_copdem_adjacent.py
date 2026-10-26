from __future__ import annotations

"""Download the eight public Copernicus DEM GLO-30 DSM tiles around N28E085."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import json
from pathlib import Path
import subprocess


ROOT = "https://copernicus-dem-30m.s3.amazonaws.com"
OUT = Path(r"D:\MathorCup_Data\External_Gyirong\P0-6_Copernicus_DEM_adjacent")
TILES = [
    "N27_00_E084_00", "N27_00_E085_00", "N27_00_E086_00",
    "N28_00_E084_00",                     "N28_00_E086_00",
    "N29_00_E084_00", "N29_00_E085_00", "N29_00_E086_00",
]


def checksum(path: Path) -> str:
    h = sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(tile: str) -> dict:
    stem = f"Copernicus_DSM_COG_10_{tile}_DEM"
    url = f"{ROOT}/{stem}/{stem}.tif"
    target = OUT / f"{stem}.tif"
    partial = target.with_suffix(".tif.part")
    if not target.exists() or target.stat().st_size == 0:
        subprocess.run(
            ["curl.exe", "--silent", "--show-error", "--location", "--continue-at", "-",
             "--retry", "8", "--retry-all-errors", "--retry-delay", "10",
             "--output", str(partial), url],
            check=True,
        )
        partial.replace(target)
    return {"tile": tile, "url": url, "target": str(target), "bytes": target.stat().st_size, "sha256": checksum(target)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(download, tile): tile for tile in TILES}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            (OUT / "download_manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"DEM {len(rows)}/{len(TILES)} {row['tile']} {row['bytes']/1024**2:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
