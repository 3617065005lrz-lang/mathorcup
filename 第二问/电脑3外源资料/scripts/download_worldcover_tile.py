#!/usr/bin/env python3
"""Download official ESA WorldCover tiles with size and SHA-256 verification."""
import argparse, hashlib
from pathlib import Path
from urllib.request import urlopen

TILES={
"ESA_WorldCover_10m_2020_v100_N27E084_Map.tif":("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v100/2020/map/ESA_WorldCover_10m_2020_v100_N27E084_Map.tif",103582423),
"ESA_WorldCover_10m_2021_v200_N27E084_Map.tif":("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N27E084_Map.tif",104225123),
}
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest().upper()
def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",type=Path,required=True); a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    for name,(url,expected) in TILES.items():
        dest=a.out/name
        if dest.exists() and dest.stat().st_size==expected:
            print(f"Exists: {dest} ({expected} bytes) SHA256={digest(dest)}"); continue
        tmp=dest.with_suffix(dest.suffix+".part")
        with urlopen(url,timeout=120) as r, tmp.open("wb") as f:
            while True:
                b=r.read(1024*1024)
                if not b: break
                f.write(b)
        size=tmp.stat().st_size
        if size!=expected: tmp.unlink(missing_ok=True); raise RuntimeError(f"{name}: expected {expected} bytes, got {size}")
        tmp.replace(dest); print(f"Saved: {dest} ({size} bytes) SHA256={digest(dest)}")
if __name__=="__main__": main()
