from __future__ import annotations

"""Build reusable source caches. Run only when raw inputs or settings change."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    "01_build_modis_source_cache.py",
    "02_build_chirps_source_cache.py",
    "02b_build_smap_source_cache.py",
    "03_build_dynamic_trigger_cache.py",
]


def main() -> None:
    for name in SCRIPTS:
        print(f"\n=== {name} ===", flush=True)
        subprocess.run([sys.executable, str(ROOT / "scripts" / name)], check=True)


if __name__ == "__main__":
    main()
