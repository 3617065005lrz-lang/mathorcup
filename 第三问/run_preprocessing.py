from __future__ import annotations

"""Build local caches only; no download is performed."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def main() -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / "03_build_gefs_point_cache.py")], check=True)


if __name__ == "__main__":
    main()
