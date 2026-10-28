from __future__ import annotations

"""Draw PNG figures from saved model outputs; models are not rerun."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def main() -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / "07_plot_results.py")], check=True)


if __name__ == "__main__":
    main()
