from __future__ import annotations

"""Run formal Question 2 models from prepared caches; no raw download occurs."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    "04_temporal_thresholds_and_uncertainty.py",
    "05_build_spatial_susceptibility.py",
    "06_integrate_spatiotemporal_risk.py",
    "07_evaluate_gfs_forecast_reliability.py",
    "08_glacial_lake_context.py",
    "09_itslive_velocity_context.py",
]


def main() -> None:
    for name in SCRIPTS:
        print(f"\n=== {name} ===", flush=True)
        subprocess.run([sys.executable, str(ROOT / "scripts" / name)], check=True)


if __name__ == "__main__":
    main()
