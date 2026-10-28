from __future__ import annotations

"""Run Q3 models from existing caches; no raw download and no plotting."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    "04_run_trigger_forecast.py",
    "05_run_propagation.py",
    "06_evaluate_models.py",
    "08_build_warning_and_mechanism_outputs.py",
    "09_audit_imerg_operational_availability.py",
]


def main() -> None:
    for name in SCRIPTS:
        print(f"\n=== {name} ===", flush=True)
        subprocess.run([sys.executable, str(ROOT / "scripts" / name)], check=True)


if __name__ == "__main__":
    main()
