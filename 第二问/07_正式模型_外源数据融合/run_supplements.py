from __future__ import annotations

"""重跑第二问补充外源数据实验；不下载数据，不重建旧缓存。"""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    "11_audit_supplemental_external_data.py",
    "12_run_remote_sensing_supplements.py",
    "13_run_static_context_supplements.py",
    "14_uncertainty_transfer_and_validation.py",
    "15_compare_mcd43_upgraded_joint_risk.py",
]


def main() -> None:
    for name in SCRIPTS:
        print(f"\n=== {name} ===", flush=True)
        subprocess.run([sys.executable, str(ROOT / "scripts" / name)], check=True)


if __name__ == "__main__":
    main()
