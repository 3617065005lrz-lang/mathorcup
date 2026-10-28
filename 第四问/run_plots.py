from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "03_analyze_and_plot.py"), run_name="__main__")
