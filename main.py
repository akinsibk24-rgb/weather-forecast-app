"""
Nigeria Weather Forecaster — entry point.

Run with:
    python main.py

On first run with no data/*.csv present, generate the sample dataset:
    python scripts/generate_sample_data.py
"""

from pathlib import Path
from app.gui import run
from app.data_loader import DEFAULT_FILE

if __name__ == "__main__":
    if not DEFAULT_FILE.exists():
        print("No sample dataset found — generating one now...")
        import subprocess
        import sys
        subprocess.run([sys.executable, str(Path(__file__).parent / "scripts" / "generate_sample_data.py")])
    run()
