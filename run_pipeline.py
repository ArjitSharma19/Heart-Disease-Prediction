"""Root runner delegating to heart-disease-prediction pipeline."""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent / "heart-disease-prediction"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from run_pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline()
