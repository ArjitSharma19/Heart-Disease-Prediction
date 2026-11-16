"""Direct Python launcher for Streamlit application.

Handles modern (streamlit >= 1.12) and legacy Streamlit imports.
Usage:
    python run_app.py
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
if PROJECT_DIR.name != "heart-disease-prediction":
    PROJECT_DIR = PROJECT_DIR / "heart-disease-prediction"

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Modern Streamlit (>= 1.12) moved cli to streamlit.web.cli
try:
    from streamlit.web.cli import main
except ImportError:
    from streamlit.cli import main

if __name__ == "__main__":
    app_path = str(PROJECT_DIR / "app.py")
    sys.argv = ["streamlit", "run", app_path]
    sys.exit(main())

