"""Configuration constants and paths for Heart Disease Prediction project."""

import os
import sys
from pathlib import Path

# Base Paths (pathlib used throughout - no hardcoded string paths)
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

# Set local Matplotlib cache directory to avoid OS permission warnings
MPL_CACHE_DIR = PROJECT_ROOT / ".matplotlib_cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

# Ensure project root is in sys.path for direct imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_PATH = RAW_DATA_DIR / "processed.cleveland.data"
CLEAN_DATA_PATH = PROCESSED_DATA_DIR / "heart_clean.csv"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"
SPLIT_INDICES_PATH = PROCESSED_DATA_DIR / "split_indices.json"

MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure runtime directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, NOTEBOOKS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data Ingestion Constants
UCI_CLEVELAND_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
)

# Random state reproducibility constant
RANDOM_STATE = 42

# Train/Test Split configuration
TEST_SIZE = 0.2

# Column names from UCI documentation
COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]

TARGET_COL = "target"

# Feature groups
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Outlier auditing features
OUTLIER_FEATURES = ["trestbps", "chol", "thalach", "oldpeak"]

# Plausible clinical and domain value ranges for validation
# Tuples indicate continuous (min, max) inclusive ranges; sets indicate discrete allowed values.
PLAUSIBLE_RANGES = {
    "age": (18.0, 100.0),           # Patient age in years
    "sex": {0.0, 1.0},              # 0: Female, 1: Male
    "cp": {1.0, 2.0, 3.0, 4.0},     # Chest pain type (1: typical angina, 2: atypical, 3: non-anginal, 4: asymptomatic)
    "trestbps": (70.0, 250.0),      # Resting blood pressure in mm Hg
    "chol": (80.0, 650.0),          # Serum cholesterol in mg/dl
    "fbs": {0.0, 1.0},              # Fasting blood sugar > 120 mg/dl (1: True, 0: False)
    "restecg": {0.0, 1.0, 2.0},     # Resting ECG (0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy)
    "thalach": (60.0, 230.0),       # Maximum heart rate achieved
    "exang": {0.0, 1.0},            # Exercise induced angina (1: Yes, 0: No)
    "oldpeak": (0.0, 8.0),          # ST depression induced by exercise relative to rest
    "slope": {1.0, 2.0, 3.0},       # Slope of the peak exercise ST segment (1: upsloping, 2: flat, 3: downsloping)
    "ca": {0.0, 1.0, 2.0, 3.0},     # Number of major vessels (0-3) colored by flourosopy
    "thal": {3.0, 6.0, 7.0},        # Thalassemia (3: normal, 6: fixed defect, 7: reversable defect)
    "target": {0, 1},               # 0: Absence of disease, 1: Presence of disease
}
