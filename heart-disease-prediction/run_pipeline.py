"""End-to-End Pipeline Runner for Heart Disease Prediction (Stage 1 & Stage 2)."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_cleaning import clean_data
from src.eda import run_eda
from src.evaluate import evaluate_models
from src.preprocessing import fit_and_validate_pipeline, get_preprocessor, get_train_test_split
from src.train import train_models
from src.utils import setup_logger

logger = setup_logger("run_pipeline")


def run_full_pipeline():
    """Execute end-to-end: Ingestion, Cleaning, EDA, Preprocessing, Training, and Evaluation."""
    print("\n" + "=" * 90)
    print(" HEART DISEASE PREDICTION PIPELINE - FULL END-TO-END EXECUTION")
    print("=" * 90)

    # 1. Data Ingestion & Cleaning
    print("\n>>> STEP 1: INGESTION, INSPECTION & DATA CLEANING")
    df_clean = clean_data()

    # 2. Exploratory Data Analysis & Visualizations
    print("\n>>> STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
    run_eda()

    # 3. Preprocessing Pipeline & Stratified Train/Test Splitting
    print("\n>>> STEP 3: PREPROCESSING & SPLIT VERIFICATION")
    X_train, X_test, y_train, y_test = get_train_test_split(df=df_clean)
    preprocessor = get_preprocessor()
    fit_and_validate_pipeline(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
    )

    # 4. Model Training & 5-Fold Stratified Cross Validation
    print("\n>>> STEP 4: MODEL TRAINING & 5-FOLD CV OPTIMIZATION")
    train_models()

    # 5. Held-Out Test Evaluation & Diagnostics
    print("\n>>> STEP 5: HELD-OUT TEST EVALUATION, THRESHOLD TUNING & PLOTS")
    evaluate_models()

    print("\n" + "=" * 90)
    print(" PIPELINE COMPLETED SUCCESSFULLY! All models, reports, and figures generated.")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    run_full_pipeline()
