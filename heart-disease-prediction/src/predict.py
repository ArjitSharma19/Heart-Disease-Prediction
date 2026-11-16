"""Batch prediction CLI and module for Heart Disease Risk Prediction.

Usage:
    python -m src.predict --input path/to/patients.csv --output path/to/predictions.csv
"""

import argparse
from pathlib import Path
import joblib
import pandas as pd

from src.config import BEST_MODEL_PATH, METADATA_PATH, TARGET_COL
from src.utils import load_json, setup_logger

logger = setup_logger("predict")


def predict_batch(
    input_df: pd.DataFrame,
    model_path: Path = BEST_MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
) -> pd.DataFrame:
    """Run batch prediction on input dataframe using the serialized pipeline."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}.")

    pipeline = joblib.load(model_path)
    metadata = load_json(metadata_path) if Path(metadata_path).exists() else {}

    threshold = float(metadata.get("chosen_threshold", 0.50))
    feature_order = metadata.get(
        "input_feature_names",
        ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    )

    # Validate that required features are present
    missing_features = [f for f in feature_order if f not in input_df.columns]
    if missing_features:
        raise ValueError(f"Input dataframe missing required clinical features: {missing_features}")

    X = input_df[feature_order].copy()
    probabilities = pipeline.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    results_df = input_df.copy()
    results_df["predicted_probability"] = probabilities.round(4)
    results_df["predicted_risk_label"] = [
        "High Risk of Heart Disease" if p == 1 else "Low Risk of Heart Disease" for p in predictions
    ]
    results_df["risk_tier"] = [
        "Low Risk (<35%)" if prob < 0.35 else ("Moderate Risk (35-65%)" if prob < 0.65 else "High Risk (>65%)")
        for prob in probabilities
    ]

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Heart Disease Batch Prediction CLI")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input CSV file containing patient records")
    parser.add_argument("--output", "-o", type=str, default="predictions.csv", help="Path to save output predictions CSV")

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    logger.info(f"Loading patients from {input_path}...")
    df = pd.read_csv(input_path)

    results = predict_batch(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    logger.info(f"Predictions saved successfully to {output_path} ({len(results)} records).")


if __name__ == "__main__":
    main()

