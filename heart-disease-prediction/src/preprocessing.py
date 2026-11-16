"""Preprocessing pipeline module for Heart Disease prediction.

Builds leak-free ColumnTransformer and stratified train/test split.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    CLEAN_DATA_PATH,
    NUMERIC_FEATURES,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    SPLIT_INDICES_PATH,
    TARGET_COL,
    TEST_DATA_PATH,
    TEST_SIZE,
    TRAIN_DATA_PATH,
)
from src.utils import save_json, setup_logger

logger = setup_logger("preprocessing")


def get_preprocessor() -> ColumnTransformer:
    """Build and return an unfitted scikit-learn ColumnTransformer.

    Architecture:
    - Numeric features: SimpleImputer(strategy='median') -> StandardScaler()
    - Categorical features: SimpleImputer(strategy='most_frequent') -> OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def get_train_test_split(
    df: Optional[pd.DataFrame] = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    save_splits: bool = True,
    clean_data_path: Path = CLEAN_DATA_PATH,
    train_output_path: Path = TRAIN_DATA_PATH,
    test_output_path: Path = TEST_DATA_PATH,
    indices_output_path: Path = SPLIT_INDICES_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform a stratified 80/20 train/test split on cleaned data and persist split artifacts.

    Ensures the test set remains strictly isolated for unbiased final evaluation.
    """
    if df is None:
        clean_data_path = Path(clean_data_path)
        logger.info(f"Loading cleaned dataset from {clean_data_path}...")
        df = pd.read_csv(clean_data_path)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
        shuffle=True,
    )

    train_class_dist = y_train.value_counts(normalize=True).to_dict()
    test_class_dist = y_test.value_counts(normalize=True).to_dict()

    logger.info("--- Stratified Train/Test Split Generated ---")
    logger.info(f"Train set: {X_train.shape[0]} samples | Class distribution: {train_class_dist}")
    logger.info(f"Test set:  {X_test.shape[0]} samples | Class distribution: {test_class_dist}")

    if save_splits:
        # Save complete split dataframes
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        Path(train_output_path).parent.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(train_output_path, index=True)
        test_df.to_csv(test_output_path, index=True)
        logger.info(f"Saved train set to {train_output_path} (Shape: {train_df.shape})")
        logger.info(f"Saved test set to {test_output_path} (Shape: {test_df.shape})")

        # Save split indices and metadata for reproducibility
        split_metadata = {
            "random_state": random_state,
            "test_size": test_size,
            "stratified": True,
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "train_indices": [int(i) for i in X_train.index.tolist()],
            "test_indices": [int(i) for i in X_test.index.tolist()],
            "train_class_counts": {int(k): int(v) for k, v in y_train.value_counts().items()},
            "test_class_counts": {int(k): int(v) for k, v in y_test.value_counts().items()},
            "train_class_proportions": {int(k): round(float(v), 4) for k, v in train_class_dist.items()},
            "test_class_proportions": {int(k): round(float(v), 4) for k, v in test_class_dist.items()},
        }
        save_json(split_metadata, indices_output_path)

    return X_train, X_test, y_train, y_test


def fit_and_validate_pipeline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: Optional[ColumnTransformer] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Fit ColumnTransformer strictly on X_train and transform both splits.

    Validates:
    1. Zero remaining NaNs in train and test transformed outputs.
    2. Identical feature dimensions matching one-hot expansion.
    3. Consistency of train and test class distributions.
    """
    if preprocessor is None:
        preprocessor = get_preprocessor()

    # Fit ONLY on training data to prevent information leakage
    logger.info("Fitting preprocessor strictly on X_train...")
    preprocessor.fit(X_train)

    # Transform both splits
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out().tolist()

    # Verification 1: Check for any remaining NaNs
    train_nans = int(np.isnan(X_train_proc).sum())
    test_nans = int(np.isnan(X_test_proc).sum())

    # Verification 2: Output shape inspection
    expected_rows_train = len(X_train)
    expected_rows_test = len(X_test)
    num_features = len(feature_names)

    # Verification 3: Class balance validation
    train_ratio_pos = float(y_train.mean())
    test_ratio_pos = float(y_test.mean())
    ratio_diff = abs(train_ratio_pos - test_ratio_pos)

    border = "=" * 80
    print(f"\n{border}")
    print(" PREPROCESSING PIPELINE VERIFICATION REPORT")
    print(f"{border}")
    print(f"1. Zero NaN Check:")
    print(f"   - Transformed X_train NaNs: {train_nans} {'[PASSED]' if train_nans == 0 else '[FAILED]'}")
    print(f"   - Transformed X_test NaNs:  {test_nans} {'[PASSED]' if test_nans == 0 else '[FAILED]'}")

    print(f"\n2. Output Shapes:")
    print(f"   - Transformed X_train Shape: {X_train_proc.shape} (Expected: ({expected_rows_train}, {num_features}))")
    print(f"   - Transformed X_test Shape:  {X_test_proc.shape} (Expected: ({expected_rows_test}, {num_features}))")
    print(f"   - Total Processed Features:  {num_features}")
    print(f"   - Feature Names: {feature_names}")

    print(f"\n3. Stratified Class Balance Check:")
    print(f"   - Train Positive Class Ratio (Disease=1): {train_ratio_pos * 100:.2f}% ({y_train.sum()}/{len(y_train)})")
    print(f"   - Test Positive Class Ratio (Disease=1):  {test_ratio_pos * 100:.2f}% ({y_test.sum()}/{len(y_test)})")
    print(f"   - Absolute Class Ratio Difference:        {ratio_diff * 100:.4f}% (< 0.1% expected for perfect stratification)")
    print(f"{border}\n")

    assert train_nans == 0, f"Error: Found {train_nans} NaNs in X_train_proc!"
    assert test_nans == 0, f"Error: Found {test_nans} NaNs in X_test_proc!"
    assert X_train_proc.shape == (expected_rows_train, num_features), "Error: X_train shape mismatch!"
    assert X_test_proc.shape == (expected_rows_test, num_features), "Error: X_test shape mismatch!"
    assert ratio_diff < 0.01, f"Error: Stratification discrepancy too high: {ratio_diff}"

    logger.info("All verification assertions successfully passed!")
    return X_train_proc, X_test_proc, feature_names


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_train_test_split()
    transformer = get_preprocessor()
    fit_and_validate_pipeline(X_train, X_test, y_train, y_test, transformer)

