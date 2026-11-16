"""Data cleaning and quality assurance module for Heart Disease prediction."""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from src.config import (
    CLEAN_DATA_PATH,
    COLUMN_NAMES,
    NUMERIC_FEATURES,
    OUTLIER_FEATURES,
    PLAUSIBLE_RANGES,
    RAW_DATA_PATH,
    TARGET_COL,
)
from src.utils import inspect_dataframe, load_raw_data, setup_logger

logger = setup_logger("data_cleaning")


def replace_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Replace '?' missing indicators with np.nan and cast all columns to numeric."""
    df_clean = df.copy()
    # Replace '?' indicator with NaN
    df_clean = df_clean.replace("?", np.nan)

    # Cast all columns to float numeric representation
    for col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    return df_clean


def report_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate and log missing value counts and percentages per column."""
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    missing_summary = pd.DataFrame(
        {
            "Missing_Count": missing_count,
            "Missing_Percentage": missing_pct.round(2),
        }
    )

    logger.info("--- Missing Values Report ---")
    columns_with_missing = missing_summary[missing_summary["Missing_Count"] > 0]
    if len(columns_with_missing) > 0:
        for col, row in columns_with_missing.iterrows():
            logger.info(
                f"Feature '{col}': {int(row['Missing_Count'])} missing values ({row['Missing_Percentage']}%)"
            )
    else:
        logger.info("No missing values detected.")

    return missing_summary


def binarize_target(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """Convert multi-class heart disease target (0-4) to binary (0: no disease, 1: disease)."""
    df_clean = df.copy()
    raw_counts = df_clean[target_col].value_counts().sort_index().to_dict()
    logger.info(f"Original target distribution (0-4): {raw_counts}")

    # 0 = absence of disease, 1-4 = presence of heart disease
    df_clean[target_col] = (df_clean[target_col] > 0).astype(int)
    bin_counts = df_clean[target_col].value_counts().sort_index().to_dict()
    logger.info(
        f"Binarized target distribution (0 vs 1): {bin_counts} "
        f"[Class 0: {bin_counts.get(0, 0)} ({bin_counts.get(0, 0)/len(df_clean)*100:.2f}%), "
        f"Class 1: {bin_counts.get(1, 0)} ({bin_counts.get(1, 0)/len(df_clean)*100:.2f}%)]"
    )
    return df_clean


def check_and_drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Audit and remove duplicate records, logging the exact count."""
    dup_count = int(df.duplicated().sum())
    logger.info(f"Duplicate rows audit: {dup_count} duplicate row(s) found.")
    if dup_count > 0:
        df_clean = df.drop_duplicates().reset_index(drop=True)
        logger.info(f"Dropped {dup_count} duplicate row(s). New shape: {df_clean.shape}")
        return df_clean
    return df


def detect_iqr_outliers(
    df: pd.DataFrame, features: List[str] = OUTLIER_FEATURES
) -> Dict[str, Dict]:
    """Detect outliers using the Interquartile Range (IQR) method and log retention decision.

    Outliers are flagged when value < Q1 - 1.5 * IQR or value > Q3 + 1.5 * IQR.
    Decision: Outliers are NOT deleted because the Cleveland dataset is small (N=303)
    and extreme physiological observations (elevated blood pressure, high cholesterol,
    severe ST depression) reflect genuine cardiovascular pathology essential for model training.
    """
    outlier_info = {}
    logger.info("--- IQR Outlier Audit (Q1 - 1.5*IQR, Q3 + 1.5*IQR) ---")

    for col in features:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_values = df.loc[outlier_mask, col].tolist()
        count = len(outlier_values)

        outlier_info[col] = {
            "q1": round(float(q1), 2),
            "q3": round(float(q3), 2),
            "iqr": round(float(iqr), 2),
            "lower_bound": round(float(lower_bound), 2),
            "upper_bound": round(float(upper_bound), 2),
            "count": count,
            "percentage": round(float(count / len(df) * 100), 2),
            "values": [round(float(v), 2) for v in outlier_values],
        }

        logger.info(
            f"Feature '{col}': Bounds=[{lower_bound:.2f}, {upper_bound:.2f}], "
            f"Outliers Count={count} ({outlier_info[col]['percentage']}%)"
        )

    logger.info(
        "OUTLIER DECISION LOG: All detected outliers are RETAINED in the dataset. "
        "Reasoning: In small clinical cohorts (N=303), extreme values in cardiac parameters "
        "(e.g., serum cholesterol up to 564 mg/dl or resting blood pressure up to 200 mm Hg) "
        "represent genuine, clinically significant manifestations of advanced coronary artery disease. "
        "Truncating or removing them would distort physiological signal and induce bias."
    )
    return outlier_info


def validate_dtypes_and_ranges(df: pd.DataFrame) -> bool:
    """Validate that all columns conform to expected numeric dtypes and clinical ranges."""
    logger.info("--- Validating Data Types and Domain Ranges ---")
    all_valid = True

    for col in COLUMN_NAMES:
        # Check dtype
        if not np.issubdtype(df[col].dtype, np.number):
            logger.error(f"Column '{col}' is not numeric! Dtype: {df[col].dtype}")
            all_valid = False

        # Check plausible range for non-null values
        valid_range = PLAUSIBLE_RANGES.get(col)
        non_null_vals = df[col].dropna()

        if isinstance(valid_range, tuple):
            min_val, max_val = valid_range
            out_of_range = non_null_vals[(non_null_vals < min_val) | (non_null_vals > max_val)]
            if len(out_of_range) > 0:
                logger.warning(
                    f"Column '{col}' has {len(out_of_range)} value(s) outside plausible range [{min_val}, {max_val}]: {out_of_range.tolist()}"
                )
                all_valid = False
            else:
                logger.info(f"Feature '{col}': Passed continuous range [{min_val}, {max_val}] check (min={non_null_vals.min()}, max={non_null_vals.max()})")

        elif isinstance(valid_range, set):
            invalid_cats = set(non_null_vals.unique()) - valid_range
            if invalid_cats:
                logger.error(f"Column '{col}' has unexpected categorical values: {invalid_cats}")
                all_valid = False
            else:
                logger.info(f"Feature '{col}': Passed discrete category check. Observed values: {sorted(non_null_vals.unique())}")

    if all_valid:
        logger.info("Validation successful: All features conform to expected dtypes and plausible ranges.")
    return all_valid


def clean_data(
    raw_path: Path = RAW_DATA_PATH,
    output_path: Path = CLEAN_DATA_PATH,
    save_clean: bool = True,
) -> pd.DataFrame:
    """Execute end-to-end data cleaning workflow and persist cleaned data.

    CRITICAL LEAKAGE RULE:
    No missing values are imputed here. Imputation must strictly occur inside the
    ColumnTransformer pipeline after the train/test split.
    """
    logger.info("Starting Data Ingestion & Cleaning Stage...")

    # Step 1: Load raw data & print inspection
    df_raw = load_raw_data(raw_path)
    inspect_dataframe(df_raw, title="Step 1: Raw Cleveland Data Inspection")

    # Step 2: Replace '?' and convert to numeric
    df_numeric = replace_missing_indicators(df_raw)

    # Step 3: Missing values audit
    report_missing_values(df_numeric)

    # Step 4: Binarize target (0 vs 1)
    df_binarized = binarize_target(df_numeric, target_col=TARGET_COL)

    # Step 5: Duplicate check & drop
    df_dedup = check_and_drop_duplicates(df_binarized)

    # Step 6: IQR outlier audit
    detect_iqr_outliers(df_dedup, features=OUTLIER_FEATURES)

    # Step 7: Domain range and dtype validation
    validate_dtypes_and_ranges(df_dedup)

    # Confirm missing values are NOT imputed
    remaining_nans = df_dedup.isnull().sum().sum()
    logger.info(
        f"Verification: {remaining_nans} NaNs remain in cleaned dataset (as required; no leakage before train/test split)."
    )

    if save_clean:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_dedup.to_csv(output_path, index=False)
        logger.info(f"Saved cleaned-but-not-imputed data to {output_path} (Shape: {df_dedup.shape})")

    return df_dedup


if __name__ == "__main__":
    clean_data()

