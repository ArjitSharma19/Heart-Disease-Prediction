"""Unit and integration tests for Heart Disease prediction pipeline (Stage 1)."""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import (
    CLEAN_DATA_PATH,
    COLUMN_NAMES,
    FIGURES_DIR,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    RAW_DATA_PATH,
    SPLIT_INDICES_PATH,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
)
from src.data_cleaning import clean_data, detect_iqr_outliers, report_missing_values
from src.preprocessing import fit_and_validate_pipeline, get_preprocessor, get_train_test_split
from src.utils import load_raw_data


class TestStage1Pipeline(unittest.TestCase):
    """Test suite for data loading, cleaning, EDA, and preprocessing."""

    def test_01_raw_data_exists_and_shape(self):
        """Verify raw Cleveland dataset is present and has 303 rows and 14 columns."""
        df_raw = load_raw_data(RAW_DATA_PATH)
        self.assertEqual(df_raw.shape, (303, 14))
        self.assertListEqual(list(df_raw.columns), COLUMN_NAMES)

    def test_02_cleaning_and_no_premature_imputation(self):
        """Verify cleaned data preserves NaNs (no premature imputation leakage)."""
        df_clean = clean_data(save_clean=False)
        self.assertEqual(df_clean.shape, (303, 14))
        # Known missing entries: 4 in ca, 2 in thal
        self.assertEqual(df_clean["ca"].isnull().sum(), 4)
        self.assertEqual(df_clean["thal"].isnull().sum(), 2)
        self.assertEqual(df_clean.isnull().sum().sum(), 6)

        # Target binarization check
        self.assertTrue(set(df_clean["target"].unique()).issubset({0, 1}))
        self.assertEqual(int(df_clean["target"].value_counts()[0]), 164)
        self.assertEqual(int(df_clean["target"].value_counts()[1]), 139)

    def test_03_outlier_audit(self):
        """Verify IQR outlier detection detects outliers and does not alter rows."""
        df_clean = pd.read_csv(CLEAN_DATA_PATH)
        outlier_info = detect_iqr_outliers(df_clean)
        self.assertIn("trestbps", outlier_info)
        self.assertIn("chol", outlier_info)
        self.assertIn("thalach", outlier_info)
        self.assertIn("oldpeak", outlier_info)
        self.assertEqual(outlier_info["trestbps"]["count"], 9)
        self.assertEqual(outlier_info["chol"]["count"], 5)
        self.assertEqual(outlier_info["thalach"]["count"], 1)
        self.assertEqual(outlier_info["oldpeak"]["count"], 5)

    def test_04_eda_figures_exist(self):
        """Verify all 5 required EDA figures were saved to reports/figures/."""
        expected_figures = [
            "target_distribution.png",
            "numeric_distributions_by_target.png",
            "categorical_distributions_by_target.png",
            "correlation_heatmap.png",
            "missing_values_summary.png",
        ]
        for fig_name in expected_figures:
            fig_path = FIGURES_DIR / fig_name
            self.assertTrue(fig_path.exists(), f"Figure missing: {fig_path}")
            self.toBeGreater(fig_path.stat().st_size, 1000)

    def toBeGreater(self, a, b):
        self.assertGreater(a, b)

    def test_05_preprocessing_pipeline_and_splits(self):
        """Verify ColumnTransformer fits solely on train, eliminates NaNs, and produces 28 features."""
        X_train, X_test, y_train, y_test = get_train_test_split(save_splits=False)
        self.assertEqual(X_train.shape, (242, 13))
        self.assertEqual(X_test.shape, (61, 13))

        # Check stratification balance
        train_pos_rate = y_train.mean()
        test_pos_rate = y_test.mean()
        self.assertAlmostEqual(train_pos_rate, test_pos_rate, delta=0.01)

        # Transformer test
        preprocessor = get_preprocessor()
        X_train_proc, X_test_proc, feature_names = fit_and_validate_pipeline(
            X_train, X_test, y_train, y_test, preprocessor
        )

        # Shapes
        self.assertEqual(X_train_proc.shape, (242, 28))
        self.assertEqual(X_test_proc.shape, (61, 28))

        # Zero NaNs
        self.assertEqual(np.isnan(X_train_proc).sum(), 0)
        self.assertEqual(np.isnan(X_test_proc).sum(), 0)


if __name__ == "__main__":
    unittest.main()

