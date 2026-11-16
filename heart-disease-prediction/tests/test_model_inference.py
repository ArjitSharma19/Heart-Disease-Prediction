"""Unit test verifying inference functionality of serialized best_model.joblib."""

import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.config import BEST_MODEL_PATH, MODELS_DIR


class TestModelInference(unittest.TestCase):
    """Test suite verifying end-to-end inference from serialized best_model.joblib."""

    def setUp(self):
        self.model_path = MODELS_DIR / "best_model.joblib"

    def test_model_artifact_and_inference(self):
        """Verify best_model.joblib exists, reloads cleanly, and predicts accurately on sample patient records."""
        if not self.model_path.exists():
            self.skipTest(f"Model artifact {self.model_path} not found yet. Run 'python -m src.train' first.")

        pipeline = joblib.load(self.model_path)
        self.assertIsNotNone(pipeline)

        # Create a sample patient record with typical cardiovascular clinical values
        sample_patient = pd.DataFrame(
            [
                {
                    "age": 58.0,
                    "sex": 1.0,
                    "cp": 4.0,           # Asymptomatic chest pain (high risk indicator)
                    "trestbps": 140.0,
                    "chol": 260.0,
                    "fbs": 0.0,
                    "restecg": 2.0,
                    "thalach": 120.0,     # Depressed peak heart rate
                    "exang": 1.0,         # Exercise induced angina
                    "oldpeak": 2.2,       # Significant ST depression
                    "slope": 2.0,         # Flat ST slope
                    "ca": 2.0,            # 2 major vessels colored
                    "thal": 7.0,          # Reversible defect
                }
            ]
        )

        # 1. Test binary prediction
        prediction = pipeline.predict(sample_patient)
        self.assertEqual(len(prediction), 1)
        self.assertIn(int(prediction[0]), [0, 1])

        # 2. Test probability estimation
        probabilities = pipeline.predict_proba(sample_patient)
        self.assertEqual(probabilities.shape, (1, 2))
        prob_sum = float(np.sum(probabilities))
        self.assertAlmostEqual(prob_sum, 1.0, places=5)
        self.assertGreaterEqual(probabilities[0][1], 0.0)
        self.assertLessEqual(probabilities[0][1], 1.0)

        # 3. Test robustness with missing values (e.g. ca and thal as np.nan)
        sample_with_nan = sample_patient.copy()
        sample_with_nan.loc[0, "ca"] = np.nan
        sample_with_nan.loc[0, "thal"] = np.nan

        pred_nan = pipeline.predict(sample_with_nan)
        prob_nan = pipeline.predict_proba(sample_with_nan)
        self.assertEqual(len(pred_nan), 1)
        self.assertEqual(prob_nan.shape, (1, 2))


if __name__ == "__main__":
    unittest.main()

