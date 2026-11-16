"""FastAPI REST Service for Heart Disease Risk Prediction.

Exposes endpoints for:
- Health checks and model metadata (/health)
- Single patient real-time prediction (/predict)
- Multi-patient batch scoring (/predict/batch)
- Interactive API documentation at /docs (Swagger UI)
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.config import BEST_MODEL_PATH, METADATA_PATH
from src.utils import load_json, setup_logger

logger = setup_logger("api")

# Load model and metadata at startup
model_pipeline = None
model_metadata = {}

DISCLAIMER_TEXT = (
    "DISCLAIMER: This API is strictly an educational/research prototype. "
    "It is NOT a medical device and must NOT replace professional clinical judgment."
)


def get_model():
    global model_pipeline, model_metadata
    if model_pipeline is None:
        if not BEST_MODEL_PATH.exists():
            raise RuntimeError(f"Model artifact not found at {BEST_MODEL_PATH}.")
        model_pipeline = joblib.load(BEST_MODEL_PATH)
        if METADATA_PATH.exists():
            model_metadata = load_json(METADATA_PATH)
    return model_pipeline, model_metadata


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up model on server boot."""
    try:
        get_model()
        logger.info("FastAPI successfully loaded best_model pipeline on startup.")
    except Exception as e:
        logger.warning(f"Could not load model on startup: {e}")
    yield


app = FastAPI(
    title="Heart Disease Prediction API",
    description="REST API for Cardiovascular Risk Stratification on the UCI Cleveland Dataset.",
    version="1.0.0",
    lifespan=lifespan,
)


class PatientData(BaseModel):
    """Clinical diagnostic observations for a single patient."""
    age: float = Field(..., ge=18.0, le=100.0, description="Age in years", example=58.0)
    sex: float = Field(..., ge=0.0, le=1.0, description="Sex (0.0: Female, 1.0: Male)", example=1.0)
    cp: float = Field(..., ge=1.0, le=4.0, description="Chest pain type (1: typical, 2: atypical, 3: non-anginal, 4: asymptomatic)", example=4.0)
    trestbps: float = Field(..., ge=70.0, le=250.0, description="Resting blood pressure in mm Hg", example=140.0)
    chol: float = Field(..., ge=80.0, le=650.0, description="Serum cholesterol in mg/dl", example=260.0)
    fbs: float = Field(..., ge=0.0, le=1.0, description="Fasting blood sugar > 120 mg/dl (0.0: No, 1.0: Yes)", example=0.0)
    restecg: float = Field(..., ge=0.0, le=2.0, description="Resting ECG (0: normal, 1: ST-T abnormality, 2: LV hypertrophy)", example=2.0)
    thalach: float = Field(..., ge=60.0, le=230.0, description="Maximum heart rate achieved in bpm", example=120.0)
    exang: float = Field(..., ge=0.0, le=1.0, description="Exercise induced angina (0.0: No, 1.0: Yes)", example=1.0)
    oldpeak: float = Field(..., ge=0.0, le=8.0, description="ST depression induced by exercise in mm", example=2.2)
    slope: float = Field(..., ge=1.0, le=3.0, description="Slope of peak exercise ST segment (1: up, 2: flat, 3: down)", example=2.0)
    ca: float = Field(..., ge=0.0, le=3.0, description="Number of major vessels (0-3) colored by fluoroscopy", example=2.0)
    thal: float = Field(..., description="Thalassemia (3.0: normal, 6.0: fixed defect, 7.0: reversible defect)", example=7.0)


class PredictionResponse(BaseModel):
    """Model output payload for a single patient risk assessment."""
    probability: float
    prediction: int
    risk_label: str
    risk_tier: str
    decision_threshold: float
    model_name: str
    disclaimer: str


class BatchPredictionRequest(BaseModel):
    """Batch scoring request payload."""
    patients: List[PatientData]


class BatchPredictionResponse(BaseModel):
    """Batch scoring response payload."""
    total_records: int
    predictions: List[PredictionResponse]



@app.get("/", tags=["General"])
def root():
    """Root welcome endpoint."""
    return {
        "message": "Welcome to the Heart Disease Prediction API",
        "documentation": "/docs",
        "health_check": "/health",
        "disclaimer": DISCLAIMER_TEXT,
    }


@app.get("/health", tags=["General"])
def health_check():
    """Health status and active model metadata."""
    try:
        _, meta = get_model()
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_name": meta.get("model_name", "Unknown"),
            "chosen_threshold": meta.get("chosen_threshold", 0.50),
            "training_date": meta.get("training_date", "Unknown"),
        }
    except Exception as e:
        return {"status": "degraded", "model_loaded": False, "error": str(e)}


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_patient(patient: PatientData):
    """Real-time heart disease risk prediction for a single patient."""
    pipeline, meta = get_model()
    threshold = float(meta.get("chosen_threshold", 0.50))
    feature_order = meta.get(
        "input_feature_names",
        ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    )

    input_df = pd.DataFrame([patient.dict()])[feature_order]

    try:
        proba = float(pipeline.predict_proba(input_df)[0, 1])
        pred = 1 if proba >= threshold else 0

        risk_label = "High Risk of Heart Disease" if pred == 1 else "Low Risk of Heart Disease"
        if proba < 0.35:
            tier = "Low Risk Tier (<35%)"
        elif proba < 0.65:
            tier = "Moderate Risk Tier (35-65%)"
        else:
            tier = "High Risk Tier (>65%)"

        return PredictionResponse(
            probability=round(proba, 4),
            prediction=pred,
            risk_label=risk_label,
            risk_tier=tier,
            decision_threshold=threshold,
            model_name=meta.get("model_name", "XGBoost"),
            disclaimer=DISCLAIMER_TEXT,
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch_patients(batch: BatchPredictionRequest):
    """Batch heart disease risk prediction for multiple patients."""
    pipeline, meta = get_model()
    threshold = float(meta.get("chosen_threshold", 0.50))
    feature_order = meta.get(
        "input_feature_names",
        ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    )

    records = [p.dict() for p in batch.patients]
    input_df = pd.DataFrame(records)[feature_order]

    try:
        probas = pipeline.predict_proba(input_df)[:, 1]
        preds = (probas >= threshold).astype(int)

        responses = []
        for proba, pred in zip(probas, preds):
            proba_flt = float(proba)
            pred_int = int(pred)
            risk_label = "High Risk of Heart Disease" if pred_int == 1 else "Low Risk of Heart Disease"
            tier = "Low Risk Tier (<35%)" if proba_flt < 0.35 else ("Moderate Risk Tier (35-65%)" if proba_flt < 0.65 else "High Risk Tier (>65%)")

            responses.append(
                PredictionResponse(
                    probability=round(proba_flt, 4),
                    prediction=pred_int,
                    risk_label=risk_label,
                    risk_tier=tier,
                    decision_threshold=threshold,
                    model_name=meta.get("model_name", "XGBoost"),
                    disclaimer=DISCLAIMER_TEXT,
                )
            )

        return BatchPredictionResponse(total_records=len(responses), predictions=responses)
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

