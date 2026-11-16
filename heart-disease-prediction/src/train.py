"""Model training, hyperparameter optimization, model benchmarking, and artifact serialization.

Step 1 of Modeling:
- Embeds unfitted ColumnTransformer and classifier inside sklearn Pipeline (zero data leakage).
- Evaluates models on training set using 5-Fold StratifiedKFold CV.
- Optimizes:
    a) Baseline: Logistic Regression (light tune over C and penalty with liblinear solver).
    b) Random Forest Classifier: tune over n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features.
    c) XGBClassifier: tune over n_estimators, max_depth, learning_rate, subsample, colsample_bytree.
- Uses RECALL as primary tuning/selection metric, while tracking ROC-AUC and F1.
- Selects winning model: highest CV Recall, with CV ROC-AUC as tiebreaker.
- Saves models/logreg.joblib, models/rf.joblib, models/xgb.joblib, models/best_model.joblib, and models/metadata.json.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import (
    CATEGORICAL_FEATURES,
    MODELS_DIR,
    NUMERIC_FEATURES,
    PLAUSIBLE_RANGES,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COL,
    TRAIN_DATA_PATH,
)
from src.preprocessing import get_preprocessor
from src.utils import save_json, setup_logger

logger = setup_logger("train")

# Paths for saved model artifacts
LOGREG_MODEL_PATH = MODELS_DIR / "logreg.joblib"
RF_MODEL_PATH = MODELS_DIR / "rf.joblib"
XGB_MODEL_PATH = MODELS_DIR / "xgb.joblib"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
CV_RESULTS_CSV = REPORTS_DIR / "cv_benchmarks.csv"
CV_RESULTS_JSON = REPORTS_DIR / "cv_benchmarks.json"


def tune_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> Tuple[Pipeline, Dict[str, Any], Dict[str, float]]:
    """Tune Baseline Logistic Regression pipeline favoring Recall.

    Uses liblinear solver to support both L1 and L2 penalties without deprecation warnings.
    """
    logger.info("Tuning Baseline Model: Logistic Regression...")
    pipeline = Pipeline(
        steps=[
            ("preprocessor", get_preprocessor()),
            ("classifier", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, solver="liblinear")),
        ]
    )

    param_grid = {
        "classifier__C": [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
        "classifier__penalty": ["l1", "l2"],
        "classifier__class_weight": [None, "balanced"],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="recall",
        n_jobs=1,
        refit="recall",
    )
    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Calculate multi-metric cross validation for stability reporting
    scoring = ["recall", "roc_auc", "f1", "precision", "accuracy"]
    cv_scores = cross_validate(best_pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)

    metrics = {
        "recall_mean": float(np.mean(cv_scores["test_recall"])),
        "recall_std": float(np.std(cv_scores["test_recall"])),
        "roc_auc_mean": float(np.mean(cv_scores["test_roc_auc"])),
        "roc_auc_std": float(np.std(cv_scores["test_roc_auc"])),
        "f1_mean": float(np.mean(cv_scores["test_f1"])),
        "f1_std": float(np.std(cv_scores["test_f1"])),
        "precision_mean": float(np.mean(cv_scores["test_precision"])),
        "precision_std": float(np.std(cv_scores["test_precision"])),
        "accuracy_mean": float(np.mean(cv_scores["test_accuracy"])),
        "accuracy_std": float(np.std(cv_scores["test_accuracy"])),
    }

    logger.info(
        f"Logistic Regression Best CV Recall: {metrics['recall_mean']:.4f} (+/- {metrics['recall_std']:.4f}), "
        f"ROC-AUC: {metrics['roc_auc_mean']:.4f}, F1: {metrics['f1_mean']:.4f}"
    )
    logger.info(f"Logistic Regression Best Params: {best_params}")

    return best_pipeline, best_params, metrics


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> Tuple[Pipeline, Dict[str, Any], Dict[str, float]]:
    """Tune Random Forest Classifier pipeline favoring Recall.

    Explores ensemble depth, estimators, and splitting thresholds.
    """
    logger.info("Tuning Random Forest Classifier...")
    pipeline = Pipeline(
        steps=[
            ("preprocessor", get_preprocessor()),
            ("classifier", RandomForestClassifier(random_state=RANDOM_STATE)),
        ]
    )

    param_distributions = {
        "classifier__n_estimators": [50, 100, 150, 200, 250],
        "classifier__max_depth": [3, 4, 5, 6, 8, None],
        "classifier__min_samples_split": [2, 4, 6, 8],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2", None],
        "classifier__class_weight": [None, "balanced", "balanced_subsample"],
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=30,
        cv=cv,
        scoring="recall",
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit="recall",
    )
    search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_
    best_params = search.best_params_

    scoring = ["recall", "roc_auc", "f1", "precision", "accuracy"]
    cv_scores = cross_validate(best_pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)

    metrics = {
        "recall_mean": float(np.mean(cv_scores["test_recall"])),
        "recall_std": float(np.std(cv_scores["test_recall"])),
        "roc_auc_mean": float(np.mean(cv_scores["test_roc_auc"])),
        "roc_auc_std": float(np.std(cv_scores["test_roc_auc"])),
        "f1_mean": float(np.mean(cv_scores["test_f1"])),
        "f1_std": float(np.std(cv_scores["test_f1"])),
        "precision_mean": float(np.mean(cv_scores["test_precision"])),
        "precision_std": float(np.std(cv_scores["test_precision"])),
        "accuracy_mean": float(np.mean(cv_scores["test_accuracy"])),
        "accuracy_std": float(np.std(cv_scores["test_accuracy"])),
    }

    logger.info(
        f"Random Forest Best CV Recall: {metrics['recall_mean']:.4f} (+/- {metrics['recall_std']:.4f}), "
        f"ROC-AUC: {metrics['roc_auc_mean']:.4f}, F1: {metrics['f1_mean']:.4f}"
    )
    logger.info(f"Random Forest Best Params: {best_params}")

    return best_pipeline, best_params, metrics


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> Tuple[Pipeline, Dict[str, Any], Dict[str, float]]:
    """Tune XGBoost Classifier pipeline favoring Recall."""
    logger.info("Tuning XGBoost Classifier...")
    pipeline = Pipeline(
        steps=[
            ("preprocessor", get_preprocessor()),
            ("classifier", XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss")),
        ]
    )

    param_distributions = {
        "classifier__n_estimators": [50, 80, 100, 150, 200],
        "classifier__max_depth": [2, 3, 4, 5],
        "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15],
        "classifier__subsample": [0.6, 0.7, 0.8, 1.0],
        "classifier__colsample_bytree": [0.6, 0.7, 0.8, 1.0],
        "classifier__scale_pos_weight": [1.0, 1.2, 1.5],
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=30,
        cv=cv,
        scoring="recall",
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit="recall",
    )
    search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_
    best_params = search.best_params_

    scoring = ["recall", "roc_auc", "f1", "precision", "accuracy"]
    cv_scores = cross_validate(best_pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)

    metrics = {
        "recall_mean": float(np.mean(cv_scores["test_recall"])),
        "recall_std": float(np.std(cv_scores["test_recall"])),
        "roc_auc_mean": float(np.mean(cv_scores["test_roc_auc"])),
        "roc_auc_std": float(np.std(cv_scores["test_roc_auc"])),
        "f1_mean": float(np.mean(cv_scores["test_f1"])),
        "f1_std": float(np.std(cv_scores["test_f1"])),
        "precision_mean": float(np.mean(cv_scores["test_precision"])),
        "precision_std": float(np.std(cv_scores["test_precision"])),
        "accuracy_mean": float(np.mean(cv_scores["test_accuracy"])),
        "accuracy_std": float(np.std(cv_scores["test_accuracy"])),
    }

    logger.info(
        f"XGBoost Best CV Recall: {metrics['recall_mean']:.4f} (+/- {metrics['recall_std']:.4f}), "
        f"ROC-AUC: {metrics['roc_auc_mean']:.4f}, F1: {metrics['f1_mean']:.4f}"
    )
    logger.info(f"XGBoost Best Params: {best_params}")

    return best_pipeline, best_params, metrics


def select_best_model(models_summary: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Select winning model using Recall first, then ROC-AUC as tiebreaker.

    DECISION RULE:
    1. Primary ranking: Recall Mean (descending) - in heart disease diagnostics, minimizing
       false negatives (missing sick patients) is clinically paramount.
    2. Tiebreaker: ROC-AUC Mean (descending) - overall discriminatory capability across all operating points.
    """
    logger.info("Executing Model Selection Rule: Recall first, ROC-AUC as tiebreaker...")
    sorted_models = sorted(
        models_summary,
        key=lambda item: (item["metrics"]["recall_mean"], item["metrics"]["roc_auc_mean"]),
        reverse=True,
    )
    winner = sorted_models[0]
    logger.info(
        f"SELECTION RESULT: '{winner['name']}' wins with CV Recall = {winner['metrics']['recall_mean']:.4f} "
        f"and CV ROC-AUC = {winner['metrics']['roc_auc_mean']:.4f}"
    )
    return winner


def train_models(train_path: Path = TRAIN_DATA_PATH) -> Dict[str, Any]:
    """Execute Stage 2 training workflow and persist all fitted model pipelines."""
    train_path = Path(train_path)
    logger.info(f"Loading training data from {train_path}...")
    train_df = pd.read_csv(train_path, index_col=0)

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]

    # Ensure 5-Fold Stratified CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Baseline Logistic Regression
    pipe_lr, params_lr, metrics_lr = tune_logistic_regression(X_train, y_train, cv)
    pipe_lr.fit(X_train, y_train)
    joblib.dump(pipe_lr, LOGREG_MODEL_PATH)
    logger.info(f"Saved fitted Logistic Regression to {LOGREG_MODEL_PATH}")

    # 2. Random Forest
    pipe_rf, params_rf, metrics_rf = tune_random_forest(X_train, y_train, cv)
    pipe_rf.fit(X_train, y_train)
    joblib.dump(pipe_rf, RF_MODEL_PATH)
    logger.info(f"Saved fitted Random Forest to {RF_MODEL_PATH}")

    # 3. XGBoost
    pipe_xgb, params_xgb, metrics_xgb = tune_xgboost(X_train, y_train, cv)
    pipe_xgb.fit(X_train, y_train)
    joblib.dump(pipe_xgb, XGB_MODEL_PATH)
    logger.info(f"Saved fitted XGBoost to {XGB_MODEL_PATH}")

    models_summary = [
        {"name": "Logistic Regression", "pipeline": pipe_lr, "path": LOGREG_MODEL_PATH, "params": params_lr, "metrics": metrics_lr},
        {"name": "Random Forest", "pipeline": pipe_rf, "path": RF_MODEL_PATH, "params": params_rf, "metrics": metrics_rf},
        {"name": "XGBoost", "pipeline": pipe_xgb, "path": XGB_MODEL_PATH, "params": params_xgb, "metrics": metrics_xgb},
    ]

    # Save CV benchmarks comparison table
    cv_table_records = []
    for item in models_summary:
        m = item["metrics"]
        cv_table_records.append(
            {
                "Model": item["name"],
                "CV Recall Mean": round(m["recall_mean"], 4),
                "CV Recall Std": round(m["recall_std"], 4),
                "CV ROC-AUC Mean": round(m["roc_auc_mean"], 4),
                "CV ROC-AUC Std": round(m["roc_auc_std"], 4),
                "CV F1 Mean": round(m["f1_mean"], 4),
                "CV F1 Std": round(m["f1_std"], 4),
                "CV Precision Mean": round(m["precision_mean"], 4),
                "CV Accuracy Mean": round(m["accuracy_mean"], 4),
                "Best Params": str(item["params"]),
            }
        )
    cv_df = pd.DataFrame(cv_table_records)
    cv_df.to_csv(CV_RESULTS_CSV, index=False)
    save_json(cv_table_records, CV_RESULTS_JSON)

    # 4. Model Selection & Saving final pipeline
    winner = select_best_model(models_summary)
    joblib.dump(winner["pipeline"], BEST_MODEL_PATH)
    logger.info(f"Saved winning pipeline to {BEST_MODEL_PATH}")

    # Feature names from fitted preprocessor
    fitted_preprocessor = winner["pipeline"].named_steps["preprocessor"]
    feature_names = fitted_preprocessor.get_feature_names_out().tolist()

    # Extract categorical option values
    categorical_options = {
        col: sorted(list(PLAUSIBLE_RANGES[col]))
        for col in CATEGORICAL_FEATURES
        if isinstance(PLAUSIBLE_RANGES.get(col), set)
    }

    metadata = {
        "model_name": winner["name"],
        "model_artifact": str(BEST_MODEL_PATH.name),
        "selection_rule": "Highest CV Recall on training set; tiebreaker: highest CV ROC-AUC",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "chosen_threshold": 0.50,  # Default threshold; updated during evaluation threshold analysis
        "cv_metrics": winner["metrics"],
        "best_hyperparameters": winner["params"],
        "input_feature_names": list(X_train.columns),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "categorical_options": categorical_options,
        "processed_feature_names": feature_names,
        "num_processed_features": len(feature_names),
    }
    save_json(metadata, METADATA_PATH)

    border = "=" * 90
    print(f"\n{border}")
    print(" 5-FOLD CROSS-VALIDATION TRAINING BENCHMARK (ON X_TRAIN ONLY)")
    print(f"{border}")
    print(cv_df[["Model", "CV Recall Mean", "CV Recall Std", "CV ROC-AUC Mean", "CV F1 Mean", "CV Accuracy Mean"]].to_string(index=False))
    print(f"\nWINNER SELECTED: {winner['name']}")
    print(f"Selection Rule: Recall first ({winner['metrics']['recall_mean']:.4f}), ROC-AUC tiebreaker ({winner['metrics']['roc_auc_mean']:.4f})")
    print(f"Artifact Saved: {BEST_MODEL_PATH}")
    print(f"{border}\n")

    return winner


if __name__ == "__main__":
    train_models()
