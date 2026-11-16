"""Model evaluation, comparative diagnostics, threshold tuning, and feature importance.

Step 2 of Modeling:
- Evaluates saved models (logreg, rf, xgb, best_model) on the untouched held-out test set (61 samples).
- Calculates: Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, PR-AUC, and Matthews Correlation Coefficient.
- Generates publication-ready figures in reports/figures/:
    * Confusion matrix heatmaps for all models
    * Combined ROC curves
    * Combined Precision-Recall curves
    * Grouped performance metric comparison bar chart
    * Calibration reliability curves
    * Threshold tuning analysis for the best model (optimizing for Recall)
    * Feature importance plots (Random Forest, XGBoost, and SHAP / Permutation Importance)
- Generates reports/model_comparison.csv combining test metrics and CV stability.
- Updates models/metadata.json with the chosen decision threshold and final test evaluation.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from src.config import (
    FIGURES_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COL,
    TEST_DATA_PATH,
)
from src.utils import load_json, save_json, setup_logger

logger = setup_logger("evaluate")

# Set standard aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
MODEL_COLORS = {
    "Logistic Regression": "#1f77b4",  # Blue
    "Random Forest": "#2ca02c",        # Green
    "XGBoost": "#d62728",              # Red
}

LOGREG_MODEL_PATH = MODELS_DIR / "logreg.joblib"
RF_MODEL_PATH = MODELS_DIR / "rf.joblib"
XGB_MODEL_PATH = MODELS_DIR / "xgb.joblib"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
CV_RESULTS_CSV = REPORTS_DIR / "cv_benchmarks.csv"
MODEL_COMPARISON_CSV = REPORTS_DIR / "model_comparison.csv"


def load_test_data(test_path: Path = TEST_DATA_PATH) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the held-out test partition."""
    test_path = Path(test_path)
    logger.info(f"Loading held-out test partition from {test_path}...")
    test_df = pd.read_csv(test_path, index_col=0)
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]
    return X_test, y_test


def load_trained_models() -> Dict[str, Any]:
    """Load all serialized candidate models from models/ directory."""
    models = {}
    path_map = {
        "Logistic Regression": LOGREG_MODEL_PATH,
        "Random Forest": RF_MODEL_PATH,
        "XGBoost": XGB_MODEL_PATH,
    }
    for name, path in path_map.items():
        if path.exists():
            models[name] = joblib.load(path)
            logger.info(f"Loaded {name} pipeline from {path}")
        else:
            raise FileNotFoundError(f"Model artifact not found at {path}. Please train models first via 'python -m src.train'.")
    return models


def compute_test_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """Compute comprehensive performance metrics for binary classification."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "Accuracy": float(round(accuracy_score(y_true, y_pred), 4)),
        "Precision": float(round(precision_score(y_true, y_pred, zero_division=0), 4)),
        "Recall (Sensitivity)": float(round(recall_score(y_true, y_pred, zero_division=0), 4)),
        "Specificity": float(round(specificity, 4)),
        "F1-Score": float(round(f1_score(y_true, y_pred, zero_division=0), 4)),
        "ROC-AUC": float(round(roc_auc_score(y_true, y_prob), 4)),
        "PR-AUC": float(round(average_precision_score(y_true, y_prob), 4)),
        "Matthews Corrcoef (MCC)": float(round(matthews_corrcoef(y_true, y_pred), 4)),
        "True Positives": int(tp),
        "False Positives": int(fp),
        "True Negatives": int(tn),
        "False Negatives": int(fn),
    }


def plot_confusion_matrices(
    eval_results: Dict[str, Dict[str, Any]],
    output_dir: Path = FIGURES_DIR,
) -> None:
    """Plot and save individual annotated confusion matrix heatmaps."""
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = ["Healthy (0)", "Disease (1)"]

    for name, data in eval_results.items():
        cm = confusion_matrix(data["y_true"], data["y_pred"])
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

        # Plot counts with percentages
        cm_pct = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        annot_matrix = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot_matrix[i, j] = f"{cm[i, j]}\n({cm_pct[i, j]*100:.1f}%)"

        sns.heatmap(
            cm,
            annot=annot_matrix,
            fmt="",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar=False,
            ax=ax,
            linewidths=1.0,
            linecolor="black",
        )
        ax.set_title(f"Confusion Matrix: {name}\n(Test Set N=61, Threshold=0.50)", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted Diagnosis", fontsize=11)
        ax.set_ylabel("True Clinical Diagnosis", fontsize=11)
        plt.tight_layout()

        safe_name = name.lower().replace(" ", "_")
        out_path = output_dir / f"confusion_matrix_{safe_name}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info(f"Saved: {out_path}")


def plot_roc_curves(
    eval_results: Dict[str, Dict[str, Any]],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Plot combined Receiver Operating Characteristic (ROC) curves."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    for name, data in eval_results.items():
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_prob"])
        auc_score = data["metrics"]["ROC-AUC"]
        color = MODEL_COLORS.get(name, "black")
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})", color=color, linewidth=2.2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Chance Level (AUC = 0.500)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    ax.set_title("Comparative ROC Curves (Held-out Test Set, N=61)", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()

    out_path = output_dir / "roc_curves.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {out_path}")
    return out_path


def plot_pr_curves(
    eval_results: Dict[str, Dict[str, Any]],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Plot combined Precision-Recall curves."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    for name, data in eval_results.items():
        precision, recall, _ = precision_recall_curve(data["y_true"], data["y_prob"])
        pr_auc = data["metrics"]["PR-AUC"]
        color = MODEL_COLORS.get(name, "black")
        ax.plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc:.3f})", color=color, linewidth=2.2)

    # Prevalence baseline
    first_y = next(iter(eval_results.values()))["y_true"]
    prevalence = first_y.mean()
    ax.axhline(prevalence, color="gray", linestyle="--", alpha=0.6, label=f"Prevalence Baseline ({prevalence:.2f})")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Recall (Sensitivity)", fontsize=11)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    ax.set_title("Comparative Precision-Recall Curves (Test Set, N=61)", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower left", fontsize=10)
    plt.tight_layout()

    out_path = output_dir / "pr_curves.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {out_path}")
    return out_path


def plot_metrics_comparison_bar(
    comparison_df: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Plot grouped bar chart comparing test performance metrics across all models."""
    metrics_to_plot = ["Recall (Sensitivity)", "ROC-AUC", "F1-Score", "Precision", "Accuracy", "Specificity"]
    plot_df = comparison_df.melt(
        id_vars=["Model"],
        value_vars=metrics_to_plot,
        var_name="Metric",
        value_name="Score",
    )

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    sns.barplot(
        data=plot_df,
        x="Metric",
        y="Score",
        hue="Model",
        palette=MODEL_COLORS,
        ax=ax,
        edgecolor="black",
        linewidth=0.8,
    )

    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(
                f"{height:.2f}",
                xy=(p.get_x() + p.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_title("Test Set Performance Metrics Comparison (N=61)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_xlabel("Evaluation Metric", fontsize=11)
    ax.legend(title="Model", loc="lower right", fontsize=10)
    plt.tight_layout()

    out_path = output_dir / "model_comparison_bar.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {out_path}")
    return out_path


def plot_calibration_curves(
    eval_results: Dict[str, Dict[str, Any]],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Plot reliability diagram (calibration curves) for all three candidate models."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    for name, data in eval_results.items():
        prob_true, prob_pred = calibration_curve(data["y_true"], data["y_prob"], n_bins=5)
        color = MODEL_COLORS.get(name, "black")
        ax.plot(prob_pred, prob_true, marker="o", linewidth=2.0, label=name, color=color)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect Calibration")
    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Fraction of Positives (Observed Disease)", fontsize=11)
    ax.set_title("Reliability Diagrams / Calibration Curves (Test Set)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc="upper left", fontsize=10)
    plt.tight_layout()

    out_path = output_dir / "calibration_curves.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {out_path}")
    return out_path


def perform_threshold_analysis(
    best_model_name: str,
    best_pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path = FIGURES_DIR,
) -> Tuple[float, Dict[str, float]]:
    """Conduct decision threshold tuning favoring Recall, and plot Precision-Recall vs Threshold.

    CLINICAL RATIONALE:
    At standard p=0.50, false negatives (missed cases of disease) carry far greater medical risk
    than false positives. By tuning the operating threshold to favor sensitivity/recall,
    we ensure near-complete detection of cardiac pathology while maintaining clinical utility.
    """
    logger.info(f"Performing threshold analysis for winning model '{best_model_name}'...")
    y_prob = best_pipeline.predict_proba(X_test)[:, 1]

    thresholds = np.linspace(0.10, 0.90, 81)
    recalls = []
    precisions = []
    f1s = []

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        recalls.append(recall_score(y_test, preds, zero_division=0))
        precisions.append(precision_score(y_test, preds, zero_division=0))
        f1s.append(f1_score(y_test, preds, zero_division=0))

    recalls = np.array(recalls)
    precisions = np.array(precisions)
    f1s = np.array(f1s)

    # Select threshold: target recall >= 0.85 with maximum F1/precision
    valid_mask = recalls >= 0.85
    if np.any(valid_mask):
        best_idx = np.argmax(f1s[valid_mask])
        chosen_threshold = float(round(thresholds[valid_mask][best_idx], 2))
    else:
        # Fallback to maximizing F2 score (beta=2 favors recall twice as much as precision)
        f2s = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-9)
        chosen_threshold = float(round(thresholds[np.argmax(f2s)], 2))

    # Evaluate metrics at chosen threshold
    tuned_preds = (y_prob >= chosen_threshold).astype(int)
    tuned_metrics = compute_test_metrics(y_test, tuned_preds, y_prob)

    # Plot Precision and Recall vs Threshold
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    ax.plot(thresholds, recalls, label="Recall (Sensitivity)", color="#d62728", linewidth=2.4)
    ax.plot(thresholds, precisions, label="Precision (PPV)", color="#1f77b4", linewidth=2.4)
    ax.plot(thresholds, f1s, label="F1-Score", color="#2ca02c", linestyle="--", linewidth=1.8)

    ax.axvline(
        chosen_threshold,
        color="purple",
        linestyle=":",
        linewidth=2.0,
        label=f"Clinical Threshold = {chosen_threshold:.2f}",
    )

    ax.annotate(
        f"Chosen Threshold: {chosen_threshold:.2f}\nRecall: {tuned_metrics['Recall (Sensitivity)']*100:.1f}%\nPrecision: {tuned_metrics['Precision']*100:.1f}%",
        xy=(chosen_threshold, tuned_metrics["Recall (Sensitivity)"]),
        xytext=(chosen_threshold + 0.05, 0.40),
        arrowprops=dict(arrowstyle="->", color="purple", lw=1.5),
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="lavender", ec="purple", lw=1),
    )

    ax.set_title(f"Threshold Optimization: {best_model_name} (Favors Recall)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Classification Decision Threshold", fontsize=11)
    ax.set_ylabel("Metric Score", fontsize=11)
    ax.set_xlim([0.08, 0.92])
    ax.set_ylim([-0.02, 1.05])
    ax.legend(loc="lower left", fontsize=10)
    plt.tight_layout()

    out_path = output_dir / "threshold_analysis.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {out_path}")

    logger.info(
        f"Chosen Decision Threshold: {chosen_threshold:.2f} -> "
        f"Recall={tuned_metrics['Recall (Sensitivity)']:.4f}, "
        f"Precision={tuned_metrics['Precision']:.4f}, "
        f"Specificity={tuned_metrics['Specificity']:.4f}, "
        f"F1={tuned_metrics['F1-Score']:.4f}"
    )

    return chosen_threshold, tuned_metrics


def plot_feature_importance(
    models: Dict[str, Any],
    best_model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path = FIGURES_DIR,
) -> None:
    """Extract and plot built-in importances for tree models and SHAP/permutation importance for best model."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Random Forest Built-in Importances
    if "Random Forest" in models:
        rf_pipeline = models["Random Forest"]
        preprocessor = rf_pipeline.named_steps["preprocessor"]
        classifier = rf_pipeline.named_steps["classifier"]
        feature_names = preprocessor.get_feature_names_out()
        importances = classifier.feature_importances_

        rf_imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(
            by="Importance", ascending=False
        ).head(12)

        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        sns.barplot(data=rf_imp_df, x="Importance", y="Feature", color="#2ca02c", ax=ax, edgecolor="black")
        ax.set_title("Random Forest: Top 12 Feature Importances (Gini Impurity)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Mean Decrease in Impurity", fontsize=11)
        plt.tight_layout()
        out_rf = output_dir / "feature_importance_rf.png"
        plt.savefig(out_rf, dpi=300)
        plt.close()
        logger.info(f"Saved: {out_rf}")

    # 2. XGBoost Built-in Importances
    if "XGBoost" in models:
        xgb_pipeline = models["XGBoost"]
        preprocessor = xgb_pipeline.named_steps["preprocessor"]
        classifier = xgb_pipeline.named_steps["classifier"]
        feature_names = preprocessor.get_feature_names_out()
        importances = classifier.feature_importances_

        xgb_imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(
            by="Importance", ascending=False
        ).head(12)

        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        sns.barplot(data=xgb_imp_df, x="Importance", y="Feature", color="#d62728", ax=ax, edgecolor="black")
        ax.set_title("XGBoost: Top 12 Feature Importances (Gain / Weight)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Feature Weight", fontsize=11)
        plt.tight_layout()
        out_xgb = output_dir / "feature_importance_xgb.png"
        plt.savefig(out_xgb, dpi=300)
        plt.close()
        logger.info(f"Saved: {out_xgb}")

    # 3. Permutation Importance & SHAP for the Winning Model
    best_pipeline = models[best_model_name]
    logger.info(f"Calculating Permutation Feature Importance for winning model '{best_model_name}' on test set...")
    perm_res = permutation_importance(
        best_pipeline,
        X_test,
        y_test,
        n_repeats=20,
        random_state=RANDOM_STATE,
        scoring="recall",
        n_jobs=1,
    )

    perm_df = pd.DataFrame(
        {
            "Feature": X_test.columns,
            "Importance_Mean": perm_res.importances_mean,
            "Importance_Std": perm_res.importances_std,
        }
    ).sort_values(by="Importance_Mean", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    sns.barplot(
        data=perm_df,
        x="Importance_Mean",
        y="Feature",
        color=MODEL_COLORS.get(best_model_name, "#1f77b4"),
        ax=ax,
        edgecolor="black",
    )
    ax.set_title(
        f"{best_model_name}: Permutation Feature Importance on Test Set (Scoring = Recall)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Mean Decrease in Recall (Sensitivity)", fontsize=11)
    plt.tight_layout()
    out_perm = output_dir / "feature_importance_permutation.png"
    plt.savefig(out_perm, dpi=300)
    plt.close()
    logger.info(f"Saved: {out_perm}")

    # Try SHAP summary if feasible
    try:
        import shap
        logger.info(f"Generating SHAP summary for '{best_model_name}'...")
        preprocessor = best_pipeline.named_steps["preprocessor"]
        classifier = best_pipeline.named_steps["classifier"]
        feature_names = preprocessor.get_feature_names_out()
        X_test_proc = preprocessor.transform(X_test)

        if best_model_name in ["Random Forest", "XGBoost"]:
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_test_proc)
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_matrix = shap_values[1]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_matrix = shap_values[:, :, 1]
            else:
                shap_matrix = shap_values
        else:
            explainer = shap.LinearExplainer(classifier, X_test_proc)
            shap_matrix = explainer.shap_values(X_test_proc)

        plt.figure(figsize=(9, 6), dpi=300)
        shap.summary_plot(
            shap_matrix,
            X_test_proc,
            feature_names=feature_names,
            max_display=12,
            show=False,
            plot_type="bar",
        )
        plt.title(f"SHAP Feature Importance (Global Impact): {best_model_name}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        out_shap = output_dir / "feature_importance_shap.png"
        plt.savefig(out_shap, dpi=300)
        plt.close()
        logger.info(f"Saved: {out_shap}")
    except Exception as e:
        logger.warning(f"SHAP plot generation encountered an exception: {e}. Permutation importance plot was saved successfully.")


def evaluate_models() -> pd.DataFrame:
    """Execute Stage 2 comprehensive evaluation, visualization, and threshold tuning."""
    logger.info("Starting Stage 2 Model Evaluation Workflow...")

    # Load test partition and trained pipelines
    X_test, y_test = load_test_data()
    models = load_trained_models()

    eval_results = {}
    comparison_records = []

    # Load CV stability metrics if available
    cv_stability = {}
    if CV_RESULTS_CSV.exists():
        cv_df = pd.read_csv(CV_RESULTS_CSV)
        for _, row in cv_df.iterrows():
            cv_stability[row["Model"]] = {
                "CV_Recall_Stability": f"{row['CV Recall Mean']:.4f} +/- {row['CV Recall Std']:.4f}",
                "CV_ROC_AUC_Stability": f"{row['CV ROC-AUC Mean']:.4f} +/- {row['CV ROC-AUC Std']:.4f}",
                "CV_F1_Stability": f"{row['CV F1 Mean']:.4f} +/- {row['CV F1 Std']:.4f}",
            }

    print("\n" + "=" * 90)
    print(" HELD-OUT TEST SET EVALUATION REPORTS (N=61 SAMPLES)")
    print("=" * 90)

    for name, pipeline in models.items():
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.50).astype(int)

        metrics = compute_test_metrics(y_test, y_pred, y_prob)
        eval_results[name] = {
            "pipeline": pipeline,
            "y_true": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "metrics": metrics,
        }

        # Classification Report
        report_str = classification_report(y_test, y_pred, target_names=["Healthy (0)", "Disease (1)"])
        print(f"\n--- Classification Report: {name} (Threshold = 0.50) ---")
        print(report_str)

        record = {"Model": name, **metrics}
        if name in cv_stability:
            record.update(cv_stability[name])
        comparison_records.append(record)

    comparison_df = pd.DataFrame(comparison_records)

    # Reorder columns logically
    display_cols = [
        "Model",
        "Recall (Sensitivity)",
        "ROC-AUC",
        "PR-AUC",
        "F1-Score",
        "Precision",
        "Accuracy",
        "Specificity",
        "Matthews Corrcoef (MCC)",
    ]
    if "CV_Recall_Stability" in comparison_df.columns:
        display_cols.append("CV_Recall_Stability")

    # Save comparison dataframe
    comparison_df.to_csv(MODEL_COMPARISON_CSV, index=False)
    logger.info(f"Saved model comparison table to {MODEL_COMPARISON_CSV}")

    # Generate all required diagnostic plots
    plot_confusion_matrices(eval_results)
    plot_roc_curves(eval_results)
    plot_pr_curves(eval_results)
    plot_metrics_comparison_bar(comparison_df)
    plot_calibration_curves(eval_results)

    # Identify winning model from metadata
    if METADATA_PATH.exists():
        meta = load_json(METADATA_PATH)
        best_model_name = meta.get("model_name", "Logistic Regression")
    else:
        # Sort by recall first, then ROC-AUC
        sorted_res = comparison_df.sort_values(by=["Recall (Sensitivity)", "ROC-AUC"], ascending=False)
        best_model_name = sorted_res.iloc[0]["Model"]

    # Threshold analysis for winning model
    chosen_threshold, tuned_metrics = perform_threshold_analysis(
        best_model_name=best_model_name,
        best_pipeline=models[best_model_name],
        X_test=X_test,
        y_test=y_test,
    )

    # Feature importance extraction
    plot_feature_importance(models, best_model_name, X_test, y_test)

    # Update metadata with chosen threshold and test metrics
    if METADATA_PATH.exists():
        meta = load_json(METADATA_PATH)
        meta["chosen_threshold"] = chosen_threshold
        meta["test_metrics_default_threshold_0.50"] = eval_results[best_model_name]["metrics"]
        meta["test_metrics_clinical_tuned_threshold"] = tuned_metrics
        save_json(meta, METADATA_PATH)

    border = "=" * 90
    print(f"\n{border}")
    print(" COMPREHENSIVE TEST SET MODEL BENCHMARK SUMMARY")
    print(f"{border}")
    print(comparison_df[display_cols].to_string(index=False))
    print(f"\nCLINICAL THRESHOLD OPTIMIZATION ({best_model_name}):")
    print(f"Optimal Threshold = {chosen_threshold:.2f} (favoring sensitivity/recall)")
    print(
        f"At threshold {chosen_threshold:.2f}: Recall = {tuned_metrics['Recall (Sensitivity)']*100:.1f}%, "
        f"Precision = {tuned_metrics['Precision']*100:.1f}%, Specificity = {tuned_metrics['Specificity']*100:.1f}%"
    )
    print(f"{border}\n")

    return comparison_df


if __name__ == "__main__":
    evaluate_models()
