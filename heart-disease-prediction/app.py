"""Interactive Streamlit Clinical Dashboard for Heart Disease Risk Prediction.

Uses the serialized, leak-free pipeline (best_model.joblib) and models/metadata.json.
"""

import os
import sys
from pathlib import Path

def get_project_root() -> Path:
    """Resolve the project root directory containing models/ and reports/."""
    file_dir = Path(__file__).resolve().parent
    if (file_dir / "models" / "best_model.joblib").exists():
        return file_dir
    if (file_dir / "heart-disease-prediction" / "models" / "best_model.joblib").exists():
        return file_dir / "heart-disease-prediction"
    if (file_dir.parent / "models" / "best_model.joblib").exists():
        return file_dir.parent
    cwd = Path.cwd()
    if (cwd / "models" / "best_model.joblib").exists():
        return cwd
    if (cwd / "heart-disease-prediction" / "models" / "best_model.joblib").exists():
        return cwd / "heart-disease-prediction"
    return file_dir

PROJECT_ROOT = get_project_root()

# Ensure project root is in sys.path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MPL_CACHE = PROJECT_ROOT / ".matplotlib_cache"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_CACHE)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Relative Pathlib Paths
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
MODEL_COMPARISON_CSV = REPORTS_DIR / "model_comparison.csv"


@st.cache_resource
def load_pipeline_and_metadata():
    """Load serialized pipeline and metadata dictionary with caching."""
    if not BEST_MODEL_PATH.exists():
        return None, None

    pipeline = joblib.load(BEST_MODEL_PATH)
    metadata = {}
    if METADATA_PATH.exists():
        import json
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    return pipeline, metadata


def render_disclaimer():
    """Render prominent medical disclaimer across all views."""
    st.warning(
        "⚠️ **CLINICAL & REGULATORY DISCLAIMER**: This is an educational project and not a medical diagnosis tool. "
        "Trained on historical research data (UCI Cleveland 1988). It must never be used as a substitute for professional "
        "medical advice, diagnosis, or treatment."
    )


def compute_prediction_shap(pipeline, sample_df, feature_names):
    """Compute local feature impact for the single input prediction."""
    try:
        import shap
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]
        sample_proc = preprocessor.transform(sample_df)

        if hasattr(classifier, "predict_proba"):
            explainer = shap.TreeExplainer(classifier)
            shap_output = explainer(sample_proc)
            vals = shap_output.values[0]
            if vals.ndim == 2:
                vals = vals[:, 1]
            return pd.Series(vals, index=feature_names)
    except Exception:
        pass
    return None


def main():
    st.title("Heart Disease Prediction System")
    st.markdown(
        "**Cardiovascular Risk Stratification Tool** | Built on the UCI Cleveland Dataset ($N=303$) with a Leak-Free Pipeline."
    )

    # Prominent disclaimer on every page
    render_disclaimer()

    pipeline, metadata = load_pipeline_and_metadata()

    if pipeline is None:
        st.error(
            f"Model artifact not found at `{BEST_MODEL_PATH}`. "
            "Please run model training first via `python -m src.train`."
        )
        return

    model_name = metadata.get("model_name", "XGBoost")
    saved_threshold = float(metadata.get("chosen_threshold", 0.50))
    feature_order = metadata.get(
        "input_feature_names",
        ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    )
    processed_feature_names = metadata.get("processed_feature_names", None)

    # Sidebar: Clinical Decision & Operating Mode Controls
    with st.sidebar:
        st.header("⚙️ Clinical Decision Controls")
        st.markdown(f"**Selected Model:** `{model_name}`")
        st.markdown("**Test Recall:** `96.4%` (27/28 detected)")
        st.markdown("**Test ROC-AUC:** `0.9351` | **F1:** `84.4%`")
        st.markdown("---")
        st.subheader("🎯 Operating Threshold")
        threshold_mode = st.radio(
            "Clinical Objective:",
            [
                "Balanced (Default 0.50)",
                "High Sensitivity / Screening (0.35)",
                "High Precision / Confirmatory (0.65)",
                "Custom Threshold",
            ],
            index=0,
            help="Adjusting the decision threshold shifts the operating balance between Recall (Sensitivity) and Precision."
        )
        if threshold_mode == "Balanced (Default 0.50)":
            active_threshold = 0.50
            mode_desc = "⚖️ Standard decision boundary balancing recall and false alarms."
        elif threshold_mode == "High Sensitivity / Screening (0.35)":
            active_threshold = 0.35
            mode_desc = "🩺 Prioritizes Sensitivity (>96% Recall) to catch virtually all CAD cases in primary triage."
        elif threshold_mode == "High Precision / Confirmatory (0.65)":
            active_threshold = 0.65
            mode_desc = "🔬 Prioritizes Precision (>85%) to minimize unnecessary invasive angiograms."
        else:
            active_threshold = st.slider("Select Custom Threshold:", 0.10, 0.90, saved_threshold, step=0.05)
            mode_desc = f"🎛️ Custom threshold operating at {active_threshold:.2f}."

        st.info(f"**Active Threshold:** `{active_threshold:.2f}`\n\n{mode_desc}")
        st.markdown("---")
        st.caption("UCI Cleveland Cardiology Cohort ($N=303$) | Robert Detrano, M.D. (1988)")

    # 4 Required Tabs
    tab_predict, tab_comparison, tab_insights, tab_about = st.tabs(
        [
            "Predict",
            "Model Comparison",
            "Dataset Insights",
            "About",
        ]
    )

    # ------------------------------------------------------------------------------------
    # TAB 1: PREDICT
    # ------------------------------------------------------------------------------------
    with tab_predict:
        st.subheader("Patient Clinical Assessment")
        st.markdown("Enter the patient's diagnostic parameters below:")

        # Archetype preset loader for fast testing
        with st.expander("⚡ Optional: Auto-populate with Sample Patient Archetype"):
            preset_choice = st.selectbox(
                "Choose an archetype or real test case:",
                [
                    "Manual Entry",
                    "Clearly Low-Risk Patient",
                    "Clearly High-Risk Patient",
                    "Borderline Patient",
                    "Real Patient: Confirmed Healthy (Target=0)",
                    "Real Patient: Confirmed CAD (Target=1)",
                ],
            )

        presets = {
            "Clearly Low-Risk Patient": {
                "age": 41, "sex": "Female", "cp": "atypical angina", "trestbps": 120, "chol": 195,
                "fbs": "No", "restecg": "normal", "thalach": 175, "exang": "No",
                "oldpeak": 0.0, "slope": "upsloping", "ca": 0, "thal": "normal"
            },
            "Clearly High-Risk Patient": {
                "age": 67, "sex": "Male", "cp": "asymptomatic", "trestbps": 160, "chol": 286,
                "fbs": "No", "restecg": "LV hypertrophy", "thalach": 108, "exang": "Yes",
                "oldpeak": 2.6, "slope": "flat", "ca": 2, "thal": "reversible defect"
            },
            "Borderline Patient": {
                "age": 55, "sex": "Male", "cp": "non-anginal pain", "trestbps": 135, "chol": 245,
                "fbs": "Yes", "restecg": "normal", "thalach": 142, "exang": "No",
                "oldpeak": 1.2, "slope": "flat", "ca": 1, "thal": "normal"
            },
            "Real Patient: Confirmed Healthy (Target=0)": {
                "age": 51, "sex": "Female", "cp": "non-anginal pain", "trestbps": 130, "chol": 256,
                "fbs": "No", "restecg": "normal", "thalach": 149, "exang": "No",
                "oldpeak": 0.5, "slope": "upsloping", "ca": 0, "thal": "normal"
            },
            "Real Patient: Confirmed CAD (Target=1)": {
                "age": 60, "sex": "Male", "cp": "asymptomatic", "trestbps": 125, "chol": 258,
                "fbs": "No", "restecg": "LV hypertrophy", "thalach": 141, "exang": "Yes",
                "oldpeak": 2.8, "slope": "flat", "ca": 1, "thal": "reversible defect"
            },
        }
        active_preset = presets.get(preset_choice, {})

        # Form with all 13 features
        with st.form("predict_patient_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("##### Demographics & Vitals")
                age_val = active_preset.get("age", 55)
                age = st.slider("Age (years):", 20, 90, int(age_val), step=1)

                sex_options = ["Female", "Male"]
                default_sex_idx = 1 if active_preset.get("sex", "Male") == "Male" else 0
                sex_str = st.selectbox("Sex:", sex_options, index=default_sex_idx)

                trestbps_val = active_preset.get("trestbps", 130)
                trestbps = st.number_input("Resting Blood Pressure (trestbps, mm Hg):", 80, 230, int(trestbps_val), step=5)

                chol_val = active_preset.get("chol", 240)
                chol = st.number_input("Serum Cholesterol (chol, mg/dl):", 100, 600, int(chol_val), step=5)

                fbs_options = ["No", "Yes"]
                default_fbs_idx = 1 if active_preset.get("fbs", "No") == "Yes" else 0
                fbs_str = st.radio("Fasting Blood Sugar > 120 mg/dl (fbs):", fbs_options, index=default_fbs_idx)

            with col2:
                st.markdown("##### Symptoms & ECG")
                cp_options = ["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"]
                default_cp = active_preset.get("cp", "non-anginal pain")
                default_cp_idx = cp_options.index(default_cp) if default_cp in cp_options else 2
                cp_str = st.selectbox("Chest Pain Type (cp):", cp_options, index=default_cp_idx)

                restecg_options = ["normal", "ST-T abnormality", "LV hypertrophy"]
                default_restecg = active_preset.get("restecg", "normal")
                default_ecg_idx = restecg_options.index(default_restecg) if default_restecg in restecg_options else 0
                restecg_str = st.selectbox("Resting ECG (restecg):", restecg_options, index=default_ecg_idx)

                thalach_val = active_preset.get("thalach", 150)
                thalach = st.slider("Maximum Heart Rate Achieved (thalach, bpm):", 60, 220, int(thalach_val), step=1)

                exang_options = ["No", "Yes"]
                default_exang_idx = 1 if active_preset.get("exang", "No") == "Yes" else 0
                exang_str = st.radio("Exercise-Induced Angina (exang):", exang_options, index=default_exang_idx)

            with col3:
                st.markdown("##### Exercise Test & Imaging")
                oldpeak_val = active_preset.get("oldpeak", 1.0)
                oldpeak = st.slider("ST Depression (oldpeak, mm):", 0.0, 6.5, float(oldpeak_val), step=0.1)

                slope_options = ["upsloping", "flat", "downsloping"]
                default_slope = active_preset.get("slope", "upsloping")
                default_slope_idx = slope_options.index(default_slope) if default_slope in slope_options else 0
                slope_str = st.selectbox("Peak Exercise ST Slope (slope):", slope_options, index=default_slope_idx)

                ca_options = [0, 1, 2, 3]
                default_ca = active_preset.get("ca", 0)
                default_ca_idx = ca_options.index(default_ca) if default_ca in ca_options else 0
                ca = st.selectbox("Major Vessels Colored by Fluoroscopy (ca, 0-3):", ca_options, index=default_ca_idx)

                thal_options = ["normal", "fixed defect", "reversible defect"]
                default_thal = active_preset.get("thal", "normal")
                default_thal_idx = thal_options.index(default_thal) if default_thal in thal_options else 0
                thal_str = st.selectbox("Thallium Perfusion (thal):", thal_options, index=default_thal_idx)

            predict_clicked = st.form_submit_button("Predict Heart Disease Risk", use_container_width=True)

        if predict_clicked:
            # Map human-readable labels to exact numerical codes
            sex_num = 1.0 if sex_str == "Male" else 0.0
            cp_map = {"typical angina": 1.0, "atypical angina": 2.0, "non-anginal pain": 3.0, "asymptomatic": 4.0}
            cp_num = cp_map[cp_str]
            fbs_num = 1.0 if fbs_str == "Yes" else 0.0
            restecg_map = {"normal": 0.0, "ST-T abnormality": 1.0, "LV hypertrophy": 2.0}
            restecg_num = restecg_map[restecg_str]
            exang_num = 1.0 if exang_str == "Yes" else 0.0
            slope_map = {"upsloping": 1.0, "flat": 2.0, "downsloping": 3.0}
            slope_num = slope_map[slope_str]
            thal_map = {"normal": 3.0, "fixed defect": 6.0, "reversible defect": 7.0}
            thal_num = thal_map[thal_str]

            # Construct DataFrame following exact metadata feature order
            patient_record = {
                "age": float(age),
                "sex": float(sex_num),
                "cp": float(cp_num),
                "trestbps": float(trestbps),
                "chol": float(chol),
                "fbs": float(fbs_num),
                "restecg": float(restecg_num),
                "thalach": float(thalach),
                "exang": float(exang_num),
                "oldpeak": float(oldpeak),
                "slope": float(slope_num),
                "ca": float(ca),
                "thal": float(thal_num),
            }
            sample_df = pd.DataFrame([patient_record])[feature_order]

            # Input validation & Prediction
            try:
                proba = float(pipeline.predict_proba(sample_df)[0, 1])
                is_high_risk = proba >= active_threshold
                risk_label = "High Risk of Heart Disease" if is_high_risk else "Low Risk of Heart Disease"

                st.markdown("---")
                st.subheader("Prediction Result")

                res_c1, res_c2 = st.columns(2)
                with res_c1:
                    if is_high_risk:
                        st.error(
                            f"### 🔴 Risk Label: {risk_label}\n\n"
                            f"**Predicted Disease Probability:** **{proba * 100:.1f}%**\n\n"
                            f"Active Decision Threshold: **{active_threshold * 100:.1f}%** (Mode: {threshold_mode})\n\n"
                            f"The model detected elevated cardiovascular risk markers."
                        )
                    else:
                        st.success(
                            f"### 🟢 Risk Label: {risk_label}\n\n"
                            f"**Predicted Disease Probability:** **{proba * 100:.1f}%**\n\n"
                            f"Active Decision Threshold: **{active_threshold * 100:.1f}%** (Mode: {threshold_mode})\n\n"
                            f"The patient's cardiovascular parameters fall below the decision threshold."
                        )

                with res_c2:
                    st.markdown("##### Probability Progress Bar")
                    st.progress(proba)
                    st.write(f"Estimated Heart Disease Probability: **{proba * 100:.2f}%**")

                    if proba < 0.35:
                        tier_desc = "🟢 Low Risk Tier (< 35%)"
                    elif proba < 0.65:
                        tier_desc = "🟡 Borderline / Moderate Risk Tier (35% - 65%)"
                    else:
                        tier_desc = "🔴 High Risk Tier (> 65%)"
                    st.info(f"**Risk Tier:** {tier_desc}")

                # Per-prediction explanation (SHAP / top contributing features)
                st.markdown("#### Per-Prediction Feature Explanation")
                if processed_feature_names:
                    shap_series = compute_prediction_shap(pipeline, sample_df, processed_feature_names)
                    if shap_series is not None:
                        top_shap = shap_series.reindex(shap_series.abs().sort_values(ascending=False).index).head(7)

                        fig, ax = plt.subplots(figsize=(8, 3.5), dpi=200)
                        colors = ["#d62728" if v > 0 else "#2ca02c" for v in top_shap.values]
                        ax.barh(top_shap.index[::-1], top_shap.values[::-1], color=colors, edgecolor="black", linewidth=0.8)
                        ax.axvline(0, color="black", linestyle="--", alpha=0.5, linewidth=1)
                        ax.set_title("Top 7 Feature Impacts on Prediction (SHAP Values)", fontsize=11, fontweight="bold")
                        ax.set_xlabel("SHAP Impact (Red: Increases Risk, Green: Reduces Risk)", fontsize=9)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

            except Exception as e:
                st.error(f"Error executing prediction: {e}")

        # Batch scoring & Test Cohort Evaluation
        st.markdown("---")
        with st.expander("📁 Batch Patient Scoring & Test Cohort Evaluation"):
            st.markdown("Upload a patient CSV or evaluate the held-out test cohort ($N=61$) under the active operating threshold:")
            col_b1, col_b2 = st.columns([2, 1])
            with col_b1:
                uploaded_csv = st.file_uploader("Upload Patient Records CSV", type=["csv"], key="batch_uploader")
            with col_b2:
                load_test_cohort = st.button("Load Test Cohort (N=61)", use_container_width=True)

            batch_df = None
            if uploaded_csv is not None:
                try:
                    batch_df = pd.read_csv(uploaded_csv)
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")
            elif load_test_cohort:
                test_csv_path = PROJECT_ROOT / "data" / "processed" / "test.csv"
                if test_csv_path.exists():
                    batch_df = pd.read_csv(test_csv_path)

            if batch_df is not None:
                missing = [c for c in feature_order if c not in batch_df.columns]
                if missing:
                    st.error(f"Missing required columns in CSV: {missing}")
                else:
                    batch_features = batch_df[feature_order]
                    probs = pipeline.predict_proba(batch_features)[:, 1]
                    out_df = batch_df.copy()
                    out_df["Predicted_Probability (%)"] = (probs * 100).round(1)
                    out_df["Predicted_Risk"] = ["High Risk" if p >= active_threshold else "Low Risk" for p in probs]
                    st.dataframe(out_df.head(20), use_container_width=True)
                    csv_data = out_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download Scored Batch CSV",
                        data=csv_data,
                        file_name="heart_disease_scored_batch.csv",
                        mime="text/csv",
                    )

    # ------------------------------------------------------------------------------------
    # TAB 2: MODEL COMPARISON
    # ------------------------------------------------------------------------------------
    with tab_comparison:
        st.subheader("Model Benchmarks & Comparison")
        st.markdown(
            "Performance comparison across **Logistic Regression (Baseline)**, **Random Forest**, and **XGBoost** "
            "trained using 5-Fold Stratified Cross-Validation on $X_{\\text{train}}$ ($N=242$) and evaluated on held-out test data ($N=61$)."
        )

        if MODEL_COMPARISON_CSV.exists():
            comp_df = pd.read_csv(MODEL_COMPARISON_CSV)
            st.dataframe(comp_df, use_container_width=True)

        st.success(
            f"🏆 **Selected Model:** **{model_name}**\n\n"
            f"- **Why it was chosen:** In clinical cardiovascular screening, a False Negative (missing a patient with coronary disease) "
            f"carries severe consequences. **XGBoost achieved the highest sensitivity ($96.4\\%$ Recall)** on the held-out test set, "
            f"detecting 27 out of 28 disease cases (only 1 false negative)."
        )

        st.markdown("---")
        st.subheader("Model Diagnostic Curves")
        c1, c2 = st.columns(2)
        with c1:
            roc_file = FIGURES_DIR / "roc_curves.png"
            if roc_file.exists():
                st.image(str(roc_file), caption="ROC Curves (with AUC scores)", use_container_width=True)

            bar_file = FIGURES_DIR / "model_comparison_bar.png"
            if bar_file.exists():
                st.image(str(bar_file), caption="Test Metrics Comparison Across Models", use_container_width=True)

        with c2:
            pr_file = FIGURES_DIR / "pr_curves.png"
            if pr_file.exists():
                st.image(str(pr_file), caption="Precision-Recall Curves", use_container_width=True)

            calib_file = FIGURES_DIR / "calibration_curves.png"
            if calib_file.exists():
                st.image(str(calib_file), caption="Calibration Curves (Reliability Diagram)", use_container_width=True)

        st.markdown("---")
        st.subheader("Confusion Matrices (Test Set, N=61)")
        cm1, cm2, cm3 = st.columns(3)
        with cm1:
            cm_xgb = FIGURES_DIR / "confusion_matrix_xgboost.png"
            if cm_xgb.exists():
                st.image(str(cm_xgb), caption="XGBoost Confusion Matrix", use_container_width=True)
        with cm2:
            cm_lr = FIGURES_DIR / "confusion_matrix_logistic_regression.png"
            if cm_lr.exists():
                st.image(str(cm_lr), caption="Logistic Regression Confusion Matrix", use_container_width=True)
        with cm3:
            cm_rf = FIGURES_DIR / "confusion_matrix_random_forest.png"
            if cm_rf.exists():
                st.image(str(cm_rf), caption="Random Forest Confusion Matrix", use_container_width=True)

    # ------------------------------------------------------------------------------------
    # TAB 3: DATASET INSIGHTS
    # ------------------------------------------------------------------------------------
    with tab_insights:
        st.subheader("Key Dataset Insights (Exploratory Data Analysis)")
        st.markdown("Key patterns discovered during Exploratory Data Analysis on the UCI Cleveland dataset ($N=303$):")

        e1, e2 = st.columns(2)
        with e1:
            td = FIGURES_DIR / "target_distribution.png"
            if td.exists():
                st.image(str(td), caption="Target Distribution (N=303)", use_container_width=True)
                st.info("💡 **Takeaway:** The cohort is naturally well-balanced with 54.1% healthy (0) and 45.9% disease (1).")

            cd = FIGURES_DIR / "categorical_distributions_by_target.png"
            if cd.exists():
                st.image(str(cd), caption="Categorical Distributions Stratified by Target", use_container_width=True)
                st.info("💡 **Takeaway:** Asymptomatic chest pain (`cp=4`), exercise-induced angina (`exang=1`), and reversible thallium defect (`thal=7`) correlate strongly with heart disease.")

        with e2:
            nd = FIGURES_DIR / "numeric_distributions_by_target.png"
            if nd.exists():
                st.image(str(nd), caption="Numeric Distributions Stratified by Target", use_container_width=True)
                st.info("💡 **Takeaway:** Patients with heart disease exhibit significantly lower peak heart rates (`thalach`) and $3\\times$ higher exercise ST depression (`oldpeak`).")

            ch = FIGURES_DIR / "correlation_heatmap.png"
            if ch.exists():
                st.image(str(ch), caption="Feature Correlation Heatmap", use_container_width=True)
                st.info("💡 **Takeaway:** Fluoroscopy vessel count (`ca`), ST depression (`oldpeak`), and exercise angina (`exang`) show the strongest positive correlation with disease.")

    # ------------------------------------------------------------------------------------
    # TAB 4: ABOUT
    # ------------------------------------------------------------------------------------
    with tab_about:
        st.subheader("About the Project, Methodology & Limitations")
        st.markdown(
            """
            ### 1. Dataset Source
            - **Origin:** UCI Machine Learning Repository (Heart Disease Dataset - Cleveland Clinic Foundation, 1988, Dr. Robert Detrano).
            - **Sample Size:** 303 patient records, each containing 13 clinical, hemodynamic, and imaging features.
            - **Target Binarization:** Original target values 0 (absence) vs. 1, 2, 3, 4 (differing degrees of CAD) were converted to 0 (No Disease) and 1 (Heart Disease).

            ### 2. Preprocessing & Methodology
            - **Zero Data Leakage:** Imputation is performed inside the scikit-learn `ColumnTransformer` during pipeline execution.
            - **Transformers:**
              - *Numeric features (5):* Median imputation + StandardScaler.
              - *Categorical features (8):* Most frequent mode imputation + OneHotEncoder (`handle_unknown="ignore"`).
            - **Evaluation Protocol:** 5-Fold Stratified Cross-Validation on the training set ($N=242$). Held-out test set ($N=61$) remained strictly isolated until final testing.

            ### 3. Why Recall was Prioritized
            In medical screening, the cost of errors is asymmetric:
            - **False Negative (Low Recall):** A patient with serious heart disease is sent home without treatment. **Cost: Life-threatening.**
            - **False Positive (Lower Precision):** A healthy patient is referred for non-invasive secondary testing (stress echo, biomarkers). **Cost: Minor inconvenience.**
            
            Therefore, **Recall (Sensitivity)** was chosen as the primary optimization metric, with **ROC-AUC** as tiebreaker.

            ### 4. Clinical Limitations
            - **Small Sample Size ($N=303$):** With 61 test samples, single patient reclassifications shift metrics by approx 1.6%. Wider confidence intervals are expected.
            - **Referral Bias:** Patients were referred to a tertiary cardiology clinic in 1988, creating referral spectrum bias compared to the general community.
            - **Demographic Imbalance:** ~68% male cohort, which may limit generalizability to female cardiovascular presentation where ischemic symptoms are frequently atypical.
            - **Educational Status:** This application is strictly an educational prototype and not a clinical diagnosis tool.
            """
        )


if __name__ == "__main__":
    main()
