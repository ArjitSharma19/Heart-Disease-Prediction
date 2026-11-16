# Heart Disease Risk Prediction (UCI Cleveland Cohort)

An end-to-end Machine Learning and clinical decision-support project predicting coronary heart disease using the UCI Heart Disease dataset (Cleveland Clinic subset, $N=303$).

---

## 1. Project Overview

Coronary artery disease (CAD) is the leading cause of mortality worldwide. Identifying individuals at elevated risk using non-invasive hemodynamic and diagnostic features allows for early lifestyle and clinical interventions.

This project delivers a **leak-free, production-ready machine learning solution**:
- **Data Ingestion & Quality Audit**: Ingests raw data from UCI, replaces missing values (`?`) with `NaN`, binarizes the target ($0$ = Healthy, $1$ = Disease), and validates physiological ranges.
- **Exploratory Data Analysis (EDA)**: Produces diagnostic plots saved to `reports/figures/` analyzing correlations, distributions, and missing patterns.
- **Leak-Free Preprocessing**: Employs scikit-learn's `ColumnTransformer` with median imputation and standardization for numeric features, and mode imputation with one-hot encoding for categorical features. All transformations are fitted strictly on the training partition ($N=242$).
- **Stratified 5-Fold Cross-Validation**: Hyperparameter tuning for Logistic Regression, Random Forest, and XGBoost on training data only.
- **Recall-First Model Optimization**: Prioritizes **Recall (Sensitivity)** because in clinical triage, a False Negative (missing a patient with coronary disease) is far more dangerous than a False Positive.
- **Interactive Streamlit Web Dashboard**: Provides real-time risk scoring, probability gauge, archetype presets, SHAP explainability, benchmark comparisons, and EDA visualizations.
- **REST API (FastAPI)**: Lightweight prediction endpoint with Swagger UI docs at `/docs`.

---

## 2. Dataset Description

- **Source**: UCI Machine Learning Repository (Cleveland Clinic Foundation, 1988, Dr. Robert Detrano).
- **Cohort Size**: 303 patient records (242 Train, 61 Test - 80/20 Stratified Split).
- **Target Distribution**: Naturally balanced cohort:
  - Class 0 (No Disease): 164 patients (54.13%)
  - Class 1 (Disease Present): 139 patients (45.87%)
- **Feature Schema (13 Attributes)**:
  - Continuous Numeric (5): `age`, `trestbps` (resting BP), `chol` (cholesterol), `thalach` (max heart rate), `oldpeak` (ST depression).
  - Categorical & Binary (8): `sex` (0=F, 1=M), `cp` (chest pain type 1-4), `fbs` (>120 mg/dl: 0/1), `restecg` (ECG 0-2), `exang` (angina 0/1), `slope` (ST slope 1-3), `ca` (major vessels 0-3), `thal` (perfusion defect 3, 6, 7).

---

## 3. Directory Structure

```text
heart-disease-prediction/
├── data/
│   ├── raw/
│   │   └── processed.cleveland.data       # Raw untouched UCI Cleveland dataset (303 rows)
│   └── processed/
│       ├── heart_clean.csv                # Cleaned, binarized, un-imputed dataset (303 rows)
│       ├── train.csv                      # Stratified training split (242 samples)
│       ├── test.csv                       # Held-out stratified test split (61 samples)
│       └── split_indices.json             # Exact row indices and stratification metadata
├── models/
│   ├── logreg.joblib                      # Fitted Baseline Logistic Regression pipeline
│   ├── rf.joblib                          # Fitted Random Forest pipeline
│   ├── xgb.joblib                         # Fitted XGBoost pipeline
│   ├── best_model.joblib                  # Winning pipeline (XGBoost, Recall=96.43%)
│   └── metadata.json                      # Operating threshold (0.50), feature order, and metrics
├── notebooks/
│   └── heart_disease_end_to_end.ipynb     # Walkthrough exploration notebook
├── reports/
│   ├── conclusion.md                      # Clinical findings, model comparison & caveats
│   ├── cv_benchmarks.csv                  # 5-fold CV stability across candidate models
│   ├── model_comparison.csv               # Held-out test set evaluation metrics
│   └── figures/
│       ├── target_distribution.png
│       ├── numeric_distributions_by_target.png
│       ├── categorical_distributions_by_target.png
│       ├── correlation_heatmap.png
│       ├── missing_values_summary.png
│       ├── confusion_matrix_logistic_regression.png
│       ├── confusion_matrix_random_forest.png
│       ├── confusion_matrix_xgboost.png
│       ├── roc_curves.png
│       ├── pr_curves.png
│       ├── model_comparison_bar.png
│       ├── calibration_curves.png
│       ├── threshold_analysis.png
│       ├── feature_importance_rf.png
│       ├── feature_importance_xgb.png
│       └── feature_importance_permutation.png
├── src/
│   ├── __init__.py
│   ├── config.py                          # Paths, random state (42), feature groups, ranges
│   ├── data_cleaning.py                   # Ingestion, missing check, outlier audit, range validation
│   ├── eda.py                             # Exploratory Data Analysis figure generator
│   ├── preprocessing.py                   # Leak-free ColumnTransformer & stratified splitter
│   ├── train.py                           # 5-fold CV tuning, model selection, serialization
│   ├── evaluate.py                        # Held-out test evaluation, threshold tuning, SHAP
│   ├── predict.py                         # Single & batch CLI inference utility
│   └── utils.py                           # Logging, metrics, JSON/CSV helpers
├── tests/
│   ├── test_pipeline.py                   # Unit & integration tests for Stage 1
│   └── test_model_inference.py            # Artifact persistence & inference tests
├── .streamlit/
│   └── config.toml                        # Streamlit UI settings (telemetry disabled)
├── api.py                                 # FastAPI REST server
├── app.py                                 # Interactive Streamlit Web Application
├── run_app.py                             # Cross-version Streamlit launcher
├── run_pipeline.py                        # Automated end-to-end execution script
├── requirements.txt                       # Pinned dependencies
└── README.md
```

---

## 4. Setup Instructions

### 4.1. Prerequisites
- Python 3.10 to 3.13
- Git

### 4.2. Virtual Environment Setup
```bash
# Clone repository and enter project root
cd heart-disease-prediction

# Create virtual environment
python -m venv .venv

# Activate environment:
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

---

## 5. Execution Workflow

### 5.1. Option A: Run the Entire Pipeline Automatically
Executes data ingestion, cleaning, EDA, preprocessing, model training, and evaluation in sequence:
```bash
python run_pipeline.py
```

### 5.2. Option B: Run Modular Stages Individually
```bash
# Stage 1: Data Ingestion & Cleaning (No leakage, retains clinical outliers)
python -m src.data_cleaning

# Stage 2: Exploratory Data Analysis (Generates reports/figures/*.png)
python -m src.eda

# Stage 3: Leak-Free Preprocessing & Stratified Splitting
python -m src.preprocessing

# Stage 4: 5-Fold Stratified CV Training & Hyperparameter Tuning
python -m src.train

# Stage 5: Evaluation on Held-Out Test Set, Threshold Tuning & SHAP
python -m src.evaluate
```

### 5.3. Run Unit & Regression Tests
```bash
python -m unittest discover tests -v
```

---

## 6. Launching the Applications

### 6.1. Streamlit Clinical Dashboard
```bash
# Direct Streamlit command
streamlit run app.py

# Or via the robust Python launcher
python run_app.py
```
Open **`http://localhost:8501`** in your browser.

#### Features of the Dashboard:
1. **Predict Tab**:
   - 13 clinical inputs with human-readable selectors and clinical defaults.
   - Quick presets: "Clearly Low-Risk", "Clearly High-Risk", and "Borderline".
   - Outputs: Binary Risk Label (🔴 High / 🟢 Low), Calibrated Probability Gauge, and Local SHAP Feature Explanations.
2. **Model Comparison Tab**:
   - Side-by-side benchmark table (`reports/model_comparison.csv`).
   - Diagnostic visualizers: ROC Curves, PR Curves, Calibration Curves, and Confusion Matrices.
3. **Dataset Insights Tab**:
   - High-resolution EDA figures with actionable one-line clinical takeaways.
4. **About Tab**:
   - Methodology breakdown, rationale for Recall prioritization, and clinical limitations.

### 6.2. FastAPI Service (Optional REST API)
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger documentation available at **`http://localhost:8000/docs`**.

---

## 7. Model Benchmarks & Final Results

Evaluated on the strictly held-out test partition ($N=61$, $28$ disease cases, $33$ healthy controls):

| Model | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | ROC-AUC | PR-AUC | MCC | False Negatives (FN) | False Positives (FP) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Winner)** | **83.61%** | **75.00%** | **96.43%** | **72.73%** | **84.38%** | **0.9351** | **0.9191** | **0.7007** | **1** | **9** |
| **Logistic Regression** | 85.25% | 80.65% | 89.29% | 81.82% | 84.75% | 0.9589 | 0.9546 | 0.7087 | 3 | 6 |
| **Random Forest** | 85.25% | 80.65% | 89.29% | 81.82% | 84.75% | 0.9545 | 0.9491 | 0.7087 | 3 | 6 |

### 5-Fold Cross-Validation Stability on $X_{\text{train}}$:
- **XGBoost**: CV Recall = $0.8644 \pm 0.0416$ | CV ROC-AUC = $0.8776 \pm 0.0284$
- **Logistic Regression**: CV Recall = $0.8372 \pm 0.0556$ | CV ROC-AUC = $0.9067 \pm 0.0110$
- **Random Forest**: CV Recall = $0.8008 \pm 0.0853$ | CV ROC-AUC = $0.9004 \pm 0.0258$

### Winning Model Decision:
**XGBoost** was selected as the operational pipeline because it maximized **Recall (96.43%)**, detecting **27 of 28 disease patients** with only a single missed case ($FN=1$). In contrast, Logistic Regression and Random Forest missed 3 disease patients each ($FN=3$).

---

## 8. Key Clinical Insights

1. **Thallium Scintigraphy Defect (`thal`)**: Reversible thallium perfusion defects represent the single most powerful predictor of obstructive coronary disease.
2. **Fluoroscopy Vessel Calcification (`ca`)**: The number of major vessels colored by fluoroscopy ($0-3$) shows strong positive correlation with disease severity ($r = +0.46$).
3. **ST Depression & Peak Exercise (`oldpeak` & `thalach`)**: Heart disease patients exhibit $3\times$ higher ST depression during exercise and significantly blunted chronotropic response (lower peak exercise heart rate).
4. **Asymptomatic Presentation (`cp=4`)**: Counter-intuitively, asymptomatic chest pain presentations in this high-risk tertiary cohort had the highest incidence of multi-vessel disease.

---

## 9. Project Limitations

1. **Small Sample Size ($N=303$)**: In a test set of 61 patients, a single case reclassification shifts metrics by $\approx 1.64\%$. Wider confidence intervals should be factored in when interpreting metrics.
2. **Referral Spectrum Bias**: The dataset comprises patients referred for coronary angiography at Cleveland Clinic in 1988, presenting a higher pre-test probability than unselected primary care populations.
3. **Gender Imbalance**: The sample is $68\%$ male, which may cause under-representation of ischemic patterns more typical in women.
4. **Regulatory Notice**: This system is strictly an academic demonstration and must **never** be used for clinical diagnostic or therapeutic purposes.
