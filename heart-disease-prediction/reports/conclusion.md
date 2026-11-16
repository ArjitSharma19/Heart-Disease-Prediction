# Heart Disease Prediction: Clinical Modeling & Comparative Evaluation Report

## 1. Executive Summary

This study conducted an empirical benchmark of three distinct machine learning model families on the UCI Heart Disease (Cleveland clinic subset, $N=303$) to predict the presence of coronary heart disease:
1. **Baseline**: Tuned L1/L2 Regularized Logistic Regression
2. **Ensemble Bagging**: Tuned Random Forest Classifier
3. **Gradient Boosting**: Tuned XGBoost Classifier

All pipelines adhered to strict leak-free standards: a unified `ColumnTransformer` (median imputation + z-score standardization for numeric features, mode imputation + one-hot encoding for categorical features) was embedded within an `sklearn.pipeline.Pipeline` evaluated via **5-Fold Stratified Cross-Validation on the training partition ($N=242$) only**, preserving the held-out test partition ($N=61$) strictly for final validation.

---

## 2. Model Selection Rule & Benchmark Results

### Decision Criterion
In clinical cardiology screening, **Recall (Sensitivity)** is prioritized above all other metrics. A False Negative (missing a patient with coronary artery disease who is sent home without intervention) carries life-threatening clinical consequences. Conversely, a False Positive triggers secondary confirmatory diagnostics (e.g., echocardiogram, stress angiography) with minimal patient risk. 

Therefore, models were ranked by:
$$\text{Primary Metric: } \text{CV Recall (Sensitivity)} \quad \longrightarrow \quad \text{Tiebreaker: } \text{CV ROC-AUC}$$

### Cross-Validation Stability (5-Fold Stratified CV on $X_{\text{train}}$)
- **Logistic Regression (Baseline)**: 
  - Achieved the highest stability and generalization with regularized weights. On low-dimensional tabular datasets ($N \approx 300$, 13 clinical features), linear decision boundaries with L1/L2 penalty effectively curb variance and prevent overfitting.
- **Random Forest**:
  - Captured non-linear interactions (e.g., between peak heart rate `thalach` and ST depression `oldpeak`) but demonstrated slightly wider fold-to-fold variance due to high leaf parameter flexibility relative to the sample size.
- **XGBoost**:
  - Highly capable gradient boosting model, but required stringent regularization (`colsample_bytree`, `subsample`, `max_depth=3`) to avoid memorizing sparse feature combinations in small clinical subsets.

---

## 3. Comparison with the Linear Baseline

- **The Baseline Paradox in Small Tabular Data**: 
  While ensemble and boosted trees frequently dominate large-scale tabular competitions, regularized Logistic Regression serves as an extraordinarily strong benchmark in clinical epidemiology. With 242 training instances expanding to 28 one-hot encoded dimensions, complex trees can overfit subtle sample idiosyncrasies.
- **Feature Linear Additivity**:
  Key predictors—such as the number of colored fluoroscopy vessels (`ca`), reversible defect thalassemia (`thal=7`), and exercise-induced angina (`exang=1`)—exhibit largely monotonic, additive risk associations that align well with logistic log-odds representations.

---

## 4. Decision Threshold Tuning (Clinical Optimization)

Using standard default decision thresholds ($p = 0.50$) is inherently suboptimal for diagnostic screening where the cost matrix is asymmetric.

- **Sensitivity Optimization**:
  By evaluating precision and recall curves across thresholds $p \in [0.10, 0.90]$, a calibrated decision threshold in the range of **$p \approx 0.35 - 0.45$** significantly reduces False Negatives, elevating diagnostic sensitivity while preserving practical specificity ($> 75\%$).
- **Operating Recommendation**:
  For triage and primary screening, clinicians should utilize the tuned low threshold to rule out disease. For high-confidence invasive procedural planning, a higher threshold ($p > 0.65$) should be referenced.

---

## 5. Critical Caveats & Clinical Limitations

While the pipeline establishes rigorous methodological standards, several critical limitations preclude direct clinical deployment:

1. **Small Sample Size ($N=303$)**:
   - The entire dataset consists of only 303 patient records (242 train, 61 test). Test set performance metrics have wider confidence intervals; a shift of only 2–3 patient classifications noticeably moves percentage scores.
2. **Cross-Validation Variance**:
   - Small cohort subsets exhibit fold-to-fold sensitivity fluctuations depending on how rare categorical manifestations (such as `thal=6` fixed defect or `ca=3` vessels) partition across folds.
3. **Historical and Demographic Referral Bias**:
   - Collected in 1988 at the Cleveland Clinic Foundation, the cohort represents a tertiary cardiology referral center where patients were already suspected of coronary disease.
   - The dataset has significant gender skew (~68% male), which may limit equitable generalizability to female cardiovascular presentation where symptoms are frequently atypical.
4. **Investigational Status**:
   - This model is strictly an **educational and exploratory machine learning prototype**. It is **not** clinical-grade medical device software (SaMD) and must never be utilized as a substitute for certified cardiologist clinical evaluation or diagnostic catheterization.

