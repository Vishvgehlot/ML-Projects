# Predicting Online Purchase Intent  
## A KNN-Based Approach to Customer Behavior Modeling in E-Commerce

---

## Project Overview

This project aims to predict whether a user will complete a purchase during an online shopping session based on behavioral and session-level features.

The dataset contains real-world e-commerce session data, including page activity, traffic sources, visitor type, and session duration metrics.

The objectives of this project are to:

- Identify high-purchase-intent sessions  
- Handle class imbalance effectively  
- Compare parametric, distance-based, and boosting models  
- Optimize for Precision-Recall AUC (PR-AUC) due to class imbalance  

---

## Dataset

- Source: UCI Machine Learning Repository  
- Rows: 12,330 sessions  
- Features: 18  
- Target Variable: `Revenue` (Purchase = 1, No Purchase = 0)

### Class Distribution

- No Purchase: ~85%
- Purchase: ~15%

Due to this imbalance, the project applies:

- SMOTE (for KNN experiments)
- Cost-sensitive learning (`scale_pos_weight`) for XGBoost
- PR-AUC as the primary evaluation metric

---

## Preprocessing Pipeline

1. One-Hot Encoding of categorical variables:
   - Month
   - VisitorType
   - Weekend

2. Feature Scaling:
   - StandardScaler (required for KNN)

3. Train-Test Split:
   - Stratified sampling to preserve class proportions

4. Class Imbalance Handling:
   - SMOTE applied to training data for KNN
   - Class weighting for XGBoost

---

## Models Implemented

### 1. Logistic Regression (Baseline)

- Hyperparameter tuning via GridSearchCV
- Optimized using F1-score
- Served as a strong linear benchmark

Approximate PR-AUC: ~0.40

---

### 2. K-Nearest Neighbors (Primary Focus)

Initial tuning using F1-score selected k = 1, which led to severe overfitting due to synthetic neighbors created by SMOTE.

Although cross-validation F1 was high, real test-set performance was significantly lower.

Re-tuning using PR-AUC (average_precision) produced a more stable model.

Best configuration:

- n_neighbors = 9  
- p = 1 (Manhattan distance)  
- weights = "distance"  

Test Performance:

- PR-AUC ≈ 0.52  
- ROC-AUC ≈ 0.80  
- Recall (Purchase) ≈ 0.43  

Key insight:
Distance-based models are highly sensitive to synthetic oversampling and require careful evaluation using imbalanced test data.

---

### 3. XGBoost (Final Model)

To better capture non-linear feature interactions and handle imbalance:

- Used `scale_pos_weight` instead of SMOTE
- Set `eval_metric = "aucpr"`
- Applied moderate regularization via subsampling

Final Performance:

PR-AUC ≈ 0.76

This significantly outperformed Logistic Regression, KNN, and Random Forest.

---

## Evaluation Metrics

Primary metric: Precision-Recall AUC (PR-AUC)

Accuracy was not used as the main metric because it is inflated by the dominant negative class.

ROC-AUC was reported but not optimized directly, as it can appear strong even when minority recall is weak.

PR-AUC better reflects real business value by focusing on identifying true purchase sessions.

---

## Final Model Comparison

| Model               | PR-AUC | Notes                              |
|---------------------|--------|-------------------------------------|
| Logistic Regression | ~0.40  | Strong linear baseline              |
| KNN (PR-optimized)  | ~0.52  | Sensitive to SMOTE                  |
| Random Forest       | ~0.60  | Improved non-linear modeling        |
| XGBoost             | ~0.76  | Best overall performance            |

---

## Key Learnings

1. Cross-validation on SMOTE-balanced data can inflate performance metrics.
2. Optimizing for F1 in imbalanced problems can be misleading.
3. PR-AUC is more appropriate than accuracy for minority-class detection.
4. Distance-based models struggle with synthetic oversampling.
5. Boosting models handle non-linearity and imbalance more effectively.
6. Model performance plateaued around PR-AUC ≈ 0.76, suggesting inherent noise in user behavior data.

---

## Business Implications

With PR-AUC ≈ 0.76, the model can:

- Identify high-intent sessions for retargeting
- Trigger real-time promotional strategies
- Improve conversion optimization
- Reduce marketing spend on low-intent traffic

---

## Future Improvements

- Probability calibration
- Advanced feature engineering
- LightGBM comparison
- Threshold optimization for business-specific recall targets
- Temporal session modeling

---

## Tech Stack

- Python  
- Scikit-learn  
- Imbalanced-learn (SMOTE)  
- XGBoost  
- Pandas  
- NumPy  
- Matplotlib  
