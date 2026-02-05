<!-- # Credit Card Fraud Detection using Machine Learning (XGBoost)

## Project Overview
This project focuses on detecting fraudulent credit card transactions in a highly imbalanced dataset, where fraudulent transactions represent only 0.17% of all data points.

Instead of relying on misleading metrics such as accuracy, the project emphasizes fraud-appropriate evaluation metrics including Precision-Recall AUC (PR-AUC), Recall, and confusion matrix analysis. Multiple machine learning models were evaluated, and XGBoost with class weighting was selected as the final model based on overall fraud detection performance.

This project was completed independently and follows a real-world applied machine learning workflow.

---

## Dataset
The dataset used is the Credit Card Fraud Detection Dataset from Kaggle.

Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  
Total Transactions: 284,807  
Fraud Cases: 492 (0.172%)

Features:
- Time – Seconds elapsed between transactions
- Amount – Transaction amount
- V1–V28 – PCA-transformed features (original features hidden for confidentiality)
- Class – Target variable (1 = Fraud, 0 = Non-Fraud)

---

## Key Challenges
- Extreme class imbalance
- Accuracy becomes misleading (>99% even for trivial models)
- High cost of false negatives (missed fraud)
- Requirement for probability-based evaluation

---

## Methodology

### 1. Train–Test Split
- Dataset split into 80% training and 20% testing
- Stratified sampling used to preserve fraud distribution

Example:
train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

---

### 2. Handling Class Imbalance
Class imbalance was handled using class weighting, not naive oversampling.

For XGBoost:
scale_pos_weight = number_of_normal_transactions / number_of_fraud_transactions

This ensures that fraud misclassifications are penalized more heavily during training.

SMOTE oversampling was also tested but did not improve PR-AUC, so the weighted approach was retained.

---

### 3. Models Evaluated
The following models were trained and compared:
- Logistic Regression (class-weighted)
- Random Forest (class-weighted)
- XGBoost (class-weighted)

Model selection was based on PR-AUC and Recall, not accuracy.

---

## Evaluation Metrics
Because fraud detection is an imbalanced classification problem, the following metrics were used:
- Precision – Correctness of fraud alerts
- Recall – Ability to detect fraudulent transactions
- F1-score – Balance between precision and recall
- ROC-AUC – Overall ranking ability
- PR-AUC – Primary metric for fraud detection

---

## Model Performance (XGBoost)

### Confusion Matrix
|                | Predicted Non-Fraud | Predicted Fraud |
|----------------|---------------------|-----------------|
| **Actual Non-Fraud** | 56,855 | 9 |
| **Actual Fraud**     | 12 | 86 |

---

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|------|-----------|--------|----------|---------|
| Non-Fraud (0) | 1.00 | 1.00 | 1.00 | 56,864 |
| Fraud (1) | 0.91 | 0.88 | 0.89 | 98 |
| **Accuracy** |  |  | **1.00** | 56,962 |
| **Macro Avg** | 0.95 | 0.94 | 0.95 | 56,962 |
| **Weighted Avg** | 1.00 | 1.00 | 1.00 | 56,962 |

---

### AUC Metrics
- **ROC-AUC:** 0.9899  
- **PR-AUC:** 0.8952  

---

### Interpretation
- The model successfully detects **88% of fraudulent transactions**
- **91% precision** ensures minimal false fraud alerts
- High **PR-AUC** confirms strong performance on an extremely imbalanced dataset
- Results demonstrate a good balance between fraud detection and operational reliability
PR-AUC confirms robust performance under extreme class imbalance

---

## Threshold & Probability Analysis
- ROC-AUC and PR-AUC were computed using predicted probabilities
- Thresholding was applied only for confusion matrix and classification report
- Probabilities were already strongly separated, limiting gains from threshold tuning

---

## Key Learnings
- Accuracy is misleading for fraud detection
- PR-AUC is more informative than ROC-AUC under class imbalance
- Class weighting is highly effective for rare-event detection
- SMOTE does not always improve real-world performance
- Model improvements plateau without richer features

---

## Limitations
- No user-level behavioral features
- No transaction history or temporal aggregation
- Real-world fraud systems require additional contextual data

---

## Possible Extensions
- Cross-validation using PR-AUC
- Feature importance analysis
- Anomaly detection models (Isolation Forest, One-Class SVM)
- Probability calibration
- Deployment as an API

---

## How to Run
1. Download creditcard.csv from Kaggle  
2. Place it in the project directory  
3. Install dependencies:
pip install numpy pandas scikit-learn xgboost matplotlib  
4. Run credit_card.ipynb

---

## Conclusion
This project demonstrates a realistic, evaluation-driven approach to fraud detection. By handling class imbalance correctly, using appropriate metrics, and comparing multiple models, the final solution reflects industry-relevant machine learning practices. -->


# 💳 Credit Card Fraud Detection (XGBoost)

## 📌 Project Overview
This project focuses on detecting fraudulent credit card transactions in a highly imbalanced dataset, where fraudulent transactions represent only **0.17%** of all data points.

Instead of relying on misleading metrics like accuracy (which can be 99.8% even if the model detects zero fraud), this project emphasizes **Precision-Recall AUC (PR-AUC)**, **Recall**, and confusion matrix analysis. 

The final model, an **XGBoost Classifier with class weighting**, achieved a **PR-AUC of 0.89**, successfully detecting 88% of fraud cases while maintaining high precision.

---

## 📂 Dataset
**Source:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
**Volume:** 284,807 Transactions  
**Fraud Cases:** 492 (0.172%)  

**Features:**
- **Time:** Seconds elapsed between transactions.
- **Amount:** Transaction amount.
- **V1–V28:** Anonymized PCA-transformed features (original features hidden for confidentiality).
- **Class:** Target variable (1 = Fraud, 0 = Non-Fraud).

---

## 🚧 Key Challenges
* **Extreme Class Imbalance:** Fraud is a "needle in a haystack" event.
* **Cost of Misclassification:** False Negatives (missed fraud) are extremely costly; False Positives (annoying the user) are undesirable but manageable.
* **Metric Trap:** Standard accuracy is useless here. A dummy classifier predicting "No Fraud" for everyone would have 99.8% accuracy but 0% recall.

---

## ⚙️ Methodology

### 1. Train–Test Split
I used **Stratified Sampling** to ensure the test set has the same proportion of fraud cases as the training set, which is crucial for valid evaluation.
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)
```

### 2. Handling Class Imbalance
Instead of using synthetic oversampling (like SMOTE), which can sometimes introduce noise, I used **Class Weighting** within the XGBoost algorithm. This penalizes the model more heavily for missing a fraud case.

```python
# Ratio of Negative to Positive samples
scale_pos_weight = 580 
```

### 3. Model Selection
I evaluated multiple approaches (including Logistic Regression and Random Forest), but **XGBoost** was selected for the final deployment due to its superior PR-AUC score and speed.

---

## 📊 Evaluation Metrics
* **Precision:** When the model predicts fraud, how often is it correct?
* **Recall (Sensitivity):** Out of all actual fraud cases, how many did we catch?
* **PR-AUC:** The Area Under the Precision-Recall Curve. (The gold standard for imbalanced classification).
* **ROC-AUC:** Used for overall ranking performance.

---

## 🏆 Final Model Performance (XGBoost)

| Metric | Score |
| :--- | :--- |
| **ROC-AUC** | 0.990 |
| **PR-AUC** | **0.895** |

### Confusion Matrix
```text
                 Predicted Non-Fraud   Predicted Fraud
Actual Non-Fraud        56,855                9
Actual Fraud               12                86
```
*The model successfully detected **86 out of 98** fraud cases in the test set, with only 9 false alarms out of 56,000+ normal transactions.*

### Classification Report
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **0 (Normal)** | 1.00 | 1.00 | 1.00 | 56,864 |
| **1 (Fraud)** | **0.91** | **0.88** | **0.89** | 98 |

---

## 📈 Visualizations

### Precision-Recall Curve
![PR Curve](PR_Curve.png)

### ROC Curve
![ROC Curve](ROC_Curve.png)

---

## 🧠 Key Learnings & Limitations
* **Accuracy is misleading:** Relying on accuracy would have hidden the model's true performance.
* **Class Weighting vs. SMOTE:** In this specific dataset, adjusting `scale_pos_weight` was sufficient and computationally faster than oversampling.
* **Limitation:** The dataset lacks user-level context (e.g., "Is this the first time the user bought from this store?"). Real-world systems would require historical features.

---

## 💻 How to Run
1.  Clone the repository.
2.  Download `creditcard.csv` from Kaggle and place it in the root directory.
3.  Install dependencies:
    ```bash
    pip install numpy pandas scikit-learn xgboost matplotlib
    ```
4.  Run the notebook:
    ```bash
    jupyter notebook credit_card.ipynb
    ```

---