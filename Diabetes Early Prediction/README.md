# 🩺 Diabetes Prediction (Random Forest)

## 📌 Project Overview
Diabetes is a chronic disease where early detection can significantly improve patient outcomes. This project focuses on building a machine learning pipeline to predict whether a patient is diabetic based on diagnostic measures.

**Goal:** Build a clinically meaningful model. In healthcare, a **False Negative** (telling a diabetic patient they are healthy) is dangerous. Therefore, this project prioritizes **Recall** (Sensitivity) over simple accuracy to ensure high-risk patients are flagged for further testing.

---

## 📊 Dataset
**Source:** [Pima Indians Diabetes Dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database)  
**Volume:** 768 Patients  
**Target:** 0 (Non-Diabetic) vs. 1 (Diabetic)  

**Key Challenge:**
The dataset contained **invalid zero values** for medical features like *Glucose*, *Blood Pressure*, and *BMI*. A living patient cannot have 0 blood pressure. These were essentially missing values hidden as zeros.

---

## 🛠️ Data Preprocessing
### 1. Handling Hidden Missing Values
Instead of dropping rows (which would lose valuable data), I replaced invalid zeros with `NaN` and imputed them using the **Median**. Median imputation is more robust to outliers than the Mean, which is crucial for medical data like Insulin levels.

### 2. Feature Scaling
Applied `StandardScaler` to normalize features. While Random Forest doesn't strictly require scaling, this ensured fair comparison when benchmarking against distance-based algorithms like KNN.

### 3. Handling Class Imbalance
The dataset is imbalanced (fewer diabetic cases). I utilized **Class Weighting (`class_weight="balanced"`)** in the Random Forest model to penalize misclassifications of the minority class more heavily.

---

## 🤖 Model Comparison
I evaluated three models using **Recall** and **PR-AUC** as the primary decision metrics.

| Model | Accuracy | Recall (Diabetic) | F1-Score | ROC-AUC | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **KNN (Baseline)** | 0.80 | 0.60 | 0.68 | 0.81 | 0.67 |
| **XGBoost** | 0.76 | 0.73 | 0.68 | 0.84 | 0.78 |
| **Random Forest** | **0.82** | **0.82** | **0.76** | **0.87** | **0.81** |

### Why Random Forest?
While XGBoost is powerful, it tended to overfit on this small dataset (768 rows). **Random Forest** proved to be more stable and robust, achieving the highest Recall (0.82) and ROC-AUC (0.87) without complex tuning.

---

## 🏆 Final Model Performance (Random Forest)
The final model was tuned using `RandomizedSearchCV` to optimize for medical sensitivity.

**Confusion Matrix:**
```text
                 Predicted Healthy   Predicted Diabetic
Actual Healthy          81                  18
Actual Diabetic         10                  45
```

**Clinical Interpretation:**
* **Recall: 82%** – The model successfully detected 45 out of 55 diabetic patients.
* **False Negatives: 10** – Only 10 cases were missed. In a real-world setting, this is a strong starting point for a screening tool.
* **Precision Trade-off:** We accept some False Positives (18 healthy people flagged) because the cost of further testing is low compared to the risk of missing a diagnosis.

---

## 🧠 Key Learnings
* **Domain Knowledge Matters:** Simply running a model without fixing the "Zero-Value" glitch would have destroyed performance.
* **Accuracy is Dangerous:** A high-accuracy model could still miss most diabetic patients. Optimizing for Recall was the correct ethical choice.
* **Small Data Strategy:** Complex boosting algorithms (XGBoost) aren't always better. On small, noisy tabular data, bagging methods (Random Forest) often generalize better.

---

## 💻 Tech Stack
* **Python** (Pandas, NumPy)
* **Scikit-Learn** (Imputation, Scaling, Modeling)
* **Matplotlib / Seaborn** (Data Visualization)

---

## 🚀 How to Run
1. Clone the repo and install dependencies.
2. Ensure `diabetes.csv` is in the directory.
3. Run the notebook:
   ```bash
   jupyter notebook diabetes.ipynb
   ```