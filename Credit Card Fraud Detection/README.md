# Credit Card Fraud Detection using XGBoost and PCA

## Project Overview
This project focuses on identifying fraudulent credit card transactions using Machine Learning techniques. The dataset contains transactions made by credit cards in September 2013 by European cardholders. The project demonstrates a pipeline involving dimensionality reduction via Principal Component Analysis (PCA) and classification using the XGBoost algorithm.

The primary objective is to build a classifier capable of distinguishing between normal and fraudulent transactions within a highly imbalanced dataset.

## Dataset
The project utilizes the **Credit Card Fraud Detection** dataset hosted on Kaggle.

* **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Description:** The dataset contains 284,807 transactions, where only 492 are frauds (0.172%).
* **Features:**
    * **Time:** Seconds elapsed between each transaction and the first transaction in the dataset.
    * **Amount:** Transaction amount.
    * **V1 - V28:** Principal components obtained via PCA (original features are not provided due to confidentiality).
    * **Class:** The target variable (1 for Fraud, 0 for Non-Fraud).

## Methodology

### 1. Data Preprocessing
* **Data Splitting:** The dataset is split into a Training set (80%) and a Test set (20%) to ensure unbiased evaluation.
* **Feature Scaling:** `StandardScaler` is applied to normalize the features, ensuring that the model is not biased by the scale of input variables.

### 2. Dimensionality Reduction
* **Principal Component Analysis (PCA):** The feature space is reduced to 2 principal components. This step facilitates visualization of the decision boundary and reduces computational complexity, though it challenges the model to find a separating hyperplane in a lower-dimensional space.

### 3. Model Selection
* **XGBoost Classifier:** Extreme Gradient Boosting (XGBoost) is selected for its high performance and efficiency in classification tasks. It is an implementation of gradient boosted decision trees designed for speed and performance.

## Implementation Details

The workflow is implemented in Python using the following libraries:
* **Pandas & NumPy:** For data manipulation and numerical operations.
* **Scikit-Learn:** For data splitting, scaling, PCA, and evaluation metrics.
* **XGBoost:** For the classification model.
* **Matplotlib:** For visualizing the decision boundary.

## Performance Evaluation

The model was evaluated on the test set using a Confusion Matrix and Accuracy Score.

* **Accuracy:** ~99.85%
* **Confusion Matrix Results:**
    * The model successfully identified the majority of non-fraudulent transactions (Class 0).
    * Due to the extreme class imbalance and the reduction to only 2 PCA components, the model exhibited a conservative prediction strategy, heavily favoring the majority class.

### Key Learnings
* **Imbalanced Data:** High accuracy in fraud detection can be misleading if the model predicts the majority class for all instances.
* **Dimensionality Reduction:** Reducing features to 2 components aids in visualization (plotting decision boundaries) but may result in information loss critical for separating highly overlapping classes in complex datasets.

## Future Improvements
To improve the sensitivity (Recall) regarding fraudulent cases, the following extensions are recommended:
* **Resampling Techniques:** Implement SMOTE (Synthetic Minority Over-sampling Technique) or random undersampling to balance the class distribution.
* **Feature Retention:** Retain more principal components or use the original features instead of reducing to 2 components.
* **Threshold Tuning:** Adjust the probability threshold of the XGBoost classifier to prioritize Recall over Precision.

## How to Run
1.  Download the `creditcard.csv` dataset from the Kaggle link above.
2.  Place the CSV file in the same directory as the notebook.
3.  Install dependencies:
    ```bash
    pip install numpy pandas matplotlib scikit-learn xgboost
    ```
4.  Run the Jupyter Notebook `credit_card.ipynb`.