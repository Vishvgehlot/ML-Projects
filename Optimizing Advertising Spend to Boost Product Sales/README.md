# Advertising Sales Prediction using Machine Learning

## Project Overview
This project predicts product **sales based on advertising budgets** using multiple machine learning regression models.

The objective is to analyze how advertising spending across different channels impacts sales and compare different regression algorithms to determine the best-performing model.

The project follows a complete machine learning workflow including:

- Exploratory Data Analysis (EDA)
- Data preprocessing
- Model training
- Model evaluation
- Hyperparameter tuning
- Model comparison

---

## Dataset

The dataset contains advertising budgets across different marketing channels and the resulting sales.

Features:
- TV Advertising Budget
- Radio Advertising Budget
- Newspaper Advertising Budget

Target Variable:
- Sales

Dataset Source (Kaggle):  
https://www.kaggle.com/datasets/ashydv/advertising-dataset

---

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

To understand the dataset, the following visualizations were used:

- Pairplot to observe relationships between variables
- Correlation heatmap to analyze feature correlations

These visualizations help identify how advertising channels influence sales.

---

### 2. Data Preprocessing

The following preprocessing steps were applied:

- Train-Test Split
- Feature scaling using **StandardScaler** (for SVR)
- Encoding categorical variables using **ColumnTransformer** and **OneHotEncoder**

---

## Machine Learning Models Used

The following regression algorithms were implemented and evaluated:

- Linear Regression
- Support Vector Regression (SVR)
- Decision Tree Regressor
- Random Forest Regressor

Each model was trained on the training dataset and evaluated using the test dataset.

---

## Evaluation Metrics

Models were evaluated using the following regression metrics:

- **Mean Absolute Error (MAE)** – average prediction error
- **Root Mean Squared Error (RMSE)** – penalizes large errors
- **R² Score** – measures how well the model explains variance in the target variable

---

## Model Performance Comparison

| Model | MAE | RMSE | R² Score |
|------|------|------|------|
| Linear Regression | 2363.43 | 2971.21 | 0.99899 |
| Decision Tree Regressor | 3385.45 | 4184.25 | 0.99799 |
| Random Forest Regressor | **1056.30** | **1422.01** | **0.99977** |
| Support Vector Regression | 2372.37 | 2976.68 | 0.99898 |

---

## Hyperparameter Tuning

Hyperparameter tuning was performed using **GridSearchCV** for the Support Vector Regression model.

Parameters tuned:

- Kernel type
- C (regularization parameter)
- Gamma
- Epsilon

Best Parameters Found:

C = 1000  
epsilon = 0.1  
kernel = linear  

Best Cross-Validation R² Score:

0.99898

---

## Best Performing Model

The **Random Forest Regressor** achieved the best performance among all models.

Results:

- MAE: 1056.30
- RMSE: 1422.01
- R² Score: 0.99977

This indicates that Random Forest captures the relationship between advertising budgets and sales most effectively.

However, the strong performance of Linear Regression and SVR suggests that the dataset exhibits a **strong linear relationship between advertising spending and sales**.

---

## Visualization

Scatter plots of **Actual vs Predicted values** were used to visually evaluate model performance.

A well-performing regression model produces predictions close to the diagonal line, indicating accurate predictions.

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## Project Structure

advertising-sales-regression/
│
├── optimizing.ipynb
├── README.md
└── dataset.csv

---

## Future Improvements

Possible improvements for this project include:

- Feature engineering
- Cross-validation across all models
- Advanced hyperparameter tuning
- Model deployment as a web application or API

---

## Author

**Vishv Gehlot**

Second-year university student exploring machine learning, data science, and full-stack development.