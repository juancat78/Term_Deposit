# Term Deposit Subscription Prediction 💼📈

## Overview

This project aims to predict whether a client will subscribe to a **term deposit** as a result of a direct marketing campaign by a European bank. Using a dataset of over **40,000 client records**, we explore a complete machine learning pipeline that includes data preprocessing, feature selection, model training, evaluation (Hold-Out and Stratified K-Fold CV), and hyperparameter tuning. The final goal is to deploy the most performant and generalizable model.

---

## 1. Project Context 🗂️

- **Dataset origin**: Marketing campaigns of a Portuguese banking institution  
- **Task**: Binary classification – Will a client subscribe to a term deposit? (`y`: yes / no)  
- **Challenge**: Strong class imbalance (`y = yes` is the minority class)

---

## 2. Data Preprocessing Pipeline 🛠️

A detailed preprocessing strategy was implemented, including:

### Encoding
- Binary variables (e.g., `default`, `housing`, `loan`): **LabelEncoder**
- Multiclass variables (e.g., `job`, `education`, `month`): **OneHotEncoder**

### Scaling
Scaling method chosen based on distribution and outliers:
- `StandardScaler` for normally distributed variables
- `RobustScaler` for skewed distributions or with outliers

### Class Imbalance
- Target imbalance addressed with **SMOTE (Synthetic Minority Over-sampling Technique)**
- SMOTE applied **only to the training set**, respecting data leakage prevention
- Implemented in both Hold-Out and Stratified K-Fold CV strategies

### Feature Selection
- Feature importance analyzed via correlation matrix
- Final models trained using **all features**, although tests with top correlated features were conducted

---

## 3. Modeling Approach 🤖

The following classification models were trained and evaluated:

- **Logistic Regression**
- **Decision Tree (CART)**
- **k-Nearest Neighbors (kNN)** – optimized with grid search
- **XGBoost** – tested both with and without hyperparameter tuning

### XGBoost Tuning
Hyperparameter optimization via **RandomizedSearchCV**, tuning:
- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`

---

## 4. Model Evaluation 📊

### Hold-Out Validation

- Data split: 80% training / 20% testing  
- Best result: **XGBoost + SMOTE + RandomizedSearchCV**  
- **F1-Score (positive class)**: **0.95**

### Stratified K-Fold Cross-Validation (5 folds)

- SMOTE applied only to the training portion of each fold  
- Metrics computed and averaged across all folds  

#### Cross-Validation Performance (Positive Class):

| Model           | Precision     | Recall        | F1-Score      | ROC-AUC       |
|----------------|---------------|---------------|---------------|---------------|
| Logistic        | 0.316 ± 0.010 | 0.768 ± 0.012 | 0.448 ± 0.011 | 0.889 ± 0.004 |
| Decision Tree   | 0.311 ± 0.017 | 0.814 ± 0.016 | 0.450 ± 0.018 | 0.867 ± 0.008 |
| kNN             | 0.351 ± 0.006 | 0.666 ± 0.005 | 0.460 ± 0.004 | 0.834 ± 0.005 |
| XGBoost         | **0.403 ± 0.005** | **0.798 ± 0.015** | **0.535 ± 0.006** | **0.939 ± 0.004** |

---

## 5. Key Visualizations 📊

- **Correlation Heatmap** – for feature exploration
- **Confusion Matrices** – for each classifier
- **ROC Curves** – to compare models visually
- **Learning Curves** – to detect overfitting (especially for XGBoost)

---

## 6. Conclusions ✅

- **XGBoost** is the most effective model, outperforming all others in F1 and ROC-AUC.
- **Stratified K-Fold CV** provides more robust performance metrics than Hold-Out validation.
- **SMOTE** significantly improves recall and F1-score for the minority class.
- **RandomizedSearchCV** further enhances XGBoost's predictive power.

### ✅ Recommendation:

Deploy the **XGBoost model** trained with **SMOTE** and optimized using **RandomizedSearchCV**.

## Author 👤

**Juan Pablo Catalano**  
Data Science & Geoscience Specialist  
📍 Argentina  
🔗 [LinkedIn](https://www.linkedin.com/in/juancatalano)  
