Term Deposit Subscription Prediction 💼📈
Overview
This project aims to predict whether a client will subscribe to a term deposit as a result of a direct marketing campaign by a European bank. Using a dataset of over 40,000 client records, we explore a full machine learning pipeline that includes data preprocessing, feature selection, model training, evaluation (Hold-Out and Stratified K-Fold CV), and hyperparameter tuning. The goal is to deploy the most performant and generalizable model for real-world application.

1. Project Context 🗂️
Dataset origin: Marketing campaigns of a Portuguese banking institution

Task: Binary classification – Will a client subscribe to a term deposit? (y: yes / no)

Challenge: Strong class imbalance (y = yes is the minority class)

2. Data Preprocessing Pipeline 🛠️
A detailed preprocessing strategy was implemented, including:

Encoding
Binary variables (e.g., default, housing, loan): LabelEncoder

Multiclass variables (e.g., job, education, month): OneHotEncoder

Scaling
Chosen per feature based on distribution and presence of outliers:

StandardScaler for normally distributed variables

RobustScaler for skewed distributions or outliers

Class Imbalance
Target imbalance addressed with SMOTE (Synthetic Minority Over-sampling Technique)

SMOTE applied only to the training set, respecting data leakage prevention

Implemented in both Hold-Out and Stratified K-Fold CV strategies

Feature Selection
Feature importance evaluated via correlation matrix

Final model used all features due to better performance, but tests were conducted using only top correlated features

3. Modeling Approach 🤖
The following classification algorithms were trained and compared:

Model	Details
Logistic Regression	Baseline model
Decision Tree (CART)	Depth-limited trees
k-Nearest Neighbors (kNN)	Optimal k via grid search
XGBoost	With and without RandomizedSearchCV

XGBoost Tuning
Hyperparameter optimization via RandomizedSearchCV on:

n_estimators, max_depth, learning_rate, subsample, colsample_bytree

Final model:

python
Copiar
Editar
XGBClassifier(learning_rate=0.1, max_depth=3, n_estimators=200, ...)
4. Model Evaluation 📊
Hold-Out Validation
Split: 80% training / 20% testing

Best model:

XGBoost with SMOTE and hyperparameter tuning

F1-score (positive class): 0.95

Stratified K-Fold Cross-Validation (n = 5)
SMOTE applied to each fold’s training set

Evaluated metrics per fold, with mean ± std

More reliable and generalizable results

Cross-Validation Performance (Positive Class):
Model	Precision	Recall	F1-Score	ROC-AUC
Logistic	0.316 ± 0.010	0.768 ± 0.012	0.448 ± 0.011	0.889 ± 0.004
Decision Tree	0.311 ± 0.017	0.814 ± 0.016	0.450 ± 0.018	0.867 ± 0.008
kNN	0.351 ± 0.006	0.666 ± 0.005	0.460 ± 0.004	0.834 ± 0.005
XGBoost	0.403 ± 0.005	0.798 ± 0.015	0.535 ± 0.006	0.939 ± 0.004

5. Key Visualizations 📊
Correlation Heatmap: Used to guide feature selection

Confusion Matrix: Visual inspection of TP, FP, FN, TN per model

ROC-AUC Curves: For visual comparison of classifier performance

Learning Curves: Validated that XGBoost does not overfit on training data

6. Conclusions ✅
XGBoost emerged as the top-performing algorithm, outperforming all others significantly on F1 and ROC-AUC scores.

Stratified K-Fold CV gave more robust performance estimates than Hold-Out validation.

SMOTE substantially improved minority class performance across all models.

RandomizedSearchCV enhanced XGBoost performance even further by finding optimal hyperparameters.

📌 Recommendation: Deploy the XGBoost + SMOTE + RandomizedSearchCV model.

Author 👤
Juan Pablo Catalano
Data Science & Geoscience Specialist
📍 Argentina
🔗 LinkedIn
