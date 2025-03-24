# Telephone-Churn-Data-Set-Analysis Report

# Churn Prediction with Machine Learning

![GitHub](https://img.shields.io/badge/Language-Python-blue)
![GitHub](https://img.shields.io/badge/Library-ScikitLearn-orange)
![GitHub](https://img.shields.io/badge/Library-XGBoost-green)
![GitHub](https://img.shields.io/badge/Library-LightGBM-yellow)
![GitHub](https://img.shields.io/badge/Library-CatBoost-red)
![GitHub](https://img.shields.io/badge/Library-RandomForest-pink)

## Overview

This project aimed to predict customer churn using machine learning models based on a telecom dataset from Kaggle. The dataset included features such as customer demographics, usage patterns, and service details. The best-performing model, XGBoost, achieved an accuracy of 0.9552, a PR AUC score of 0.8534, and a recall of 68.4%, making it the top model for detecting churn while maintaining 100% precision. LGBM followed closely with an accuracy of 0.9328 and a PR AUC of 0.8320. Feature importance analysis revealed that customer service calls, total day minutes, and international plan were the most critical predictors of churn. SMOTE effectively handled class imbalance, and hyperparameter tuning via GridSearchCV optimized model performance. 

## Dataset

The dataset used in this project is the Telecom Churn Dataset from Kaggle. You can find the dataset ( https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets ).

## Key Steps

1. **Data Loading and Preprocessing**: Load the dataset, handle missing values, and drop irrelevant columns.
2. **Exploratory Data Analysis (EDA)**: Visualize data distributions, class imbalance, and churn rates by state.
3. **Feature Engineering**: Create new features and encode categorical variables.
4. **Model Training**: Train multiple machine learning models (Random Forest, LightGBM, XGBoost, CatBoost).
5. **Hyperparameter Tuning**: Use GridSearchCV to optimize model hyperparameters.
6. **Model Evaluation**: Evaluate models using accuracy, ROC-AUC, precision-recall curves, and confusion matrices.
7. **Model Interpretation**: Use SHAP values to explain model predictions and visualize feature importance.

## Libraries Used

### Data Processing & Preprocessing
- `pandas` - Data handling, reading CSV files.
- `numpy` - Numerical computations.
- `sklearn.preprocessing` - Label encoding, standardization.
- `imblearn` - SMOTE for handling class imbalance.

### Data Visualization
- `matplotlib` - Basic plotting.
- `seaborn` - Advanced statistical visualizations.
- `plotly` - Interactive visualizations.
- `geopandas` & `us` - Mapping churn data by state.

### Machine Learning Models
- `scikit-learn (sklearn)` - LogisticRegression, RandomForestClassifier, GradientBoostingClassifier.
- `XGBoost (xgboost)` - Powerful gradient boosting model.
- `LightGBM (lightgbm)` - Efficient boosting model.
- `CatBoost (catboost)` - Boosting model optimized for categorical features.

### Model Evaluation & Feature Importance
- `sklearn.metrics` - Accuracy, confusion matrix, ROC-AUC, precision-recall.
- `shap` - Explainability (SHAP values).
- `GridSearchCV` / `RandomizedSearchCV` - Hyperparameter tuning.

Results
Model Performance
| Model         | Accuracy | ROC-AUC | Precision | Recall | F1-Score |
|--------------|----------|---------|-----------|--------|----------|
| RandomForest | 0.9328   | 0.9778  | 1.00      | 0.526  | 0.690    |
| LGBM        | 0.9328   | 0.9820  | 0.917     | 0.579  | 0.710    |
| XGBoost     | 0.9552   | 0.9854  | 1.00      | 0.684  | 0.812    |
| CatBoost    | 0.9179   | 0.9796  | 0.786     | 0.579  | 0.667    |

Feature Importance
Feature importance was analyzed using SHAP values. The top features influencing churn prediction include:
- Customer Service Calls: High number of service calls is strongly associated with churn.
- Total Day Minutes: Customers with higher day minutes are more likely to churn.
- International Plan: Customers with an international plan are more likely to churn.

Conclusion
- Best Performing Model: XGBoost achieved the highest accuracy (95.52%) and ROC-AUC score (0.9854).

- Key Insights: Customer service calls and total day minutes are the most significant predictors of churn.

Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
