# Pakistan Digital Banking & Payment Analytics

## 1. Problem

Digital banking in Pakistan is growing rapidly. This project analyzes historical digital banking activity and forecasts future mobile banking users using Machine Learning.

## 2. Objectives

- Analyze mobile and internet banking trends
- Measure transaction activity and user engagement
- Calculate transaction-related indicators
- Forecast mobile banking users for 2026
- Present results through an interactive Streamlit dashboard

## 3. Description

This project combines **Data Analysis, Visualization, and Machine Learning** to analyze Pakistan's digital banking data.

Main metrics:
- Mobile Banking Users
- Internet Banking Users
- Mobile Banking Transactions
- Mobile Banking Transaction Value
- Transactions per User
- Average Transaction Value

Dataset period: **2007–2025**

## 4. Variables

- `Observation Date`
- `Series Key`
- `Series Name`
- `Observation Value`
- `Unit`
- `Observation Status`
- `Time_Index`
- Mobile Banking Users
- Internet Banking Users
- Mobile Banking Transactions
- Mobile Banking Transaction Value

## 5. Mathematical Statements

**Transactions per User**

Transactions per User = Mobile Transactions / Mobile Users

**Average Transaction Value**

Average Transaction Value = Transaction Value / Transaction Volume

**Linear Regression**

y = β₀ + β₁X

Where `y` is predicted Mobile Banking Users and `X` is Time_Index.

**Model Evaluation**

MAE = (1/n) × Σ |Actual − Predicted|

RMSE = √[(1/n) × Σ(Actual − Predicted)²]

## 6. Code

### `app.py`

Provides the Streamlit dashboard, visualizations, KPIs, ML evaluation, and 2026 forecast.

### `file2.py`

Performs data analysis, calculations, visualizations, Linear Regression, model evaluation, and forecast generation.

## 7. Solution

The project analyzes historical digital banking data and uses **Linear Regression** with `Time_Index` to forecast Mobile Banking Users.

The final **8 observations** are used for testing, while the remaining observations are used for training. Model performance is evaluated using **MAE and RMSE**.

## Streamlit Dashboard

The dashboard includes:

- Latest Digital Banking Indicators
- Banking Growth Trends
- Transaction Volume and Value
- Transactions per User
- Average Transaction Value
- Actual vs Predicted Results
- 2026 Forecast

## Machine Learning Forecast

Quarterly forecasts are generated for:

- Q1 2026
- Q2 2026
- Q3 2026
- Q4 2026

Output file:

`mobile_banking_users_forecast_2026.csv`

These are model-generated estimates based on historical trends.

## Dataset

Main dataset:

`dataset.csv`

The dataset contains historical digital banking observations from approximately **2007–2025**.

## Visualizations

- Mobile Banking Users vs Transactions
- Transaction Value Trends
- Transactions per User
- Average Transaction Value
- Actual vs Predicted Values
- 2026 Forecast

## Business Insights

The project helps analyze:

- Digital banking adoption
- Transaction growth
- User engagement
- Transaction value patterns
- Future mobile banking user estimates

## Limitations and Future Improvements

The current model mainly uses historical time trends and does not include external factors such as population growth, smartphone adoption, internet penetration, economic conditions, or banking policies.

Future improvements can include **Random Forest, XGBoost, and advanced time-series models**.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- CSV

## Project Structure

ml_project/
├── dataset.csv
├── app.py
├── file2.py
├── mobile_banking_users_forecast_2026.csv
└── README.md

## Installation

`pip install pandas numpy matplotlib scikit-learn streamlit`

## How to Run

`streamlit run app.py`

`python file2.py`

## Conclusion

This project demonstrates how historical digital banking data can be used for **data analysis, visualization, user engagement measurement, and Machine Learning-based forecasting** in Pakistan.

## Academic Project

**Pakistan Digital Banking & Payment Analytics + ML-Based Forecasting**

A Python-based Data Analytics and Machine Learning project using historical digital banking data.