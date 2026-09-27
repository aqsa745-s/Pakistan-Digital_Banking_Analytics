"""
Pakistan Digital Banking & Payment Analytics
--------------------------------------------
Cleaned analysis and machine-learning script.

This version keeps the original analysis flow and calculations while:
- removing repeated imports and duplicate calculations
- grouping related work into clear sections/functions
- keeping the same main metrics, ML approach, charts, and forecast output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================================================
# DATA LOADING & PROFILE
# =========================================================

def load_and_profile_data(file_path="dataset.csv"):
    """Load the dataset, convert dates, and print the basic profile."""
    df = pd.read_csv(file_path)

    print("===== DATASET PROFILE =====")
    print("\nRows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("unique series names:", df["Series name"].nunique())

    df["Observation Date"] = pd.to_datetime(
        df["Observation Date"],
        format="%d-%b-%Y",
    )

    return df


def check_dates(df):
    """Print the date range and frequency information."""
    print("\n===== DATE RANGE =====")
    print("First Date:")
    print(df["Observation Date"].min())

    print("\nLast Date:")
    print(df["Observation Date"].max())

    print("\nUnique Dates:")
    print(df["Observation Date"].nunique())

    print("\n===== DATE FREQUENCY CHECK =====")

    dates = pd.to_datetime(df["Observation Date"])
    date_gaps = (
        dates.sort_values()
        .drop_duplicates()
        .diff()
        .dt.days
        .dropna()
    )

    print("Minimum gap:", date_gaps.min(), "days")
    print("Maximum gap:", date_gaps.max(), "days")

    print("\nMost common gaps:")
    print(date_gaps.value_counts().head())


def check_metric_availability(df):
    """Print availability statistics for all and digital metrics."""
    print("\n===== METRIC DATA AVAILABILITY =====")

    metric_check = df.groupby("Series name")["Observation Value"].agg(
        Total_Observations="size",
        Available_Values="count",
    )

    metric_check["Missing_Values"] = (
        metric_check["Total_Observations"]
        - metric_check["Available_Values"]
    )

    metric_check["Availability_%"] = (
        metric_check["Available_Values"]
        / metric_check["Total_Observations"]
        * 100
    )

    print(
        metric_check
        .sort_values("Available_Values", ascending=False)
        .head(20)
        .to_string()
    )

    digital_metrics = df[
        df["Series name"].str.contains(
            "Mobile|Internet|e-Banking|e-Commerce",
            case=False,
            na=False,
        )
    ]

    digital_check = digital_metrics.groupby(
        "Series name"
    )["Observation Value"].agg(
        Total_Observations="size",
        Available_Values="count",
    )

    digital_check["Missing_Values"] = (
        digital_check["Total_Observations"]
        - digital_check["Available_Values"]
    )

    digital_check["Availability_%"] = (
        digital_check["Available_Values"]
        / digital_check["Total_Observations"]
        * 100
    )

    print(digital_check.to_string())


# =========================================================
# SERIES PREPARATION
# =========================================================

def get_series(df, series_name, include_unit=False):
    """Return one cleaned series sorted by observation date."""
    series = df[df["Series name"] == series_name].copy()

    series = series[
        series["Observation Value"].notnull()
    ].sort_values("Observation Date")

    columns = ["Observation Date", "Observation Value"]

    if include_unit:
        columns.append("Unit")

    return series, columns


def display_selected_series(series, columns):
    """Print selected series observations."""
    print(
        series[columns].to_string(index=False)
    )


def prepare_yearly_last(series):
    """Return the last observation available for each year."""
    yearly = (
        series.groupby("Year")["Observation Value"]
        .last()
        .reset_index()
    )
    return yearly


# =========================================================
# YEARLY USER & TRANSACTION ANALYSIS
# =========================================================

def prepare_yearly_user_analysis(series):
    """Prepare yearly user values and YoY growth."""
    series = series.copy()
    series["Year"] = series["Observation Date"].dt.year

    yearly = prepare_yearly_last(series)

    yearly["YoY_Growth_%"] = (
        yearly["Observation Value"].pct_change() * 100
    )

    return yearly


def prepare_yearly_transaction_analysis(series):
    """Prepare yearly transaction values and YoY growth."""
    series = series.copy()
    series["Year"] = series["Observation Date"].dt.year

    yearly = prepare_yearly_last(series)

    yearly["YoY_Growth_%"] = (
        yearly["Observation Value"].pct_change() * 100
    )

    return yearly


def prepare_mobile_activity(mobile_users, mobile_transactions):
    """Calculate yearly mobile users, transactions, and transactions per user."""
    users = mobile_users.copy()
    transactions = mobile_transactions.copy()

    users["Year"] = users["Observation Date"].dt.year
    transactions["Year"] = transactions["Observation Date"].dt.year

    mobile_users_yearly = (
        users.groupby("Year")["Observation Value"]
        .last()
        .reset_index()
        .rename(
            columns={
                "Observation Value": "Mobile_Users",
            }
        )
    )

    mobile_transactions_yearly = (
        transactions.groupby("Year")["Observation Value"]
        .last()
        .reset_index()
        .rename(
            columns={
                "Observation Value": "Mobile_Transactions_Million",
            }
        )
    )

    mobile_activity = pd.merge(
        mobile_users_yearly,
        mobile_transactions_yearly,
        on="Year",
        how="inner",
    )

    mobile_activity["Mobile_Users_Million"] = (
        mobile_activity["Mobile_Users"] / 1_000_000
    )

    mobile_activity["Transactions_Per_User"] = (
        mobile_activity["Mobile_Transactions_Million"]
        / mobile_activity["Mobile_Users_Million"]
    )

    return mobile_activity


def prepare_transaction_analysis(mobile_value, mobile_transactions):
    """Calculate yearly transaction value and average transaction value."""
    value = mobile_value.copy()
    transactions = mobile_transactions.copy()

    value["Year"] = value["Observation Date"].dt.year
    transactions["Year"] = transactions["Observation Date"].dt.year

    mobile_value_yearly = (
        value.groupby("Year")["Observation Value"]
        .sum()
        .reset_index()
        .rename(
            columns={
                "Observation Value": "Transaction_Value_Billion_PKR",
            }
        )
    )

    mobile_transactions_yearly = (
        transactions.groupby("Year")["Observation Value"]
        .sum()
        .reset_index()
        .rename(
            columns={
                "Observation Value": "Transactions_Million",
            }
        )
    )

    transaction_analysis = pd.merge(
        mobile_value_yearly,
        mobile_transactions_yearly,
        on="Year",
        how="inner",
    )

    transaction_analysis["Average_Transaction_Value_PKR"] = (
        transaction_analysis["Transaction_Value_Billion_PKR"]
        * 1_000_000_000
        / (
            transaction_analysis["Transactions_Million"]
            * 1_000_000
        )
    )

    return (
        mobile_value_yearly,
        mobile_transactions_yearly,
        transaction_analysis,
    )


def print_analysis_results(
    yearly_mobile_users,
    yearly_internet_users,
    yearly_mobile_transactions,
    mobile_activity,
    transaction_analysis,
):
    """Print yearly analysis tables and derived metrics."""
    print(yearly_mobile_users.to_string(index=False))
    print(yearly_internet_users.to_string(index=False))
    print(yearly_mobile_transactions.to_string(index=False))

    print(
        mobile_activity[
            [
                "Year",
                "Mobile_Users_Million",
                "Mobile_Transactions_Million",
                "Transactions_Per_User",
            ]
        ].to_string(index=False)
    )

    print("\n===== AVERAGE TRANSACTION VALUE =====")
    print(
        transaction_analysis[
            [
                "Year",
                "Transaction_Value_Billion_PKR",
                "Transactions_Million",
                "Average_Transaction_Value_PKR",
            ]
        ].to_string(index=False)
    )


def calculate_growth_and_correlation(mobile_activity):
    """Calculate user/transaction growth and their correlation."""
    correlation = mobile_activity[
        ["Mobile_Users_Million", "Mobile_Transactions_Million"]
    ].corr()

    print("\n===== CORRELATION ANALYSIS =====")
    print(correlation)

    growth_analysis = mobile_activity[
        [
            "Year",
            "Mobile_Users_Million",
            "Mobile_Transactions_Million",
        ]
    ].copy()

    growth_analysis["Users_Growth_%"] = (
        growth_analysis["Mobile_Users_Million"].pct_change() * 100
    )

    growth_analysis["Transactions_Growth_%"] = (
        growth_analysis["Mobile_Transactions_Million"].pct_change()
        * 100
    )

    print("\n===== USERS VS TRANSACTIONS GROWTH =====")
    print(
        growth_analysis[
            [
                "Year",
                "Users_Growth_%",
                "Transactions_Growth_%",
            ]
        ].to_string(index=False)
    )

    return growth_analysis


# =========================================================
# VISUALIZATION
# =========================================================

def plot_growth_analysis(growth_analysis):
    """Plot mobile banking users against transactions."""
    plt.figure(figsize=(10, 6))

    plt.plot(
        growth_analysis["Year"],
        growth_analysis["Mobile_Users_Million"],
        marker="o",
        label="Mobile Banking Users",
    )

    plt.plot(
        growth_analysis["Year"],
        growth_analysis["Mobile_Transactions_Million"],
        marker="o",
        label="Mobile Banking Transactions",
    )

    plt.title("Mobile Banking Users vs Transactions (2016–2025)")
    plt.xlabel("Year")
    plt.ylabel("Million")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_transaction_value(mobile_value_yearly):
    """Plot yearly mobile banking transaction value."""
    plt.figure(figsize=(10, 6))

    plt.plot(
        mobile_value_yearly["Year"],
        mobile_value_yearly["Transaction_Value_Billion_PKR"],
        marker="o",
    )

    plt.title("Mobile Banking Transaction Value (2007–2025)")
    plt.xlabel("Year")
    plt.ylabel("Transaction Value (Billion PKR)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_transactions_per_user(mobile_activity):
    """Plot yearly transactions per mobile banking user."""
    plt.figure(figsize=(10, 6))

    plt.plot(
        mobile_activity["Year"],
        mobile_activity["Transactions_Per_User"],
        marker="o",
    )

    plt.title("Mobile Banking Transactions per User (2016–2025)")
    plt.xlabel("Year")
    plt.ylabel("Transactions per User")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_average_transaction_value(transaction_analysis):
    """Plot average mobile banking transaction value."""
    plt.figure(figsize=(10, 6))

    plt.plot(
        transaction_analysis["Year"],
        transaction_analysis["Average_Transaction_Value_PKR"],
        marker="o",
    )

    plt.title("Average Mobile Banking Transaction Value (2007–2025)")
    plt.xlabel("Year")
    plt.ylabel("Average Transaction Value (PKR)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# =========================================================
# MACHINE LEARNING: FORECAST PREPARATION
# =========================================================

def prepare_forecast_data(mobile_users):
    """Prepare mobile banking users for ML forecasting."""
    forecast_data = mobile_users[
        ["Observation Date", "Observation Value"]
    ].copy()

    forecast_data["Observation Date"] = pd.to_datetime(
        forecast_data["Observation Date"]
    )

    forecast_data = (
        forecast_data
        .sort_values("Observation Date")
        .reset_index(drop=True)
    )

    forecast_data.rename(
        columns={
            "Observation Date": "Date",
            "Observation Value": "Mobile_Users",
        },
        inplace=True,
    )

    print("\n===== FORECASTING DATA =====")
    print(forecast_data.head())

    print("\nLast 5 observations:")
    print(forecast_data.tail())

    print("\nRows:", len(forecast_data))
    print("Missing values:")
    print(forecast_data.isnull().sum())

    return forecast_data


def split_forecast_data(forecast_data):
    """Split the final eight observations into the test set."""
    train = forecast_data.iloc[:-8].copy()
    test = forecast_data.iloc[-8:].copy()

    print("\n===== TRAIN DATA =====")
    print("Rows:", len(train))
    print("From:", train["Date"].min())
    print("To:", train["Date"].max())

    print("\n===== TEST DATA =====")
    print("Rows:", len(test))
    print("From:", test["Date"].min())
    print("To:", test["Date"].max())

    return train, test


# =========================================================
# MACHINE LEARNING: BASELINE MODEL
# =========================================================

def evaluate_baseline(train, test):
    """Create and evaluate the last-value baseline forecast."""
    last_train_value = train["Mobile_Users"].iloc[-1]

    test["Baseline_Prediction"] = last_train_value

    print("\n===== BASELINE FORECAST =====")
    print(
        test[
            ["Date", "Mobile_Users", "Baseline_Prediction"]
        ].to_string(index=False)
    )

    mae = mean_absolute_error(
        test["Mobile_Users"],
        test["Baseline_Prediction"],
    )

    rmse = np.sqrt(
        mean_squared_error(
            test["Mobile_Users"],
            test["Baseline_Prediction"],
        )
    )

    print("\n===== BASELINE MODEL EVALUATION =====")
    print("MAE:", mae)
    print("RMSE:", rmse)

    return test, mae, rmse


# =========================================================
# MACHINE LEARNING: LINEAR REGRESSION
# =========================================================

def train_linear_regression(train, test):
    """Train and evaluate the linear regression model."""
    train = train.copy()
    test = test.copy()

    train["Time_Index"] = range(len(train))
    test["Time_Index"] = range(
        len(train),
        len(train) + len(test),
    )

    X_train = train[["Time_Index"]]
    y_train = train["Mobile_Users"]

    X_test = test[["Time_Index"]]
    y_test = test["Mobile_Users"]

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    model = LinearRegression()
    model.fit(X_train, y_train)

    print("\n===== LINEAR REGRESSION MODEL =====")
    print("Slope:", model.coef_[0])
    print("Intercept:", model.intercept_)

    test["Linear_Regression_Prediction"] = model.predict(X_test)

    print("\n===== LINEAR REGRESSION PREDICTIONS =====")
    print(
        test[
            [
                "Date",
                "Mobile_Users",
                "Linear_Regression_Prediction",
            ]
        ]
    )

    linear_mae = mean_absolute_error(
        test["Mobile_Users"],
        test["Linear_Regression_Prediction"],
    )

    linear_rmse = np.sqrt(
        mean_squared_error(
            test["Mobile_Users"],
            test["Linear_Regression_Prediction"],
        )
    )

    print("\n===== LINEAR REGRESSION MODEL EVALUATION =====")
    print("MAE:", linear_mae)
    print("RMSE:", linear_rmse)

    return test, model, linear_mae, linear_rmse


def plot_model_predictions(test):
    """Plot actual values against baseline and linear regression."""
    plt.figure(figsize=(12, 6))

    plt.plot(
        test["Date"],
        test["Mobile_Users"],
        marker="o",
        label="Actual",
    )

    plt.plot(
        test["Date"],
        test["Baseline_Prediction"],
        marker="o",
        label="Baseline",
    )

    plt.plot(
        test["Date"],
        test["Linear_Regression_Prediction"],
        marker="o",
        label="Linear Regression",
    )

    plt.title("Mobile Banking Users: Actual vs Model Predictions")
    plt.xlabel("Date")
    plt.ylabel("Number of Mobile Banking Users")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


# =========================================================
# FUTURE FORECAST
# =========================================================

def create_future_forecast(model, forecast_data, future_periods=4):
    """Generate and save the 2026 mobile banking users forecast."""
    last_time_index = len(forecast_data)

    future_time_index = np.arange(
        last_time_index,
        last_time_index + future_periods,
    )

    future_predictions = model.predict(
        future_time_index.reshape(-1, 1)
    )

    future_dates = pd.date_range(
        start=forecast_data["Date"].max()
        + pd.DateOffset(months=3),
        periods=future_periods,
        freq="QE",
    )

    future_forecast = pd.DataFrame({
        "Date": future_dates,
        "Predicted_Mobile_Users": future_predictions,
    })

    print("\n===== FUTURE MOBILE BANKING USERS FORECAST =====")
    print(future_forecast.to_string(index=False))

    return future_forecast


def plot_future_forecast(future_forecast):
    """Plot the 2026 mobile banking users forecast."""
    plt.figure(figsize=(10, 5))

    plt.plot(
        future_forecast["Date"],
        future_forecast["Predicted_Mobile_Users"],
        marker="o",
        label="Forecasted Mobile Users",
    )

    plt.title("Forecast of Mobile Banking Users - 2026")
    plt.xlabel("Date")
    plt.ylabel("Predicted Mobile Banking Users")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def save_forecast(future_forecast, file_path="mobile_banking_users_forecast_2026.csv"):
    """Save the 2026 forecast to CSV."""
    future_forecast.to_csv(
        file_path,
        index=False,
    )

    print("\nForecast file saved successfully.")


def print_final_ml_summary(
    linear_mae,
    linear_rmse,
    future_forecast,
):
    """Print the final machine-learning summary."""
    print("\n========== FINAL ML SUMMARY ==========")

    print("\nModel: Linear Regression")

    print("\nEvaluation:")
    print("MAE:", linear_mae)
    print("RMSE:", linear_rmse)

    print("\n2026 Forecast:")
    print(future_forecast.to_string(index=False))

    print("\n======================================")


# =========================================================
# MAIN ANALYSIS PIPELINE
# =========================================================

def main():
    """Run the complete banking analytics and ML workflow."""
    df = load_and_profile_data()
    check_dates(df)
    check_metric_availability(df)

    mobile_users, mobile_user_columns = get_series(
        df,
        "Number of Mobile Phone Banking Users",
    )
    display_selected_series(mobile_users, mobile_user_columns)

    internet_users, internet_user_columns = get_series(
        df,
        "Number of Internet Banking Users",
    )
    display_selected_series(internet_users, internet_user_columns)

    mobile_transactions, mobile_transaction_columns = get_series(
        df,
        "eBanking - Total Number of Mobile Phone Banking Transactions",
    )
    display_selected_series(
        mobile_transactions,
        mobile_transaction_columns,
    )

    print(
        df[
            df["Series name"]
            == "eBanking - Total Number of Mobile Phone Banking Transactions"
        ][
            ["Series name", "Unit"]
        ].drop_duplicates().to_string(index=False)
    )

    mobile_value, mobile_value_columns = get_series(
        df,
        "Total Value of Mobile Phone Banking Transactions",
        include_unit=True,
    )
    display_selected_series(mobile_value, mobile_value_columns)

    internet_transactions, internet_transaction_columns = get_series(
        df,
        "eBanking - Total Number of Internet Banking Transactions",
        include_unit=True,
    )
    display_selected_series(
        internet_transactions,
        internet_transaction_columns,
    )

    yearly_mobile_users = prepare_yearly_user_analysis(mobile_users)
    yearly_internet_users = prepare_yearly_user_analysis(internet_users)
    yearly_mobile_transactions = prepare_yearly_transaction_analysis(
        mobile_transactions
    )

    mobile_activity = prepare_mobile_activity(
        mobile_users,
        mobile_transactions,
    )

    (
        mobile_value_yearly,
        mobile_transactions_yearly,
        transaction_analysis,
    ) = prepare_transaction_analysis(
        mobile_value,
        mobile_transactions,
    )

    print_analysis_results(
        yearly_mobile_users,
        yearly_internet_users,
        yearly_mobile_transactions,
        mobile_activity,
        transaction_analysis,
    )

    growth_analysis = calculate_growth_and_correlation(
        mobile_activity
    )

    plot_growth_analysis(growth_analysis)
    plot_transaction_value(mobile_value_yearly)
    plot_transactions_per_user(mobile_activity)
    plot_average_transaction_value(transaction_analysis)

    forecast_data = prepare_forecast_data(mobile_users)

    train, test = split_forecast_data(forecast_data)

    test, baseline_mae, baseline_rmse = evaluate_baseline(
        train,
        test,
    )

    test, model, linear_mae, linear_rmse = train_linear_regression(
        train,
        test,
    )

    plot_model_predictions(test)

    future_forecast = create_future_forecast(
        model,
        forecast_data,
    )

    plot_future_forecast(future_forecast)
    save_forecast(future_forecast)

    print_final_ml_summary(
        linear_mae,
        linear_rmse,
        future_forecast,
    )


if __name__ == "__main__":
    main()
