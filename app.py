import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Pakistan Digital Banking Analytics",
    page_icon="🏦",
    layout="wide",
)


# =========================================================
# DATA LOADING AND PREPARATION
# =========================================================

@st.cache_data
def load_data():
    """Load the dataset and convert the observation date."""
    data = pd.read_csv("dataset.csv")

    data["Observation Date"] = pd.to_datetime(
        data["Observation Date"],
        errors="coerce",
    )

    return data


def get_series(data, series_name):
    """Select, clean, and sort one series."""
    series = data[data["Series name"] == series_name].copy()

    return (
        series
        .dropna(subset=["Observation Value"])
        .sort_values("Observation Date")
    )


def prepare_yearly_last(series):
    """Get the last available observation for each year."""
    return (
        series
        .set_index("Observation Date")["Observation Value"]
        .resample("YE")
        .last()
        .dropna()
    )


def prepare_yearly_sum(series):
    """Get the yearly sum of observations."""
    return (
        series
        .set_index("Observation Date")["Observation Value"]
        .resample("YE")
        .sum()
        .dropna()
    )


def calculate_engagement(mobile_users_yearly, mobile_transactions_yearly):
    """Calculate mobile transactions per registered user."""
    engagement = pd.DataFrame({
        "Mobile Users (M)": mobile_users_yearly / 1_000_000,
        "Mobile Transactions (M)": mobile_transactions_yearly,
    }).dropna()

    engagement["Transactions per User"] = (
        engagement["Mobile Transactions (M)"]
        / engagement["Mobile Users (M)"]
    )

    return engagement


def calculate_average_transaction_value(
    mobile_value_yearly,
    mobile_transactions_yearly,
):
    """Calculate average mobile transaction value in PKR."""
    average_value = pd.DataFrame()

    average_value["Average Transaction Value (PKR)"] = (
        mobile_value_yearly * 1_000_000_000
    ) / (
        mobile_transactions_yearly * 1_000_000
    )

    return average_value.dropna()


# =========================================================
# MACHINE LEARNING
# =========================================================

def prepare_forecast_data(mobile_users):
    """Prepare mobile banking users for forecasting."""
    forecast_data = mobile_users[
        mobile_users["Observation Date"] >= "2016-09-30"
    ].copy()

    forecast_data = forecast_data[
        ["Observation Date", "Observation Value"]
    ].dropna()

    forecast_data = (
        forecast_data
        .sort_values("Observation Date")
        .reset_index(drop=True)
    )

    forecast_data["Time_Index"] = np.arange(len(forecast_data))

    return forecast_data


def train_and_evaluate_models(forecast_data):
    """Train baseline and linear regression models and calculate metrics."""
    train = forecast_data.iloc[:-8].copy()
    test = forecast_data.iloc[-8:].copy()

    x_train = train[["Time_Index"]]
    y_train = train["Observation Value"]

    x_test = test[["Time_Index"]]
    y_test = test["Observation Value"]

    # Baseline: repeat the last training value.
    baseline_prediction = np.repeat(
        y_train.iloc[-1],
        len(y_test),
    )

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_prediction,
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            baseline_prediction,
        )
    )

    # Linear regression.
    model = LinearRegression()
    model.fit(x_train, y_train)

    test_prediction = model.predict(x_test)

    linear_mae = mean_absolute_error(
        y_test,
        test_prediction,
    )

    linear_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_prediction,
        )
    )

    metrics = {
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
        "linear_mae": linear_mae,
        "linear_rmse": linear_rmse,
    }

    return (
        train,
        test,
        model,
        y_test,
        test_prediction,
        metrics,
    )


def create_2026_forecast(model, forecast_data):
    """Generate the four quarterly mobile banking user forecasts for 2026."""
    future_dates = pd.date_range(
        start="2026-03-31",
        periods=4,
        freq="QE",
    )

    future_index = np.arange(
        len(forecast_data),
        len(forecast_data) + 4,
    )

    future_x = pd.DataFrame({
        "Time_Index": future_index,
    })

    future_predictions = model.predict(future_x)

    forecast_2026 = pd.DataFrame({
        "Date": future_dates,
        "Forecasted Mobile Banking Users": future_predictions,
    })

    forecast_2026["Forecasted Users (M)"] = (
        forecast_2026["Forecasted Mobile Banking Users"]
        / 1_000_000
    )

    return forecast_2026


# =========================================================
# SIDEBAR
# =========================================================

def display_sidebar():
    """Display the dashboard sidebar."""
    st.sidebar.title("🏦 Dashboard Menu")

    st.sidebar.markdown("""
### Project
**Pakistan Digital Banking & Payment Analytics**

### Analysis Areas
- 📊 Digital Banking Growth
- 💳 Payment Activity
- 👥 User Engagement
- 🤖 ML Forecasting
- 📌 Business Insights
""")

    st.sidebar.divider()

    st.sidebar.info(
        "This dashboard analyzes Pakistan's digital banking "
        "indicators and provides a model-based forecast for 2026."
    )


# =========================================================
# KPI SECTION
# =========================================================

def display_latest_kpis(
    mobile_users,
    internet_users,
    mobile_transactions,
    mobile_transaction_value,
):
    """Display the latest digital banking indicators."""
    latest_mobile_users = mobile_users.iloc[-1]["Observation Value"]
    latest_internet_users = internet_users.iloc[-1]["Observation Value"]
    latest_mobile_transactions = (
        mobile_transactions.iloc[-1]["Observation Value"]
    )
    latest_mobile_value = (
        mobile_transaction_value.iloc[-1]["Observation Value"]
    )
    latest_date = mobile_users.iloc[-1]["Observation Date"]

    st.divider()
    st.subheader("📌 Latest Digital Banking Indicators")
    st.caption(
        "Most recent available digital banking indicators in the dataset"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "📱 Mobile Banking Users",
            f"{latest_mobile_users / 1_000_000:.2f} M",
        )

    with col2:
        st.metric(
            "🌐 Internet Banking Users",
            f"{latest_internet_users / 1_000_000:.2f} M",
        )

    with col3:
        st.metric(
            "💳 Mobile Transactions",
            f"{latest_mobile_transactions:.2f} M",
        )

    with col4:
        st.metric(
            "💰 Transaction Value",
            f"{latest_mobile_value:,.2f} B PKR",
        )

    st.caption(
        f"Latest available observation: {latest_date.date()}"
    )


# =========================================================
# DIGITAL BANKING GROWTH
# =========================================================

def display_digital_banking_growth(
    mobile_users_yearly,
    internet_users_yearly,
):
    """Display yearly mobile and internet banking user trends."""
    st.divider()
    st.header("📊 Digital Banking Growth")
    st.caption("Yearly trend in mobile and internet banking users")

    user_chart = pd.DataFrame({
        "Mobile Banking Users (M)": (
            mobile_users_yearly / 1_000_000
        ),
        "Internet Banking Users (M)": (
            internet_users_yearly / 1_000_000
        ),
    })

    st.line_chart(
        user_chart,
        use_container_width=True,
    )


# =========================================================
# PAYMENT ACTIVITY
# =========================================================

def display_payment_activity(
    mobile_transactions_yearly,
    mobile_value_yearly,
):
    """Display mobile transaction volume and value."""
    st.divider()
    st.header("💳 Digital Payment Activity")
    st.caption(
        "Growth in mobile banking transaction volume and transaction value"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Mobile Banking Transactions")

        transaction_chart = pd.DataFrame({
            "Transactions (Million)": mobile_transactions_yearly,
        })

        st.line_chart(
            transaction_chart,
            use_container_width=True,
        )

    with col2:
        st.subheader("💰 Mobile Transaction Value")

        value_chart = pd.DataFrame({
            "Transaction Value (B PKR)": mobile_value_yearly,
        })

        st.line_chart(
            value_chart,
            use_container_width=True,
        )


# =========================================================
# USER ENGAGEMENT
# =========================================================

def display_user_engagement(
    mobile_users_yearly,
    mobile_transactions_yearly,
):
    """Display transactions per mobile banking user."""
    st.divider()
    st.header("👥 Mobile Banking User Engagement")
    st.caption(
        "Estimated mobile banking transactions per registered "
        "mobile banking user"
    )

    engagement = calculate_engagement(
        mobile_users_yearly,
        mobile_transactions_yearly,
    )

    st.line_chart(
        engagement["Transactions per User"],
        use_container_width=True,
    )


# =========================================================
# AVERAGE TRANSACTION VALUE
# =========================================================

def display_average_transaction_value(
    mobile_value_yearly,
    mobile_transactions_yearly,
):
    """Display average mobile transaction value."""
    st.subheader("💵 Average Mobile Transaction Value")

    average_value = calculate_average_transaction_value(
        mobile_value_yearly,
        mobile_transactions_yearly,
    )

    st.line_chart(
        average_value,
        use_container_width=True,
    )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

def display_model_performance(metrics):
    """Display baseline and linear regression error metrics."""
    st.subheader("📊 Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Baseline MAE",
            f"{metrics['baseline_mae'] / 1_000_000:.2f} M",
        )

    with col2:
        st.metric(
            "Linear Regression MAE",
            f"{metrics['linear_mae'] / 1_000_000:.2f} M",
        )

    with col3:
        st.metric(
            "Baseline RMSE",
            f"{metrics['baseline_rmse'] / 1_000_000:.2f} M",
        )

    with col4:
        st.metric(
            "Linear Regression RMSE",
            f"{metrics['linear_rmse'] / 1_000_000:.2f} M",
        )


# =========================================================
# ML FORECASTING
# =========================================================

def display_ml_forecast(
    forecast_data,
    model,
    y_test,
    test_prediction,
    metrics,
):
    """Display model performance, validation results, and 2026 forecast."""
    st.divider()
    st.header("🤖 Machine Learning Forecast")
    st.caption(
        "Linear Regression model used to estimate future mobile banking users"
    )

    display_model_performance(metrics)

    st.subheader("📈 Actual vs Predicted Mobile Banking Users")

    comparison = pd.DataFrame({
        "Actual": y_test.values / 1_000_000,
        "Predicted": test_prediction / 1_000_000,
    }, index=forecast_data.iloc[-8:]["Observation Date"])

    st.line_chart(
        comparison,
        use_container_width=True,
    )

    st.subheader("🔮 Mobile Banking Users Forecast — 2026")

    forecast_2026 = create_2026_forecast(
        model,
        forecast_data,
    )

    col1, col2, col3, col4 = st.columns(4)

    quarter_labels = ["Q1 2026", "Q2 2026", "Q3 2026", "Q4 2026"]

    for column, label, value in zip(
        [col1, col2, col3, col4],
        quarter_labels,
        forecast_2026["Forecasted Users (M)"],
    ):
        with column:
            st.metric(label, f"{value:.2f} M")

    forecast_chart = forecast_2026.set_index(
        "Date"
    )["Forecasted Users (M)"]

    st.line_chart(
        forecast_chart,
        use_container_width=True,
    )

    st.subheader("📋 2026 Forecast Details")

    forecast_table = forecast_2026[
        ["Date", "Forecasted Users (M)"]
    ].copy()

    forecast_table["Date"] = forecast_table["Date"].dt.strftime(
        "%d-%b-%Y"
    )

    forecast_table["Forecasted Users (M)"] = (
        forecast_table["Forecasted Users (M)"].round(2)
    )

    st.dataframe(
        forecast_table,
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# BUSINESS INSIGHTS
# =========================================================

def display_business_insights():
    """Display project business insights."""
    st.divider()
    st.header("📌 Key Business Insights")

    st.markdown("""
### 1. 📱 Strong Mobile Banking Growth
Mobile banking users increased substantially over the analyzed period,
indicating continued expansion of digital banking adoption.

### 2. 💳 Rapid Growth in Digital Transactions
Mobile banking transaction activity has increased strongly,
showing greater usage of digital payment channels.

### 3. 👥 Increasing User Engagement
Transactions per mobile banking user increased over time,
indicating that users are not only joining digital channels
but are also using them more frequently.

### 4. 💰 Increasing Transaction Size
Average mobile transaction value has generally increased,
indicating growth in the monetary value handled through
mobile banking channels.

### 5. 🤖 Machine Learning Forecast
The Linear Regression model provides a model-based estimate
of future mobile banking users for 2026.
""")


# =========================================================
# LIMITATIONS
# =========================================================

def display_limitations():
    """Display forecast limitations."""
    with st.expander("⚠️ Forecast Limitations"):
        st.write("""
The 2026 forecast is a model-based estimate and should not
be interpreted as an official forecast.

The current Linear Regression model uses time as the main
predictor. It does not include factors such as population
growth, smartphone adoption, internet penetration, economic
conditions, regulatory changes, seasonality, or technological
developments.

A future version could use more advanced models such as
Random Forest, XGBoost, or time-series models and could
incorporate additional economic and digital adoption
indicators.
""")


# =========================================================
# DATASET INFORMATION
# =========================================================

def display_dataset_information(data):
    """Display basic dataset information."""
    st.divider()
    st.header("📁 Dataset Information")

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        st.metric(
            "Total Records",
            f"{len(data):,}",
        )

    with info_col2:
        st.metric(
            "Number of Series",
            f"{data['Series Display Name'].nunique():,}",
        )

    with info_col3:
        st.metric(
            "Data Period",
            f"{data['Observation Date'].min().year}–"
            f"{data['Observation Date'].max().year}",
        )


# =========================================================
# MAIN APPLICATION
# =========================================================

def main():
    """Run the Streamlit dashboard."""
    display_sidebar()

    st.title("🏦 Pakistan Digital Banking & Payment Analytics")
    st.caption(
        "Digital Banking Performance Analysis & ML-Based Forecasting | "
        "2007–2025"
    )

    # Load and prepare data.
    data = load_data()

    mobile_users = get_series(
        data,
        "Number of Mobile Phone Banking Users",
    )

    internet_users = get_series(
        data,
        "Number of Internet Banking Users",
    )

    mobile_transactions = get_series(
        data,
        "eBanking - Total Number of Mobile Phone Banking Transactions",
    )

    mobile_transaction_value = get_series(
        data,
        "Total Value of Mobile Phone Banking Transactions",
    )

    # Prepare yearly analysis data.
    mobile_users_yearly = prepare_yearly_last(mobile_users)
    internet_users_yearly = prepare_yearly_last(internet_users)
    mobile_transactions_yearly = prepare_yearly_sum(
        mobile_transactions
    )
    mobile_value_yearly = prepare_yearly_sum(
        mobile_transaction_value
    )

    # Dashboard sections.
    display_latest_kpis(
        mobile_users,
        internet_users,
        mobile_transactions,
        mobile_transaction_value,
    )

    display_digital_banking_growth(
        mobile_users_yearly,
        internet_users_yearly,
    )

    display_payment_activity(
        mobile_transactions_yearly,
        mobile_value_yearly,
    )

    display_user_engagement(
        mobile_users_yearly,
        mobile_transactions_yearly,
    )

    display_average_transaction_value(
        mobile_value_yearly,
        mobile_transactions_yearly,
    )

    # Prepare and evaluate ML model.
    forecast_data = prepare_forecast_data(mobile_users)

    (
        train,
        test,
        model,
        y_test,
        test_prediction,
        metrics,
    ) = train_and_evaluate_models(forecast_data)

    display_ml_forecast(
        forecast_data,
        model,
        y_test,
        test_prediction,
        metrics,
    )

    display_business_insights()
    display_limitations()
    display_dataset_information(data)

    # Footer.
    st.divider()
    st.caption(
        "Pakistan Digital Banking & Payment Analytics | "
        "Python • Pandas • Scikit-learn • Streamlit"
    )


if __name__ == "__main__":
    main()
