# app.py
import os
import glob
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Prophet import with friendly error
try:
    from prophet import Prophet
    prophet_available = True
except Exception as e:
    prophet_available = False

# -----------------------
# CONFIG
# -----------------------
DATA_FOLDER = "data"        # place your processed_*.csv files here
MODEL_FOLDER = "models"     # trained model cache will be stored here
os.makedirs(MODEL_FOLDER, exist_ok=True)

st.set_page_config(page_title="Market Price Predictor (Prophet)", layout="wide")

# -----------------------
# Helpers
# -----------------------
def safe_name(s: str):
    return str(s).strip().replace(" ", "_").replace("/", "_")

@st.cache_data
def list_processed_files():
    files = sorted(glob.glob(os.path.join(DATA_FOLDER, "processed_*.csv")))
    commodities = [os.path.basename(fp).replace("processed_", "").replace(".csv", "") for fp in files]
    return files, commodities

@st.cache_data
def load_commodity_df(fp):
    df = pd.read_csv(fp, parse_dates=["price_date"])
    # normalize market names
    if "market_name" in df.columns:
        df["market_name"] = df["market_name"].astype(str).str.strip().str.title()
    return df

@st.cache_resource
def load_or_train_model(series_df, model_path_local):
    """
    Loads saved Prophet model if present; otherwise trains a new one and saves (joblib).
    series_df must contain ['price_date', 'modal_price_rs_per_kg'].
    """
    if not prophet_available:
        raise RuntimeError("Prophet package not available. Install 'prophet' for forecasting.")

    # Try to load pre-saved model
    if os.path.exists(model_path_local):
        try:
            m = joblib.load(model_path_local)
            return m
        except Exception:
            # fallback to retraining if load fails
            pass

    # Prepare data for Prophet
    prophet_df = series_df[["price_date", "modal_price_rs_per_kg"]].dropna().rename(
        columns={"price_date": "ds", "modal_price_rs_per_kg": "y"}
    )
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
    prophet_df = prophet_df.sort_values("ds").reset_index(drop=True)

    # Train model
    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    # optionally add holidays/regressors here
    m.fit(prophet_df)

    # Save model if possible
    try:
        joblib.dump(m, model_path_local)
    except Exception:
        # if saving fails (environment limitations), ignore
        pass

    return m

# -----------------------
# UI: sidebar inputs
# -----------------------
st.title("Vegetable/Market Price Predictor — Prophet")
st.sidebar.header("Inputs & Options")

files, available_commodities = list_processed_files()
if not files:
    st.error(f"No processed CSVs found in `{DATA_FOLDER}`. Place `processed_<Commodity>.csv` files there.")
    st.stop()

commodity = st.sidebar.selectbox("Commodity", ["-- choose --"] + available_commodities)
if commodity == "-- choose --":
    st.info("Select a commodity from the dropdown (processed CSV needed in `/data`).")
    st.stop()

# Load commodity data
chosen_file = [fp for fp in files if os.path.basename(fp).startswith(f"processed_{commodity}")][0]
df = load_commodity_df(chosen_file)

if "market_name" not in df.columns:
    st.error("`market_name` column missing in the processed CSV. Ensure preprocessing produced this column.")
    st.stop()

markets = sorted(df["market_name"].unique().tolist())
market = st.sidebar.selectbox("Market", ["-- choose --"] + markets)
if market == "-- choose --":
    st.info("Select a market from the dropdown.")
    st.stop()

# Date input
min_date = df["price_date"].min().date()
max_date = (df["price_date"].max().date() + pd.Timedelta(days=365))
target_date = st.sidebar.date_input("Target date", value=max_date, min_value=min_date, max_value=max_date)

# Evaluation options
eval_horizon = st.sidebar.number_input("Evaluation horizon (days holdout)", min_value=7, max_value=180, value=30, step=1)
run_cv = st.sidebar.checkbox("Enable Prophet cross-validation (slow)", value=False)
show_components = st.sidebar.checkbox("Show Prophet components plot", value=False)

# -----------------------
# Prepare series
# -----------------------
series = df[df["market_name"] == market].sort_values("price_date").reset_index(drop=True)
st.subheader(f"{commodity} @ {market}")
st.write(f"Rows: {len(series)} | Date range: {series['price_date'].min().date()} → {series['price_date'].max().date()}")

if series.empty:
    st.error("No data available for this commodity-market.")
    st.stop()

# If target date exists historically, show actual and stop
if target_date >= series["price_date"].min().date() and target_date <= series["price_date"].max().date():
    row = series[series["price_date"].dt.date == target_date]
    if not row.empty:
        actual = float(row.iloc[-1]["modal_price_rs_per_kg"])
        st.success(f"Actual modal price on {target_date}: Rs {actual:.2f} per kg")
        # plot small history with date marker
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(series["price_date"], series["modal_price_rs_per_kg"], label="History")
        ax.axvline(pd.to_datetime(target_date), color="red", linestyle="--", label="Target date")
        ax.set_ylabel("modal_price_rs_per_kg")
        ax.legend()
        st.pyplot(fig)
        st.stop()

# -----------------------
# Load / train model
# -----------------------
model_path = os.path.join(MODEL_FOLDER, f"prophet_{safe_name(commodity)}_{safe_name(market)}.joblib")

with st.spinner("Training or loading Prophet model..."):
    try:
        model = load_or_train_model(series, model_path)
    except Exception as e:
        st.error(f"Prophet model step failed: {e}")
        st.stop()

# -----------------------
# Forecast to target date
# -----------------------
prophet_df_full = series[["price_date", "modal_price_rs_per_kg"]].dropna().rename(
    columns={"price_date": "ds", "modal_price_rs_per_kg": "y"}
)
prophet_df_full["ds"] = pd.to_datetime(prophet_df_full["ds"])
prophet_df_full = prophet_df_full.sort_values("ds").reset_index(drop=True)

last_hist_date = prophet_df_full["ds"].max().date()
days_to_target = (target_date - last_hist_date).days
periods = max(days_to_target, 0)

future = model.make_future_dataframe(periods=periods, freq="D")
forecast = model.predict(future)
forecast["ds_date"] = forecast["ds"].dt.date

# pick row for exactly target_date or nearest
if target_date in forecast["ds_date"].values:
    row_fore = forecast[forecast["ds_date"] == target_date].iloc[0]
else:
    # pick nearest date (fallback)
    forecast["abs_diff_days"] = (forecast["ds"].dt.date - target_date).apply(lambda x: abs(x.days))
    row_fore = forecast.sort_values("abs_diff_days").iloc[0]

yhat = float(row_fore["yhat"])
yhat_lower = float(row_fore.get("yhat_lower", np.nan))
yhat_upper = float(row_fore.get("yhat_upper", np.nan))
pred_date_shown = pd.to_datetime(row_fore["ds"]).date()

st.markdown("### Prediction")
st.metric(label=f"Predicted modal price for {commodity} at {market} on {target_date}", value=f"Rs {yhat:.2f}/kg")
st.write(f"Prediction date shown: **{pred_date_shown}** (if exact target not available, nearest date used)")
if not np.isnan(yhat_lower):
    st.write(f"Confidence interval (approx): [{yhat_lower:.2f}, {yhat_upper:.2f}] per kg")

# -----------------------
# Plot history + forecast (up to shown date)
# -----------------------
fig, ax = plt.subplots(figsize=(12,4))
ax.plot(prophet_df_full["ds"], prophet_df_full["y"], label="History")
mask = forecast["ds"] <= pd.to_datetime(pred_date_shown)
ax.plot(forecast.loc[mask, "ds"], forecast.loc[mask, "yhat"], label="Forecast")
if "yhat_lower" in forecast.columns and "yhat_upper" in forecast.columns:
    ax.fill_between(forecast.loc[mask, "ds"],
                    forecast.loc[mask, "yhat_lower"],
                    forecast.loc[mask, "yhat_upper"],
                    alpha=0.2)
ax.axvline(pd.to_datetime(target_date), color="red", linestyle="--", label="Target date")
ax.set_ylabel("modal_price_rs_per_kg")
ax.legend()
st.pyplot(fig)

# -----------------------
# Holdout evaluation (train/test split) — robust to NaNs
# -----------------------
st.subheader("Model evaluation (holdout)")

prophet_df_eval = prophet_df_full.copy()
do_eval = len(prophet_df_eval) > eval_horizon + 10

if do_eval:
    train_df = prophet_df_eval.iloc[:-eval_horizon].copy()
    test_df = prophet_df_eval.iloc[-eval_horizon:].copy()

    # Train fresh model on train split (honest eval)
    with st.spinner("Computing holdout evaluation..."):
        m_eval = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
        m_eval.fit(train_df)
        future_eval = m_eval.make_future_dataframe(periods=eval_horizon, freq="D")
        fc_eval = m_eval.predict(future_eval)

        # Merge predictions with test set on date
        preds_eval = fc_eval[['ds','yhat','yhat_lower','yhat_upper']].merge(
            test_df[['ds','y']], on='ds', how='right'
        ).sort_values('ds')

    # Convert to numeric and drop rows with missing values
    preds_eval['yhat'] = pd.to_numeric(preds_eval['yhat'], errors='coerce')
    preds_eval['y'] = pd.to_numeric(preds_eval['y'], errors='coerce')
    before_drop = len(preds_eval)
    preds_eval = preds_eval.dropna(subset=['y','yhat']).reset_index(drop=True)
    after_drop = len(preds_eval)

    if after_drop == 0:
        st.warning("No valid evaluation rows after aligning predictions and test set (y or yhat are NaN). "
                   "Try reducing the eval horizon or check the series date alignment.")
    else:
        # compute metrics safely
        y_true = preds_eval['y'].values
        y_pred = preds_eval['yhat'].values

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-9, y_true))) * 100
        r2 = r2_score(y_true, y_pred)

        # show metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE (Rs/kg)", f"{mae:.3f}")
        col2.metric("RMSE (Rs/kg)", f"{rmse:.3f}")
        col3.metric("MAPE (%)", f"{mape:.2f}")
        col4.metric("R²", f"{r2:.3f}")

        # residuals and diagnostics plots
        preds_eval["residual"] = preds_eval["y"] - preds_eval["yhat"]
        fig2, ax2 = plt.subplots(1, 2, figsize=(12,4))
        ax2[0].plot(preds_eval["ds"], preds_eval["residual"], marker="o")
        ax2[0].axhline(0, color="k", linewidth=0.8)
        ax2[0].set_title("Residuals over time")
        ax2[0].set_xlabel("Date")
        ax2[0].set_ylabel("Residual (Actual - Pred)")

        ax2[1].scatter(preds_eval["yhat"], preds_eval["residual"], alpha=0.6)
        ax2[1].axhline(0, color="k", linewidth=0.8)
        ax2[1].set_xlabel("Predicted (yhat)")
        ax2[1].set_ylabel("Residual")
        ax2[1].set_title("Residuals vs Predictions")
        st.pyplot(fig2)

        # Practical accuracy bands: percent within +/-5% and +/-10%
        pct_errors = np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-9, y_true))
        within_5 = (pct_errors <= 0.05).mean() * 100
        within_10 = (pct_errors <= 0.10).mean() * 100
        st.write(f"Rows before drop: {before_drop}, after drop: {after_drop}")
        st.write(f"Percentage of holdout predictions within ±5%: **{within_5:.2f}%**, within ±10%: **{within_10:.2f}%**")

else:
    st.info("Not enough data for holdout evaluation with the chosen horizon. Reduce the horizon or use more historical data.")

# -----------------------
# Model details & components
# -----------------------
st.subheader("Model details")

try:
    st.write(f"- Daily seasonality: {model.daily_seasonality}")
    st.write(f"- Weekly seasonality: {model.weekly_seasonality}")
    st.write(f"- Yearly seasonality: {model.yearly_seasonality}")
    # changepoints
    cp = getattr(model, "changepoints", None)
    if cp is not None:
        st.write(f"- Number of changepoints: {len(cp)}")
        if len(cp) > 0 and st.checkbox("Show first 10 changepoints"):
            st.write(cp[:10].tolist())
    else:
        st.write("- Changepoints: unavailable")
except Exception:
    st.write("- Model meta information not available")

if show_components:
    try:
        with st.spinner("Rendering Prophet components..."):
            fig_comp = model.plot_components(forecast)
            st.pyplot(fig_comp)
    except Exception as e:
        st.error(f"Could not render components plot: {e}")

# -----------------------
# Optional: Prophet cross_validation (slow)
# -----------------------
if run_cv:
    try:
        from prophet.diagnostics import cross_validation, performance_metrics
        st.subheader("Prophet Cross-Validation (may be slow)")
        with st.spinner("Running Prophet cross-validation..."):
            initial = f"{int(len(prophet_df_full) * 0.6)} days"
            period = f"{max(1, int(len(prophet_df_full) * 0.1))} days"
            horizon = f"{eval_horizon} days"
            df_cv = cross_validation(model, initial=initial, period=period, horizon=horizon, parallel="processes")
            perf = performance_metrics(df_cv)
            st.write("Cross-validation performance (sample):")
            st.dataframe(perf.round(4))
    except Exception as e:
        st.error(f"Cross-validation failed or not available: {e}")

# -----------------------
# Optional: show forecast table
# -----------------------
if st.checkbox("Show forecast table (last 60 rows)"):
    st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(60).reset_index(drop=True))
