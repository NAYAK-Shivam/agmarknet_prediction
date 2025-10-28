import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from prophet import Prophet
from lightgbm import LGBMRegressor
from neuralforecast import NeuralForecast
from neuralforecast.models import TFT
from neuralforecast.utils import AirPassengersDF
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------
# Helper functions
# ------------------------------
def load_data(commodity):
    path = f"data/processed_{commodity}.csv"
    if not os.path.exists(path):
        st.error(f"❌ Data file not found for {commodity}")
        return None
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, mape, r2

# ------------------------------
# Prophet Model
# ------------------------------
def prophet_forecast(df, date):
    df_prophet = df.rename(columns={"Date": "ds", "Modal Price (Rs./Quintal)": "y"})
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    pred_row = forecast[forecast["ds"] == pd.to_datetime(date)]
    if pred_row.empty:
        pred_row = forecast.iloc[-1:]
    return pred_row["yhat"].values[0]

# ------------------------------
# LightGBM Model
# ------------------------------
def lightgbm_forecast(df, date):
    df["Date"] = pd.to_datetime(df["Date"])
    df["dayofyear"] = df["Date"].dt.dayofyear
    X = df[["dayofyear"]]
    y = df["Modal Price (Rs./Quintal)"]

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = LGBMRegressor(n_estimators=200, learning_rate=0.1)
    model.fit(X_train, y_train)

    mae, rmse, mape, r2 = evaluate_model(y_test, model.predict(X_test))

    dayofyear = pd.to_datetime(date).timetuple().tm_yday
    pred = model.predict(pd.DataFrame({"dayofyear": [dayofyear]}))[0]
    return pred, (mae, rmse, mape, r2)

# ------------------------------
# TFT Model
# ------------------------------
def tft_forecast(df, date):
    df_tft = df.rename(columns={"Date": "ds", "Modal Price (Rs./Quintal)": "y"})
    df_tft["unique_id"] = "series_1"
    df_tft = df_tft[["unique_id", "ds", "y"]]

    nf = NeuralForecast(models=[TFT(input_size=30, h=7, max_steps=100, scaler_type='robust')],
                        freq='D')
    nf.fit(df_tft)
    forecast = nf.predict()
    forecast.reset_index(inplace=True)
    pred_row = forecast[forecast["ds"] == pd.to_datetime(date)]
    if pred_row.empty:
        pred_row = forecast.iloc[-1:]
    return float(pred_row["TFT"].values[0])

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🌾 Agricultural Price Forecasting Dashboard")
st.write("Predict future market prices using multiple models (Prophet, LightGBM, or TFT).")

commodity = st.selectbox("Select Commodity", [f.split("_")[1].split(".")[0] for f in os.listdir("data") if f.startswith("processed_")])
market = st.text_input("Enter Market Name", "Harur(Uzhavar Sandhai)")
date = st.date_input("Select Date for Prediction", datetime(2025, 1, 30))
model_choice = st.selectbox("Select Model", ["Prophet", "LightGBM", "TFT"])

if st.button("🔮 Predict Price"):
    df = load_data(commodity)
    if df is not None:
        try:
            if model_choice == "Prophet":
                pred = prophet_forecast(df, date)
                st.success(f"📈 Prophet Predicted Price: Rs {pred:.2f}/kg")

            elif model_choice == "LightGBM":
                pred, metrics = lightgbm_forecast(df, date)
                mae, rmse, mape, r2 = metrics
                st.success(f"💡 LightGBM Predicted Price: Rs {pred:.2f}/kg")
                st.write(f"**MAE:** {mae:.3f} | **RMSE:** {rmse:.3f} | **MAPE:** {mape:.2f}% | **R²:** {r2:.3f}")

            elif model_choice == "TFT":
                pred = tft_forecast(df, date)
                st.success(f"🤖 TFT Predicted Price: Rs {pred:.2f}/kg")

        except Exception as e:
            st.error(f"❌ Error running model: {e}")
