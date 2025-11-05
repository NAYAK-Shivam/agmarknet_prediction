# app.py
import os
import glob
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from tqdm import tqdm
import openai
import json
import google.generativeai as genai
import streamlit as st

# Optional fallback
from lightgbm import LGBMRegressor

# === Import your base forecasting utilities ===
from model_utils import (
    train_prophet_model_meta, predict_prophet_meta,
    train_lightgbm_model_meta, iterative_forecast_lgb_meta,
    train_sarima_model_meta, predict_sarima_meta
)

# Try to import Prophet and LightGBM
try:
    from prophet import Prophet
    prophet_available = True
except Exception:
    prophet_available = False

try:
    import lightgbm as lgb
    lgb_available = True
except Exception:
    lgb_available = False

try:
    import statsmodels.api as sm
    sarima_available = True
except Exception:
    sarima_available = False

# -----------------------
# CONFIG
# -----------------------
DATA_FOLDER = "data"            # place processed_*.csv here
MODEL_FOLDER = "models"         # model cache
os.makedirs(MODEL_FOLDER, exist_ok=True)

DATA_DIR = "data"
META_DIR = "meta_data"
META_MODEL_DIR = "meta_models"
os.makedirs(META_DIR, exist_ok=True)
os.makedirs(META_MODEL_DIR, exist_ok=True)

st.set_page_config(page_title="Market Price Predictor", layout="wide")
st.title("🌾 Agri Price Forecasting (Meta-Learner Integrated)")
st.markdown("Dynamically generate, train and forecast for selected commodity and market.")

# -----------------------
# Helpers & cached loaders
# -----------------------
def safe_name(s: str):
    return str(s).strip().replace(" ", "_").replace("/", "_")

@st.cache_data
def list_processed_files():
    files = sorted(glob.glob(os.path.join(DATA_FOLDER, "processed_*.csv")))
    commodities = [os.path.basename(fp).replace("processed_", "").replace(".csv", "") for fp in files]
    return files, commodities

@st.cache_data
def load_csv(fp):
    return pd.read_csv(fp, parse_dates=["price_date"])


# === Helper: Sliding Window Simulation ===
@st.cache_data(show_spinner=False)
def generate_meta_training_data_for_market(df, horizon, series_key):
    """
    Perform sliding window forecasting for a single time series (one market).
    Uses precomputed features already in CSV.
    Retrains models at each step.
    """
    results = []

    df = df.copy()
    df["price_date"] = pd.to_datetime(df["price_date"])
    df = df.sort_values("price_date").reset_index(drop=True)

    target_col = "modal_price_rs_per_kg"

    # Features are already computed in your processed CSVs
    feature_cols = [
        "month", "day_of_week", "is_weekend",
        "roll_mean_7", "roll_std_14",
        "lag_1", "lag_7", "lag_30"
    ]

    min_train_size = 90  # can reduce if small dataset

    for end_idx in tqdm(range(min_train_size, len(df) - horizon),
                        desc=f"{series_key}-H{horizon}", leave=False):
        train_df = df.iloc[:end_idx].copy()
        test_df = df.iloc[end_idx : end_idx + horizon].copy()
        if test_df.empty:
            continue

        forecast_date = test_df["price_date"].iloc[-1]

        # --- Prophet ---
        try:
            m_prophet = train_prophet_model_meta(train_df)
            pred_prophet_df = predict_prophet_meta(m_prophet, train_df, forecast_date.date())
            prophet_pred = float(pred_prophet_df["prophet_pred"].iloc[-1])
        except Exception:
            prophet_pred = np.nan

        # --- LightGBM ---
        try:
            m_lgb, _ = train_lightgbm_model_meta(train_df, feature_cols)
            pred_lgb_list = iterative_forecast_lgb_meta(m_lgb, train_df, horizon, feature_cols)
            lgbm_pred = float(pred_lgb_list[-1][1])
        except Exception:
            lgbm_pred = np.nan

        # --- SARIMA ---
        try:
            m_sarima = train_sarima_model_meta(train_df)
            pred_sarima_df = predict_sarima_meta(m_sarima, train_df, forecast_date.date())
            sarima_pred = float(pred_sarima_df["sarima_pred"].iloc[-1])

        except Exception:
            sarima_pred = np.nan

        # --- Actual ---
        actual_value = df.loc[df["price_date"] == forecast_date, target_col].values
        actual_value = float(actual_value[0]) if len(actual_value) else np.nan

        # --- Context features ---
        row = df.loc[df["price_date"] == forecast_date, feature_cols]
        if not row.empty:
            row_dict = row.iloc[0].to_dict()
        else:
            row_dict = {col: np.nan for col in feature_cols}

        results.append({
            "series_key": series_key,
            "price_date": forecast_date,
            "prophet_pred": prophet_pred,
            "lgbm_pred": lgbm_pred,
            "sarima_pred": sarima_pred,
            "actual": actual_value,
            **row_dict
        })

    meta_df = pd.DataFrame(results)
    meta_df = meta_df[
        ["series_key", "price_date", "prophet_pred", "lgbm_pred", "sarima_pred", "actual",
         "month", "day_of_week", "is_weekend",
         "roll_mean_7", "roll_std_14", "lag_1", "lag_7", "lag_30"]
    ]
    return meta_df

# -------------------------------------------------------------------
# === Meta Learner Training ===
# -------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_meta_learner(meta_df, commodity, market, horizon):
    X = meta_df[["prophet_pred", "lgbm_pred", "sarima_pred",
                 "month", "day_of_week", "is_weekend",
                 "roll_mean_7", "roll_std_14", "lag_1", "lag_7", "lag_30"]]
    y = meta_df["actual"]

    # Clean NaNs
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]

    # Try Ridge first (robust for small data)
    try:
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        algo_used = "Ridge Regression"
    except Exception:
        model = LGBMRegressor(n_estimators=200, learning_rate=0.05)
        model.fit(X, y)
        algo_used = "LightGBM Regressor"

    # Save model
    model_path = os.path.join(
        META_MODEL_DIR, f"meta_model_{commodity}_{market}_h{horizon}.pkl"
    )
    joblib.dump(model, model_path)

    return model, algo_used, model_path

@st.cache_resource
def train_prophet_model(series_df, model_path):
    if not prophet_available:
        raise RuntimeError("Prophet not installed.")
    # try load
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception:
            pass
    df = series_df[["price_date", "modal_price_rs_per_kg"]].dropna().rename(columns={"price_date":"ds","modal_price_rs_per_kg":"y"})
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(df)
    try:
        joblib.dump(m, model_path)
    except Exception:
        pass
    return m

def train_lightgbm_model(df, feature_cols, target_col="modal_price_rs_per_kg", min_rows=50, horizon_days=30):
    if not lgb_available:
        raise RuntimeError("LightGBM not installed.")
    df = df.sort_values("price_date").reset_index(drop=True).copy()
    # ensure features exist
    present_feats = [c for c in feature_cols if c in df.columns]
    if len(present_feats) == 0:
        raise ValueError("No LightGBM feature columns found in data. Check preprocessing.")
    # drop rows lacking required features/target
    df_train = df.dropna(subset=present_feats + [target_col]).copy()
    if len(df_train) < min_rows:
        raise ValueError(f"Not enough rows to train LightGBM ({len(df_train)} < {min_rows})")
    # train/test split (last horizon_days as test)
    if len(df_train) <= horizon_days + 1:
        raise ValueError("Not enough rows for holdout; increase data or reduce horizon.")
    train = df_train.iloc[:-horizon_days]
    test = df_train.iloc[-horizon_days:]
    X_train = train[present_feats].astype(float)
    y_train = train[target_col].astype(float)
    X_test = test[present_feats].astype(float)
    y_test = test[target_col].astype(float)
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_test, label=y_test, reference=dtrain)
    params = {
        "objective":"regression",
        "metric":"rmse",
        "boosting_type":"gbdt",
        "learning_rate":0.05,
        "num_leaves":31,
        "verbose": -1,
        "seed": 42
    }
    # Train with compatibility for new/old LightGBM
    try:
        model = lgb.train(params, dtrain, num_boost_round=1000, valid_sets=[dvalid],
                          early_stopping_rounds=50, verbose_eval=False)
    except TypeError:
        # new API: pass early stopping as callback
        callbacks = [lgb.early_stopping(50)]
        model = lgb.train(params, dtrain, num_boost_round=1000, valid_sets=[dvalid], callbacks=callbacks)
    # Save model (optional)
    # return model + test predictions + metrics
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test == 0, 1e-9, y_test))) * 100
    r2 = r2_score(y_test, y_pred)
    metrics = {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2, "before": len(test)+len(train), "after": len(y_test)}
    results_df = pd.DataFrame({"price_date": test["price_date"].values, "y": y_test.values, "yhat": y_pred})
    return model, results_df, metrics

@st.cache_resource
def train_sarima_model(series_df, model_path, order=(1,1,1), seasonal_order=(1,1,1,7)):
    if not sarima_available:
        raise RuntimeError("statsmodels not installed.")
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception:
            pass
    y = series_df["modal_price_rs_per_kg"].dropna().values
    if len(y) < 30:
        raise ValueError("Not enough data to train TFT model.")
    model = sm.tsa.statespace.SARIMAX(
        y, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False
    )
    results = model.fit(disp=False)
    try:
        joblib.dump(results, model_path)
    except Exception:
        pass
    return results

def iterative_forecast_lgb(model, recent_df, days_ahead, feature_cols, target_col="modal_price_rs_per_kg"):
    """Simple iterative forecasting using predicted target as lag input for next steps."""
    df_buf = recent_df.copy().sort_values("price_date").reset_index(drop=True)
    preds = []
    for _ in range(days_ahead):
        next_date = df_buf["price_date"].max() + pd.Timedelta(days=1)
        row = {}
        # compute simple lags from df_buf
        for lag in [1,7,30]:
            col = f"lag_{lag}"
            if col in feature_cols:
                idx = df_buf[df_buf["price_date"] == (next_date - pd.Timedelta(days=lag))]
                if not idx.empty:
                    row[col] = float(idx.iloc[0][target_col])
                else:
                    row[col] = float(df_buf[target_col].iloc[-1])
        if "roll_mean_7" in feature_cols:
            row["roll_mean_7"] = float(df_buf[target_col].tail(7).mean()) if len(df_buf) >= 1 else float(df_buf[target_col].iloc[-1])
        if "roll_std_14" in feature_cols:
            row["roll_std_14"] = float(df_buf[target_col].tail(14).std()) if len(df_buf) >= 2 else 0.0
        # temporal
        row["day"] = next_date.day
        row["month"] = next_date.month
        row["day_of_week"] = next_date.dayofweek
        row["is_weekend"] = int(next_date.dayofweek in [5,6])
        # categorical encodings if present
        for cat in ["market_name_enc","commodity_enc","district_name_enc"]:
            if cat in feature_cols and cat in df_buf.columns:
                row[cat] = int(df_buf[cat].iloc[-1])
        # create dataframe with feature_cols order
        feat_row = pd.DataFrame([{c: row.get(c, np.nan) for c in feature_cols}])
        feat_row = feat_row.astype(float)
        pred = model.predict(feat_row)[0]
        # append to buffer
        new = { "price_date": next_date, target_col: pred }
        for k,v in row.items():
            new[k] = v
        df_buf = pd.concat([df_buf, pd.DataFrame([new])], ignore_index=True, sort=False)
        preds.append((next_date.date(), float(pred)))
    return preds

# -----------------------
# UI: file & commodity selection
# -----------------------
files, available_commodities = list_processed_files()
if not files:
    st.error(f"No processed_*.csv files found in `{DATA_FOLDER}` folder. Put your per-commodity processed CSVs there.")
    st.stop()

st.sidebar.header("Inputs")
commodity = st.sidebar.selectbox("Commodity", ["-- choose --"] + available_commodities)
if commodity == "-- choose --":
    st.stop()

# safe case-insensitive file matching
matching_files = [fp for fp in files if os.path.basename(fp).lower().startswith(f"processed_{commodity.lower()}")]
if not matching_files:
    st.error(f"No processed file found for commodity '{commodity}'. Files found: {files}")
    st.stop()
chosen_fp = matching_files[0]
df = load_csv(chosen_fp)
if "market_name" not in df.columns:
    st.error("processed CSV must contain column 'market_name'.")
    st.stop()

markets = sorted(df["market_name"].astype(str).str.title().unique().tolist())
market = st.sidebar.selectbox("Market", ["-- choose --"] + markets)
if market == "-- choose --":
    st.stop()

# options
model_choice = st.sidebar.selectbox("Model", ["Prophet", "LightGBM", "TFT", "All", "Meta-Learner"])
eval_horizon = st.sidebar.number_input("Evaluation horizon (days holdout)", min_value=7, max_value=180, value=30, step=1)
cap_ci_zero = st.sidebar.checkbox("Cap Prophet CI lower bound at 0 (display)", value=True)
target_date = st.sidebar.date_input("Target date", value=(df["price_date"].max().date() + pd.Timedelta(days=30)),
                                   min_value=df["price_date"].min().date(),
                                   max_value=(df["price_date"].max().date() + pd.Timedelta(days=365)))

# prepare series for chosen market
series = df[df["market_name"].astype(str).str.title() == market].sort_values("price_date").reset_index(drop=True)
st.subheader(f"{commodity} @ {market}")
st.write(f"Rows: {len(series)} | Date range: {series['price_date'].min().date()} → {series['price_date'].max().date()}")

if series.empty:
    st.error("No data for this market.")
    st.stop()

# Automatically detect feature columns if present (lags / rolls / enc)
possible_feat_cols = [c for c in series.columns if c.startswith("lag_") or c.startswith("roll_") or c.endswith("_enc") or c in ["day","month","day_of_week","is_weekend"]]
# sensible default order
FEATURE_COLS = [c for c in ["lag_1","lag_7","lag_30","roll_mean_7","roll_std_14","day","month","day_of_week","is_weekend","market_name_enc","commodity_enc","district_name_enc"] if c in possible_feat_cols]

# -----------------------
# Prophet path
# -----------------------
prophet_pred = None
prophet_forecast = None
prophet_model = None
if model_choice in ("Prophet","Both"):
    if not prophet_available:
        st.warning("Prophet not installed. Install 'prophet' to use this model.")
    else:
        p_model_path = os.path.join(MODEL_FOLDER, f"prophet_{safe_name(commodity)}_{safe_name(market)}.joblib")
        with st.spinner("Training/loading Prophet..."):
            try:
                prophet_model = train_prophet_model(series, p_model_path)
                last_hist = series["price_date"].max().date()
                days_to_target = (target_date - last_hist).days
                periods = max(days_to_target, 0)
                future = prophet_model.make_future_dataframe(periods=periods, freq="D")
                prophet_forecast = prophet_model.predict(future)
                prophet_forecast["ds_date"] = prophet_forecast["ds"].dt.date
                if target_date in prophet_forecast["ds_date"].values:
                    r = prophet_forecast[prophet_forecast["ds_date"] == target_date].iloc[0]
                else:
                    prophet_forecast["abs_diff"] = (prophet_forecast["ds"].dt.date - target_date).apply(lambda x: abs(x.days))
                    r = prophet_forecast.sort_values("abs_diff").iloc[0]
                yhat_p = float(r["yhat"])
                lower_p = float(r.get("yhat_lower", np.nan))
                upper_p = float(r.get("yhat_upper", np.nan))
                if cap_ci_zero and not np.isnan(lower_p):
                    lower_p = max(0.0, lower_p)
                prophet_pred = {"date": pd.to_datetime(r["ds"]).date(), "yhat": yhat_p, "lower": lower_p, "upper": upper_p}
            except Exception as e:
                st.error(f"Prophet error: {e}")
                prophet_model = None

# -----------------------
# LightGBM path
# -----------------------
lgb_pred = None
lgb_model = None
if model_choice in ("LightGBM","Both"):
    if not lgb_available:
        st.warning("LightGBM not installed. Install 'lightgbm' to use this model.")
    else:
        lgb_model_path = os.path.join(MODEL_FOLDER, f"lgb_{safe_name(commodity)}_{safe_name(market)}.joblib")
        # train or load LGB model (train with detected FEATURE_COLS)
        try:
            lgb_model, lgb_results_df, lgb_metrics = None, None, None
            # Attempt to train (will raise helpful errors if insufficient features/data)
            with st.spinner("Training LightGBM (uses lag/rolling features)..."):
                lgb_model, lgb_results_df, lgb_metrics = train_lightgbm_model(series, FEATURE_COLS, target_col="modal_price_rs_per_kg", min_rows=40, horizon_days=eval_horizon)
                # save model
                try:
                    joblib.dump(lgb_model, lgb_model_path)
                except Exception:
                    pass
        except Exception as e:
            st.write("LightGBM training warning/error:", e)
            lgb_model = None
            lgb_results_df = None
            lgb_metrics = None

        # Prediction using LGB
        if lgb_model is not None:
            last_hist_date = series["price_date"].max().date()
            if target_date <= last_hist_date:
                # return actual if exists
                exact = series[series["price_date"].dt.date == target_date]
                if not exact.empty:
                    lgb_pred = {"date": target_date, "yhat": float(exact.iloc[-1]["modal_price_rs_per_kg"]), "lower": np.nan, "upper": np.nan, "note": "actual"}
                else:
                    # if features for that date exist, try direct prediction
                    feat_row = series[series["price_date"].dt.date == target_date]
                    if not feat_row.empty and set(FEATURE_COLS).issubset(feat_row.columns):
                        X = feat_row[FEATURE_COLS].astype(float)
                        yhat = float(lgb_model.predict(X)[0])
                        lgb_pred = {"date": target_date, "yhat": yhat, "lower": np.nan, "upper": np.nan}
                    else:
                        # fallback: use iterative forecast for 1 day from last
                        preds = iterative_forecast_lgb(lgb_model, series, (target_date - last_hist_date).days if target_date>last_hist_date else 1, FEATURE_COLS, target_col="modal_price_rs_per_kg")
                        if preds:
                            lgb_pred = {"date": preds[-1][0], "yhat": preds[-1][1], "lower": np.nan, "upper": np.nan}
                        else:
                            lgb_pred = None
            else:
                days_ahead = (target_date - last_hist_date).days
                try:
                    preds = iterative_forecast_lgb(lgb_model, series, days_ahead, FEATURE_COLS, target_col="modal_price_rs_per_kg")
                    lgb_pred = {"date": preds[-1][0], "yhat": preds[-1][1], "lower": np.nan, "upper": np.nan}
                except Exception as e:
                    st.write("Iterative LGB forecast failed:", e)
                    lgb_pred = None

# -----------------------
# SARIMA (TFT) path
# -----------------------
sarima_pred = None
sarima_model = None
if model_choice in ("TFT", "All"):
    if not sarima_available:
        st.warning("statsmodels not installed. Install 'statsmodels' to use TFT.")
    else:
        sarima_model_path = os.path.join(MODEL_FOLDER, f"sarima_{safe_name(commodity)}_{safe_name(market)}.joblib")
        with st.spinner("Training/loading TFT..."):
            try:
                sarima_model = train_sarima_model(series, sarima_model_path)
                y = series["modal_price_rs_per_kg"].dropna().values
                last_hist_date = series["price_date"].max().date()
                steps = (target_date - last_hist_date).days
                if steps <= 0:
                    steps = 1
                forecast = sarima_model.get_forecast(steps=steps)
                fc_mean = np.array(forecast.predicted_mean)
                conf_int = np.array(forecast.conf_int(alpha=0.05))
                yhat_s = float(fc_mean[-1])
                lower_s = float(conf_int[-1, 0])
                upper_s = float(conf_int[-1, 1])
                sarima_pred = {"date": target_date, "yhat": yhat_s, "lower": lower_s, "upper": upper_s}
            except Exception as e:
                st.error(f"TFT error: {e}")
                sarima_model = None

# ==========================================
# 🧠 META-LEARNER ENSEMBLE MODEL
# ==========================================
if model_choice == "Meta-Learner":
    st.subheader("🧠 Meta-Learner (Ensemble Model)")
    
    series_key = f"{commodity}___{market}"
    meta_csv_path = os.path.join(META_DIR, f"meta_{commodity}_{market}_h{eval_horizon}.csv")
    meta_model_path = os.path.join(META_MODEL_DIR, f"meta_model_{commodity}_{market}_h{eval_horizon}.pkl")

    # --- Step 1: Generate Meta-Training Data ---
    if not os.path.exists(meta_csv_path):
        with st.spinner(f"⏳ Generating meta-training data for {series_key}..."):
            meta_df = generate_meta_training_data_for_market(series, eval_horizon, series_key)
            meta_df.to_csv(meta_csv_path, index=False)
            st.success(f"Meta-training data generated and saved → {meta_csv_path}")
    else:
        st.info("📁 Using cached meta-training data")
        meta_df = pd.read_csv(meta_csv_path, parse_dates=["price_date"])

    # --- Step 2: Train Meta-Learner (Ridge/LightGBM fallback) ---
    if not os.path.exists(meta_model_path):
        with st.spinner("🧩 Training meta-learner model..."):
            model_meta, algo_used, model_path = train_meta_learner(meta_df, commodity, market, eval_horizon)
            st.success(f"Meta-Learner trained using {algo_used} and saved → {model_path}")
    else:
        model_meta = joblib.load(meta_model_path)
        st.info("📁 Loaded existing meta-learner model")

    # --- Step 3: Predict and Evaluate Meta-Learner ---
    X = meta_df[["prophet_pred", "lgbm_pred", "sarima_pred",
                 "month", "day_of_week", "is_weekend",
                 "roll_mean_7", "roll_std_14", "lag_1", "lag_7", "lag_30"]]
    y = meta_df["actual"]

    # Drop NaNs for safety
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]

    y_pred_meta = model_meta.predict(X)

    rmse = np.sqrt(mean_squared_error(y, y_pred_meta))
    mae = mean_absolute_error(y, y_pred_meta)
    r2 = r2_score(y, y_pred_meta)

    st.markdown("### 📊 Ensemble Model Performance")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**R² Score:** {r2:.3f}")


    # --- Step 4: Plot Comparison ---
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(meta_df["price_date"], meta_df["actual"], label="Actual", color="black", linewidth=2)
    ax.plot(meta_df["price_date"], meta_df["prophet_pred"], label="Prophet", linestyle="--", alpha=0.6)
    ax.plot(meta_df["price_date"], meta_df["lgbm_pred"], label="LightGBM", linestyle="--", alpha=0.6)
    ax.plot(meta_df["price_date"], meta_df["sarima_pred"], label="TFT", linestyle="--", alpha=0.6)
    ax.plot(meta_df["price_date"], y_pred_meta, label="Meta-Learner Ensemble", color="blue", linewidth=2.5)
    ax.set_title(f"{commodity} – {market} (Horizon: {eval_horizon})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)

    # --- Step 5: Display sample predictions ---
    display_df = meta_df[["price_date", "actual", "prophet_pred", "lgbm_pred", "sarima_pred"]].copy()
    display_df["meta_pred"] = y_pred_meta
    st.markdown("### 🧾 Sample Meta Predictions")
    st.dataframe(display_df.tail(15))


# -----------------------
# Display predictions
# -----------------------
st.markdown("## Predictions")
col1, col2 = st.columns(2)
if prophet_pred is not None:
    col1.metric(f"Prophet → {commodity} @ {market} ({prophet_pred['date']})", f"Rs {prophet_pred['yhat']:.2f}/kg")
    col1.write(f"CI: [{prophet_pred['lower']:.2f}, {prophet_pred['upper']:.2f}]")
else:
    col1.write("Prophet: unavailable")

if lgb_pred is not None:
    col2.metric(f"LightGBM → {commodity} @ {market} ({lgb_pred['date']})", f"Rs {lgb_pred['yhat']:.2f}/kg")
else:
    col2.write("LightGBM: unavailable")

if sarima_pred is not None:
    st.metric(f"TFT → {commodity} @ {market} ({sarima_pred['date']})", f"Rs {sarima_pred['yhat']:.2f}/kg")
    st.write(f"CI: [{sarima_pred['lower']:.2f}, {sarima_pred['upper']:.2f}]")
else:
    if model_choice in ("SARIMA (TFT)", "All"):
        st.write("SARIMA (TFT): unavailable")

# comparison
if model_choice == "Both" and prophet_pred is not None and lgb_pred is not None:
    diff = lgb_pred["yhat"] - prophet_pred["yhat"]
    pct = diff / prophet_pred["yhat"] * 100 if prophet_pred["yhat"] != 0 else np.nan
    st.write(f"Difference (LGB - Prophet): Rs {diff:.2f} ({pct:.2f}%)")

# -----------------------
# Plotting
# -----------------------
def plot_history(history_df, forecast_df=None, pred_date=None, model_label="Model"):
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(history_df["price_date"], history_df["modal_price_rs_per_kg"], label="History")
    if forecast_df is not None:
        mask = forecast_df["ds"] <= pd.to_datetime(pred_date)
        ax.plot(forecast_df.loc[mask,"ds"], forecast_df.loc[mask,"yhat"], label=f"{model_label} forecast")
        if "yhat_lower" in forecast_df.columns and "yhat_upper" in forecast_df.columns:
            ax.fill_between(forecast_df.loc[mask,"ds"], forecast_df.loc[mask,"yhat_lower"], forecast_df.loc[mask,"yhat_upper"], alpha=0.2)
    ax.axvline(pd.to_datetime(pred_date), color="red", linestyle="--", label="Target date")
    ax.set_ylabel("modal_price_rs_per_kg")
    ax.legend()
    st.pyplot(fig)

if prophet_forecast is not None and prophet_pred is not None:
    st.subheader("Prophet: History + Forecast")
    plot_history(series, prophet_forecast, prophet_pred["date"], model_label="Prophet")

if lgb_pred is not None:
    st.subheader("LightGBM: History + Predicted Point")
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(series["price_date"], series["modal_price_rs_per_kg"], label="History")
    ax.scatter(pd.to_datetime(lgb_pred["date"]), lgb_pred["yhat"], color="orange", label="LGB prediction")
    ax.axvline(pd.to_datetime(lgb_pred["date"]), color="red", linestyle="--", label="Target")
    ax.set_ylabel("modal_price_rs_per_kg")
    ax.legend()
    st.pyplot(fig)

if sarima_pred is not None:
    st.subheader("TFT: History + Forecast")
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(series["price_date"], series["modal_price_rs_per_kg"], label="History")
    ax.axvline(pd.to_datetime(sarima_pred["date"]), color="red", linestyle="--", label="Target date")
    ax.scatter(pd.to_datetime(sarima_pred["date"]), sarima_pred["yhat"], color="green", label="TFT forecast")
    ax.fill_between(
        [pd.to_datetime(sarima_pred["date"])],
        [sarima_pred["lower"]],
        [sarima_pred["upper"]],
        color="green", alpha=0.2, label="95% CI"
    )
    ax.legend()
    st.pyplot(fig)

# -----------------------
# Evaluation: holdout metrics for each model (robust, drops NaNs)
# -----------------------
st.subheader("Model Evaluation (Holdout)")

def safe_metrics_from_preds(df_preds):
    df = df_preds.dropna(subset=["y","yhat"]).copy()
    if df.empty:
        return None
    y_true = df["y"].values
    y_pred = df["yhat"].values
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true==0, 1e-9, y_true))) * 100
    r2 = r2_score(y_true, y_pred)
    within5 = (np.abs((y_true - y_pred)/np.where(y_true==0,1e-9,y_true)) <= 0.05).mean()*100
    within10 = (np.abs((y_true - y_pred)/np.where(y_true==0,1e-9,y_true)) <= 0.10).mean()*100
    return {"mae":mae,"rmse":rmse,"mape":mape,"r2":r2,"within5":within5,"within10":within10,"n":len(df)}

if model_choice in ("Prophet","Both") and prophet_model is not None:
    # create honest holdout eval for Prophet
    df_prophet = series[["price_date","modal_price_rs_per_kg"]].dropna().rename(columns={"price_date":"ds","modal_price_rs_per_kg":"y"})
    df_prophet = df_prophet.sort_values("ds").reset_index(drop=True)
    if len(df_prophet) > eval_horizon + 10:
        train = df_prophet.iloc[:-eval_horizon]
        test = df_prophet.iloc[-eval_horizon:]
        m_eval = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
        m_eval.fit(train)
        fut = m_eval.make_future_dataframe(periods=eval_horizon, freq="D")
        fc = m_eval.predict(fut)
        preds_eval = fc[["ds","yhat"]].merge(test[["ds","y"]], on="ds", how="right").sort_values("ds").rename(columns={"ds":"price_date"})
        pm = safe_metrics_from_preds(preds_eval.rename(columns={"price_date":"ds"}).rename(columns={"ds":"ds","y":"y","yhat":"yhat"}).rename(columns={"ds":"price_date"}))
        if pm:
            col1,col2,col3,col4 = st.columns(4)
            col1.metric("Prophet MAE", f"{pm['mae']:.3f}")
            col2.metric("Prophet RMSE", f"{pm['rmse']:.3f}")
            col3.metric("Prophet MAPE%", f"{pm['mape']:.2f}")
            col4.metric("Prophet R2", f"{pm['r2']:.3f}")
            st.write(f"Prophet eval points: {pm['n']}. Within ±5%: {pm['within5']:.2f}%, ±10%: {pm['within10']:.2f}%")
        else:
            st.info("Prophet: No valid rows after alignment for holdout metrics.")
    else:
        st.info("Prophet: not enough historical rows for holdout evaluation.")

if model_choice in ("LightGBM","Both") and lgb_model is not None:
    # we already trained a lgb model and have lgb_results_df, lgb_metrics if successful
    try:
        if 'lgb_results_df' in locals() and lgb_results_df is not None:
            lm = safe_metrics_from_preds(lgb_results_df.rename(columns={"price_date":"price_date","y":"y","yhat":"yhat"}))
            if lm:
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("LGB MAE", f"{lm['mae']:.3f}")
                c2.metric("LGB RMSE", f"{lm['rmse']:.3f}")
                c3.metric("LGB MAPE%", f"{lm['mape']:.2f}")
                c4.metric("LGB R2", f"{lm['r2']:.3f}")
                st.write(f"LGB eval points: {lm['n']}. Within ±5%: {lm['within5']:.2f}%, ±10%: {lm['within10']:.2f}%")
            else:
                st.info("LightGBM: No valid rows for evaluation after alignment.")
        else:
            st.info("LightGBM: no evaluation results available (training may have failed).")
    except Exception as e:
        st.write("LightGBM evaluation error:", e)

if model_choice in ("TFT", "All") and sarima_model is not None:
    try:
        y = series["modal_price_rs_per_kg"].dropna().values
        if len(y) > eval_horizon + 10:
            train_y = y[:-eval_horizon]
            test_y = y[-eval_horizon:]
            m_eval = sm.tsa.statespace.SARIMAX(train_y, order=(1,1,1), seasonal_order=(1,1,1,7),
                                               enforce_stationarity=False, enforce_invertibility=False)
            r_eval = m_eval.fit(disp=False)
            fc_eval = r_eval.get_forecast(steps=eval_horizon)
            yhat_eval = fc_eval.predicted_mean
            mae = mean_absolute_error(test_y, yhat_eval)
            rmse = np.sqrt(mean_squared_error(test_y, yhat_eval))
            mape = np.mean(np.abs((test_y - yhat_eval) / np.where(test_y==0, 1e-9, test_y))) * 100
            r2 = r2_score(test_y, yhat_eval)
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("TFT MAE", f"{mae:.3f}")
            c2.metric("TFT RMSE", f"{rmse:.3f}")
            c3.metric("TFT MAPE%", f"{mape:.2f}")
            c4.metric("TFT R2", f"{r2:.3f}")
        else:
            st.info("TFT: not enough historical rows for holdout evaluation.")
    except Exception as e:
        st.error(f"TFT evaluation error: {e}")

# -----------------------
# Optional displays
# -----------------------
if st.checkbox("Show raw series sample (first 60 rows)"):
    st.dataframe(series.head(60))

if st.checkbox("Show detected feature columns"):
    st.write("Detected feature columns used for LightGBM:", FEATURE_COLS)

if st.checkbox("Show Prophet forecast tail"):
    if prophet_forecast is not None:
        st.dataframe(prophet_forecast[["ds","yhat","yhat_lower","yhat_upper"]].tail(60))
    else:
        st.info("No Prophet forecast available.")

def generate_forecast_summary(commodity, market, horizon):
    """
    Load meta and processed CSVs, include meta-learner blended prediction,
    and build a structured summary for LLM interpretation.
    """
    try:
        meta_path = f"meta_data/meta_{commodity}_{market}_h{horizon}.csv"
        proc_path = f"data/processed_{commodity}.csv"
        meta_model_path = f"meta_models/meta_model_{commodity}_{market}_h{horizon}.pkl"

        meta_df = pd.read_csv(meta_path)
        proc_df = pd.read_csv(proc_path)
        meta_df = meta_df.sort_values("price_date").reset_index(drop=True)
        proc_df = proc_df.sort_values("price_date").reset_index(drop=True)

        latest_meta = meta_df.tail(1)
        latest_date = latest_meta["price_date"].iloc[0]
        latest_actual = float(latest_meta["actual"].iloc[0])

        # === Compute meta-learner ensemble prediction ===
        try:
            model = joblib.load(meta_model_path)
            feature_cols = [
                "prophet_pred", "lgbm_pred", "sarima_pred",
                "month", "day_of_week", "is_weekend",
                "roll_mean_7", "roll_std_14", "lag_1", "lag_7", "lag_30"
            ]
            X_latest = latest_meta[feature_cols]
            meta_pred = float(model.predict(X_latest)[0])
        except Exception as e:
            meta_pred = np.nan

        # === Compute average of base models ===
        avg_pred = latest_meta[["prophet_pred", "lgbm_pred", "sarima_pred"]].mean(axis=1).iloc[0]

        summary = {
            "commodity": commodity,
            "market": market,
            "horizon_days": horizon,
            "latest_date": latest_date,
            "meta_ensemble_forecast": round(meta_pred, 2) if pd.notnull(meta_pred) else None,
            "average_base_forecast": round(avg_pred, 2),
            "actual_latest": round(latest_actual, 2),
            "change_vs_actual": round(meta_pred - latest_actual, 2) if pd.notnull(meta_pred) else None,
            "context_features": {
                "roll_mean_7": float(latest_meta["roll_mean_7"].iloc[0]),
                "roll_std_14": float(latest_meta["roll_std_14"].iloc[0]),
                "lag_1": float(latest_meta["lag_1"].iloc[0]),
                "lag_7": float(latest_meta["lag_7"].iloc[0]),
                "is_weekend": int(latest_meta["is_weekend"].iloc[0]),
                "day_of_week": int(latest_meta["day_of_week"].iloc[0]),
            }
        }

        return summary

    except Exception as e:
        return {"error": str(e)}
# =====================================================
# 🤖 LLM-Driven Insight Chatbot Integration (Gemini + Ollama)
# =====================================================
import google.generativeai as genai
import ollama
import json
import streamlit as st

# --- Configure Gemini (if key available)
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    gemini_available = True
else:
    gemini_available = False

# --- Define Gemini Insight Function
def get_gemini_insight(prompt):
    try:
        model = genai.GenerativeModel("gemini-2.0-flash-001")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise RuntimeError(f"Gemini error: {e}")

# --- Define Ollama (Local) Insight Function
def get_local_insight(prompt):
    try:
        response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Ollama error: {e}")

# --- Chatbot UI ---
st.markdown("---")
st.header("🤖 AI Insight Chatbot")

user_query = st.text_input("Ask about price trends, forecasts, or comparisons:")

if user_query and commodity and market:
    with st.spinner("Analyzing trends..."):
        try:
            # 1️⃣ Generate structured summary from your forecast data
            forecast_summary = generate_forecast_summary(
                commodity.replace(".csv", ""), market, eval_horizon
            )
            summary_text = json.dumps(forecast_summary, indent=2)

            # 2️⃣ Construct unified prompt
            prompt = f"""
            You are AgriAI, an intelligent assistant for agricultural forecasting.

            Forecast Summary:
            {summary_text}

            User Query:
            {user_query}

            Based on the ensemble model results and contextual trends, provide:
            - A concise insight into the forecasted price trend.
            - Mention relevant seasonal, lag, or volatility indicators.
            - Offer a practical recommendation if applicable (e.g., sell delay, stock up, etc.).

            Explain:"
            - Expected price direction & confidence"
            - Trend vs recent history"
            - Any caution based on volatility / seasonality"
            - Actionable insight for farmers/traders
            Provide your response in clear, professional language suitable for agricultural stakeholders.
            """

            # 3️⃣ Try Gemini first, fallback to Ollama
            if gemini_available:
                try:
                    insight = get_gemini_insight(prompt)
                except Exception as e:
                    st.warning(f"⚠️ Gemini unavailable ({e}) → using local Llama3 fallback")
                    insight = get_local_insight(prompt)
            else:
                st.info("💡 Gemini API key not found → using local Llama3 model")
                insight = get_local_insight(prompt)

            st.success(insight)

        except Exception as e:
            st.error(f"Error fetching AI insight: {e}")
