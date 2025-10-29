import os 
import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm import tqdm

# === Import base forecasting utilities ===
from model_utils import (
    train_prophet_model, predict_prophet,
    train_lightgbm_model, iterative_forecast_lgb,
    train_sarima_model, predict_sarima
)

# === CONFIG ===
DATA_DIR = "data"
META_DIR = "meta_data"
FORECAST_HORIZONS = [7, 14, 20, 30]
COMMODITIES = ["Banana", "Tomato", "Potato", "Coconut", "Brinjal"]

os.makedirs(META_DIR, exist_ok=True)


# === Helper: Sliding Window Simulation ===
def sliding_window_forecast(df, horizon, series_key):
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
            m_prophet = train_prophet_model(train_df)
            pred_prophet_df = predict_prophet(m_prophet, train_df, horizon)
            prophet_pred = float(pred_prophet_df["prophet_pred"].iloc[-1])
        except Exception:
            prophet_pred = np.nan

        # --- LightGBM ---
        try:
            m_lgb, _ = train_lightgbm_model(train_df, feature_cols)
            pred_lgb_list = iterative_forecast_lgb(m_lgb, train_df, horizon, feature_cols)
            lgbm_pred = float(pred_lgb_list[-1][1])
        except Exception:
            lgbm_pred = np.nan

        # --- SARIMA ---
        try:
            m_sarima = train_sarima_model(train_df)
            pred_sarima_df = predict_sarima(m_sarima, train_df, horizon)
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


# === Main Driver ===
def generate_meta_training_data():
    for commodity in COMMODITIES:
        file_path = os.path.join(DATA_DIR, f"processed_{commodity}.csv")
        if not os.path.exists(file_path):
            print(f"⚠️ Missing file for {commodity}, skipping.")
            continue

        print(f"\n📘 Generating meta-training data for {commodity}")
        df = pd.read_csv(file_path)

        if "series_key" not in df.columns:
            print(f"⚠️ No series_key column in {commodity} file — skipping.")
            continue

        for horizon in FORECAST_HORIZONS:
            all_meta = []
            for series_key, series_df in df.groupby("series_key"):
                if len(series_df) < 100:
                    continue  # skip too-short series
                meta_df = sliding_window_forecast(series_df, horizon, series_key)
                if not meta_df.empty:
                    all_meta.append(meta_df)

            if all_meta:
                combined_df = pd.concat(all_meta, ignore_index=True)
                save_path = os.path.join(META_DIR, f"meta_training_{commodity}_h{horizon}.csv")
                combined_df.to_csv(save_path, index=False)
                print(f"✅ Saved: {save_path} ({len(combined_df)} rows)")
            else:
                print(f"⚠️ No valid series for {commodity} (horizon={horizon})")


if __name__ == "__main__":
    generate_meta_training_data()
