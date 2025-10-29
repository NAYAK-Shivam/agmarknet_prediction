# models_utils.py

import os
import numpy as np
import pandas as pd
import joblib
from datetime import timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Try optional imports
try:
    from prophet import Prophet
    prophet_available = True
except ImportError:
    prophet_available = False

try:
    import lightgbm as lgb
    lgb_available = True
except ImportError:
    lgb_available = False

try:
    import statsmodels.api as sm
    sarima_available = True
except ImportError:
    sarima_available = False


# -------------------------------
# PROPHET MODEL
# -------------------------------
def train_prophet_model_meta(series_df, model_path=None):
    if not prophet_available:
        raise RuntimeError("Prophet not installed.")
    
    if model_path and os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception:
            pass

    df = series_df[["price_date", "modal_price_rs_per_kg"]].dropna().rename(columns={"price_date": "ds", "modal_price_rs_per_kg": "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    
    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(df)

    if model_path:
        try:
            joblib.dump(m, model_path)
        except Exception:
            pass
    
    return m


def predict_prophet_meta(model, df, target_date, cap_ci_zero=True):
    """
    Predict price for a given target_date using trained Prophet model.
    Compatible with meta-learner CSV generation.
    """
    try:
        if model is None or df.empty:
            raise ValueError("Model or dataframe is empty")

        # Prepare data
        df = df.copy().sort_values("price_date")
        last_hist_date = df["price_date"].max().date()
        days_ahead = max((target_date - last_hist_date).days, 0)

        # Create future dataframe and predict
        future = model.make_future_dataframe(periods=days_ahead, freq="D")
        forecast = model.predict(future)
        forecast["ds_date"] = forecast["ds"].dt.date

        # Find closest forecast date to target_date
        if target_date in forecast["ds_date"].values:
            r = forecast[forecast["ds_date"] == target_date].iloc[0]
        else:
            forecast["abs_diff"] = forecast["ds_date"].apply(lambda x: abs((x - target_date).days))
            r = forecast.sort_values("abs_diff").iloc[0]

        # Extract values
        yhat_p = float(r["yhat"])
        lower_p = float(r.get("yhat_lower", np.nan))
        upper_p = float(r.get("yhat_upper", np.nan))

        if cap_ci_zero and not np.isnan(lower_p):
            lower_p = max(0.0, lower_p)

        # Construct output
        preds = pd.DataFrame({
            "price_date": [target_date],
            "prophet_pred": [yhat_p],
            "lower": [lower_p],
            "upper": [upper_p]
        })

        return preds

    except Exception as e:
        print(f"[Prophet] Prediction error for {target_date}: {e}")
        return pd.DataFrame(columns=["price_date", "prophet_pred", "lower", "upper"])

# -------------------------------
# LIGHTGBM MODEL
# -------------------------------
def train_lightgbm_model_meta(df, feature_cols, target_col="modal_price_rs_per_kg", min_rows=50, horizon_days=30):
    if not lgb_available:
        raise RuntimeError("LightGBM not installed.")
    
    df = df.sort_values("price_date").reset_index(drop=True).copy()
    present_feats = [c for c in feature_cols if c in df.columns]
    df_train = df.dropna(subset=present_feats + [target_col]).copy()

    if len(df_train) < min_rows:
        raise ValueError(f"Not enough rows to train LightGBM ({len(df_train)} < {min_rows})")
    if len(df_train) <= horizon_days + 1:
        raise ValueError("Not enough rows for holdout; increase data or reduce horizon.")

    train = df_train.iloc[:-horizon_days]
    test = df_train.iloc[-horizon_days:]
    X_train, y_train = train[present_feats].astype(float), train[target_col].astype(float)
    X_test, y_test = test[present_feats].astype(float), test[target_col].astype(float)

    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_test, label=y_test, reference=dtrain)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbose": -1,
        "seed": 42
    }

    try:
        model = lgb.train(params, dtrain, num_boost_round=1000, valid_sets=[dvalid],
                          early_stopping_rounds=50, verbose_eval=False)
    except TypeError:
        callbacks = [lgb.early_stopping(50)]
        model = lgb.train(params, dtrain, num_boost_round=1000, valid_sets=[dvalid], callbacks=callbacks)

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    results_df = pd.DataFrame({"price_date": test["price_date"].values, "y": y_test.values, "yhat": y_pred})
    return model, results_df

def iterative_forecast_lgb_meta(model, recent_df, days_ahead, feature_cols, target_col="modal_price_rs_per_kg"):
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

def predict_lightgbm_meta(model, df, target_date, feature_cols, target_col="modal_price_rs_per_kg"):
    try:
        if model is None or df.empty:
            raise ValueError("Model or dataframe is empty")

        df = df.sort_values("price_date").reset_index(drop=True)
        last_hist_date = df["price_date"].max().date()

        if target_date <= last_hist_date:
            # return actual if exists
            exact = df[df["price_date"].dt.date == target_date]
            if not exact.empty:
                yhat = float(exact.iloc[-1][target_col])
                note = "actual"
            else:
                # if features exist for that date, use direct predict
                feat_row = df[df["price_date"].dt.date == target_date]
                if not feat_row.empty and set(feature_cols).issubset(feat_row.columns):
                    X = feat_row[feature_cols].astype(float)
                    yhat = float(model.predict(X)[0])
                    note = "feature_predict"
                else:
                    # fallback iterative
                    preds = iterative_forecast_lgb_meta(
                        model, df,
                        (target_date - last_hist_date).days if target_date > last_hist_date else 1,
                        feature_cols, target_col=target_col
                    )
                    if preds:
                        yhat = preds[-1][1]
                        note = "iterative_fallback"
                    else:
                        return pd.DataFrame(columns=["price_date", "lgbm_pred"])
        else:
            # Forecast ahead using iterative forecast
            days_ahead = (target_date - last_hist_date).days
            preds = iterative_forecast_lgb_meta(model, df, days_ahead, feature_cols, target_col=target_col)
            if preds:
                yhat = preds[-1][1]
                note = "iterative_future"
            else:
                return pd.DataFrame(columns=["price_date", "lgbm_pred"])

        return pd.DataFrame({
            "price_date": [target_date],
            "lgbm_pred": [yhat],
            "note": [note]
        })

    except Exception as e:
        print(f"[LightGBM] Prediction error for {target_date}: {e}")
        return pd.DataFrame(columns=["price_date", "lgbm_pred"])

# -------------------------------
# SARIMA MODEL
# -------------------------------
def train_sarima_model_meta(series_df, model_path=None, order=(1,1,1), seasonal_order=(1,1,1,7)):
    if not sarima_available:
        raise RuntimeError("statsmodels not installed.")

    if model_path and os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception:
            pass

    y = series_df["modal_price_rs_per_kg"].dropna().values
    if len(y) < 30:
        raise ValueError("Not enough data to train SARIMA.")

    model = sm.tsa.statespace.SARIMAX(
        y, order=order, seasonal_order=seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False
    )
    results = model.fit(disp=False)

    if model_path:
        try:
            joblib.dump(results, model_path)
        except Exception:
            pass

    return results


def predict_sarima_meta(model, df, target_date):
    try:
        y = df["modal_price_rs_per_kg"].dropna().values
        last_hist_date = df["price_date"].max().date()

        # Compute forecast steps dynamically based on the target date
        steps = (target_date - last_hist_date).days
        if steps <= 0:
            steps = 1

        forecast = model.get_forecast(steps=steps)
        fc_mean = np.array(forecast.predicted_mean)
        conf_int = np.array(forecast.conf_int(alpha=0.05))

        # Take the last forecasted value (for the target date)
        yhat_s = float(fc_mean[-1])
        lower_s = float(conf_int[-1, 0])
        upper_s = float(conf_int[-1, 1])

        preds = pd.DataFrame({
            "price_date": [target_date],
            "sarima_pred": [yhat_s],
            "lower": [lower_s],
            "upper": [upper_s]
        })

        return preds

    except Exception as e:
        print(f"[SARIMA] Prediction error: {e}")
        return pd.DataFrame(columns=["price_date", "sarima_pred", "lower", "upper"])
