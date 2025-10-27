# TFT_full_pipeline.py
# Run in Colab / script. Uses merged_all_commodities.csv as input.

import os
import pickle
from datetime import timedelta
import pandas as pd
import numpy as np
import torch

# pytorch-forecasting imports
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pytorch_lightning as pl

# -------------------------
# CONFIG
# -------------------------
MERGED_CSV = "/content/drive/MyDrive/agmarknet/merged_all_commodities.csv"  # update path
OUTDIR = "/content/drive/MyDrive/agmarknet_models/tft"  # where to save model & dataset
os.makedirs(OUTDIR, exist_ok=True)

MAX_ENCODER_LENGTH = 30        # history window (t-30 ... t-1)
MAX_PREDICTION_LENGTH = 7      # forecast horizon (t ... t+6)
SUBSET_FRACTION = None         # set 0.2 for quick tests, None for full dataset
MAX_EPOCHS = 6                 # increase for production (e.g., 30-100)
BATCH_SIZE = 64

# -------------------------
# 1) Load merged dataset & basic cleaning
# -------------------------
df = pd.read_csv(MERGED_CSV, parse_dates=["price_date"])
# ensure needed columns exist: commodity, market_name, modal_price_rs_per_kg
assert "commodity" in df.columns and "market_name" in df.columns and "modal_price_rs_per_kg" in df.columns, \
    "merged CSV must contain commodity, market_name, modal_price_rs_per_kg"

# Create series_key if not present
if "series_key" not in df.columns:
    df["series_key"] = df["commodity"].astype(str) + "___" + df["market_name"].astype(str)

# sort and fill small gaps if needed
df = df.sort_values(["series_key", "price_date"]).reset_index(drop=True)

# create time_idx as days since global min date (ensures consistent time_idx across groups)
global_min = df["price_date"].min()
df["time_idx"] = (df["price_date"] - global_min).dt.days.astype(int)

# create simple temporal features used as known covariates
df["day"] = df["price_date"].dt.day
df["month"] = df["price_date"].dt.month
df["day_of_week"] = df["price_date"].dt.dayofweek
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# optionally subsample for quick tests
if SUBSET_FRACTION is not None and 0 < SUBSET_FRACTION < 1.0:
    df = df.sample(frac=SUBSET_FRACTION, random_state=42).sort_values(["series_key","time_idx"]).reset_index(drop=True)

# drop rows without target or time_idx
df = df.dropna(subset=["modal_price_rs_per_kg", "time_idx"]).reset_index(drop=True)

# -------------------------
# 2) Create TimeSeriesDataSet (training and validation)
# -------------------------
max_encoder_length = MAX_ENCODER_LENGTH
max_prediction_length = MAX_PREDICTION_LENGTH

# choose cutoff for training/validation (leave last max_prediction_length days out for validation)
training_cutoff = df["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    df[df.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="modal_price_rs_per_kg",
    group_ids=["series_key"],
    min_encoder_length=1,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["series_key"],
    time_varying_known_reals=["time_idx", "day", "month", "day_of_week", "is_weekend"],
    time_varying_unknown_reals=["modal_price_rs_per_kg"],
    target_normalizer=GroupNormalizer(groups=["series_key"], transformation="softplus"),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

# dataloaders
train_dataloader = training.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

# Save the training dataset object for later inference (pickle)
with open(os.path.join(OUTDIR, "tft_training_dataset.pkl"), "wb") as f:
    pickle.dump(training, f)

# -------------------------
# 3) Build & train TemporalFusionTransformer
# -------------------------
pl.seed_everything(42)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=1e-3,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,  # 7 quantiles; if you only want mean, you can adjust
    loss=torch.nn.MSELoss(),
    logging_metrics=[],
)

checkpoint_callback = ModelCheckpoint(
    dirpath=OUTDIR,
    filename="tft-{epoch:02d}-{val_loss:.4f}",
    monitor="val_loss",
    mode="min",
    save_top_k=1,
)

early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min")

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    gpus=1 if torch.cuda.is_available() else 0,
    callbacks=[checkpoint_callback, early_stop_callback],
    enable_model_summary=True,
    logger=False,
)

trainer.fit(tft, train_dataloader, val_dataloader)

# load best checkpoint
best_ckpt = checkpoint_callback.best_model_path
print("TFT best checkpoint:", best_ckpt)
tft = TemporalFusionTransformer.load_from_checkpoint(best_ckpt)

# save model's path and metadata
with open(os.path.join(OUTDIR, "tft_metadata.pkl"), "wb") as f:
    pickle.dump({"global_min": global_min, "max_encoder_length": max_encoder_length, "max_prediction_length": max_prediction_length}, f)

# -------------------------
# 4) Robust single-series prediction function
# -------------------------
from pytorch_forecasting.data import TimeSeriesDataSet as _TSD

def predict_single_series(tft_model, training_dataset, full_df, series_key, target_date, global_min, max_encoder_length, max_prediction_length):
    """
    tft_model: trained TemporalFusionTransformer
    training_dataset: the TimeSeriesDataSet object saved from training
    full_df: merged dataframe with at least columns ['series_key','price_date','time_idx', 'modal_price_rs_per_kg', ...]
    series_key: string like 'Brinjal___Chennai'
    target_date: python date (date object) to forecast
    returns: DataFrame of predictions for the model's prediction horizon (dates and yhat, optionally quantiles)
    """
    # ensure datatypes and time_idx
    full_df = full_df.copy()
    full_df['price_date'] = pd.to_datetime(full_df['price_date'])
    full_df['time_idx'] = (full_df['price_date'] - pd.to.to_datetime(global_min)).dt.days.astype(int) if not isinstance(global_min, pd.Timestamp) else (full_df['price_date'] - global_min).dt.days.astype(int)

    # retrieve series data
    series_df = full_df[full_df['series_key'] == series_key].sort_values('time_idx').reset_index(drop=True)
    if series_df.empty:
        raise ValueError(f"series_key {series_key} not found in data")

    # last observed time index for that series
    last_time_idx = int(series_df['time_idx'].max())
    # compute required encoder start idx
    encoder_start = max(0, last_time_idx - max_encoder_length + 1)
    # build rows covering encoder window up to last observation, and future rows to cover prediction horizon
    # We need rows for time_idx: encoder_start ... last_time_idx (existing) and last_time_idx+1 ... last_time_idx+max_prediction_length (future)
    time_idx_range = list(range(encoder_start, last_time_idx + max_prediction_length + 1))
    # Build a dataframe for this series with time_idx_range
    pred_df = pd.DataFrame({"time_idx": time_idx_range})
    pred_df['series_key'] = series_key
    pred_df['price_date'] = pd.to_datetime(global_min) + pd.to_timedelta(pred_df['time_idx'], unit='D')

    # merge existing known columns from full_df where available
    cols_to_merge = [c for c in full_df.columns if c not in ['time_idx','price_date']]
    # merge on (series_key, time_idx)
    # create helper
    merge_df = series_df.set_index('time_idx')
    pred_df = pred_df.join(merge_df[cols_to_merge], on='time_idx')

    # For future rows (time_idx > last_time_idx), we must supply known covariates (day, month, day_of_week, is_weekend)
    future_mask = pred_df['time_idx'] > last_time_idx
    if future_mask.any():
        pred_df.loc[future_mask, 'day'] = pred_df.loc[future_mask, 'price_date'].dt.day
        pred_df.loc[future_mask, 'month'] = pred_df.loc[future_mask, 'price_date'].dt.month
        pred_df.loc[future_mask, 'day_of_week'] = pred_df.loc[future_mask, 'price_date'].dt.dayofweek
        pred_df.loc[future_mask, 'is_weekend'] = (pred_df.loc[future_mask, 'day_of_week'] >= 5).astype(int)
        # for unknown reals (target) we leave modal_price_rs_per_kg as NaN (the model will predict it)
        pred_df.loc[future_mask, 'modal_price_rs_per_kg'] = np.nan

    # Now construct a TimeSeriesDataSet for prediction using TimeSeriesDataSet.from_dataset()
    predict_dataset = TimeSeriesDataSet.from_dataset(training_dataset, pred_df, predict=True, stop_randomization=True)

    # create dataloader for this single series
    predict_dataloader = predict_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

    # use tft_model.predict
    preds = tft_model.predict(predict_dataloader)
    # preds shape: (n_rows, prediction_length) or returns numpy with quantiles if configured
    # Build result dataframe: prediction corresponds to last encoder window -> returns array for prediction length
    # extract predicted window corresponding to target_date: find index of target_date in pred_df
    # pred_df contains encoder + prediction window; the prediction vector corresponds to prediction window starting at last_time_idx+1
    # So construct result date list:
    pred_start = last_time_idx + 1
    pred_dates = [ (pd.to_datetime(global_min) + pd.Timedelta(days=int(i))).date() for i in range(pred_start, pred_start + max_prediction_length) ]
    # preds may be numpy array shape (1, prediction_length) or dict for quantiles; handle common cases
    if isinstance(preds, np.ndarray):
        yhat = preds[0]  # shape: (prediction_length,)
        results = pd.DataFrame({"date": pred_dates, "yhat": yhat})
    else:
        # pytorch-forecasting may return xarray or pandas; try to convert
        try:
            arr = np.array(preds)
            yhat = arr[0]
            results = pd.DataFrame({"date": pred_dates, "yhat": yhat})
        except Exception:
            # fallback: return last observed
            results = pd.DataFrame({"date": pred_dates, "yhat": [series_df['modal_price_rs_per_kg'].iloc[-1]]*len(pred_dates)})
    return results

# -------------------------
# 5) Example usage
# -------------------------
# after training above (tft and training pickled), we can call predict_single_series:
# load back training dataset and model to demonstrate
# tft already loaded as 'tft' and training pickled object path saved earlier

# Example: predict Banana___Harur for 7 days ahead
example_series_key = df['series_key'].unique()[0]  # replace with desired
target_date = (df['price_date'].max() + pd.Timedelta(days=7)).date()

# load training dataset object if needed
with open(os.path.join(OUTDIR, "tft_training_dataset.pkl"), "rb") as f:
    training_loaded = pickle.load(f)

# call prediction
pred_df = predict_single_series(tft, training_loaded, df, example_series_key, target_date, global_min, max_encoder_length, max_prediction_length)
print(pred_df)
# save predictions
pred_df.to_csv(os.path.join(OUTDIR, f"tft_pred_{example_series_key.replace('/','_')}.csv"), index=False)
