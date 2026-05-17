"""
04_predict_future.py
--------------------
Production Inference Module (Recursive Forecasting)

This script demonstrates how to predict prices into the true future, outside of the known dataset.
It uses a "Recursive Forecasting" loop: it predicts 1 month ahead, adds that prediction back
into the dataset, recalculates the lagged features (e.g., price_lag_1), and then predicts the next month.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from dateutil.relativedelta import relativedelta

# ── Paths ──
BASE_DIR = Path(__file__).parent.parent
MODELS_CACHE = BASE_DIR / "data" / "group_models_v4.pkl"
DATA_FILE = BASE_DIR / "data" / "processed" / "panel_food_prices_ph_clean.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)

# ── Configuration ──
FORECAST_HORIZON = 6  # How many months into the future to predict
COASTAL_REGIONS = {"Region III", "Region IV-A", "Region IV-B", "Region VII", "Region VIII", "Region IX", "Region XI", "Region XII"}
CORRIDOR_REGIONS = {"Region III", "Region IV-A"}

def feature_engineering(df):
    """
    Recalculates all features for the entire dataframe. 
    Because we use groupby and shift(), it perfectly handles generating features for newly appended future rows.
    """
    grp_key = ["region", "commodity"]
    
    # Target scale
    df["log_price"] = np.log(df["price_php"].astype(float))
    
    # 1. Autoregressive Lags
    for lag in [1, 2, 3, 6, 12]:
        df[f"price_lag_{lag}"] = df.groupby(grp_key)["log_price"].shift(lag)
    
    df["price_yoy"] = df["log_price"] - df.groupby(grp_key)["log_price"].shift(12)
    df["price_yoy"] = df.groupby(grp_key)["price_yoy"].ffill()
    
    df["price_vol3"] = df.groupby(grp_key)["log_price"].transform(lambda x: x.shift(1).rolling(3).std())
    df["price_trend6"] = df.groupby(grp_key)["log_price"].transform(lambda x: x.shift(1).rolling(6).mean())
    
    # 2. Climate Lags
    for lag in [3, 6]:
        df[f"oni_lag_{lag}"] = df.groupby(grp_key)["oni"].shift(lag)
    
    # 3. Calendar Features
    df["month"] = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["year"] = df["date"].dt.year
    df["typhoon_season"] = df["month"].isin([6, 7, 8, 9, 10, 11]).astype(float)
    df["fishing_ban"] = df["month"].isin([11, 12, 1, 2, 3]).astype(float)
    df["is_january"] = (df["month"] == 1).astype(float)
    
    # 4. Regional Flags
    df["is_coastal"] = df["region"].isin(COASTAL_REGIONS).astype(float)
    df["is_luzon_corridor"] = df["region"].isin(CORRIDOR_REGIONS).astype(float)
    
    # 5. Climate Interactions
    df["oni_x_month_sin"] = df["oni"] * df["month_sin"]
    df["oni_x_month_cos"] = df["oni"] * df["month_cos"]
    
    return df

def main():
    print(f"Loading trained models from: {MODELS_CACHE}")
    try:
        with open(MODELS_CACHE, "rb") as f:
            GROUP_MODELS = pickle.load(f)
    except FileNotFoundError:
        print("ERROR: group_models_v4.pkl not found. Please run 03_model.py first to train the models.")
        return

    print(f"Loading historical data from: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    df = df.sort_values(["region", "commodity_group", "commodity", "date"]).reset_index(drop=True)
    
    # Pre-calculate Regional Encoding (Must perfectly match how 03_model.py did it)
    le_region = LabelEncoder()
    le_region.fit(sorted(df["region"].unique()))
    df["region_enc"] = le_region.transform(df["region"])
    
    # Target scale
    df["log_price"] = np.log(df["price_php"].astype(float))
    
    # Pre-calculate Commodity Scale stats (mean/std) based on historical data
    # In production, we anchor this to the known historical distribution
    commodity_stats = df.groupby("commodity")["log_price"].agg(comm_mean="mean", comm_std="std").reset_index()
    global_std = df["log_price"].std()
    global_mean = df["log_price"].mean()
    commodity_stats["comm_std"] = commodity_stats["comm_std"].fillna(global_std)
    commodity_stats["comm_mean"] = commodity_stats["comm_mean"].fillna(global_mean)
    
    df = df.merge(commodity_stats, on="commodity", how="left")
    
    # Get baseline information
    last_known_date = df["date"].max()
    print(f"Latest historical data is up to: {last_known_date.date()}")
    
    # Identify unique entities to forecast
    unique_entities = df[["region", "commodity_group", "commodity"]].drop_duplicates()
    
    # ── RECURSIVE FORECASTING LOOP ──
    print(f"\nStarting Recursive Forecast for {FORECAST_HORIZON} months ahead...")
    
    current_df = df.copy()
    
    for step in range(1, FORECAST_HORIZON + 1):
        target_date = last_known_date + relativedelta(months=step)
        print(f"  -> Predicting for: {target_date.date()}")
        
        # 1. Create empty rows for the target date for every region/commodity
        new_rows = unique_entities.copy()
        new_rows["date"] = target_date
        
        # 2. Real-World Climate Forecast (NOAA May 2026 Outlook)
        # We are transitioning into an El Nino summer
        oni_map = {4: 0.2, 5: 0.4, 6: 0.6, 7: 0.8, 8: 1.0, 9: 1.1}
        current_oni = oni_map.get(target_date.month, 0.5)
        current_phase = "El Nino" if current_oni >= 0.5 else ("La Nina" if current_oni <= -0.5 else "Neutral")
        
        new_rows["oni"] = current_oni
        new_rows["enso_phase"] = current_phase
        
        # Set target variable as NaN initially
        new_rows["price_php"] = np.nan
        new_rows["log_price"] = np.nan
        
        # 3. Append to the main dataframe
        current_df = pd.concat([current_df, new_rows], ignore_index=True)
        current_df = current_df.sort_values(["region", "commodity", "date"]).reset_index(drop=True)
        
        # 4. Generate all autoregressive features (this looks back at the historical rows automatically!)
        current_df = feature_engineering(current_df)
        
        # Get ENSO dummies to match the training set exactly (drop_first=True drops El Nino)
        current_df_dummy = current_df.copy()
        current_df_dummy["enso_phase_La Nina"] = (current_df_dummy["enso_phase"] == "La Nina").astype(float)
        current_df_dummy["enso_phase_Neutral"] = (current_df_dummy["enso_phase"] == "Neutral").astype(float)
        
        enso_dummy_cols = ["enso_phase_La Nina", "enso_phase_Neutral"]
        
        feature_cols = [
            "price_lag_1", "price_lag_2", "price_lag_3", "price_lag_6", "price_lag_12",
            "price_yoy", "price_vol3", "price_trend6",
            "oni", "oni_lag_3", "oni_lag_6",
            "oni_x_month_sin", "oni_x_month_cos",
            "month_sin", "month_cos", "year",
            "typhoon_season", "fishing_ban", "is_coastal", "is_luzon_corridor", "is_january",
            "region_enc"
        ] + enso_dummy_cols + ["comm_mean", "comm_std"]
        
        # 5. Predict for the new rows
        target_mask = current_df_dummy["date"] == target_date
        
        for grp_name, model in GROUP_MODELS.items():
            grp_mask = target_mask & (current_df_dummy["commodity_group"] == grp_name)
            
            if grp_mask.sum() == 0:
                continue
                
            X_infer = current_df_dummy.loc[grp_mask, feature_cols].values
            
            # Predict
            pred_log_price = model.predict(X_infer)
            pred_price = np.exp(pred_log_price)
            
            # Update the main dataframe with the predicted values so the NEXT month can use it
            current_df.loc[grp_mask, "log_price"] = pred_log_price
            current_df.loc[grp_mask, "price_php"] = pred_price

    # ── Extraction & Saving ──
    # Extract only the future dates we just predicted
    future_predictions = current_df[current_df["date"] > last_known_date].copy()
    
    # Save the output
    out_file = OUTPUTS_DIR / "true_future_forecast.csv"
    future_predictions[["date", "region", "commodity_group", "commodity", "price_php", "oni", "enso_phase"]].to_csv(out_file, index=False)
    
    print(f"\n[SUCCESS] Forecast complete!")
    print(f"Predicted {len(future_predictions)} rows spanning from {(last_known_date + relativedelta(months=1)).date()} to {target_date.date()}.")
    print(f"Results saved to: {out_file}")

if __name__ == "__main__":
    main()
