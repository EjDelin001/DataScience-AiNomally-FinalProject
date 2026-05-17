"""
04_forecast.py
--------------
One-time batch script: generates recursive autoregressive forecasts
from Apr 2026 → Dec 2050 for all region × commodity combinations
across three ENSO scenarios (Neutral, El Niño, La Niña).

Run this ONCE before launching app.py:
    python src/04_forecast.py

Output: outputs/food_price_forecast_2050.csv
        data/forecast_state_v4.pkl  (conformal regressors + metadata)
"""

import warnings
warnings.filterwarnings("ignore")

import pickle
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

try:
    from mapie.regression import SplitConformalRegressor
except ImportError:
    raise ImportError("Run: pip install mapie")

# ── Add src/ to path so forecast_engine is importable ───────────────
sys.path.insert(0, str(Path(__file__).parent))
from forecast_engine import recursive_forecast, COASTAL_REGIONS, CORRIDOR_REGIONS
from oni_generator import generate_oni_series, SCENARIO_PARAMS

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)

MODELS_CACHE  = DATA_DIR / "group_models_v4.pkl"
STATE_CACHE   = DATA_DIR / "forecast_state_v4.pkl"
OUTPUT_CSV    = OUTPUTS_DIR / "food_price_forecast_2050.csv"

KEEP_GROUPS = ["Vegetables", "Fish", "Meat", "Rice"]
START_DATE  = pd.Timestamp("2026-04-01")
END_DATE    = pd.Timestamp("2050-12-01")

# ════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD & FEATURE-ENGINEER PANEL (mirrors 03_model.py exactly)
# ════════════════════════════════════════════════════════════════════
print("Step 1: Loading and feature-engineering the panel...")

df = pd.read_csv(
    DATA_DIR / "processed" / "panel_food_prices_ph_clean.csv",
    parse_dates=["date"]
)
df = df.sort_values(["region", "commodity_group", "commodity", "date"]).reset_index(drop=True)
print(f"  Loaded {len(df):,} rows | {df['date'].min().date()} -> {df['date'].max().date()}")

grp_key = ["region", "commodity"]

# Log-transform target
df["log_price"] = np.log(df["price_php"])

# Price lags
for lag in [1, 2, 3, 6, 12]:
    df[f"price_lag_{lag}"] = df.groupby(grp_key)["log_price"].shift(lag)

# YoY change
df["price_yoy"] = df["log_price"] - df.groupby(grp_key)["log_price"].shift(12)

# Rolling 3-month volatility
df["price_vol3"] = (
    df.groupby(grp_key)["log_price"]
    .transform(lambda x: x.shift(1).rolling(3).std())
)

# Rolling 6-month trend
df["price_trend6"] = (
    df.groupby(grp_key)["log_price"]
    .transform(lambda x: x.shift(1).rolling(6).mean())
)

# ONI lags
for lag in [3, 6]:
    df[f"oni_lag_{lag}"] = df.groupby(grp_key)["oni"].shift(lag)

# Cyclic month encoding
df["month"]     = df["date"].dt.month
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["year"]      = df["date"].dt.year

# Supply-side seasonal dummies
df["typhoon_season"]    = df["month"].isin([6, 7, 8, 9, 10, 11]).astype(float)
df["fishing_ban"]       = df["month"].isin([11, 12, 1, 2, 3]).astype(float)
df["is_coastal"]        = df["region"].isin(COASTAL_REGIONS).astype(float)
df["is_luzon_corridor"] = df["region"].isin(CORRIDOR_REGIONS).astype(float)
df["is_january"]        = (df["month"] == 1).astype(float)

# ONI × season interactions
df["oni_x_month_sin"] = df["oni"] * df["month_sin"]
df["oni_x_month_cos"] = df["oni"] * df["month_cos"]

# ENSO phase dummies
df = pd.get_dummies(df, columns=["enso_phase"], drop_first=True, dtype=float)
enso_dummy_cols = sorted([c for c in df.columns if c.startswith("enso_phase_")])

# Label-encode region (fit on sorted unique values — no split leakage)
le_region = LabelEncoder()
le_region.fit(sorted(df["region"].unique()))
df["region_enc"] = le_region.transform(df["region"])

feature_cols = [
    "price_lag_1", "price_lag_2", "price_lag_3", "price_lag_6", "price_lag_12",
    "price_yoy", "price_vol3", "price_trend6",
    "oni", "oni_lag_3", "oni_lag_6",
    "oni_x_month_sin", "oni_x_month_cos",
    "month_sin", "month_cos", "year",
    "typhoon_season",
    "fishing_ban", "is_coastal",
    "is_luzon_corridor",
    "is_january",
    "region_enc",
] + enso_dummy_cols

df_clean = df.dropna(subset=feature_cols + ["log_price"]).reset_index(drop=True)
print(f"  After dropping NaN: {len(df_clean):,} rows | Features: {len(feature_cols)}")

# ════════════════════════════════════════════════════════════════════
# STEP 2 — CHRONOLOGICAL SPLIT (same cutoffs as 03_model.py)
# ════════════════════════════════════════════════════════════════════
print("\nStep 2: Reproducing chronological split...")

sorted_dates  = df_clean.sort_values("date")["date"].reset_index(drop=True)
n_total       = len(sorted_dates)
cutoff_90_row = int(n_total * 0.90)
CUTOFF_UNSEEN = sorted_dates.iloc[cutoff_90_row].replace(day=1)

known_df  = df_clean[df_clean["date"] < CUTOFF_UNSEEN].copy().reset_index(drop=True)
unseen_df = df_clean[df_clean["date"] >= CUTOFF_UNSEEN].copy().reset_index(drop=True)

known_sorted  = known_df.sort_values("date").reset_index(drop=True)
n_known       = len(known_sorted)
cutoff_80_row = int(n_known * 0.80)
CUTOFF_TEST   = known_sorted["date"].iloc[cutoff_80_row].replace(day=1)

train_df = known_df[known_df["date"] < CUTOFF_TEST].copy().reset_index(drop=True)
print(f"  CUTOFF_TEST   : {CUTOFF_TEST.date()}")
print(f"  CUTOFF_UNSEEN : {CUTOFF_UNSEEN.date()}")
print(f"  Train rows    : {len(train_df):,}")

# ════════════════════════════════════════════════════════════════════
# STEP 3 — COMMODITY STATS (training set only — no leakage)
# ════════════════════════════════════════════════════════════════════
print("\nStep 3: Computing commodity scale features from training set...")

train_mask = df_clean["date"] < CUTOFF_TEST
commodity_stats = (
    df_clean[train_mask]
    .groupby("commodity")["log_price"]
    .agg(comm_mean="mean", comm_std="std")
    .reset_index()
)
global_mean = df_clean.loc[train_mask, "log_price"].mean()
global_std  = df_clean.loc[train_mask, "log_price"].std()

commodity_stats["comm_std"]  = commodity_stats["comm_std"].fillna(global_std)
commodity_stats["comm_mean"] = commodity_stats["comm_mean"].fillna(global_mean)
feature_cols += ["comm_mean", "comm_std"]

# Apply back to df_clean
df_clean = df_clean.merge(commodity_stats, on="commodity", how="left")
df_clean["comm_std"]  = df_clean["comm_std"].fillna(global_std)
df_clean["comm_mean"] = df_clean["comm_mean"].fillna(global_mean)

# Re-slice after merge
train_df  = df_clean[df_clean["date"] < CUTOFF_TEST].copy().reset_index(drop=True)
print(f"  Commodity stats computed for {len(commodity_stats)} commodities.")

# ════════════════════════════════════════════════════════════════════
# STEP 4 — TRAIN FRESH MODELS WITH SAVED BEST HYPERPARAMETERS
#
# NOTE: The cached group_models_v4.pkl was built from an older code
# version with a different feature set (34 features vs our 26).
# Instead of loading the mismatched pkl, we retrain lightweight
# XGBoost models using the best hyperparameters already saved in
# optuna_best_params_v4.csv — no Optuna search needed (~30 seconds).
# Tab 1 (historical view) is unaffected as it reads pre-generated CSVs.
# ════════════════════════════════════════════════════════════════════
print("\nStep 4: Training fresh XGBoost models from saved best hyperparameters...")

BEST_PARAMS_CSV = OUTPUTS_DIR / "optuna_best_params_v4.csv"
if not BEST_PARAMS_CSV.exists():
    raise FileNotFoundError(f"Best params CSV not found: {BEST_PARAMS_CSV}")

best_params_df = pd.read_csv(BEST_PARAMS_CSV).set_index("group")

PARAM_COLS = ["n_estimators", "max_depth", "learning_rate", "subsample",
              "colsample_bytree", "min_child_weight", "reg_alpha", "reg_lambda"]

# find_min_alpha — identical to 03_model.py
def find_min_alpha(model, X_cal, y_cal, target_cov=0.90,
                   alpha_min=0.90, alpha_max=0.99, step=0.01):
    cal_split  = int(len(X_cal) * 0.80)
    X_cf, X_ch = X_cal[:cal_split], X_cal[cal_split:]
    y_cf, y_ch = y_cal[:cal_split], y_cal[cal_split:]
    y_hat_cf   = model.predict(X_cf)
    scores     = np.abs(y_cf - y_hat_cf)
    y_hat_ch   = model.predict(X_ch)
    abs_err_ch = np.abs(y_ch - y_hat_ch)
    n          = len(scores)
    candidates = np.round(np.arange(alpha_min, alpha_max + step / 2, step), 2)
    for alpha in candidates:
        q_level   = min(np.ceil((n + 1) * alpha) / n, 1.0)
        threshold = np.quantile(scores, q_level)
        coverage  = float(np.mean(abs_err_ch <= threshold))
        if coverage >= target_cov:
            return alpha, coverage
    return alpha_max, float(np.mean(abs_err_ch <= np.quantile(scores, 1.0)))

train_df_sorted = train_df.sort_values("date").reset_index(drop=True)
cal_idx_full    = int(len(train_df_sorted) * 0.80)

GROUP_MODELS        = {}
GROUP_CONFIDENCE    = {}
conformal_regressors = {}

for grp_name in KEEP_GROUPS:
    print(f"\n  -- {grp_name} --")
    grp_train_mask = train_df_sorted["commodity_group"] == grp_name
    X_tr_g = train_df_sorted.loc[grp_train_mask, feature_cols].values
    y_tr_g = train_df_sorted.loc[grp_train_mask, "log_price"].values

    cal_idx_g = int(len(X_tr_g) * 0.80)
    X_fit_g   = X_tr_g[:cal_idx_g]
    y_fit_g   = y_tr_g[:cal_idx_g]
    X_cal_g   = X_tr_g[cal_idx_g:]
    y_cal_g   = y_tr_g[cal_idx_g:]

    if len(X_cal_g) < 10:
        X_cal_g = train_df_sorted.loc[:, feature_cols].values[cal_idx_full:]
        y_cal_g = train_df_sorted.loc[:, "log_price"].values[cal_idx_full:]

    # Load saved best hyperparameters and train fresh model
    row    = best_params_df.loc[grp_name]
    params = {col: row[col] for col in PARAM_COLS}
    params["n_estimators"]     = int(params["n_estimators"])
    params["max_depth"]        = int(params["max_depth"])
    params["min_child_weight"] = int(params["min_child_weight"])
    print(f"  Params: n_estimators={params['n_estimators']}, lr={params['learning_rate']:.4f}")

    model_g = XGBRegressor(random_state=42, tree_method="hist", device="cpu", **params)
    model_g.fit(X_fit_g, y_fit_g)
    GROUP_MODELS[grp_name] = model_g
    print(f"  Trained. Features in model: {model_g.n_features_in_}")

    alpha_g, cal_cov = find_min_alpha(model_g, X_cal_g, y_cal_g)
    GROUP_CONFIDENCE[grp_name] = alpha_g

    cr_g = SplitConformalRegressor(estimator=model_g, confidence_level=alpha_g, prefit=True)
    cr_g.conformalize(X_cal_g, y_cal_g)
    conformal_regressors[grp_name] = cr_g
    print(f"  Conformal: alpha={alpha_g:.2f}, cal-holdout coverage={cal_cov:.1%}")

# Save forecast state for future use by app.py if needed
forecast_state = {
    "conformal_regressors": conformal_regressors,
    "commodity_stats":      commodity_stats,
    "feature_cols":         feature_cols,
    "le_region":            le_region,
    "group_confidence":     GROUP_CONFIDENCE,
    "enso_dummy_cols":      enso_dummy_cols,
    "global_mean":          global_mean,
    "global_std":           global_std,
}
pickle.dump(forecast_state, open(STATE_CACHE, "wb"))
print(f"\n  Forecast state saved -> {STATE_CACHE}")

# ====================================================================
# STEP 5 - BUILD SEED HISTORY PER (region, commodity)
# ====================================================================
print("\nStep 5: Building seed price histories...")

# Use the last 13 months of the full dataset as seed (covers all 12 lags)
seed_cutoff = df_clean["date"].max() - pd.DateOffset(months=12)
seed_base   = df_clean[df_clean["date"] >= seed_cutoff][
    ["region", "commodity", "date", "log_price"]
].copy()

def get_seed(region, commodity):
    mask = (seed_base["region"] == region) & (seed_base["commodity"] == commodity)
    s    = seed_base[mask].sort_values("date")
    return {pd.Timestamp(r.date): r.log_price for _, r in s.iterrows()}

# Build historical ONI lookup for lag seeding
# (actual ONI values from the panel for dates before START_DATE)
historical_oni = (
    df_clean[["date", "oni"]]
    .drop_duplicates("date")
    .set_index("date")["oni"]
    .to_dict()
)
print(f"  Historical ONI seed: {len(historical_oni)} months of actual data.")

# ====================================================================
# STEP 6 - RECURSIVE FORECASTING (all series x 3 scenarios)
# ====================================================================
print("\nStep 6: Running recursive forecasts...")
print(f"  Horizon: {START_DATE.date()} -> {END_DATE.date()} ({len(pd.date_range(START_DATE, END_DATE, freq='MS'))} months)")

all_rows = []
regions      = sorted(df_clean["region"].unique())
commodities_by_region = {
    r: sorted(df_clean[df_clean["region"] == r]["commodity"].unique())
    for r in regions
}

comm_stat_lookup = commodity_stats.set_index("commodity")[["comm_mean", "comm_std"]].to_dict("index")

# Pre-generate ONI series for each scenario (deterministic, seed=42)
SCENARIOS = list(SCENARIO_PARAMS.keys())  # ["El Nino", "Neutral", "La Nina"]

for scenario_name in SCENARIOS:
    print(f"\n  Scenario: {scenario_name}")

    # Generate realistic monthly ONI series for this scenario
    oni_df = generate_oni_series(
        scenario=scenario_name,
        start_date=START_DATE,
        end_date=END_DATE,
        seed=42,
        historical_oni=historical_oni,
    )
    print(f"  ONI stats: mean={oni_df['oni'].mean():+.2f}, "
          f"min={oni_df['oni'].min():.2f}, max={oni_df['oni'].max():.2f}")
    print(f"  Phase breakdown: "
          f"El Nino={( oni_df['enso_phase']=='El Nino').mean()*100:.0f}%, "
          f"Neutral={( oni_df['enso_phase']=='Neutral').mean()*100:.0f}%, "
          f"La Nina={( oni_df['enso_phase']=='La Nina').mean()*100:.0f}%")

    n_series = 0
    for region in regions:
        region_enc = int(le_region.transform([region])[0])
        for commodity in commodities_by_region[region]:
            series_meta = df_clean[
                (df_clean["region"] == region) &
                (df_clean["commodity"] == commodity)
            ]
            if len(series_meta) < 12:
                continue

            commodity_group = series_meta["commodity_group"].iloc[0]
            cr              = conformal_regressors[commodity_group]

            stats   = comm_stat_lookup.get(commodity, {"comm_mean": global_mean, "comm_std": global_std})
            c_mean  = stats["comm_mean"]
            c_std   = stats["comm_std"]
            seed_lp = get_seed(region, commodity)

            if len(seed_lp) < 3:
                continue

            fc_df = recursive_forecast(
                region=region,
                commodity=commodity,
                commodity_group=commodity_group,
                conformal_reg=cr,
                seed_log_prices=seed_lp,
                start_date=START_DATE,
                end_date=END_DATE,
                oni_df=oni_df,
                region_enc=region_enc,
                comm_mean=c_mean,
                comm_std=c_std,
                feature_cols=feature_cols,
            )

            fc_df["region"]          = region
            fc_df["commodity"]       = commodity
            fc_df["commodity_group"] = commodity_group
            fc_df["scenario"]        = scenario_name
            all_rows.append(fc_df)
            n_series += 1

    print(f"    -> {n_series} series forecasted.")

# ====================================================================
# STEP 7 - SAVE OUTPUT
# ====================================================================
print("\nStep 7: Saving output CSV...")

final_df = pd.concat(all_rows, ignore_index=True)
final_df["date"] = pd.to_datetime(final_df["date"])

col_order = ["date", "region", "commodity", "commodity_group", "scenario",
             "pred", "lower", "upper", "oni", "enso_phase"]
final_df  = final_df[col_order].sort_values(["scenario", "region", "commodity", "date"])
final_df.to_csv(OUTPUT_CSV, index=False)

print(f"\n  Saved {len(final_df):,} rows -> {OUTPUT_CSV}")
print(f"     Scenarios  : {final_df['scenario'].unique().tolist()}")
print(f"     Regions    : {final_df['region'].nunique()}")
print(f"     Commodities: {final_df['commodity'].nunique()}")
print(f"     Date range : {final_df['date'].min().date()} -> {final_df['date'].max().date()}")

# Quick sanity check
assert final_df["pred"].isna().sum() == 0, "NaN predictions found!"
assert (final_df["lower"] <= final_df["pred"]).all(), "lower > pred found!"
assert (final_df["pred"] <= final_df["upper"]).all(), "pred > upper found!"
print("\n  Sanity checks passed -- no NaN, all intervals valid.")
print("\nDone! You can now launch the app: python -m streamlit run src/app.py")
