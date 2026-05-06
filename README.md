# Philippine Food Price Forecasting

A machine learning pipeline that forecasts food prices across Philippine regions 
using WFP market data and ENSO climate indicators (ONI index).

## Pipeline
1. `src/01_pipeline.py` — Merges WFP food price data with ONI climate index
2. `src/02_clean.py` — Cleans and filters commodities by completeness
3. `src/03_model.py` — XGBoost forecasting model with conformal prediction intervals

## Data
- WFP Food Prices (Philippines) — sourced from Humanitarian Data Exchange
- Oceanic Niño Index (ONI) — sourced from NOAA

## Model
- Per-group XGBoost models (Rice, Fish, Meat, Vegetables)
- Conformal prediction intervals targeting ≥90% coverage
- Evaluated on a true unseen holdout set