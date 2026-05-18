# ENSO-Aware Long-Range Price Forecasting: Technical Implementation Report

This document details the technical architecture and software engineering decisions behind the newly implemented long-range 2050 food price forecasting system, written from the perspective of a Computer Science developer. 

## 1. Synthetic Climate Data Generation (`src/oni_generator.py`)
To forecast up to 2050, we needed a realistic projection of future Oceanic Nino Index (ONI) cycles. Instead of using static values, we implemented a procedural generation module using trigonometric functions to mimic the actual geophysical properties of the ENSO climate oscillation.

* **Sinusoidal Oscillation:** A base sine wave simulates the ~4-year (48-month) periodicity of ENSO cycles. 
* **Boreal Winter Phase Locking:** A secondary cosine wave (`1.0 + 0.35 * cos(...)`) acts as a seasonal modulator. ENSO events naturally peak during Northern Hemisphere winters and weaken in the spring (the "spring predictability barrier"). The modulator scales the sine wave amplitude dynamically based on the month.
* **Deterministic Noise:** We added a fixed-seed Gaussian noise term (`rng.normal(0, 0.12)`) to introduce realistic variability without compromising the determinism required for consistent Streamlit dashboard rendering.
* **Scenario Biasing:** The engine shifts the mean and amplitude based on the selected scenario (`bias = 0.40` for El Niño, `bias = -0.40` for La Niña) to simulate long-term climate epochs.

## 2. Recursive Autoregressive Engine (`src/forecast_engine.py`)
Machine learning models trained on time-series panels cannot natively predict 24 years into the future in one shot. We engineered a recursive step-forward state machine to solve this.

* **Dynamic State Management:** The engine maintains a mutable dictionary (`price_history`) containing the last 12+ months of log-prices.
* **On-the-fly Feature Engineering:** At step $t$, the system pulls historical lags ($t-1$, $t-3$, $t-12$). From these, it dynamically computes rolling statistics (3-month volatility, 6-month trends, YoY change).
* **Feedback Loop:** Once the XGBoost model predicts the $t$ log-price, this scalar is appended back to `price_history`. For step $t+1$, the prediction from step $t$ becomes the `lag_1` feature. This allows the model to "walk forward" indefinitely.
* **Conformal Prediction Propagation:** The MAPIE `SplitConformalRegressor` wraps the XGBoost model. At every step, it generates not just a point prediction, but `lower` and `upper` bounds dynamically mapped from the calibration set residuals.

## 3. High-Performance Batch Processing (`src/04_forecast.py`)
Generating granular monthly forecasts across all regions and commodities out to 2050 is computationally heavy. `04_forecast.py` serves as a dedicated batch-processing pipeline.

* **Hyperparameter Injection:** Rather than re-running the expensive Optuna hyperparameter optimization (which takes ~15 minutes), the script directly loads the cached optimal trees (`optuna_best_params_v4.csv`) and refits lightweight XGBoost models in ~30 seconds.
* **Seed History Extraction:** The script isolates the final 13 months of the empirical dataset to act as the "seed" state to prime the recursive engine's lag arrays.
* **Dimensionality & Serialization:** The script iterates over 3 ENSO scenarios $\times$ Regions $\times$ Commodities $\times$ 297 future months, batching the resulting vectors into a massive cross-joined pandas DataFrame. This is serialized to CSV so the Streamlit UI does not have to recompute the simulation.

## 4. Interactive UI & Reactive Design (`src/app.py`)
We upgraded the Streamlit prototype to visually map the complex long-range predictions into an intuitive dashboard.

* **Reactive State:** The UI uses `@st.cache_data` heavily to load the massive 2050 CSV file into memory just once. Scenario selections in the sidebar instantly filter the DataFrame without disk I/O.
* **Algorithmic Banding:** To render the climate impacts, the script runs `itertools.groupby` over the generated ONI time-series to group identical consecutive phases into continuous time blocks. 
* **Layered Plotly Rendering:** It injects Plotly `add_vrect` elements underneath the main line traces (`layer="below"`) using the grouped blocks to create the seamless red (El Niño) and blue (La Niña) background shading, perfectly aligning the climate state with the projected price impacts.

## 5. Predictive Mechanics: How ENSO Influences the Forecast
To understand how the mathematical generation of ENSO translates into actual price fluctuations on the Streamlit graph, we must look at the feature engineering within `forecast_engine.py` and the behavior of the XGBoost trees:

* **Feature Vector Construction:** At each step $t$, the XGBoost model does not merely extrapolate past prices. It evaluates a 26-dimensional feature vector. Crucially, this vector includes the current synthetic ONI value, historical ONI lags (3-month and 6-month), and one-hot encoded ENSO phase dummies (`enso_phase_La Nina`, `enso_phase_Neutral`).
* **Non-Linear Climate Interactions:** Because XGBoost is a tree-based ensemble, it captures non-linear relationships. For instance, if the ENSO phase is El Niño (meaning `enso_phase_La Nina = 0` and `enso_phase_Neutral = 0`) and the commodity is "Rice" in a drought-prone region, the model's decision trees will route the prediction to higher price leaves based on patterns it learned from historical El Niño events in the empirical training data.
* **Seasonal Amplification:** We engineered interaction terms like `oni_x_month_sin` and `oni_x_month_cos`. This signals to the model not just that an ENSO event is happening, but *when* it is happening within the calendar year. An El Niño peaking in April (dry season) impacts agricultural yields (and thus prices) much differently than one peaking in October (wet/typhoon season).
* **Visualizing the Impact:** When a user selects the "El Nino Dominant" scenario in the UI, `oni_generator.py` produces a time-series with prolonged, severe El Niño epochs. The recursive engine feeds these elevated ONI metrics into the XGBoost model month-by-month. Consequently, the model naturally projects higher prices for climate-susceptible commodities. On the UI graph, the user sees the projected price line swell simultaneously with the red background shading (the active El Niño phase block), explicitly demonstrating the cause-and-effect relationship between the climate input and the price forecast output.
