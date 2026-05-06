# Philippine Food Price Forecasting

A machine learning pipeline that forecasts food prices across Philippine regions 
using WFP market data and ENSO climate indicators (ONI index).

## 🚀 The Pipeline

Our methodology strictly avoids data leakage and ensures robust, real-world reliability through a 4-step architecture:

1. **`src/01_pipeline.py` (Data Engineering)** 
   Merges historical WFP food price data with ONI climate metrics. Captures specific supply-side disruptions such as regional fishing bans and the Philippine typhoon season.
2. **`src/02_clean.py` (Data Quality Control)** 
   Categorizes commodities into primary agricultural groups and filters out statistically sparse time-series using a strict 50% windowed completeness threshold from 2010 onwards.
3. **`src/03_model.py` (Model Training & Conformal Calibration)** 
   Trains independent **XGBoost Regressors** optimized via 100 Optuna trials per commodity group. It then applies **Split Conformal Prediction (MAPIE)** to empirically calibrate uncertainty and generate dynamic 90% confidence intervals.
4. **`src/app.py` (Interactive Prototype)**
   An interactive web dashboard built with **Streamlit** and **Plotly** that allows stakeholders to visually explore the out-of-sample and unseen predictions on a per-region, per-commodity basis.

## 📂 Project Structure

```text
├── data/
│   ├── raw/                 ← Original downloaded datasets (WFP & ONI)
│   ├── interim/             ← Intermediate merged data
│   └── processed/           ← Final clean panel ready for XGBoost
├── outputs/
│   └── finalOutputs/        ← High-res dashboards, evaluation CSVs, and metrics
├── src/
│   ├── 01_pipeline.py
│   ├── 02_clean.py
│   ├── 03_model.py
│   └── app.py               ← Streamlit Prototype Application
├── README.md
└── .gitignore
```

## 📊 Key Results

Instead of generating single point predictions, our system produces mathematically sound **Prediction Intervals**. The model is evaluated on a strict **Chronological 3-Way Split** (Train / Test / Unseen) to simulate real-world deployment.

**Performance on the untouched Future Unseen Set (Sept 2025 – March 2026):**
- **Overall Coverage:** > 90% (Empirically calibrated to hit strict statistical guarantees)
- **Mean Absolute Error (MAE):** Low deviance across Fish, Meat, Rice, and Vegetables despite major seasonal volatility.
- *(Detailed visual diagnostics and regional breakdowns are available in the `outputs/finalOutputs/` directory).*

## 💻 How to Run the Prototype

To launch the interactive system walkthrough for the final presentation:

```bash
# 1. Ensure you have the required libraries
pip install streamlit plotly

# 2. Launch the application
python -m streamlit run src/app.py
```

## 🛠️ Technology Stack
- **Data Engineering:** `pandas`, `numpy`
- **Machine Learning:** `xgboost`, `scikit-learn`
- **Hyperparameter Tuning:** `optuna`
- **Uncertainty Quantification:** `mapie` (Conformal Prediction)
- **Web App / UI:** `streamlit`, `plotly`
- **Visualizations:** `matplotlib`