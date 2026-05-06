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

## ☁️ Running in Google Colab (Foolproof Guide)
Because this project uses a strict local folder structure (`data/processed`, `outputs/finalOutputs`), you must make a few tiny tweaks to run it in the flat `/content/` folder of Google Colab. 

Here is exactly what to do:

### Step 1: Upload your files
1. Go to [Google Colab](https://colab.research.google.com/) and create a New Notebook.
2. Click the **Folder icon** on the left sidebar.
3. Drag and drop **both** `03_model.py` and `panel_food_prices_ph_clean.csv` directly into that sidebar.

### Step 2: Install required libraries
Create a new code cell at the top of your notebook and run this exact command to install the missing libraries:
```python
!pip install optuna mapie xgboost
```

### Step 3: Change the Paths in `03_model.py`
Double-click `03_model.py` in the Colab sidebar to open it in the editor.

**A. Find the "LOAD DATA" section (around line 77):**
Delete these 5 lines:
```python
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent
FINAL_OUTPUTS = BASE_DIR / "outputs" / "finalOutputs"
FINAL_OUTPUTS.mkdir(exist_ok=True, parents=True)
df = pd.read_csv(BASE_DIR / "data" / "processed" / "panel_food_prices_ph_clean.csv", parse_dates=["date"])
```
Replace them with these 4 lines:
```python
import os
os.makedirs("finalOutputs", exist_ok=True)
FINAL_OUTPUTS = "finalOutputs"
df = pd.read_csv("panel_food_prices_ph_clean.csv", parse_dates=["date"])
```

**B. Find the "CACHE" section (around line 315):**
Delete these 2 lines:
```python
MODELS_CACHE  = BASE_DIR / "data" / "group_models_v4.pkl"
STUDIES_CACHE = BASE_DIR / "data" / "group_studies_v4.pkl"
```
Replace them with these 2 lines:
```python
MODELS_CACHE  = "group_models_v4.pkl"
STUDIES_CACHE = "group_studies_v4.pkl"
```

### Step 4: Turn on the GPU and Run!
1. In the top menu, click **Runtime** > **Change runtime type**.
2. Select **T4 GPU** and click Save.
3. Create a new code cell and run:
```python
!python 03_model.py
```
All your dashboards and CSVs will pop up in a new `finalOutputs` folder in the sidebar!