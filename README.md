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
│   ├── processed/           ← Final clean panel ready for XGBoost
│   ├── group_models_v4.pkl  ← Pre-trained model cache (Required for app.py)
│   └── group_studies_v4.pkl ← Optuna studies cache
├── outputs/                 ← High-res dashboards, evaluation CSVs, and metrics
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

### Data Split
| Set | Rows | Date Range |
|---|---|---|
| Train | 21,113 | Jun 2001 → Aug 2024 |
| Test | 5,446 | Sep 2024 → Aug 2025 |
| Unseen (holdout) | 3,217 | Sep 2025 → Mar 2026 |

### Test Set Performance (Sep 2024 – Aug 2025)
| Metric | Overall | Vegetables | Fish | Meat | Rice |
|---|---|---|---|---|---|
| RMSE (₱) | 7.43 | 4.10 | 11.82 | 8.57 | 2.38 |
| MAE (₱) | 3.08 | 1.52 | 5.44 | 5.25 | 1.75 |
| Coverage | 90.9% | 90.7% | 91.0% | 91.1% | 92.3% |
| PI Width (₱) | 11.88 | 5.06 | 21.64 | 20.92 | 8.70 |

> Coverage target ≥ 90%. All four groups meet the target on the test set. ✅

### Unseen Holdout Performance (Sep 2025 – Mar 2026)
| Metric | Overall |
|---|---|
| RMSE (₱) | 7.94 |
| MAE (₱) | 3.58 |
| Coverage | 88.4% |
| PI Width (₱) | 12.27 |

> The 88.4% unseen coverage reflects genuine temporal distribution shift — the unseen set was never consulted during calibration, making this an honest out-of-sample benchmark.

### Auto-Calibrated Confidence Levels (no test leakage)
| Group | α | Cal-holdout coverage |
|---|---|---|
| Vegetables | 0.91 | 90.7% |
| Fish | 0.91 | 90.3% |
| Meat | 0.92 | 92.4% |
| Rice | 0.95 | 91.1% |

*(Detailed visual diagnostics and regional breakdowns are in the `outputs/` directory.)*

## 💻 How to Run the Prototype

To launch the interactive system walkthrough for the final presentation:

```bash
# 1. Setup the Model Files
# Place the pre-trained models (.pkl files) directly inside the data/ folder:
# - data/group_models_v4.pkl
# - data/group_studies_v4.pkl

# 2. Ensure you have the required libraries
pip install streamlit plotly

# 3. Launch the application
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
Because this project uses a strict local folder structure (`data/processed`, `outputs`), you must make a few tiny tweaks to run it in the flat `/content/` folder of Google Colab. 

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
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)
df = pd.read_csv(BASE_DIR / "data" / "processed" / "panel_food_prices_ph_clean.csv", parse_dates=["date"])
```
Replace them with these 4 lines:
```python
import os
os.makedirs("outputs", exist_ok=True)
OUTPUTS_DIR = "outputs"
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
All your dashboards and CSVs will pop up in a new `outputs` folder in the sidebar!