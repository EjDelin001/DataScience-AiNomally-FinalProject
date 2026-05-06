import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# Page config
st.set_page_config(page_title="AiNomally Food Price Forecast", layout="wide", initial_sidebar_state="expanded")

st.title("🇵🇭 Philippine Food Price Forecasting")
st.markdown("**Developed by Team AiNomally for IT 322 Final Project**")
st.markdown("This prototype demonstrates our machine learning pipeline predicting agricultural commodity prices across Philippine regions, complete with conformal prediction uncertainty intervals.")

# Setup paths
BASE_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"

@st.cache_data
def load_data():
    # Load Test and Unseen sets
    test_df = pd.read_csv(OUTPUTS_DIR / "food_price_predictions_v4.csv", parse_dates=["date"])
    unseen_df = pd.read_csv(OUTPUTS_DIR / "food_price_predictions_unseen_v4.csv", parse_dates=["date"])
    
    test_df["Dataset"] = "Test Set (Known)"
    unseen_df["Dataset"] = "Unseen Set (Future)"
    
    # Combine them for the full timeline
    df = pd.concat([test_df, unseen_df], ignore_index=True)
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Output files not found. Please ensure you have run 03_model.py so the CSV files are in outputs/")
    st.stop()

# ── SIDEBAR CONTROLS ──
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3050/3050525.png", width=100) # Generic food icon
st.sidebar.header("Filter Options")

region = st.sidebar.selectbox("Select Region", sorted(df["region"].unique()))

# Filter commodities based on the selected region
region_df = df[df["region"] == region]
commodity_group = st.sidebar.selectbox("Select Commodity Group", sorted(region_df["commodity_group"].unique()))

group_df = region_df[region_df["commodity_group"] == commodity_group]
commodity = st.sidebar.selectbox("Select Specific Commodity", sorted(group_df["commodity"].unique()))

filtered_df = group_df[group_df["commodity"] == commodity].sort_values("date")

# ── MAIN LAYOUT ──
st.subheader(f"Price Forecast: {commodity} in {region}")

# Build interactive Plotly chart
fig = go.Figure()

# 1. Confidence Interval (Shaded Area)
fig.add_trace(go.Scatter(
    x=pd.concat([filtered_df["date"], filtered_df["date"][::-1]]),
    y=pd.concat([filtered_df["upper"], filtered_df["lower"][::-1]]),
    fill='toself',
    fillcolor='rgba(56, 189, 248, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=True,
    name='90% Conformal Interval'
))

# 2. Actual Prices
fig.add_trace(go.Scatter(
    x=filtered_df["date"],
    y=filtered_df["actual"],
    mode='lines+markers',
    name='Actual Price',
    line=dict(color='#fb923c', width=2),
    marker=dict(size=6)
))

# 3. Predicted Prices
fig.add_trace(go.Scatter(
    x=filtered_df["date"],
    y=filtered_df["pred"],
    mode='lines',
    line=dict(color='#38bdf8', width=2, dash='dash'),
    name='Predicted Price'
))

# Highlight Unseen cutoff boundary
cutoff_date = "2025-09-01"
fig.add_shape(type="line", x0=cutoff_date, y0=0, x1=cutoff_date, y1=1, yref="paper", line=dict(color="#ef4444", dash="dash"))
fig.add_annotation(x=cutoff_date, y=1.05, yref="paper", text="← Test | Unseen", showarrow=False, xanchor="right", font=dict(color="#ef4444"))

# Formatting
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price (PHP ₱)",
    template="plotly_dark",
    hovermode="x unified",
    height=550,
    margin=dict(l=0, r=0, t=30, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# ── METRICS ──
st.subheader("Local Performance Metrics")
col1, col2, col3 = st.columns(3)

# Calculate metrics for the specific slice they selected
rmse = (filtered_df["abs_err"]**2).mean()**0.5
mae = filtered_df["abs_err"].mean()
coverage = filtered_df["covered"].mean() * 100

col1.metric("Root Mean Squared Error (RMSE)", f"₱{rmse:.2f}")
col2.metric("Mean Absolute Error (MAE)", f"₱{mae:.2f}")
col3.metric("True Model Coverage", f"{coverage:.1f}%")

st.markdown("---")
st.markdown("**Methodology:** XGBoost Regressor optimized via Optuna with Split Conformal Prediction (MAPIE) for robust uncertainty quantification.")
