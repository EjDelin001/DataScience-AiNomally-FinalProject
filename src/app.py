import sys
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
from itertools import groupby

# ── Make src/ importable ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from oni_generator import generate_oni_series, SCENARIO_PARAMS, get_scenario_summary

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="AiNomally Food Price Forecast",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Philippine Food Price Forecasting")
st.markdown("**Developed by Team AiNomally for CS 322: Data Science**")
st.markdown(
    "This prototype demonstrates our machine learning pipeline predicting agricultural "
    "commodity prices across Philippine regions, complete with conformal prediction "
    "uncertainty intervals."
)

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"

FORECAST_START = pd.Timestamp("2026-04-01")
FORECAST_END   = pd.Timestamp("2050-12-01")

# ── ENSO visual config ────────────────────────────────────────────────
ENSO_COLORS = {
    "El Nino": "rgba(248, 113, 113, 0.18)",   # warm red, transparent
    "La Nina": "rgba(56, 189, 248, 0.18)",    # cool blue, transparent
    "Neutral": "rgba(0, 0, 0, 0)",            # invisible
}
ENSO_LINE_COLORS = {
    "El Nino": "#f87171",
    "Neutral": "#94a3b8",
    "La Nina": "#38bdf8",
}
ENSO_PLAIN_LABEL = {
    "El Nino": "El Nino (Drier / Drought Risk)",
    "La Nina": "La Nina (Wetter / Flood Risk)",
    "Neutral": "Neutral",
}

# ── Scenario display names ────────────────────────────────────────────
SCENARIO_KEYS = list(SCENARIO_PARAMS.keys())   # ["El Nino","Neutral","La Nina"]
SCENARIO_DISPLAY = {
    "El Nino": "El Nino Dominant — more droughts, higher temperatures",
    "Neutral":  "Typical Weather — near-average conditions",
    "La Nina":  "La Nina Dominant — more rainfall, flooding risk",
}
SCENARIO_DISPLAY_LIST = [SCENARIO_DISPLAY[k] for k in SCENARIO_KEYS]

# ── Data loaders ─────────────────────────────────────────────────────
@st.cache_data
def load_historical():
    test_df   = pd.read_csv(OUTPUTS_DIR / "food_price_predictions_v4.csv",        parse_dates=["date"])
    unseen_df = pd.read_csv(OUTPUTS_DIR / "food_price_predictions_unseen_v4.csv", parse_dates=["date"])
    test_df["Dataset"]   = "Test Set (Known)"
    unseen_df["Dataset"] = "Unseen Set (Future)"
    return pd.concat([test_df, unseen_df], ignore_index=True)

@st.cache_data
def load_forecast():
    path = OUTPUTS_DIR / "food_price_forecast_2050.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["date"])

@st.cache_data
def get_oni_series_cached(scenario: str):
    return generate_oni_series(
        scenario=scenario,
        start_date=FORECAST_START,
        end_date=FORECAST_END,
        seed=42,
    )

@st.cache_data
def load_group_confidence():
    return {"Fish": "91%", "Rice": "95%", "Meat": "92%", "Vegetables": "91%"}

try:
    hist_df = load_historical()
except FileNotFoundError:
    st.error("Output files not found. Please run `03_model.py` first.")
    st.stop()

fc_df            = load_forecast()
group_confidence = load_group_confidence()
FORECAST_MISSING = fc_df is None


# ── Helper: group consecutive same-phase months into periods ──────────
def get_enso_periods(fc_sub: pd.DataFrame):
    """
    Returns list of (start_date, end_date, phase) for consecutive runs
    of the same ENSO phase. Used to draw background shading bands.
    """
    periods = []
    if "enso_phase" not in fc_sub.columns or fc_sub.empty:
        return periods
    data  = fc_sub.sort_values("date")[["date", "enso_phase"]].values.tolist()
    for phase, group in groupby(data, key=lambda x: x[1]):
        rows  = list(group)
        start = pd.Timestamp(rows[0][0])
        end   = pd.Timestamp(rows[-1][0]) + pd.DateOffset(months=1)
        periods.append((start, end, phase))
    return periods


# ════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════
st.sidebar.header("Filter Options")

region = st.sidebar.selectbox("Select Region", sorted(hist_df["region"].unique()))

region_hist     = hist_df[hist_df["region"] == region]
commodity_group = st.sidebar.selectbox(
    "Select Commodity Group", sorted(region_hist["commodity_group"].unique())
)
group_hist = region_hist[region_hist["commodity_group"] == commodity_group]
commodity  = st.sidebar.selectbox(
    "Select Specific Commodity", sorted(group_hist["commodity"].unique())
)

# ── Climate Scenario selector ────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("Climate Outlook")
st.sidebar.caption(
    "Choose a long-term climate scenario for the forecast. Each scenario "
    "creates a realistic oscillating pattern — not a single constant value."
)

scenario_display = st.sidebar.selectbox(
    "Climate Scenario",
    options=SCENARIO_DISPLAY_LIST,
    index=1,
    label_visibility="collapsed",
)
scenario = SCENARIO_KEYS[SCENARIO_DISPLAY_LIST.index(scenario_display)]

# ── ENSO Calendar (sidebar) ──────────────────────────────────────────
oni_series = get_oni_series_cached(scenario)
summary    = get_scenario_summary(oni_series)

# Build yearly dominant phase for calendar view
oni_series_reset = oni_series.reset_index()
oni_yearly = (
    oni_series_reset
    .assign(year=lambda d: d["date"].dt.year)
    .groupby("year")["oni"]
    .mean()
    .reset_index()
)
oni_yearly["phase"] = oni_yearly["oni"].apply(
    lambda v: "El Nino" if v >= 0.5 else ("La Nina" if v <= -0.5 else "Neutral")
)

bar_colors = [ENSO_LINE_COLORS[p] for p in oni_yearly["phase"]]
cal_fig = go.Figure(go.Bar(
    x=oni_yearly["year"],
    y=[1] * len(oni_yearly),
    marker=dict(color=bar_colors, line=dict(width=0)),
    hovertemplate="<b>%{x}</b><br>" +
                  "<extra></extra>",
    customdata=oni_yearly["phase"],
    hovertext=oni_yearly["phase"],
))
cal_fig.update_layout(
    height=55,
    margin=dict(l=0, r=0, t=18, b=0),
    template="plotly_dark",
    showlegend=False,
    bargap=0.05,
    yaxis=dict(visible=False, showticklabels=False),
    xaxis=dict(tickfont=dict(size=9), tickangle=0, dtick=5),
    title=dict(text="Annual Climate Outlook (red=El Nino, blue=La Nina, gray=Neutral)",
               font=dict(size=9, color="#94a3b8"), x=0),
)
st.sidebar.plotly_chart(cal_fig, use_container_width=True)

# Phase breakdown pills
st.sidebar.markdown(
    f"<small>"
    f"<span style='color:#f87171'>El Nino: {summary['pct_el_nino']:.0f}% of months</span> &nbsp;"
    f"<span style='color:#94a3b8'>Neutral: {summary['pct_neutral']:.0f}%</span> &nbsp;"
    f"<span style='color:#38bdf8'>La Nina: {summary['pct_la_nina']:.0f}%</span>"
    f"</small>",
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    "<small style='color:#64748b'>"
    "El Nino = warmer/drier → drought risk for crops. "
    "La Nina = cooler/wetter → flooding, more typhoons."
    "</small>",
    unsafe_allow_html=True,
)


# ════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["Historical Analysis", "2050 Long-Range Forecast"])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — HISTORICAL ANALYSIS
# ════════════════════════════════════════════════════════════════════
with tab1:
    filtered_df = (
        group_hist[group_hist["commodity"] == commodity]
        .sort_values("date")
    )

    st.subheader(f"Price Forecast: {commodity} in {region}")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=pd.concat([filtered_df["date"], filtered_df["date"][::-1]]),
        y=pd.concat([filtered_df["upper"], filtered_df["lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(56, 189, 248, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name=f"{group_confidence.get(commodity_group, '>=90%')} Conformal Interval"
    ))
    fig.add_trace(go.Scatter(
        x=filtered_df["date"],
        y=filtered_df["actual"],
        mode="lines+markers",
        name="Actual Price",
        line=dict(color="#fb923c", width=2),
        marker=dict(size=6)
    ))
    fig.add_trace(go.Scatter(
        x=filtered_df["date"],
        y=filtered_df["pred"],
        mode="lines",
        line=dict(color="#38bdf8", width=2, dash="dash"),
        name="Predicted Price"
    ))

    cutoff_date = "2025-09-01"
    fig.add_shape(
        type="line", x0=cutoff_date, y0=0, x1=cutoff_date, y1=1,
        yref="paper", line=dict(color="#ef4444", dash="dash")
    )
    fig.add_annotation(
        x=cutoff_date, y=1.05, yref="paper",
        text="<- Test | Unseen", showarrow=False,
        xanchor="right", font=dict(color="#ef4444")
    )
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Price (PHP)",
        template="plotly_dark", hovermode="x unified",
        height=520, margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Local Performance Metrics")
    col1, col2, col3 = st.columns(3)
    rmse     = (filtered_df["abs_err"] ** 2).mean() ** 0.5
    mae      = filtered_df["abs_err"].mean()
    coverage = filtered_df["covered"].mean() * 100
    col1.metric("RMSE",     f"P{rmse:.2f}")
    col2.metric("MAE",      f"P{mae:.2f}")
    col3.metric("Coverage", f"{coverage:.1f}%")

    st.markdown("---")
    st.markdown(
        "**Methodology:** XGBoost Regressor optimized via Optuna with Split Conformal "
        "Prediction (MAPIE) for robust uncertainty quantification."
    )

# ════════════════════════════════════════════════════════════════════
# TAB 2 — 2050 LONG-RANGE FORECAST
# ════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"Long-Range Forecast to 2050: {commodity} in {region}")

    # Plain-language scenario explanation
    SCENARIO_EXPLAINER = {
        "El Nino": (
            "**El Nino scenario** — El Nino events bring warmer temperatures and drier conditions "
            "to the Philippines, raising drought risk and potentially reducing crop yields. "
            "Red-shaded periods below indicate when El Nino conditions are active."
        ),
        "Neutral": (
            "**Typical Weather scenario** — Climate oscillates near normal conditions with "
            "no dominant El Nino or La Nina pattern. Gray shading indicates neutral periods."
        ),
        "La Nina": (
            "**La Nina scenario** — La Nina events bring heavier rainfall and increased typhoon "
            "activity, raising flood risk and disrupting fishing seasons. "
            "Blue-shaded periods below indicate when La Nina conditions are active."
        ),
    }
    st.info(SCENARIO_EXPLAINER[scenario])

    if FORECAST_MISSING:
        st.warning(
            "Forecast data not found. "
            "Please run `python src/04_forecast.py` once to generate "
            "`outputs/food_price_forecast_2050.csv`, then restart the app."
        )
    else:
        fc_filtered = fc_df[
            (fc_df["region"] == region) &
            (fc_df["commodity"] == commodity) &
            (fc_df["scenario"] == scenario)
        ].sort_values("date")

        if fc_filtered.empty:
            st.info("No forecast data available for this combination.")
        else:
            hist_anchor = (
                hist_df[
                    (hist_df["region"] == region) &
                    (hist_df["commodity"] == commodity)
                ]
                .sort_values("date")
                .tail(12)
            )

            fig2 = go.Figure()

            # ── Set axis type immediately so vrects work ───────────
            fig2.update_layout(xaxis=dict(type="date"))

            # ── Historical anchor (add FIRST to anchor axis range) ─
            if not hist_anchor.empty:
                fig2.add_trace(go.Scatter(
                    x=hist_anchor["date"],
                    y=hist_anchor["actual"],
                    mode="lines+markers",
                    name="Historical Price",
                    line=dict(color="#fb923c", width=2.5),
                    marker=dict(size=5),
                ))

            # ── Confidence band ────────────────────────────────────
            fig2.add_trace(go.Scatter(
                x=pd.concat([fc_filtered["date"], fc_filtered["date"][::-1]]),
                y=pd.concat([fc_filtered["upper"], fc_filtered["lower"][::-1]]),
                fill="toself",
                fillcolor="rgba(167, 139, 250, 0.12)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="90% Confidence Interval",
            ))

            # ── Single forecast line ───────────────────────────────
            fig2.add_trace(go.Scatter(
                x=fc_filtered["date"],
                y=fc_filtered["pred"],
                mode="lines",
                name="Projected Price",
                line=dict(color="#a78bfa", width=2.5),
                hovertemplate=(
                    "<b>%{x|%b %Y}</b><br>"
                    "Projected: <b>P%{y:.2f}</b><br>"
                    "<extra></extra>"
                ),
            ))

            # ── ENSO background shading ────────────────────────────
            enso_periods = get_enso_periods(fc_filtered)
            legend_phases_added = set()

            for start, end, phase in enso_periods:
                if phase == "Neutral":
                    continue

                fig2.add_vrect(
                    x0=str(start.date()), x1=str(end.date()),
                    fillcolor=ENSO_COLORS[phase],
                    layer="below",
                    line_width=0,
                )
                if phase not in legend_phases_added:
                    legend_phases_added.add(phase)
                    fig2.add_trace(go.Scatter(
                        x=[None], y=[None],
                        mode="markers",
                        marker=dict(symbol="square", size=12,
                                    color=ENSO_LINE_COLORS[phase]),
                        name=ENSO_PLAIN_LABEL[phase],
                        showlegend=True,
                    ))

            # Forecast start boundary
            fig2.add_shape(
                type="line", x0="2026-04-01", y0=0, x1="2026-04-01", y1=1,
                yref="paper", line=dict(color="#a78bfa", dash="dot", width=1.5)
            )
            fig2.add_annotation(
                x="2026-04-01", y=1.05, yref="paper",
                text="Forecast starts here",
                showarrow=False, xanchor="center",
                font=dict(color="#a78bfa", size=11)
            )

            fig2.update_layout(
                xaxis_title="Year",
                yaxis_title="Price (PHP)",
                template="plotly_dark",
                hovermode="x unified",
                height=580,
                margin=dict(l=0, r=0, t=40, b=0),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=11)
                ),
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    rangeselector=dict(
                        buttons=[
                            dict(count=5,  label="5 yrs",  step="year", stepmode="backward"),
                            dict(count=10, label="10 yrs", step="year", stepmode="backward"),
                            dict(count=25, label="25 yrs", step="year", stepmode="backward"),
                            dict(step="all", label="All"),
                        ],
                        bgcolor="#1e293b",
                        activecolor="#38bdf8",
                        font=dict(color="#e2e8f0"),
                    )
                ),
            )
            st.plotly_chart(fig2, use_container_width=True)

            # ── Legend explanation box ─────────────────────────────
            with st.expander("What do the shaded areas mean?", expanded=False):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(
                        "🔴 **Red shading = El Nino period**  \n"
                        "Warmer sea surface temperatures in the Pacific lead to drier, "
                        "hotter weather in the Philippines — increasing drought risk and "
                        "potentially reducing agricultural yields."
                    )
                with col_b:
                    st.markdown(
                        "🔵 **Blue shading = La Nina period**  \n"
                        "Cooler Pacific temperatures bring heavier rainfall, more typhoons, "
                        "and flooding risk — disrupting fishing seasons and damaging crops."
                    )
                st.markdown(
                    "**Unshaded areas = Neutral conditions** — typical weather with no "
                    "dominant El Nino or La Nina pattern."
                )

            # ── Summary metrics ────────────────────────────────────
            st.subheader("Price Projections")
            col1, col2, col3, col4 = st.columns(4)

            last_hist = hist_anchor["actual"].iloc[-1] if not hist_anchor.empty else None
            pred_2030 = fc_filtered[fc_filtered["date"].dt.year == 2030]["pred"].mean()
            pred_2040 = fc_filtered[fc_filtered["date"].dt.year == 2040]["pred"].mean()
            pred_2050 = fc_filtered[fc_filtered["date"].dt.year == 2050]["pred"].mean()

            def pct_delta(future, base):
                if base and not pd.isna(future):
                    return f"{((future/base)-1)*100:+.1f}% vs today"
                return None

            col1.metric("Last Known Price", f"P{last_hist:.2f}" if last_hist else "N/A",
                        help="Most recent actual price in the dataset (March 2026)")
            col2.metric("Avg Price in 2030",
                        f"P{pred_2030:.2f}" if not pd.isna(pred_2030) else "N/A",
                        delta=pct_delta(pred_2030, last_hist))
            col3.metric("Avg Price in 2040",
                        f"P{pred_2040:.2f}" if not pd.isna(pred_2040) else "N/A",
                        delta=pct_delta(pred_2040, last_hist))
            col4.metric("Avg Price in 2050",
                        f"P{pred_2050:.2f}" if not pd.isna(pred_2050) else "N/A",
                        delta=pct_delta(pred_2050, last_hist))

            st.caption(
                "Long-range forecasts accumulate uncertainty over time — the confidence "
                "interval widens the further into the future you look. These are projections "
                "based on historical price patterns and climate data, not certainties."
            )
