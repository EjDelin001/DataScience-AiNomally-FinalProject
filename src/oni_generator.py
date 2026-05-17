"""
oni_generator.py
----------------
Generates synthetic monthly ONI (Oceanic Nino Index) time series
for long-range forecast scenarios.

Real ENSO behaviour this model captures:
  1. ~4-year oscillation cycle between warm and cool events
  2. Boreal-winter phase locking — events peak in Dec-Feb,
     weaken in Apr-Jun (the "spring predictability barrier")
  3. Scenario-specific mean bias and amplitude
  4. Small realistic noise (sigma ~0.12 degC)

Deterministic output (fixed seed=42) ensures the mini-chart in the
app shows the EXACT same ONI series used to generate the forecasts.
"""

import numpy as np
import pandas as pd

# ── Scenario parameters ──────────────────────────────────────────────
SCENARIO_PARAMS = {
    "El Nino": {
        "bias":           0.40,   # positive mean — more warm events
        "amplitude":      1.30,   # strong oscillation
        "period_months":  48,     # 4-year cycle
        "label":          "El Nino Dominant",
        "color":          "#f87171",   # red
    },
    "Neutral": {
        "bias":           0.00,
        "amplitude":      0.55,   # weak oscillation — near-average
        "period_months":  44,
        "label":          "Typical Variability",
        "color":          "#94a3b8",   # slate
    },
    "La Nina": {
        "bias":          -0.40,   # negative mean — more cool events
        "amplitude":      1.30,
        "period_months":  52,     # slightly longer cycle
        "label":          "La Nina Dominant",
        "color":          "#38bdf8",   # sky blue
    },
}

EL_NINO_THRESHOLD =  0.5   # ONI >= +0.5 → El Nino
LA_NINA_THRESHOLD = -0.5   # ONI <= -0.5 → La Nina


def _oni_to_phase(oni: float) -> str:
    if oni >= EL_NINO_THRESHOLD:
        return "El Nino"
    elif oni <= LA_NINA_THRESHOLD:
        return "La Nina"
    return "Neutral"


def generate_oni_series(
    scenario: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    seed: int = 42,
    historical_oni: dict = None,
) -> pd.DataFrame:
    """
    Generate a realistic synthetic monthly ONI time series.

    Parameters
    ----------
    scenario : str
        One of "El Nino", "Neutral", "La Nina"
    start_date : pd.Timestamp
        First month of forecast (e.g. 2026-04-01)
    end_date : pd.Timestamp
        Last month of forecast  (e.g. 2050-12-01)
    seed : int
        Fixed seed for reproducibility. Use 42 everywhere.
    historical_oni : dict, optional
        {pd.Timestamp -> float}  Actual ONI values from the dataset,
        used to correctly populate oni_lag_3 and oni_lag_6 for the
        first months of the forecast (before synthetic values exist).

    Returns
    -------
    pd.DataFrame
        Columns: date, oni, enso_phase, oni_lag_3, oni_lag_6
    """
    params = SCENARIO_PARAMS[scenario]
    dates  = pd.date_range(start=start_date, end=end_date, freq="MS")
    rng    = np.random.default_rng(seed)

    # Unified ONI lookup: historical actuals + synthetic future values
    all_oni: dict = dict(historical_oni) if historical_oni else {}

    # ── Generate synthetic ONI month-by-month ─────────────────────
    synthetic_oni: dict = {}
    for i, dt in enumerate(dates):
        month = dt.month

        # Base ENSO oscillation (sinusoidal, phase-biased by scenario)
        base_signal = params["amplitude"] * np.sin(
            2 * np.pi * i / params["period_months"]
        )

        # Seasonal phase-locking:
        #   cos(0)  = 1   at month = 1  (Jan) → amplifies events in DJF
        #   cos(pi) = -1  at month = 7  (Jul) → suppresses events in JJA
        seasonal_mod = 1.0 + 0.35 * np.cos(2 * np.pi * (month - 1) / 12)

        # Combine bias + modulated oscillation + noise
        oni_val = float(np.clip(
            params["bias"] + base_signal * seasonal_mod + rng.normal(0, 0.12),
            -3.0, 3.0
        ))

        all_oni[dt]      = oni_val
        synthetic_oni[dt] = oni_val

    # ── Build DataFrame with lags ──────────────────────────────────
    records = []
    for dt in dates:
        oni_val = synthetic_oni[dt]

        # Lag dates
        lag3_dt = pd.Timestamp(
            (dt - pd.DateOffset(months=3)).year,
            (dt - pd.DateOffset(months=3)).month, 1
        )
        lag6_dt = pd.Timestamp(
            (dt - pd.DateOffset(months=6)).year,
            (dt - pd.DateOffset(months=6)).month, 1
        )

        records.append({
            "date":       dt,
            "oni":        round(oni_val, 3),
            "enso_phase": _oni_to_phase(oni_val),
            "oni_lag_3":  round(all_oni.get(lag3_dt, oni_val), 3),
            "oni_lag_6":  round(all_oni.get(lag6_dt, oni_val), 3),
        })

    return pd.DataFrame(records).set_index("date")


def get_scenario_summary(oni_df: pd.DataFrame) -> dict:
    """
    Compute summary statistics for displaying in the sidebar.
    """
    counts = oni_df["enso_phase"].value_counts()
    total  = len(oni_df)
    return {
        "mean_oni":    round(oni_df["oni"].mean(), 2),
        "max_oni":     round(oni_df["oni"].max(),  2),
        "min_oni":     round(oni_df["oni"].min(),  2),
        "pct_el_nino": round(counts.get("El Nino", 0) / total * 100, 1),
        "pct_neutral": round(counts.get("Neutral",  0) / total * 100, 1),
        "pct_la_nina": round(counts.get("La Nina",  0) / total * 100, 1),
    }
