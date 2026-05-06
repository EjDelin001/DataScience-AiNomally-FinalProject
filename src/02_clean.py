"""
02_clean.py
-----------
Cleans intermediate datasets and produces the final analysis-ready panel.

Architecture:
1. Categorizes commodities into primary groups (Rice, Meat, Fish, Vegetables).
2. Drops statistically sparse time-series using a 50% windowed completeness threshold from 2010.
3. Filters redundant, misclassified, or noisy commodity identifiers.
4. Outputs final `panel_food_prices_ph_clean.csv`.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
INPUT  = BASE_DIR / "data" / "interim" / "panel_wfp_oni.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT = OUTPUT_DIR / "panel_food_prices_ph_clean.csv"

# ── 1. load ─────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT, parse_dates=["date"])
print(f"Loaded  {len(df):,} rows × {df.shape[1]} columns")

# ── 2. drop unwanted categories outright ────────────────────────────────────
DROP_CATEGORIES = {"pulses and nuts", "oil and fats", "miscellaneous food"}
df = df[~df["category"].isin(DROP_CATEGORIES)].copy()
print(f"After dropping sparse categories: {len(df):,} rows")

# ── 3. map category → commodity_group ───────────────────────────────────────
FISH_KEYWORDS = ("fish", "shrimp", "crab", "anchov", "tuna", "mackerel",
                 "milkfish", "tilapia", "fusilier", "roundscad",
                 "slipmouth", "threadfin")

def assign_group(row) -> str | None:
    cat  = row["category"]
    name = row["commodity"].lower()

    if cat == "cereals and tubers":
        return "Rice" if "rice" in name else None

    if cat == "vegetables and fruits":
        return "Vegetables"

    if cat == "meat, fish and eggs":
        return "Fish" if any(kw in name for kw in FISH_KEYWORDS) else "Meat"

    return None

df["commodity_group"] = df.apply(assign_group, axis=1)

before = len(df)
df = df[df["commodity_group"].notna()].copy()
print(f"After non-rice cereal removal: {len(df):,} rows  (dropped {before - len(df):,})")

# ── 4. drop specific bad / duplicate commodities ─────────────────────────────
DROP_COMMODITIES = {
    # Meat group
    "Chicken",                  # near-duplicate of Meat (chicken, whole)
    "Meat (pork, with fat)",    # correlation 0.989 with Meat (pork)
    "Meat (pork, hock)",        # correlation 0.942–0.950 with other pork
    # Fish group
    "Fish (fresh)",             # discontinued 2019, too sparse
    "Shrimp (endeavor)",        # too sparse; Shrimp (tiger) is representative
    # Rice group
    "Rice (milled, superior)",  # only 6 of 17 regions
    # Vegetables group
    "Garlic (small)",           # too sparse
    "Garlic (large)",           # too sparse
    "Onions (white)",           # only 0/17 regions pass completeness
    "Onions (red)",             # only 2/17 regions pass completeness
    "Sweet potatoes",           # only 0/17 regions pass completeness
    "Tomatoes",                 # only 2/17 regions pass completeness
    # Misclassified
    "Mangoes (carabao)",        # misclassified into meat, fish and eggs
    "Eggs (duck)",              # misclassified as Meat, contaminates Meat calibration
}

before = len(df)
df = df[~df["commodity"].isin(DROP_COMMODITIES)].copy()
print(f"After dropping bad/duplicate commodities: {len(df):,} rows  (dropped {before - len(df):,})")

# ── 5. windowed completeness filter ─────────────────────────────────────────
#
#   WHY WINDOWED:
#   The original filter measured completeness from each series' own first
#   observation. For commodities tracked since 2000, this produces a 26-year
#   expected span. Even a well-observed series with 160 months of data scores
#   only 51% completeness — just under the old 60% threshold — and gets
#   dropped entirely, leaving us with only post-2020 data.
#
#   FIX: Measure completeness from WINDOW_START (2010-01-01) onwards.
#   Series with data before 2010 simply get their pre-2010 rows kept as a
#   bonus; their survival is judged on recent consistency only.
#   Threshold relaxed to 50% to match the slightly noisier older data.

WINDOW_START = pd.Timestamp("2010-01-01")
THRESHOLD    = 0.50

def completeness_windowed(grp: pd.DataFrame) -> float:
    grp_win = grp[grp["date"] >= WINDOW_START]
    if len(grp_win) == 0:
        return 0.0
    first    = grp_win["date"].min()
    last     = grp_win["date"].max()
    expected = (last.year - first.year) * 12 + (last.month - first.month) + 1
    return grp_win["date"].nunique() / expected if expected > 0 else 0.0

completeness_df = (
    df.groupby(["region", "commodity"])
    .apply(completeness_windowed, include_groups=False)
    .reset_index(name="completeness")
)

sparse_pairs = completeness_df.loc[
    completeness_df["completeness"] < THRESHOLD, ["region", "commodity"]
]
print(f"\nRegion-commodity series failing < {THRESHOLD:.0%} windowed completeness: "
      f"{len(sparse_pairs):,}")

before = len(df)
df = df.merge(
    sparse_pairs.assign(_drop=True),
    on=["region", "commodity"],
    how="left"
)
df = df[df["_drop"].isna()].drop(columns=["_drop"]).copy()
print(f"After completeness filter: {len(df):,} rows  (dropped {before - len(df):,})")

# ── 6. final column order & save ─────────────────────────────────────────────
FINAL_COLS = [
    "date", "region", "commodity", "commodity_group",
    "category", "oni", "enso_phase", "price_php"
]
df = df[FINAL_COLS].sort_values(["date", "region", "commodity"]).reset_index(drop=True)

df.to_csv(OUTPUT, index=False)

# ── 7. summary ───────────────────────────────────────────────────────────────
print(f"\n{'─'*55}")
print(f"Output saved → {OUTPUT}")
print(f"{'─'*55}")
print(f"  Rows            : {len(df):,}")
print(f"  Columns         : {list(df.columns)}")
print(f"  Date range      : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"  Regions         : {df['region'].nunique()}")
print(f"  Commodities     : {df['commodity'].nunique()}")

print(f"\nCommodity groups:")
print(
    df.groupby("commodity_group")["commodity"]
    .nunique()
    .rename("n_commodities")
    .to_string()
)

print(f"\nCommodities per group (with earliest date and region count):")
for grp, sub in df.groupby("commodity_group"):
    print(f"\n  [{grp}]")
    stats = (
        sub.groupby("commodity")
        .agg(first=("date", "min"), regions=("region", "nunique"))
        .sort_values("first")
    )
    for comm, row in stats.iterrows():
        print(f"    • {comm:<45s}  from {row['first'].date()}  ({row['regions']:2d} regions)")

print(f"\nNull check:\n{df.isnull().sum().to_string()}")