# core/preprocess_overview.py
from __future__ import annotations

import pandas as pd
import streamlit as st
from ui.components import section, render_table, kpi_row  # render_table kept for parity/consistency


# ---------- small helpers ----------

def _type_counts(df: pd.DataFrame) -> tuple[int, int, int, int]:
    """
    Return counts of columns by broad type buckets:
      - numeric (integers/floats)
      - categorical (object/string/category)
      - datetime (naive or tz-aware)
      - boolean
    """
    num = df.select_dtypes(include=["number"]).shape[1]
    cat = df.select_dtypes(include=["object", "string", "category"]).shape[1]
    dt  = df.select_dtypes(include=["datetime", "datetimetz"]).shape[1]
    bl  = df.select_dtypes(include=["bool"]).shape[1]
    return num, cat, dt, bl


# ---------- main UI ----------

def render_preprocess_overview(ss) -> None:
    """
    Preprocess ▸ Overview
    - Shows dataset KPIs and a preview (head/tail/sample).
    - Intentionally does not include quick-fix controls here.
    """
    # Guard: require an active dataset in session state
    if not ss.active_ds or ss.active_ds not in ss.datasets:
        st.info("Pick a dataset to begin.")
        st.stop()

    df = ss.datasets[ss.active_ds]
    rows, cols = df.shape
    num, cat, dt, bl = _type_counts(df)

    # Compact, plain-English glossary (optional read)
    with st.expander("Key terms (quick guide)", expanded=False):
        st.markdown(
            """
- **Data Preprocessing**: Preparing raw data for analysis
- **Data Cleaning**: Removing errors and inconsistencies
- **Transformation**: Converting data to suitable formats
- **Feature Engineering**: Creating new variables from existing ones
- **Normalization**: Scaling data to common ranges
- **KPIs** — Quick numbers about your dataset (rows, columns, etc.).
- **Numeric columns** — Numbers you can aggregate/plot as continuous values (ints/floats).
- **Categorical columns** — Labels or text-like values (e.g., product category, city).
- **Datetime columns** — Timestamps/dates you can resample by day/week/month.
- **Boolean columns** — True/False (or 0/1) flags.
- **Head / Tail / Random sample** — Show the first rows, last rows, or a random slice for quick inspection.
            """
        )

    # ---- KPIs ----
    kpi_row([
        ("Rows", f"{rows:,}"),
        ("Cols", f"{cols:,}"),
        ("Numeric cols", num),
        ("Categorical cols", cat),
        ("Datetime cols", dt),
        ("Boolean cols", bl),
    ])

    # ---- 1) Preview (first) ----
    # Keep controls/keys identical to avoid breaking existing state
    with section("Current dataset (preview)", expandable=False):
        view = st.radio(
            "View",
            ["Head", "Tail", "Random sample"],
            horizontal=True,
            key="pp_view",
        )
        nmax = max(5, min(100, rows))
        n = st.slider("Rows to show", 5, nmax, value=min(25, nmax), key="pp_n")

        if view == "Head":
            out = df.head(n)
        elif view == "Tail":
            out = df.tail(n)
        else:
            out = df.sample(n=min(n, rows), random_state=0)

        # Use Streamlit native table for a familiar preview; width autosizes to container
        st.dataframe(out, use_container_width=True)
