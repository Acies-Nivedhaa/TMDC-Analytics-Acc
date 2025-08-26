# core/preprocess_overview.py
from __future__ import annotations
import pandas as pd
import streamlit as st
from ui.components import section, render_table, kpi_row

def _type_counts(df: pd.DataFrame):
    num = df.select_dtypes(include=["number"]).shape[1]
    cat = df.select_dtypes(include=["object", "string", "category"]).shape[1]
    dt  = df.select_dtypes(include=["datetime", "datetimetz"]).shape[1]
    bl  = df.select_dtypes(include=["bool"]).shape[1]
    return num, cat, dt, bl

def render_preprocess_overview(ss) -> None:
    """Preprocess ▸ Overview: show dataset KPIs and a preview first, then light schema/missingness.
    (No quick-fix UI here by request.)"""
    if not ss.active_ds or ss.active_ds not in ss.datasets:
        st.info("Pick a dataset to begin.")
        st.stop()

    df = ss.datasets[ss.active_ds]
    rows, cols = df.shape
    num, cat, dt, bl = _type_counts(df)

    # KPIs
    kpi_row([
        ("Rows", f"{rows:,}"),
        ("Cols", f"{cols:,}"),
        ("Numeric cols", num),
        ("Categorical cols", cat),
        ("Datetime cols", dt),
        ("Boolean cols", bl),
    ])

    # ---- 1) Preview (first) ----
    with section("Current dataset (preview)", expandable=False):
        view = st.radio("View", ["Head", "Tail", "Random sample"], horizontal=True, key="pp_view")
        nmax = max(5, min(100, rows))
        n = st.slider("Rows to show", 5, nmax, value=min(25, nmax), key="pp_n")
        if view == "Head":
            out = df.head(n)
        elif view == "Tail":
            out = df.tail(n)
        else:
            out = df.sample(n=min(n, rows), random_state=0)
        st.dataframe(out, use_container_width=True)


