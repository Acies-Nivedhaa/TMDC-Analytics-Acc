# core/eda_missingness.py
from __future__ import annotations

import streamlit as st
import pandas as pd

from ui.components import render_table  # shared table helper

# How many columns to visualize in the bar chart (the table shows all)
TOP_K = 5


def _missing_by_col(df: pd.DataFrame) -> pd.Series:
    """Return missing counts per column, sorted descending."""
    return df.isna().sum().sort_values(ascending=False)


def render_missingness(df: pd.DataFrame) -> None:
    """
    Missingness subtab:
      - Headline with total nulls & affected columns
      - Count/Percent toggle for the bar chart
      - Bar chart for top-K columns
      - Complete table (all columns with missing values) inside an expander
    """
    miss = _missing_by_col(df)
    total_nulls = int(miss.sum())
    affected_cols = int((miss > 0).sum())

    st.markdown("**Missingness**")
    if affected_cols == 0:
        st.caption("No missing values detected.")
        return

    st.caption(
        f"{total_nulls:,} total nulls across {affected_cols} column(s). "
        f"Chart shows up to {TOP_K} columns automatically."
    )

    # Choose the metric for the bar chart
    metric = st.radio(
        "Column chart metric",
        ["Count", "Percent"],
        index=0,
        horizontal=True,
    )

    # Prepare chart data (top-K columns by missing count)
    chart_s = miss.head(TOP_K)
    if metric == "Percent":
        chart_s = (chart_s / max(1, len(df)) * 100).round(2)

    st.markdown("**By column**")

    # Streamlit bar_chart expects a numeric Series/DataFrame indexed by category
    chart_df = chart_s.rename("value").reset_index().rename(columns={"index": "column"})
    st.bar_chart(chart_df.set_index("column")["value"])

    # Full table with all columns that have any missing values
    with st.expander("See all columns with missing values", expanded=False):
        tbl = pd.DataFrame({
            "column": miss.index,
            "missing_count": miss.values,
            "missing_percent": (miss / max(1, len(df)) * 100).round(2),
            "dtype": [str(t) for t in df.dtypes.reindex(miss.index)]
        })
        tbl = tbl[tbl["missing_count"] > 0]
        render_table(tbl)
