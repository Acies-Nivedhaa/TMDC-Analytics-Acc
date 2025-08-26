# core/eda_missingness.py
from __future__ import annotations
import streamlit as st
import pandas as pd
import pandas.api.types as ptypes
from ui.components import render_table  # you already have this

TOP_K = 5  # columns shown in the chart

def _missing_by_col(df: pd.DataFrame) -> pd.Series:
    """Return missing counts per column, sorted desc."""
    return df.isna().sum().sort_values(ascending=False)

def render_missingness(df: pd.DataFrame) -> None:
    """
    Missingness subtab:
      - headline with total nulls & affected columns
      - Count/Percent toggle
      - bar chart (top-K columns)
      - full table in an expander
    """
    miss = _missing_by_col(df)
    total_nulls = int(miss.sum())
    affected = int((miss > 0).sum())

    st.markdown("**Missingness**")
    if affected == 0:
        st.caption("No missing values detected.")
        return

    st.caption(f"{total_nulls:,} total nulls across {affected} column(s). "
               f"Chart shows up to {TOP_K} columns automatically.")

    # metric toggle
    metric = st.radio("Column chart metric", ["Count", "Percent"],
                      index=0, horizontal=True)

    # chart data
    chart_s = miss.head(TOP_K)
    if metric == "Percent":
        chart_s = (chart_s / max(1, len(df)) * 100).round(2)

    st.markdown("**By column**")
    chart_df = chart_s.rename("value").reset_index().rename(columns={"index": "column"})
    # bar chart expects index -> value
    st.bar_chart(chart_df.set_index("column")["value"])

    # full table
    with st.expander("See all columns with missing values", expanded=False):
        tbl = pd.DataFrame({
            "column": miss.index,
            "missing_count": miss.values,
            "missing_percent": (miss / max(1, len(df)) * 100).round(2),
            "dtype": [str(t) for t in df.dtypes.reindex(miss.index)]
        })
        tbl = tbl[tbl["missing_count"] > 0]
        render_table(tbl)
