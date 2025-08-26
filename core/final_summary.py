# core/final_summary.py
from __future__ import annotations
import pandas as pd
import numpy as np
import streamlit as st
from ui.components import section, kpi_row

def render_final_summary(df: pd.DataFrame, ss) -> None:
    # KPIs
    rows, cols = df.shape
    mem_mb = df.memory_usage(deep=True).sum() / (1024**2)
    kpi_row([
        ("Rows", f"{rows:,}"),
        ("Cols", cols),
        ("Memory (MB)", f"{mem_mb:.2f}"),
    ])

    # Outstanding issues (quick heuristics)
    with section("Outstanding issues", expandable=True):
        issues = []

        # Missing values
        na_cols = df.columns[df.isna().any()].tolist()
        if na_cols:
            issues.append(f"Missing values in: {', '.join(na_cols[:8])}" + (" …" if len(na_cols) > 8 else ""))

        # Suspected date strings not parsed
        cand = [c for c in df.select_dtypes(include=["object"]).columns]
        looks_dt = []
        for c in cand:
            s = pd.to_datetime(df[c], errors="coerce")
            share = s.notna().mean()
            if share >= 0.9:
                looks_dt.append(c)
        if looks_dt:
            issues.append(f"Text columns that look like datetimes: {', '.join(looks_dt[:8])}" + (" …" if len(looks_dt) > 8 else ""))

        # Low-cardinality strings (likely need encoding)
        low_card = [c for c in cand if df[c].nunique(dropna=False) <= 12]
        if low_card:
            issues.append(f"Low-cardinality text columns (consider encoding): {', '.join(low_card[:8])}" + (" …" if len(low_card) > 8 else ""))

        if not issues:
            st.success("No obvious outstanding issues detected.")
        else:
            for it in issues:
                st.markdown(f"- {it}")

    # One combined Preview & Export
    with section("Preview & Export", expandable=False):
        n = st.slider("Rows to preview", min_value=5, max_value=200, value=100, key="final_prev_rows")
        st.dataframe(df.head(n), use_container_width=True)
        st.download_button(
            "⬇️ Download processed CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name=f"{ss.active_ds}_processed.csv",
            mime="text/csv",
        )
