# core/preprocess_outliers.py
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from pandas.api.types import is_numeric_dtype

from ui.components import section, kpi_row, render_table

# ---------- helpers ----------

def _as_float_series(s: pd.Series) -> pd.Series:
    """Coerce to numeric float (handles booleans/extension dtypes)"""
    return pd.to_numeric(s, errors="coerce").astype("float64")

def _iqr_bounds(s: pd.Series, k: float):
    s = _as_float_series(s).dropna()
    if s.empty:
        return (-np.inf, np.inf)
    q1, q3 = np.nanpercentile(s.to_numpy(), [25, 75])
    iqr = q3 - q1
    return (q1 - k * iqr, q3 + k * iqr)

def _zscore_bounds(s: pd.Series, k: float):
    s = _as_float_series(s).dropna()
    if s.empty:
        return (-np.inf, np.inf)
    mu, sd = s.mean(), s.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return (-np.inf, np.inf)
    return (mu - k * sd, mu + k * sd)

def _modified_z_bounds(s: pd.Series, k: float):
    s = _as_float_series(s).dropna()
    if s.empty:
        return (-np.inf, np.inf)
    med = s.median()
    mad = (s - med).abs().median()
    if pd.isna(mad) or mad == 0:
        return (-np.inf, np.inf)
    scale = 1.4826 * mad  # consistent with std for normal
    return (med - k * scale, med + k * scale)

def _percentile_bounds(s: pd.Series, p_low: float, p_high: float):
    s = _as_float_series(s).dropna()
    if s.empty:
        return (-np.inf, np.inf)
    lo = np.clip(p_low, 0.0, 100.0)
    hi = np.clip(p_high, 0.0, 100.0)
    if hi <= lo:
        return (-np.inf, np.inf)
    lo_v, hi_v = np.nanpercentile(s.to_numpy(), [lo, hi])
    return (lo_v, hi_v)

def _calc_bounds(s: pd.Series, method: str, params: dict[str, float]):
    if method == "IQR (Tukey fences)":
        return _iqr_bounds(s, params.get("k_iqr", 1.5))
    if method == "Z-score (μ ± k·σ)":
        return _zscore_bounds(s, params.get("k_z", 3.0))
    if method == "Modified Z (median ± k·1.4826·MAD)":
        return _modified_z_bounds(s, params.get("k_mad", 3.5))
    if method == "Percentiles (p_low / p_high)":
        return _percentile_bounds(s, params.get("p_low", 1.0), params.get("p_high", 99.0))
    return (-np.inf, np.inf)

def _apply_action(col: pd.Series, bounds: tuple[float, float], action: str):
    lo, hi = bounds
    if action == "Clip to bounds (winsorize)":
        return _as_float_series(col).clip(lo, hi)
    if action == "Set outliers to NaN":
        s = _as_float_series(col)
        mask = (s < lo) | (s > hi)
        return s.mask(mask)
    if action == "Drop rows with outliers":
        # handled at frame-level
        return _as_float_series(col)
    return _as_float_series(col)

def _box_plot(df: pd.DataFrame, column: str, title: str):
    plot_df = pd.DataFrame({"value": _as_float_series(df[column])}).dropna()
    if plot_df.empty:
        return st.caption("No data to plot.")
    chart = (
        alt.Chart(plot_df)
        .mark_boxplot(size=60)
        .encode(y=alt.Y("value:Q", title=column))
        .properties(height=220, title=title)
    )
    st.altair_chart(chart, use_container_width=True)

# ---------- main UI ----------

def render_preprocess_outliers(ss) -> None:
    """Preprocess ▸ Outliers — multiple methods, preview, box-plot, english summary."""
    if not ss.active_ds or ss.active_ds not in ss.datasets:
        st.info("Pick a dataset to begin.")
        st.stop()

    df = ss.datasets[ss.active_ds]
    num_cols = [c for c in df.columns if is_numeric_dtype(df[c]) or df[c].dtype == "boolean"]

    if not num_cols:
        st.info("No numeric/boolean columns found.")
        return

    kpi_row([
        ("Numeric columns", len(num_cols)),
        ("Rows", f"{len(df):,}"),
    ])

    # defaults
    ss.setdefault("out_cols", num_cols[: min(8, len(num_cols))])
    ss.setdefault("out_method", "IQR (Tukey fences)")
    ss.setdefault("out_action", "Clip to bounds (winsorize)")
    ss.setdefault("out_k_iqr", 1.5)
    ss.setdefault("out_k_z", 3.0)
    ss.setdefault("out_k_mad", 3.5)
    ss.setdefault("out_p_low", 1.0)
    ss.setdefault("out_p_high", 99.0)
    ss.setdefault("out_vis_col", (num_cols[0] if num_cols else None))

    with section("Setup", expandable=False):
        st.caption("Pick columns, a detection method, and what to do with detected outliers.")
        c1, c2 = st.columns([1, 1])
        with c1:
            out_cols = st.multiselect(
                "Columns to process",
                options=num_cols,
                default=ss.out_cols,
                key="out_cols",
            )
        with c2:
            method = st.selectbox(
                "Method",
                [
                    "IQR (Tukey fences)",
                    "Z-score (μ ± k·σ)",
                    "Modified Z (median ± k·1.4826·MAD)",
                    "Percentiles (p_low / p_high)",
                ],
                index=[
                    "IQR (Tukey fences)",
                    "Z-score (μ ± k·σ)",
                    "Modified Z (median ± k·1.4826·MAD)",
                    "Percentiles (p_low / p_high)",
                ].index(ss.out_method),
                key="out_method",
            )

        c3, c4 = st.columns([1, 1])
        with c3:
            action = st.selectbox(
                "Action",
                ["Clip to bounds (winsorize)", "Set outliers to NaN", "Drop rows with outliers"],
                index=[
                    "Clip to bounds (winsorize)",
                    "Set outliers to NaN",
                    "Drop rows with outliers",
                ].index(ss.out_action),
                key="out_action",
            )

        # method params
        if method == "IQR (Tukey fences)":
            st.slider("IQR multiplier (k)", 1.0, 4.0, float(ss.out_k_iqr), step=0.25, key="out_k_iqr")
        elif method == "Z-score (μ ± k·σ)":
            st.slider("Std-dev multiplier (k)", 1.0, 6.0, float(ss.out_k_z), step=0.1, key="out_k_z")
        elif method == "Modified Z (median ± k·1.4826·MAD)":
            st.slider("MAD multiplier (k)", 1.0, 10.0, float(ss.out_k_mad), step=0.1, key="out_k_mad")
        else:
            lo_c, hi_c = st.columns(2)
            with lo_c:
                st.slider("Lower percentile", 0.0, 20.0, float(ss.out_p_low), step=0.5, key="out_p_low")
            with hi_c:
                st.slider("Upper percentile", 80.0, 100.0, float(ss.out_p_high), step=0.5, key="out_p_high")
            if ss.out_p_high <= ss.out_p_low:
                st.warning("Upper percentile must be greater than lower percentile.")

    # plain-English plan
    param_txt = {
        "IQR (Tukey fences)": f"k={ss.out_k_iqr:.2f}",
        "Z-score (μ ± k·σ)": f"k={ss.out_k_z:.2f}",
        "Modified Z (median ± k·1.4826·MAD)": f"k={ss.out_k_mad:.2f}",
        "Percentiles (p_low / p_high)": f"p_low={ss.out_p_low:.1f}%, p_high={ss.out_p_high:.1f}%",
    }[method]
    st.info(f"**Plan:** Detect outliers in **{len(out_cols)}** column(s) using **{method}** ({param_txt}); then **{action}**.")

    # bounds table
    def _compute_bounds_for_all():
        params = dict(
            k_iqr=ss.out_k_iqr,
            k_z=ss.out_k_z,
            k_mad=ss.out_k_mad,
            p_low=ss.out_p_low,
            p_high=ss.out_p_high,
        )
        rows = []
        for c in out_cols:
            s = _as_float_series(df[c])
            lo, hi = _calc_bounds(s, method, params)
            n_out = int(((s < lo) | (s > hi)).sum())
            rows.append({"column": c, "lower": lo, "upper": hi, "outliers": n_out})
        return pd.DataFrame(rows)

    with section("What will change", expandable=False):
        if not out_cols:
            st.caption("Select at least one numeric column.")
        else:
            render_table(_compute_bounds_for_all(), height=240)

    # visualization
    with section("Preview (box plot)", expandable=True):
        vis_options = out_cols or num_cols
        vis_col = st.selectbox("Visualize a single column", options=vis_options, index=0, key="out_vis_col")
        if vis_col:
            _box_plot(df, vis_col, title="Before")
            params = dict(
                k_iqr=ss.out_k_iqr, k_z=ss.out_k_z, k_mad=ss.out_k_mad,
                p_low=ss.out_p_low, p_high=ss.out_p_high
            )
            lo, hi = _calc_bounds(df[vis_col], method, params)
            s_after = _apply_action(df[vis_col], (lo, hi), action)
            if action == "Drop rows with outliers":
                s_before = _as_float_series(df[vis_col])
                mask_keep = (s_before >= lo) & (s_before <= hi)
                df_after = df.loc[mask_keep].copy()
                _box_plot(df_after, vis_col, title="After (drop rows)")
            else:
                df_tmp = df.copy()
                df_tmp[vis_col] = s_after
                _box_plot(df_tmp, vis_col, title="After")

    # apply
    def _apply_all(dfin: pd.DataFrame) -> pd.DataFrame:
        params = dict(
            k_iqr=ss.out_k_iqr, k_z=ss.out_k_z, k_mad=ss.out_k_mad,
            p_low=ss.out_p_low, p_high=ss.out_p_high
        )
        d = dfin.copy()
        if action == "Drop rows with outliers":
            keep_mask = pd.Series(True, index=d.index)
            for c in out_cols:
                s = _as_float_series(d[c])
                lo, hi = _calc_bounds(s, method, params)
                keep_mask &= (s >= lo) & (s <= hi)
            return d.loc[keep_mask].reset_index(drop=True)
        for c in out_cols:
            s = _as_float_series(d[c])
            lo, hi = _calc_bounds(s, method, params)
            d[c] = _apply_action(s, (lo, hi), action)
        return d

    cprev, capply = st.columns([1, 1])
    with cprev:
        if st.button("Preview", key="out_preview_btn"):
            prev = _apply_all(df)
            st.caption(f"Result: **{prev.shape[0]:,} × {prev.shape[1]:,}** (was {df.shape[0]:,} × {df.shape[1]:,})")
            st.dataframe(prev.head(25), use_container_width=True)
    with capply:
        if st.button("Apply", key="out_apply_btn"):
            out = _apply_all(df)
            ss.df_history.append(df.copy())
            ss.datasets[ss.active_ds] = out
            st.success("Outlier handling applied.")
