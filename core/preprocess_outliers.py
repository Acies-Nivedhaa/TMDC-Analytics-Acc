from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from pandas.api.types import is_numeric_dtype

from ui.components import section, kpi_row, render_table


# ---------- helpers ----------

def _iqr_bounds(s: pd.Series, k: float):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (q1 - k * iqr, q3 + k * iqr)

def _zscore_bounds(s: pd.Series, k: float):
    mu, sd = s.mean(), s.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return (-np.inf, np.inf)
    return (mu - k * sd, mu + k * sd)

def _modified_z_bounds(s: pd.Series, k: float):
    med = s.median()
    mad = (s - med).abs().median()
    if pd.isna(mad) or mad == 0:
        return (-np.inf, np.inf)
    # 1.4826 makes MAD consistent with std for normal data
    scale = 1.4826 * mad
    return (med - k * scale, med + k * scale)

def _percentile_bounds(s: pd.Series, p_low: float, p_high: float):
    lo = np.clip(p_low / 100.0, 0.0, 1.0)
    hi = np.clip(p_high / 100.0, 0.0, 1.0)
    if hi <= lo:
        return (-np.inf, np.inf)
    return tuple(s.quantile([lo, hi]).tolist())

def _calc_bounds(s: pd.Series, method: str, params: dict[str, float]):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return (-np.inf, np.inf)

    if method == "IQR (Tukey fences)":
        return _iqr_bounds(s, params.get("k_iqr", 1.5))
    if method == "Z-score (μ ± k·σ)":
        return _zscore_bounds(s, params.get("k_z", 3.0))
    if method == "Modified Z (median ± k·1.4826·MAD)":
        return _modified_z_bounds(s, params.get("k_mad", 3.5))
    if method == "Percentiles (p_low / p_high)":
        return _percentile_bounds(s, params.get("p_low", 1.0), params.get("p_high", 99.0))

    # fallback
    return (-np.inf, np.inf)

def _apply_action(col: pd.Series, bounds: tuple[float, float], action: str):
    lo, hi = bounds
    if action == "Clip to bounds (winsorize)":
        return col.clip(lo, hi)
    if action == "Set outliers to NaN":
        mask = (col < lo) | (col > hi)
        return col.mask(mask)
    if action == "Drop rows with outliers":
        # Handled at frame-level (we just return original col here)
        return col
    return col


def _box_plot(df: pd.DataFrame, column: str, title: str):
    # altair boxplot; df[column] should be numeric
    plot_df = df[[column]].rename(columns={column: "value"}).dropna()
    if plot_df.empty:
        return st.caption("No data to plot.")
    chart = alt.Chart(plot_df).mark_boxplot(size=60).encode(y=alt.Y("value:Q", title=column)).properties(height=220, title=title)
    st.altair_chart(chart, use_container_width=True)


# ---------- main UI ----------

def render_preprocess_outliers(ss) -> None:
    """Preprocess ▸ Outliers — multiple methods, preview, box-plot, english summary."""
    if not ss.active_ds or ss.active_ds not in ss.datasets:
        st.info("Pick a dataset to begin.")
        st.stop()

    df = ss.datasets[ss.active_ds]
    num_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
    total_num = len(num_cols)

    if total_num == 0:
        st.info("No numeric columns found.")
        return

    kpi_row([
        ("Numeric columns", total_num),
        ("Rows", f"{len(df):,}"),
    ])

    # Persist choices
    ss.setdefault("out_cols", num_cols[:min(8, total_num)])
    ss.setdefault("out_method", "IQR (Tukey fences)")
    ss.setdefault("out_action", "Clip to bounds (winsorize)")
    ss.setdefault("out_k_iqr", 1.5)
    ss.setdefault("out_k_z", 3.0)
    ss.setdefault("out_k_mad", 3.5)
    ss.setdefault("out_p_low", 1.0)
    ss.setdefault("out_p_high", 99.0)
    ss.setdefault("out_vis_col", num_cols[0])

    with section("Setup", expandable=False):
        st.caption("Pick columns, a detection method, and what to do with detected outliers.")
        cols_row1 = st.columns([1, 1])
        with cols_row1[0]:
            out_cols = st.multiselect(
                "Columns to process",
                options=num_cols,
                default=ss.out_cols,
                key="out_cols",
            )

        with cols_row1[1]:
            method = st.selectbox(
                "Method",
                [
                    "IQR (Tukey fences)",
                    "Z-score (μ ± k·σ)",
                    "Modified Z (median ± k·1.4826·MAD)",
                    "Percentiles (p_low / p_high)",
                ],
                index=["IQR (Tukey fences)", "Z-score (μ ± k·σ)", "Modified Z (median ± k·1.4826·MAD)", "Percentiles (p_low / p_high)"].index(ss.out_method),
                key="out_method",
            )

        cols_row2 = st.columns([1, 1])
        with cols_row2[0]:
            action = st.selectbox(
                "Action",
                ["Clip to bounds (winsorize)", "Set outliers to NaN", "Drop rows with outliers"],
                index=["Clip to bounds (winsorize)", "Set outliers to NaN", "Drop rows with outliers"].index(ss.out_action),
                key="out_action",
            )

        # Method-specific parameters
        if method == "IQR (Tukey fences)":
            st.slider("IQR multiplier (k)", 1.0, 4.0, float(ss.out_k_iqr), step=0.25, key="out_k_iqr")
        elif method == "Z-score (μ ± k·σ)":
            st.slider("Std-dev multiplier (k)", 1.0, 6.0, float(ss.out_k_z), step=0.1, key="out_k_z")
        elif method == "Modified Z (median ± k·1.4826·MAD)":
            st.slider("MAD multiplier (k)", 1.0, 10.0, float(ss.out_k_mad), step=0.1, key="out_k_mad")
        else:  # Percentiles
            c_lo, c_hi = st.columns(2)
            with c_lo:
                st.slider("Lower percentile", 0.0, 20.0, float(ss.out_p_low), step=0.5, key="out_p_low")
            with c_hi:
                st.slider("Upper percentile", 80.0, 100.0, float(ss.out_p_high), step=0.5, key="out_p_high")
            if ss.out_p_high <= ss.out_p_low:
                st.warning("Upper percentile must be greater than lower percentile.")

    # ----- Plain-English summary (top) -----
    param_txt = {
        "IQR (Tukey fences)": f"k={ss.out_k_iqr:.2f}",
        "Z-score (μ ± k·σ)": f"k={ss.out_k_z:.2f}",
        "Modified Z (median ± k·1.4826·MAD)": f"k={ss.out_k_mad:.2f}",
        "Percentiles (p_low / p_high)": f"p_low={ss.out_p_low:.1f}%, p_high={ss.out_p_high:.1f}%",
    }[method]

    st.info(
        f"**Plan:** Detect outliers in **{len(out_cols)}** column(s) using **{method}** ({param_txt}); "
        f"then **{action}**."
    )

    # ----- Preview + Stats -----
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
            s = pd.to_numeric(df[c], errors="coerce")
            lo, hi = _calc_bounds(s, method, params)
            n_out = int(((s < lo) | (s > hi)).sum())
            rows.append({"column": c, "lower": lo, "upper": hi, "outliers": n_out})
        return pd.DataFrame(rows)

    with section("What will change", expandable=False):
        if not out_cols:
            st.caption("Select at least one numeric column.")
        else:
            tbl = _compute_bounds_for_all()
            render_table(tbl, height=240)

    # ----- Visualization (box plot: before vs after for a single column) -----
    with section("Preview (box plot)", expandable=True):
        vis_col = st.selectbox("Visualize a single column", options=(out_cols or num_cols), index=0, key="out_vis_col")
        if vis_col:
            # Before
            _box_plot(df, vis_col, title="Before")
            # After (simulate)
            params = dict(k_iqr=ss.out_k_iqr, k_z=ss.out_k_z, k_mad=ss.out_k_mad, p_low=ss.out_p_low, p_high=ss.out_p_high)
            lo, hi = _calc_bounds(pd.to_numeric(df[vis_col], errors="coerce"), method, params)
            s_after = _apply_action(pd.to_numeric(df[vis_col], errors="coerce"), (lo, hi), action)
            if action == "Drop rows with outliers":
                mask_keep = (pd.to_numeric(df[vis_col], errors="coerce") >= lo) & (pd.to_numeric(df[vis_col], errors="coerce") <= hi)
                df_after = df.loc[mask_keep].copy()
                _box_plot(df_after, vis_col, title="After (drop rows)")
            else:
                df_tmp = df.copy()
                df_tmp[vis_col] = s_after
                _box_plot(df_tmp, vis_col, title="After")

    # ----- Apply -----
    def _apply_all(dfin: pd.DataFrame) -> pd.DataFrame:
        params = dict(k_iqr=ss.out_k_iqr, k_z=ss.out_k_z, k_mad=ss.out_k_mad, p_low=ss.out_p_low, p_high=ss.out_p_high)
        d = dfin.copy()
        if action == "Drop rows with outliers":
            keep_mask = pd.Series(True, index=d.index)
            for c in out_cols:
                s = pd.to_numeric(d[c], errors="coerce")
                lo, hi = _calc_bounds(s, method, params)
                keep_mask &= (s >= lo) & (s <= hi)
            return d.loc[keep_mask].reset_index(drop=True)

        # Column-wise replacement
        for c in out_cols:
            s = pd.to_numeric(d[c], errors="coerce")
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
