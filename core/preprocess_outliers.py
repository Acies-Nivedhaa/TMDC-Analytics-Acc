# core/preprocess_outliers.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype

from ui.components import section, render_table, kpi_row


# -------- helpers --------
def _bounds_for_series(
    s: pd.Series,
    method: str,
    *,
    iqr_k: float = 1.5,
    z_thr: float = 3.0,
    mad_thr: float = 3.5,
    q_lo: float = 0.01,
    q_hi: float = 0.99,
) -> tuple[float, float]:
    """Compute lower/upper bounds for outlier detection on a numeric series."""
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return (np.nan, np.nan)

    if method == "IQR (Tukey fences)":
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        return (q1 - iqr_k * iqr, q3 + iqr_k * iqr)

    if method == "Z-score":
        m, sd = x.mean(), x.std(ddof=0)
        sd = sd if sd and np.isfinite(sd) else 0.0
        return (m - z_thr * sd, m + z_thr * sd)

    if method == "MAD (robust)":
        med = x.median()
        mad = (x - med).abs().median()
        # 1.4826 converts MAD to std.dev equivalent under normality
        sigma = 1.4826 * mad if np.isfinite(mad) else 0.0
        return (med - mad_thr * sigma, med + mad_thr * sigma)

    # Quantile range
    lo = float(np.clip(q_lo, 0, 1))
    hi = float(np.clip(q_hi, 0, 1))
    if hi < lo:
        lo, hi = hi, lo
    return (x.quantile(lo), x.quantile(hi))


def _summarize_outliers(df: pd.DataFrame, cols: list[str], **params) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = df[c]
        if not is_numeric_dtype(s):
            continue
        lo, hi = _bounds_for_series(s, **params)
        x = pd.to_numeric(s, errors="coerce")
        mask = (x < lo) | (x > hi)
        if np.isnan(lo) or np.isnan(hi):
            n_out = 0
            frac = 0.0
        else:
            n_out = int(mask.sum())
            frac = float(mask.mean()) if len(mask) else 0.0

        rows.append({
            "column": c,
            "lower": np.round(lo, 6) if np.isfinite(lo) else np.nan,
            "upper": np.round(hi, 6) if np.isfinite(hi) else np.nan,
            "outliers": n_out,
            "outliers_%": round(100 * frac, 2),
        })
    if not rows:
        return pd.DataFrame(columns=["column", "lower", "upper", "outliers", "outliers_%"])
    return pd.DataFrame(rows).sort_values(["outliers", "column"], ascending=[False, True]).reset_index(drop=True)


def _apply_outliers(
    df: pd.DataFrame,
    cols: list[str],
    *,
    method: str,
    action: str,
    iqr_k: float = 1.5,
    z_thr: float = 3.0,
    mad_thr: float = 3.5,
    q_lo: float = 0.01,
    q_hi: float = 0.99,
) -> pd.DataFrame:
    out = df.copy()
    drop_mask_any = pd.Series(False, index=out.index)

    for c in cols:
        if c not in out.columns or not is_numeric_dtype(out[c]):
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        lo, hi = _bounds_for_series(
            s, method,
            iqr_k=iqr_k, z_thr=z_thr, mad_thr=mad_thr, q_lo=q_lo, q_hi=q_hi
        )
        if np.isnan(lo) or np.isnan(hi):
            continue

        mask = (s < lo) | (s > hi)

        if action == "Clip to bounds (winsorize)":
            out[c] = s.clip(lower=lo, upper=hi)

        elif action == "Drop rows outside bounds":
            drop_mask_any |= mask

        elif action == "Mark as NaN":
            s2 = s.copy()
            s2[mask] = np.nan
            out[c] = s2

    if drop_mask_any.any() and action == "Drop rows outside bounds":
        out = out.loc[~drop_mask_any].reset_index(drop=True)

    return out


# -------- main render --------
def render_preprocess_outliers(ss) -> None:
    """Preprocess ▸ Outliers (numeric columns only)."""
    if not ss.active_ds or ss.active_ds not in ss.datasets:
        st.info("Pick a dataset to begin.")
        st.stop()

    df = ss.datasets[ss.active_ds]
    rows, cols = df.shape
    num_cols = [c for c in df.columns if is_numeric_dtype(df[c])]

    kpi_row([
        ("Rows", f"{rows:,}"),
        ("Cols", cols),
        ("Numeric cols", len(num_cols)),
    ])

    if not num_cols:
        st.info("No numeric columns found.")
        return

    ss.setdefault("pp_out_cols", num_cols.copy())
    ss.setdefault("pp_out_method", "IQR (Tukey fences)")
    ss.setdefault("pp_out_action", "Clip to bounds (winsorize)")
    ss.setdefault("pp_out_k", 1.5)
    ss.setdefault("pp_out_z", 3.0)
    ss.setdefault("pp_out_mad", 3.5)
    ss.setdefault("pp_out_q", (0.01, 0.99))
    ss.setdefault("pp_out_vis", "Business view")
    ss.setdefault("pp_out_vizcol", "—")

    with section("Outlier handling (all numeric columns)", expandable=False):
        cols_pick = st.multiselect(
            "Columns to process",
            options=num_cols,
            default=[c for c in ss.pp_out_cols if c in num_cols],
            key="pp_out_cols",
        )

        method = st.selectbox(
            "Method",
            ["IQR (Tukey fences)", "Z-score", "MAD (robust)", "Quantile range"],
            index=["IQR (Tukey fences)", "Z-score", "MAD (robust)", "Quantile range"].index(ss.pp_out_method),
            key="pp_out_method",
        )

        action = st.selectbox(
            "Action",
            ["Clip to bounds (winsorize)", "Drop rows outside bounds", "Mark as NaN"],
            index=["Clip to bounds (winsorize)", "Drop rows outside bounds", "Mark as NaN"].index(ss.pp_out_action),
            key="pp_out_action",
        )

        # method-specific controls
        if method == "IQR (Tukey fences)":
            st.slider("IQR multiplier (k)", 1.0, 4.0, key="pp_out_k", step=0.05)
        elif method == "Z-score":
            st.slider("Z threshold", 1.0, 6.0, key="pp_out_z", step=0.1)
        elif method == "MAD (robust)":
            st.slider("MAD threshold", 1.0, 10.0, key="pp_out_mad", step=0.1)
        else:  # Quantile
            lo, hi = ss.pp_out_q
            ss.pp_out_q = st.slider("Central quantile range",
                                    0.0, 1.0, (lo, hi), step=0.01, key="pp_out_q")

        viz_col = st.selectbox("Visualize a single column (optional)",
                               ["—"] + cols_pick if cols_pick else ["—"],
                               key="pp_out_vizcol")
        vis_mode = st.radio("Visualization", ["Business view", "Box plot"],
                            horizontal=True, key="pp_out_vis")

    # ---- Summary table (current bounds + counts)
    params = dict(
        method=ss.pp_out_method,
        iqr_k=ss.pp_out_k,
        z_thr=ss.pp_out_z,
        mad_thr=ss.pp_out_mad,
        q_lo=ss.pp_out_q[0] if isinstance(ss.pp_out_q, (tuple, list)) else 0.01,
        q_hi=ss.pp_out_q[1] if isinstance(ss.pp_out_q, (tuple, list)) else 0.99,
    )
    with section("Detected outliers (before apply)", expandable=False):
        summ = _summarize_outliers(df, cols_pick, **params)
        render_table(summ, height=280)

    # ---- optional visualization
    if viz_col and viz_col != "—" and viz_col in cols_pick:
        s = pd.to_numeric(df[viz_col], errors="coerce").dropna()
        if not s.empty:
            import matplotlib.pyplot as plt

            lo, hi = _bounds_for_series(s, ss.pp_out_method,
                                        iqr_k=ss.pp_out_k, z_thr=ss.pp_out_z,
                                        mad_thr=ss.pp_out_mad,
                                        q_lo=params["q_lo"], q_hi=params["q_hi"])

            with section(f"Preview: {viz_col}", expandable=True):
                if ss.pp_out_vis == "Business view":
                    fig, ax = plt.subplots(figsize=(6.5, 3.2))
                    ax.hist(s, bins="auto")
                    ax.axvline(lo, linestyle="--")
                    ax.axvline(hi, linestyle="--")
                    ax.set_title(f"{viz_col} (bounds: {np.round(lo,3)} … {np.round(hi,3)})")
                    st.pyplot(fig, clear_figure=True)
                else:
                    fig, ax = plt.subplots(figsize=(4.5, 3.6))
                    ax.boxplot(s.values, vert=True, showfliers=True)
                    ax.set_xticklabels([viz_col])
                    st.pyplot(fig, clear_figure=True)

    # ---- Preview / Apply
    with section("Actions", expandable=False):
        c1, c2 = st.columns([0.65, 0.35])

        with c1:
            if st.button("Preview", key="pp_out_preview"):
                out = _apply_outliers(df, cols_pick, action=ss.pp_out_action, **params)
                st.write(f"Result: **{out.shape[0]:,} × {out.shape[1]:,}** (was {rows:,} × {cols})")
                render_table(_summarize_outliers(out, cols_pick, **params), height=260)
                st.dataframe(out.head(20), use_container_width=True)

        with c2:
            if st.button("Apply", type="primary", key="pp_out_apply"):
                new_df = _apply_outliers(df, cols_pick, action=ss.pp_out_action, **params)
                # Undo support
                ss.df_history.append(df.copy())
                ss.datasets[ss.active_ds] = new_df
                st.success(f"Applied outlier handling to **{ss.active_ds}**. Use **Undo** to revert.")
                ss.activity_log.append(f"Outlier handling applied on '{ss.active_ds}' ({ss.pp_out_method}, {ss.pp_out_action}).")
                st.rerun()
