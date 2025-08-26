# core/preprocess_outliers.py
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype

from ui.components import section, render_table, kpi_row


# ---------- helpers ----------
def _iqr_bounds(s: pd.Series, k: float) -> tuple[float | None, float | None]:
    x = pd.to_numeric(s, errors="coerce")
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return None, None
    return (q1 - k * iqr, q3 + k * iqr)

def _clip_both(s: pd.Series, lo: float | None, hi: float | None) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if lo is not None:
        x = np.where(x < lo, lo, x)
    if hi is not None:
        x = np.where(x > hi, hi, x)
    return pd.Series(x, index=s.index, dtype=s.dtype)

def _affect_mask(s: pd.Series, lo: float | None, hi: float | None, tail: str) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if tail == "both":
        m = ( (lo is not None) & (x < lo) ) | ( (hi is not None) & (x > hi) )
    elif tail == "high":
        m = (hi is not None) & (x > hi)
    else:  # "low"
        m = (lo is not None) & (x < lo)
    return pd.Series(np.where(pd.isna(x), False, m), index=s.index)


# ---------- main ----------
def render_preprocess_outliers(ss) -> None:
    """Preprocess ▸ Outliers with top plain-English summary + box-plot-only preview."""
    import numpy as np
    import pandas as pd
    import streamlit as st
    import matplotlib.pyplot as plt
    from pandas.api.types import is_numeric_dtype
    from ui.components import section, render_table, kpi_row

    # ---- guards ----
    if not ss.active_ds or ss.active_ds not in ss.datasets:
        st.info("Pick a dataset to begin.")
        st.stop()

    df = ss.datasets[ss.active_ds]
    num_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
    if not num_cols:
        st.info("No numeric columns found.")
        st.stop()

    rows, _ = df.shape
    kpi_row([("Rows", f"{rows:,}"), ("Numeric cols", len(num_cols))])

    # ---------- TOP SUMMARY PLACEHOLDER ----------
    # We fill this AFTER reading the controls below so it always reflects current choices.
    summary_ph = st.empty()

    # ---------- SETUP ----------
    with section("Setup", expandable=False):
        cols = st.multiselect(
            "Columns to process",
            options=num_cols,
            default=num_cols,
            help="Only numeric columns are shown.",
            key="out_cols",
        )

        method = st.selectbox(
            "Method",
            ["IQR (Tukey fences)"],
            index=0,
            key="out_method",
        )

        action = st.selectbox(
            "Action",
            [
                "Clip to bounds (winsorize)",
                "Cap high tail only",
                "Flag outliers (add boolean columns)",
                "Remove outlier rows",
            ],
            index=0,
            key="out_action",
        )

        k = st.slider(
            "IQR multiplier (k)",
            min_value=1.0, max_value=4.0, step=0.25, value=1.5,
            key="out_k",
            help="Larger k = looser bounds (fewer outliers)."
        )

    # ---------- HELPERS ----------
    def _iqr_bounds(s: pd.Series, k: float) -> tuple[float | None, float | None]:
        x = pd.to_numeric(s, errors="coerce")
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr == 0:
            return None, None
        return (q1 - k * iqr, q3 + k * iqr)

    def _clip_both(s: pd.Series, lo: float | None, hi: float | None) -> pd.Series:
        x = pd.to_numeric(s, errors="coerce")
        if lo is not None:
            x = np.where(x < lo, lo, x)
        if hi is not None:
            x = np.where(x > hi, hi, x)
        return pd.Series(x, index=s.index, dtype=s.dtype)

    def _affect_mask(s: pd.Series, lo: float | None, hi: float | None, tail: str) -> pd.Series:
        x = pd.to_numeric(s, errors="coerce")
        if tail == "both":
            m = ((lo is not None) & (x < lo)) | ((hi is not None) & (x > hi))
        elif tail == "high":
            m = (hi is not None) & (x > hi)
        else:
            m = (lo is not None) & (x < lo)
        return pd.Series(np.where(pd.isna(x), False, m), index=s.index)

    # ---------- PLAIN-ENGLISH SUMMARY (TOP) ----------
    def _summary_text() -> str:
        if not cols:
            return "No columns selected yet. Choose one or more numeric columns below."
        if action == "Clip to bounds (winsorize)":
            act = "cap extremely low and high values to reasonable limits"
        elif action == "Cap high tail only":
            act = "cap only unusually high values to a reasonable limit"
        elif action == "Flag outliers (add boolean columns)":
            act = "add a new true/false column per field indicating outliers"
        else:
            act = "remove rows that contain outlier values"

        col_list = ", ".join(cols[:5]) + ("…" if len(cols) > 5 else "")
        return (
            f"We’ll analyze **{len(cols)}** column(s) ({col_list}) with the **IQR** method. "
            f"Values outside the IQR range using *k = {k}* are flagged as outliers, then we’ll **{act}**.\n\n"
            "IQR uses the middle 50% (25th–75th percentiles), which is robust to extreme values."
        )

    # Fill the top summary now that we have the current control values
    summary_ph.info(_summary_text())

    # ---------- PREVIEW (with box plot only) ----------
    def _preview(df_in: pd.DataFrame):
        out = df_in.copy()
        per_col = []
        tail = "both" if action in ["Clip to bounds (winsorize)", "Remove outlier rows", "Flag outliers (add boolean columns)"] else "high"

        for c in cols:
            s = out[c]
            lo, hi = _iqr_bounds(s, k)
            mask = _affect_mask(s, lo, hi, tail if action != "Flag outliers (add boolean columns)" else "both")
            n_aff = int(mask.sum())

            if action == "Clip to bounds (winsorize)":
                out[c] = _clip_both(s, lo, hi)
            elif action == "Cap high tail only":
                out[c] = _clip_both(s, None, hi)
            elif action == "Flag outliers (add boolean columns)":
                out[f"is_outlier_{c}"] = mask.astype(bool)
            elif action == "Remove outlier rows":
                # defer actual drop after computing per-col stats
                pass

            per_col.append({
                "column": c,
                "lower_bound": None if lo is None else float(lo),
                "upper_bound": None if hi is None else float(hi),
                "rows_outside": n_aff,
                "pct_rows": round(100 * (n_aff / len(out)) if len(out) else 0.0, 2),
            })

        if action == "Remove outlier rows":
            drop_mask = pd.Series(False, index=out.index)
            for c in cols:
                lo, hi = _iqr_bounds(out[c], k)
                drop_mask |= _affect_mask(out[c], lo, hi, tail="both")
            out = out.loc[~drop_mask].reset_index(drop=True)

        return out, pd.DataFrame(per_col)

    with section("Preview", expandable=False):
        if st.button("Preview changes", key="out_preview_btn", type="primary"):
            if not cols:
                st.warning("Pick at least one column.")
            else:
                prev_df, stats = _preview(df)
                ss["out_prev_df"] = prev_df
                ss["out_prev_stats"] = stats
                ss["out_prev_ready"] = True

        if ss.get("out_prev_ready"):
            stats = ss.get("out_prev_stats")
            prev_df = ss.get("out_prev_df")

            if isinstance(stats, pd.DataFrame) and not stats.empty:
                st.subheader("What will change")
                render_table(stats)
                st.caption("`rows_outside` = rows that would be clipped/flagged/removed per column.")

            # ----- BOX PLOT ONLY: before vs after -----
            plot_cols = [c for c in cols if c in df.columns]
            if plot_cols:
                plot_col = st.selectbox("Visualize column", plot_cols, key="out_plot_col")
                before = pd.to_numeric(df[plot_col], errors="coerce").dropna()
                after  = pd.to_numeric(prev_df[plot_col], errors="coerce").dropna()

                fig, ax = plt.subplots()
                ax.boxplot([before, after], labels=["before", "after"], showfliers=True)
                ax.set_ylabel(plot_col)
                st.pyplot(fig)

            st.subheader("Preview data (first 20 rows)")
            st.dataframe(prev_df.head(20), use_container_width=True)

    # ---------- APPLY ----------
    with section("Apply", expandable=False):
        disabled = not ss.get("out_prev_ready", False)
        if disabled:
            st.caption("Run **Preview changes** first. You can always **Undo** from the top bar.")
        if st.button("Apply", key="out_apply_btn", disabled=disabled):
            out_df = ss.get("out_prev_df")
            if out_df is None:
                st.warning("No preview found. Click **Preview changes** first.")
            else:
                ss.df_history.append(df.copy())   # undo
                ss.datasets[ss.active_ds] = out_df
                st.success(f"Applied to **{ss.active_ds}**.")
                for k in ["out_prev_df", "out_prev_stats", "out_prev_ready"]:
                    ss.pop(k, None)
