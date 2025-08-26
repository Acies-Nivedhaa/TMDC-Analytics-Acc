# core/eda_univariate.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import streamlit as st

from ui.components import render_table  # already in your project


# ---------- helpers ----------

def _cell_key(v):
    """Stable key for lists/dicts/sets etc. (for value_counts on object cols)."""
    try:
        hash(v)
        return v
    except TypeError:
        pass
    try:
        import json
        if isinstance(v, (dict, list, tuple, set)):
            return json.dumps(v, sort_keys=True, default=str)
    except Exception:
        pass
    return str(v)


def _safe_value_counts(s: pd.Series, top: int | None = None) -> pd.DataFrame:
    if s.dtype == "object":
        vc = s.map(_cell_key).value_counts(dropna=False)
    else:
        try:
            vc = s.value_counts(dropna=False)
        except TypeError:
            vc = s.astype(str).value_counts(dropna=False)
    if top:
        vc = vc.head(top)
    total = max(1, len(s))
    out = vc.rename_axis("value").reset_index(name="count")
    out["percent"] = (out["count"] / total * 100).round(2)
    return out


def _numeric_summary(s: pd.Series) -> pd.DataFrame:
    # (count/mean/std/min/25%/50%/75%/max)
    desc = s.describe(percentiles=[0.25, 0.5, 0.75])
    return pd.DataFrame(desc).T


def _clip_tails(s: pd.Series, mode: str) -> pd.Series:
    """Winsorize tails by percent on each side."""
    if mode == "None":
        return s
    pct = float(mode.replace("%", "")) / 100.0  # e.g. "0.5%" -> 0.005
    lo, hi = s.quantile(pct), s.quantile(1 - pct)
    return s.clip(lo, hi)


def _hist_counts(s: pd.Series, bins_rule: str, logx: bool) -> pd.DataFrame:
    x = s.dropna().to_numpy()
    if x.size == 0:
        return pd.DataFrame({"bin": [], "count": []})
    if logx:
        x = np.where(x > 0, np.log10(x), np.nan)
        x = x[~np.isnan(x)]
    bins = "auto" if bins_rule == "auto" else bins_rule
    counts, edges = np.histogram(x, bins=bins)
    mids = (edges[:-1] + edges[1:]) / 2.0
    df = pd.DataFrame({"bin": mids, "count": counts})
    return df


def _infer_univariate_dtype(s: pd.Series) -> str:
    if ptypes.is_numeric_dtype(s):
        return "numeric"
    if ptypes.is_datetime64_any_dtype(s):
        return "datetime"
    return "categorical"

# --- Business insights helpers (Univariate) ---

def _iqr_outlier_share(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return 0.0
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return float(((s < lo) | (s > hi)).mean())

def _insights_univariate(col_name: str, s_full: pd.Series, kind: str) -> list[str]:
    tips: list[str] = []
    n = len(s_full)
    miss_pct = float(s_full.isna().mean() * 100)

    if miss_pct > 0:
        tips.append(f"**{miss_pct:.1f}%** values missing in **{col_name}** → consider imputation or data quality checks.")

    if kind == "numeric":
        sn = pd.to_numeric(s_full, errors="coerce").dropna()
        if sn.empty:
            return tips
        skew = float(sn.skew())
        if skew > 1:
            tips.append("Right-skewed distribution → try log/yeo-johnson transform or robust statistics.")
        elif skew < -1:
            tips.append("Left-skewed distribution → consider transformation or winsorization.")

        out_share = _iqr_outlier_share(sn) * 100
        if out_share >= 5:
            tips.append(f"≈ **{out_share:.1f}%** potential outliers (IQR rule) → cap at p1/p99 or investigate upstream causes.")

        mean, std = float(sn.mean()), float(sn.std(ddof=1))
        if mean != 0 and std != 0:
            cv = abs(std / mean)
            if cv >= 1.0:
                tips.append("High variability (CV ≥ 1) → aggregation/segmentation may stabilize behavior.")

        zero_share = float((sn == 0).mean() * 100)
        if zero_share >= 20:
            tips.append(f"**{zero_share:.1f}%** zeros → check for sparsity, consider zero-inflated models or two-part features.")

    elif kind == "categorical":
        vc = s_full.astype("string").fillna("__NA__").value_counts(dropna=False, normalize=True)
        if not vc.empty:
            top_cat, top_share = vc.index[0], float(vc.iloc[0] * 100)
            tips.append(f"Top category **{top_cat}** covers **{top_share:.1f}%** → watch for dominance/target leakage.")
            if vc.size > 30:
                tips.append("High cardinality → consider grouping rare categories or hashing encoders.")

    else:  # datetime
        sdt = pd.to_datetime(s_full, errors="coerce").dropna()
        if sdt.empty:
            return tips
        span = (sdt.max() - sdt.min()).days
        tips.append(f"Time span ≈ **{span}** days → choose resampling granularity accordingly.")
        by_dow = sdt.dt.day_name().value_counts(normalize=True)
        if not by_dow.empty:
            tips.append(f"Busiest day: **{by_dow.index[0]}** → consider staffing/capacity alignment.")

    return tips

def _insight_box(title: str, bullets: list[str]) -> None:
    import streamlit as st
    st.markdown("----")
    st.markdown(f"**{title}**")
    if not bullets:
        st.caption("No standout issues detected for this column.")
        return
    for b in bullets:
        st.markdown(f"- {b}")



# ---------- main renderer ----------

def render_univariate(df: pd.DataFrame, sample_n: int | None = None) -> None:
    """
    Univariate profile for a single column.
    - Works for numeric / categorical / datetime
    - Handles nested JSON-like values gracefully
    """
    st.markdown("**Univariate**")

    # Column picker (all columns)
    cols = df.columns.tolist()
    # Prefer a numeric column as default if available
    default_col = next((c for c in cols if ptypes.is_numeric_dtype(df[c])), cols[0])
    col = st.selectbox("Pick a column", cols, index=cols.index(default_col), key="uni_col")

    s_full = df[col]
    missing_n = int(s_full.isna().sum())
    st.caption(f"Missing values: {missing_n:,} | Dtype: {s_full.dtype}")

    # optional sampling for speed on huge data
    if sample_n:
        # robust sample size
        n = min(sample_n, len(s_full))
        if n < len(s_full):
            s = s_full.sample(n, random_state=0)
        else:
            s = s_full
    else:
        s = s_full

    kind = _infer_univariate_dtype(s)

    # ---------- NUMERIC ----------
    if kind == "numeric":
        st.markdown("**Numeric summary**")
        render_table(_numeric_summary(pd.to_numeric(s, errors="coerce")), height=160)

        c1, c2, c3 = st.columns([0.4, 0.2, 0.4])
        with c1:
            bins_rule = st.selectbox("Binning", ["auto", "sqrt", "sturges", "rice", "fd", "doane"], index=0)
        with c2:
            logx = st.checkbox("Log scale (x)", value=False)
        with c3:
            clip = st.selectbox("Clip tails", ["None", "0.1%", "0.5%", "1%", "5%"], index=0)

        s_num = pd.to_numeric(s, errors="coerce").dropna()
        s_num = _clip_tails(s_num, clip)

        hist = _hist_counts(s_num, bins_rule=bins_rule, logx=logx)
        st.bar_chart(hist.set_index("bin")["count"])

        st.caption("Unique values: {} — showing top 100".format(int(s_num.nunique(dropna=False))))
        top_tbl = _safe_value_counts(s_num, top=100)
        render_table(top_tbl)

    # ---------- CATEGORICAL ----------
    elif kind == "categorical":
        nunique = int(s.nunique(dropna=False))
        st.markdown("**Categorical summary**")
        st.caption(f"Unique values: {nunique:,} — showing top 50")

        top_tbl = _safe_value_counts(s, top=50)
        # Chart first, then table
        st.bar_chart(top_tbl.set_index("value")["count"])
        render_table(top_tbl)

    # ---------- DATETIME ----------
    else:
        s_dt = pd.to_datetime(s, errors="coerce")
        vmin, vmax = s_dt.min(), s_dt.max()
        st.markdown("**Datetime summary**")
        st.caption(f"Range: {vmin} → {vmax}")

        freq = st.selectbox("Resample frequency", ["Auto", "Day", "Week", "Month", "Quarter", "Year"], index=2)
        if freq == "Auto":
            span_days = max(1, (vmax - vmin).days if pd.notna(vmax) and pd.notna(vmin) else 1)
            if span_days <= 31:
                rule = "D"
            elif span_days <= 180:
                rule = "W"
            elif span_days <= 730:
                rule = "M"
            else:
                rule = "Q"
        else:
            rule = {"Day": "D", "Week": "W", "Month": "M", "Quarter": "Q", "Year": "Y"}[freq]

        ts = s_dt.dropna().sort_values()
        if ts.empty:
            st.info("No valid datetime values to display.")
            return

        counts = ts.dt.to_period(rule).value_counts().sort_index()
        # convert PeriodIndex to timestamp-ish labels
        idx = counts.index.astype(str)
        chart_df = pd.DataFrame({"period": idx, "count": counts.values})
        st.bar_chart(chart_df.set_index("period")["count"])

        tbl = chart_df.rename(columns={"period": "bucket"})
        render_table(tbl)

                # --- Business insights (auto-updates with the selected column) ---
    insights = _insights_univariate(col, s_full, kind)
    _insight_box("Business insights", insights)



