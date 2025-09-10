# core/preprocess_missing.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

from ui.components import section, render_table, kpi_row
from core.insights_extractors import (
    capture_missingness_insight,
    capture_missing_outlier_overview,
)


# ---------- small helpers ----------

def _mode_safe(s: pd.Series):
    """Return the most frequent value (mode) or None if not available."""
    try:
        m = s.mode(dropna=True)
        return m.iloc[0] if not m.empty else None
    except Exception:
        return None


def _missing_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing counts/percentages sorted by severity (desc)."""
    return (
        pd.DataFrame(
            {
                "column": df.columns,
                "missing": [int(df[c].isna().sum()) for c in df.columns],
                "missing_%": [round(100 * df[c].isna().mean(), 2) for c in df.columns],
                "dtype": [str(df[c].dtype) for c in df.columns],
            }
        )
        .query("missing > 0")
        .sort_values(["missing", "column"], ascending=[False, True])
        .reset_index(drop=True)
    )


def _choices_for(s: pd.Series) -> list[str]:
    """Offer handling strategies tailored to the series dtype."""
    if is_datetime64_any_dtype(s):
        return [
            "Leave as-is",
            "Drop column",
            "Drop rows where missing",
            "Forward fill (ffill)",
            "Backward fill (bfill)",
            "Fill with most frequent",
            "Fill with min timestamp",
            "Fill with max timestamp",
            "Constant…",
        ]
    if is_numeric_dtype(s):
        return [
            "Leave as-is",
            "Drop column",
            "Drop rows where missing",
            "Fill with mean",
            "Fill with median",
            "Fill with mode",
            "Forward fill (ffill)",
            "Backward fill (bfill)",
            "Interpolate (linear)",
            "Constant…",
        ]
    # categorical / text
    return [
        "Leave as-is",
        "Drop column",
        "Drop rows where missing",
        "Fill with mode",
        "Forward fill (ffill)",
        "Backward fill (bfill)",
        "Constant…",
    ]


# Human-readable labels → pandas offset aliases (used for datetime ffill/bfill buckets)
_DT_FREQS = {
    "Auto (no grouping)": None,
    "Hourly": "H",
    "Daily": "D",
    "Weekly": "W",
    "Monthly": "M",
    "Quarterly": "Q",
    "Yearly": "Y",
    "Minute": "T",
}


# ---------- main UI ----------

def render_preprocess_missing(ss) -> None:
    """
    Preprocess ▸ Missing values
    - Dtype-aware controls per column.
    - Optional datetime frequency bucketing for ffill/bfill.
    - Preview before apply; logs a succinct insight after apply.
    """
    # Guard: require an active dataset
    if not ss.active_ds or ss.active_ds not in ss.datasets:
        st.info("Pick a dataset to begin.")
        st.stop()

    df = ss.datasets[ss.active_ds]
    rows, cols = df.shape

    # Compact glossary (plain English, quick reference)
    with st.expander("Key terms (quick guide)", expanded=False):
        st.markdown(
            """
- **Missing / NA** — Empty cells (NaN/NaT) the model/tools can’t use directly.
- **Drop rows / column** — Remove affected rows or the whole column (irreversible unless you undo).
- **Mean / Median / Mode** — Numeric averages; for text/categories, *mode* = most frequent value.
- **Forward/Backward fill (ffill/bfill)** — Propagate the last/next known value to nearby gaps.
- **Interpolate (linear)** — Estimate numeric gaps along a straight line between known points.
- **Constant** — Replace gaps with a fixed value (e.g., 0, “Unknown”, or a specific date).
- **Datetime frequency** — When filling timestamps, keep fills *within* a period (e.g., per day or per month).
            """
        )

    # KPIs
    kpi_row(
        [
            ("Rows", f"{rows:,}"),
            ("Cols", cols),
            ("Cols with NA", int((df.isna().sum() > 0).sum())),
            ("Total NA", int(df.isna().sum().sum())),
        ]
    )

    mt = _missing_table(df)
    if mt.empty:
        with section("Missing values"):
            st.success("No missing values found 🎉")
        return

    # Persist user choices across reruns
    ss.setdefault("pp_missing_choice", {})   # column -> chosen strategy
    ss.setdefault("pp_missing_const", {})    # column -> constant (as text)
    ss.setdefault("pp_missing_dt_freq", {})  # column -> dt frequency label

    # ----- per-column controls -----
    with section("Per-column handling", expandable=False):
        st.caption("Only columns that have missing values are shown.")
        for _, row in mt.iterrows():
            c = row["column"]
            s = df[c]
            choices = _choices_for(s)
            label = f"{c} — {row['missing']} nulls ({row['missing_%']}%) [{s.dtype}]"

            col_sel, col_extra = st.columns([0.62, 0.38])

            # Strategy select
            with col_sel:
                choice = st.selectbox(
                    label,
                    choices,
                    key=f"pp_mv_{c}",
                    index=choices.index(ss.pp_missing_choice.get(c, "Leave as-is"))
                    if ss.pp_missing_choice.get(c) in choices
                    else 0,
                )
            ss.pp_missing_choice[c] = choice

            # Extra options: either datetime frequency or constant input
            with col_extra:
                if is_datetime64_any_dtype(s) and choice in (
                    "Forward fill (ffill)",
                    "Backward fill (bfill)",
                ):
                    freq_label_default = ss.pp_missing_dt_freq.get(c, "Auto (no grouping)")
                    freq_label = st.selectbox(
                        f"Frequency for {c}",
                        list(_DT_FREQS.keys()),
                        index=list(_DT_FREQS.keys()).index(freq_label_default)
                        if freq_label_default in _DT_FREQS
                        else 0,
                        key=f"pp_mv_dt_freq_{c}",
                        help="Restrict fill within calendar buckets (e.g., per day/month).",
                    )
                    ss.pp_missing_dt_freq[c] = freq_label
                else:
                    # Constant input (enabled only when strategy is Constant…)
                    show_const = choice == "Constant…"
                    if is_numeric_dtype(s):
                        placeholder = "e.g., 0 or 99.9"
                    elif is_datetime64_any_dtype(s):
                        placeholder = "e.g., 2024-01-01 09:30"
                    else:
                        placeholder = "e.g., Unknown"

                    const_val = st.text_input(
                        f"Constant for {c}",
                        value=ss.pp_missing_const.get(c, ""),
                        key=f"pp_mv_const_{c}",
                        placeholder=placeholder,
                        disabled=not show_const,
                    )
                    if show_const:
                        ss.pp_missing_const[c] = const_val

    # Current missingness table (pre-apply)
    with section("Current missingness (before apply)", expandable=True):
        render_table(mt, height=260)

    # ----- application logic (pure function) -----
    def _apply(df_in: pd.DataFrame) -> pd.DataFrame:
        out = df_in.copy()
        rows_drop_mask = pd.Series(False, index=out.index)

        for c, choice in ss.pp_missing_choice.items():
            if c not in out.columns or not out[c].isna().any():
                continue
            s = out[c]

            # DATETIME handling
            if is_datetime64_any_dtype(s):
                s_dt = pd.to_datetime(s, errors="coerce")

                if choice == "Leave as-is":
                    continue
                if choice == "Drop column":
                    out = out.drop(columns=[c]); continue
                if choice == "Drop rows where missing":
                    rows_drop_mask |= s_dt.isna(); continue

                # Frequency-aware ffill/bfill within calendar buckets
                if choice in ("Forward fill (ffill)", "Backward fill (bfill)"):
                    freq_label = ss.pp_missing_dt_freq.get(c, "Auto (no grouping)")
                    freq = _DT_FREQS.get(freq_label)
                    if freq:
                        tmp = pd.DataFrame({"s": s_dt})
                        tmp["_bucket"] = tmp["s"].dt.to_period(freq)
                        if choice == "Forward fill (ffill)":
                            tmp["s"] = tmp.groupby("_bucket", group_keys=False)["s"].apply(lambda x: x.ffill())
                        else:
                            tmp["s"] = tmp.groupby("_bucket", group_keys=False)["s"].apply(lambda x: x.bfill())
                        out[c] = tmp["s"]
                    else:
                        out[c] = s_dt.fillna(method="ffill" if choice.startswith("Forward") else "bfill")
                    continue

                if choice == "Fill with most frequent":
                    mv = _mode_safe(s_dt)
                    if mv is not None:
                        out[c] = s_dt.fillna(mv)
                    continue
                if choice == "Fill with min timestamp":
                    out[c] = s_dt.fillna(s_dt.min(skipna=True)); continue
                if choice == "Fill with max timestamp":
                    out[c] = s_dt.fillna(s_dt.max(skipna=True)); continue
                if choice == "Constant…":
                    raw = ss.pp_missing_const.get(c, "")
                    val = pd.to_datetime(raw, errors="coerce")
                    out[c] = s_dt.fillna(val); continue

            # NUMERIC handling
            if is_numeric_dtype(s):
                if choice == "Leave as-is":
                    continue
                if choice == "Drop column":
                    out = out.drop(columns=[c]); continue
                if choice == "Drop rows where missing":
                    rows_drop_mask |= s.isna(); continue
                if choice == "Fill with mean":
                    out[c] = s.fillna(s.mean()); continue
                if choice == "Fill with median":
                    out[c] = s.fillna(s.median()); continue
                if choice == "Fill with mode":
                    mv = _mode_safe(s)
                    if mv is not None:
                        out[c] = s.fillna(mv)
                    continue
                if choice == "Forward fill (ffill)":
                    out[c] = s.fillna(method="ffill"); continue
                if choice == "Backward fill (bfill)":
                    out[c] = s.fillna(method="bfill"); continue
                if choice == "Interpolate (linear)":
                    out[c] = s.interpolate(method="linear"); continue
                if choice == "Constant…":
                    raw = ss.pp_missing_const.get(c, "")
                    try:
                        val = float(raw)
                    except Exception:
                        val = np.nan
                    out[c] = s.fillna(val); continue

            # CATEGORICAL / TEXT handling
            if choice == "Leave as-is":
                continue
            if choice == "Drop column":
                out = out.drop(columns=[c]); continue
            if choice == "Drop rows where missing":
                rows_drop_mask |= s.isna(); continue
            if choice == "Fill with mode":
                mv = _mode_safe(s)
                if mv is not None:
                    out[c] = s.fillna(mv)
                continue
            if choice == "Forward fill (ffill)":
                out[c] = s.fillna(method="ffill"); continue
            if choice == "Backward fill (bfill)":
                out[c] = s.fillna(method="bfill"); continue
            if choice == "Constant…":
                out[c] = s.fillna(ss.pp_missing_const.get(c, "")); continue

        # Apply any row drops collected along the way
        if rows_drop_mask.any():
            out = out.loc[~rows_drop_mask].reset_index(drop=True)
        return out

    # ----- preview / apply -----
    with section("Apply", expandable=False):
        if st.button("Preview result", key="pp_mv_preview"):
            preview = _apply(df)
            st.write(
                f"Result: **{preview.shape[0]:,} × {preview.shape[1]:,}** "
                f"(was {rows:,} × {cols})"
            )
            render_table(_missing_table(preview), height=260)
            st.dataframe(preview.head(20), use_container_width=True)

        if st.button("Apply Missing Handling", type="primary", key="pp_mv_apply"):
            out = _apply(df)
            ss.df_history.append(df.copy())  # enable Undo
            ss.datasets[ss.active_ds] = out

            # Log high-level insights (best-effort; never block the UI)
            try:
                ds_name = st.session_state.get("active_ds") or "dataset"
                capture_missingness_insight(st.session_state, ds_name, out)
                capture_missing_outlier_overview(st.session_state, ds_name, out)
            except Exception:
                pass

            st.success(f"Applied to **{ss.active_ds}**.")
