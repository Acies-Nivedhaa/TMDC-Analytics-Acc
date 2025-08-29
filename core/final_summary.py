# core/final_summary.py
from __future__ import annotations
import json
import io
import numpy as np
import pandas as pd
import streamlit as st

from ui.components import section, kpi_row, render_table
from core.summary import overview_stats, suggest_actions, nunique_safe


# ---------- small helpers ----------

def _stable_str(v):
    try:
        if isinstance(v, (dict, list, tuple, set)):
            return json.dumps(v, sort_keys=True, ensure_ascii=False)
        return str(v)
    except Exception:
        return str(v)

def _safe_value_counts(s: pd.Series, top: int = 5) -> pd.Series:
    try:
        return s.value_counts(dropna=False).head(top)
    except TypeError:
        return s.map(_stable_str).value_counts(dropna=False).head(top)

def _maybe_dict(ss, keys: list[str]) -> dict:
    for k in keys:
        v = ss.get(k)
        if isinstance(v, dict):
            return v
    return {}

def _maybe_list(ss, keys: list[str]) -> list:
    for k in keys:
        v = ss.get(k)
        if isinstance(v, list):
            return v
    return []

def _non_none_items(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None and v != "None"}

# ---------- main renderer ----------

def render_final_summary(df: pd.DataFrame, ss) -> None:
    """
    Final Summary (lean):
      - KPI cards (once)
      - Outstanding issues (heuristics)
      - What changed in previous steps (from session state)
      - Preview & Export
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        st.info("No data to summarize.")
        return

    # --- KPIs (no extra title here so we don't duplicate your page title) ---
    ov = overview_stats(df)
    kpi_row([
        ("Rows", f"{ov['rows']:,}"),
        ("Cols", f"{ov['cols']:,}"),
        ("Memory (MB)", f"{ov['memory_mb']:.2f}"),
        ("Duplicate rows", f"{ov['n_duplicates']:,}"),
    ])

    # --- Outstanding issues (heuristics) ---
    with section("Outstanding issues (heuristics)", expandable=False):
        tips = suggest_actions(df)
        if not tips:
            st.success("No major issues detected.")
        else:
            for t in tips:
                st.markdown(f"- {t}")

    # --- What changed in previous steps ---
    with section("What changed in previous steps", expandable=False):
        # Encoding (from our encoding tab implementation)
        enc_choices = _maybe_dict(ss, ["enc_choices", "encoding_choices"])
        enc_choices = _non_none_items(enc_choices)
        if enc_choices:
            enc_rows = [{"column": c, "method": m} for c, m in enc_choices.items() if m and m != "None"]
            if enc_rows:
                st.markdown("**Encoding**")
                render_table(pd.DataFrame(enc_rows).sort_values("column").reset_index(drop=True), height=200)

        # Missing/null handling (support a few likely keys)
        null_plans = (
            _maybe_dict(ss, ["mv_plans", "missing_plans", "null_plans", "impute_plans"])
            or {}
        )
        if null_plans:
            st.markdown("**Missing values**")
            rows = []
            for col, plan in null_plans.items():
                # plan can be a string or dict depending on your earlier UI
                if isinstance(plan, dict):
                    method = plan.get("method", "—")
                    extra  = ", ".join([f"{k}={v}" for k, v in plan.items() if k != "method"])
                else:
                    method = str(plan)
                    extra = ""
                rows.append({"column": col, "method": method, "params": extra})
            if rows:
                render_table(pd.DataFrame(rows).sort_values("column").reset_index(drop=True), height=220)

        # Outliers (from the outliers tab—only show if user configured anything)
        out_cols = _maybe_list(ss, ["out_cols"])
        if out_cols:
            st.markdown("**Outliers**")
            out_method = ss.get("out_method", "IQR (Tukey fences)")
            out_action = ss.get("out_action", "Clip to bounds (winsorize)")
            params = {
                "k_iqr": ss.get("out_k_iqr"),
                "k_z": ss.get("out_k_z"),
                "k_mad": ss.get("out_k_mad"),
                "p_low": ss.get("out_p_low"),
                "p_high": ss.get("out_p_high"),
            }
            ptxt = ", ".join([f"{k}={v}" for k, v in params.items() if v is not None])
            st.write(f"- **Columns**: {len(out_cols)} selected")
            st.write(f"- **Method**: {out_method}")
            st.write(f"- **Action**: {out_action}")
            if ptxt:
                st.write(f"- **Params**: {ptxt}")

        # Text processing (best-effort: show selected columns if your text tab saved them)
        text_cols = _maybe_list(ss, ["text_cols", "ppt_text_cols", "pp_text_cols"])
        if text_cols:
            st.markdown("**Text processing**")
            st.write(f"- **Columns**: {', '.join(map(str, text_cols))}")

        # Calendar / features (if your time series tab saved anything, list it)
        added_feats = _maybe_list(ss, ["ts_added_features", "ts_feat_added"])
        if added_feats:
            st.markdown("**Calendar features**")
            render_table(pd.DataFrame({"feature": added_feats}), height=160)

        # If nothing was detected:
        if not any([enc_choices, null_plans, out_cols, text_cols, added_feats]):
            st.caption("No recorded changes from previous steps (or nothing was applied).")

    # --- Preview & Export (single table, single download button) ---
    with section("Preview / Export", expandable=False):
        st.caption(f"Result: **{df.shape[0]:,} × {df.shape[1]:,}**")
        st.dataframe(df.head(50), use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download processed CSV",
            data=csv,
            file_name="processed.csv",
            mime="text/csv",
            key="final_summary_download_csv",
            use_container_width=False,
        )
