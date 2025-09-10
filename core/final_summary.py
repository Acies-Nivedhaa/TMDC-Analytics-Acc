# core/final_summary.py
# ---------------------------------------------------------------------
# Final Summary page:
# - Technical KPIs
# - Outstanding data issues (heuristics)
# - Recap of changes from previous steps (pulled from session_state)
# - Auto-scan insights + downloads (JSON / Markdown / PDF)
# - Data preview + CSV export
# ---------------------------------------------------------------------

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.components import section, kpi_row, render_table
from core.overview import overview_stats, suggest_actions
from core.insights_extractors import (
    init_insights_store,
    scan_dataset_for_insights,
    insights_to_json_bytes,
    generate_markdown_report,
    insights_pdf_bytes,
    render_collected_insights_pretty,  # pretty renderer for collected insights
)

# ---------------------------------------------------------------------
# Small helpers to read structured bits from st.session_state safely
# ---------------------------------------------------------------------
def _maybe_dict(ss, keys: list[str]) -> dict:
    """
    Return the first dict found in `ss` among `keys`, else {}.
    Useful when prior steps may store under different key names.
    """
    for k in keys:
        v = ss.get(k)
        if isinstance(v, dict):
            return v
    return {}


def _maybe_list(ss, keys: list[str]) -> list:
    """
    Return the first list found in `ss` among `keys`, else [].
    """
    for k in keys:
        v = ss.get(k)
        if isinstance(v, list):
            return v
    return []


def _non_none_items(d: dict) -> dict:
    """
    Filter out None / "None" values from a dict (cosmetic for display).
    """
    return {k: v for k, v in d.items() if v is not None and v != "None"}


# ---------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------
def render_final_summary(df: pd.DataFrame, ss) -> None:
    """
    Render the Final Summary page.

    Parameters
    ----------
    df : pd.DataFrame
        The current (possibly processed) dataset to summarize.
    ss : streamlit.session_state
        Shared state used to pull prior-step settings and store insights.
    """
    # Defensive guard: nothing to show
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        st.info("No data to summarize.")
        return

    ds_name = ss.get("active_ds", "dataset")
    init_insights_store(ss)  # ensure legacy store is present

    # --- KPI cards ----------------------------------------------------
    st.subheader("Technical Summary")
    ov = overview_stats(df)
    kpi_row(
        [
            ("Rows", f"{ov['rows']:,}"),
            ("Cols", f"{ov['cols']:,}"),
            ("Memory (MB)", f"{ov['memory_mb']:.2f}"),
            ("Duplicate rows", f"{ov['n_duplicates']:,}"),
        ]
    )

    # --- Outstanding issues (heuristics) ------------------------------
    with section("Outstanding issues (heuristics)", expandable=False):
        tips = suggest_actions(df)
        if not tips:
            st.success("No major issues detected.")
        else:
            for t in tips:
                st.markdown(f"- {t}")

    # --- Recap: what changed in previous steps ------------------------
    with section("What changed in previous steps", expandable=False):
        # Encoding choices (column -> method)
        enc_choices = _non_none_items(_maybe_dict(ss, ["enc_choices", "encoding_choices"]))
        if enc_choices:
            st.markdown("**Encoding**")
            enc_rows = [{"column": c, "method": m} for c, m in enc_choices.items()]
            render_table(
                pd.DataFrame(enc_rows).sort_values("column").reset_index(drop=True),
                height=200,
            )

        # Missing value handling plans
        null_plans = _maybe_dict(ss, ["mv_plans", "missing_plans", "null_plans", "impute_plans"])
        if null_plans:
            st.markdown("**Missing values**")
            rows = []
            for col, plan in null_plans.items():
                if isinstance(plan, dict):
                    method = plan.get("method", "—")
                    extra = ", ".join([f"{k}={v}" for k, v in plan.items() if k != "method"])
                else:
                    method, extra = str(plan), ""
                rows.append({"column": col, "method": method, "params": extra})
            if rows:
                render_table(
                    pd.DataFrame(rows).sort_values("column").reset_index(drop=True),
                    height=220,
                )

        # Outlier handling summary
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

        # Text processing columns
        text_cols = _maybe_list(ss, ["text_cols", "ppt_text_cols", "pp_text_cols"])
        if text_cols:
            st.markdown("**Text processing**")
            st.write(f"- **Columns**: {', '.join(map(str, text_cols))}")

        # Time-series calendar features added
        added_feats = _maybe_list(ss, ["ts_added_features", "ts_feat_added"])
        if added_feats:
            st.markdown("**Calendar features**")
            render_table(pd.DataFrame({"feature": added_feats}), height=160)

        if not any([enc_choices, null_plans, out_cols, text_cols, added_feats]):
            st.caption("No recorded changes from previous steps (or nothing was applied).")

    # --- Insights: auto-scan + export --------------------------------
    with section("Insights (auto-scan & export)", expandable=False):
        c1, c2, c3 = st.columns([1, 1, 2])
        default_sample = max(1, min(5000, len(df)))
        sample_rows = c1.number_input(
            "Sample rows (for speed)",
            min_value=1,
            value=default_sample,
            step=500,
            key="ins_sample",
        )
        max_pairs = c2.number_input(
            "Max pairs per type", min_value=5, value=30, step=5, key="ins_pairs"
        )

        # Run a lightweight scan to collect structured + legacy insights
        if st.button("Scan dataset for insights", type="primary", key="btn_scan_insights"):
            added = scan_dataset_for_insights(
                ss,
                ds_name,
                df,
                sample_rows=int(sample_rows),
                max_num_num_pairs=int(max_pairs),
                max_cat_num_pairs=int(max_pairs),
                min_abs_corr=0.6,
                max_cat_cardinality=50,
                include_cat_cat=False,  # keep noise down
            )
            st.success(f"Captured {added} insight(s).")

        # Quick counters to show what we have in store
        n_struct = len(st.session_state.get("_insights", {}).get(ds_name, {}).get("items", []))
        from core.insights_extractors import get_insights  # local import to avoid circularity
        n_legacy = len(get_insights(st.session_state, ds_name))
        st.caption(f"Stored insights — structured: {n_struct}, legacy: {n_legacy}")

        # Pretty on-page summary of collected insights
        tabs = st.tabs(["Insight Summary"])
        with tabs[0]:
            render_collected_insights_pretty(ss, ds_name)

        # Downloads: JSON
        json_bytes = insights_to_json_bytes(ss, ds_name)
        st.download_button(
            "Download insights (JSON)",
            data=json_bytes,
            file_name=f"{ds_name}_insights.json",
            mime="application/json",
            key="dl_insights_json",
        )

        # Downloads: Markdown
        md_text = generate_markdown_report(ss, ds_name)
        st.download_button(
            "Download insights (Markdown)",
            data=md_text.encode("utf-8"),
            file_name=f"{ds_name}_insights.md",
            mime="text/markdown",
            key="dl_insights_md",
        )

        # Downloads: PDF (optional dependency)
        pdf_bytes = insights_pdf_bytes(md_text)
        if pdf_bytes:
            st.download_button(
                "Download insights (PDF)",
                data=pdf_bytes,
                file_name=f"{ds_name}_insights.pdf",
                mime="application/pdf",
                key="dl_insights_pdf",
            )
        else:
            st.caption("📎 Install `reportlab` to enable PDF export: `pip install reportlab`.")

    # --- Data preview & CSV export -----------------------------------
    with section("Preview / Export", expandable=False):
        st.caption(f"Result: **{df.shape[0]:,} × {df.shape[1]:,}**")
        st.dataframe(df.head(50), use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download processed CSV",
            data=csv,
            file_name="processed.csv",
            mime="text/csv",
            key="final_summary_download_csv",
        )
