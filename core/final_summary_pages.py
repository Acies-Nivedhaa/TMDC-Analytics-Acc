# core/final_pages.py
from __future__ import annotations

import io
import json
import zipfile
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from ui.components import render_table, kpi_row, section
from core.summary import (
    build_summary_pdf,  # used in Report & Export
    overview_stats, suggest_actions, nunique_safe,  # reused by Technical Summary
)
from core.auto_insights import (
    _ai_compute, _ai_choose_metric, _ai_trend, _pct_fmt,
)

# -------------------- small shared utils --------------------

def _dtype_bucket(s: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(s): return "numeric"
    if pd.api.types.is_datetime64_any_dtype(s): return "datetime"
    if pd.api.types.is_bool_dtype(s): return "boolean"
    if pd.api.types.is_categorical_dtype(s): return "categorical"
    return "text"

def _dtype_icon(kind: str) -> str:
    return {
        "numeric": "🔢", "datetime": "⏱️", "boolean": "🔁",
        "categorical": "🏷️", "text": "📝"
    }.get(kind, "📦")

def _robust_slider(label: str, total_rows: int, key: str, default_cap: int = 200_000) -> int:
    total = int(total_rows)
    cap_max   = int(min(500_000, max(1, total)))
    min_val   = int(min(50_000, cap_max))                 # if total < 50k, min == max == total
    default_v = int(max(min_val, min(default_cap, cap_max)))
    step_val  = int(max(1, min(50_000, max(1, (cap_max - min_val) // 4))))
    return st.slider(label, min_value=min_val, max_value=cap_max, value=default_v, step=step_val, key=key)

def _tables_multiselect(ss) -> List[str]:
    names = sorted(ss.datasets.keys())
    return st.multiselect(
        "Tables (multi-select, searchable)", options=names, default=names, placeholder="Pick one or more tables…"
    )

def _safe_sum_numeric(df: pd.DataFrame) -> Tuple[str, float]:
    col = _ai_choose_metric(df)
    if not col: return ("", np.nan)
    try:
        val = pd.to_numeric(df[col], errors="coerce").sum()
        return (col, float(val))
    except Exception:
        return (col, np.nan)

def _concat_if_possible(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    if not dfs: return None
    try:
        return pd.concat(dfs.values(), axis=0, ignore_index=True, sort=False)
    except Exception:
        return None

# ---------- helpers ported from old final_summary.py ----------

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

# -------------------- Page 0: Technical Summary (old final_summary.py) --------------------

def render_technical_summary_page(ss):
    st.markdown("### Technical Summary")

    if not ss.datasets:
        st.info("No data to summarize.")
        return

    # keep active valid
    names_all = sorted(ss.datasets.keys())
    if ss.active_ds not in names_all:
        ss.active_ds = names_all[0]

    df = ss.datasets[ss.active_ds]
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        st.info("No data to summarize.")
        return

    # KPIs
    ov = overview_stats(df)
    kpi_row([
        ("Rows", f"{ov['rows']:,}"),
        ("Cols", f"{ov['cols']:,}"),
        ("Memory (MB)", f"{ov['memory_mb']:.2f}"),
        ("Duplicate rows", f"{ov['n_duplicates']:,}"),
    ])

    # Outstanding issues
    with section("Outstanding issues (heuristics)", expandable=False):
        tips = suggest_actions(df)
        if not tips:
            st.success("No major issues detected.")
        else:
            for t in tips:
                st.markdown(f"- {t}")

    # What changed in previous steps (reads SS the same way your old page did)
    with section("What changed in previous steps", expandable=False):
        # Encoding
        enc_choices = _maybe_dict(ss, ["enc_choices", "encoding_choices"])
        enc_choices = _non_none_items(enc_choices)
        if enc_choices:
            enc_rows = [{"column": c, "method": m} for c, m in enc_choices.items() if m and m != "None"]
            if enc_rows:
                st.markdown("**Encoding**")
                render_table(pd.DataFrame(enc_rows).sort_values("column").reset_index(drop=True), height=200)

        # Missing/null handling
        null_plans = (
            _maybe_dict(ss, ["mv_plans", "missing_plans", "null_plans", "impute_plans"])
            or {}
        )
        if null_plans:
            st.markdown("**Missing values**")
            rows = []
            for col, plan in null_plans.items():
                if isinstance(plan, dict):
                    method = plan.get("method", "—")
                    extra  = ", ".join([f"{k}={v}" for k, v in plan.items() if k != "method"])
                else:
                    method = str(plan)
                    extra = ""
                rows.append({"column": col, "method": method, "params": extra})
            if rows:
                render_table(pd.DataFrame(rows).sort_values("column").reset_index(drop=True), height=220)

        # Outliers
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

        # Text processing
        text_cols = _maybe_list(ss, ["text_cols", "ppt_text_cols", "pp_text_cols"])
        if text_cols:
            st.markdown("**Text processing**")
            st.write(f"- **Columns**: {', '.join(map(str, text_cols))}")

        # Time features
        added_feats = _maybe_list(ss, ["ts_added_features", "ts_feat_added"])
        if added_feats:
            st.markdown("**Calendar features**")
            render_table(pd.DataFrame({"feature": added_feats}), height=160)

        if not any([enc_choices, null_plans, out_cols, text_cols, added_feats]):
            st.caption("No recorded changes from previous steps (or nothing was applied).")

    # Preview & Export
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
        )

# -------------------- Page 1: Overview --------------------

def render_overview_page(ss):
    st.markdown("### Overview")

    chosen = _tables_multiselect(ss)
    if not chosen:
        st.info("Select at least one table to summarize.")
        return

    total_rows = sum(len(ss.datasets[nm]) for nm in chosen)
    total_tables = len(chosen)

    # Key metric (auto)
    key_metric_label, key_metric_sum = "", np.nan
    candidates = [ss.active_ds] + [nm for nm in chosen if nm != ss.active_ds]
    for nm in candidates:
        if nm in ss.datasets:
            m, v = _safe_sum_numeric(ss.datasets[nm])
            if m:
                key_metric_label, key_metric_sum = m, v
                break

    # Data health (on active for speed)
    act = ss.datasets[ss.active_ds] if ss.active_ds in ss.datasets else ss.datasets[chosen[0]]
    miss_overall = float(act.isna().mean().mean())
    dups = int(act.duplicated().sum())

    # dtype counts across selected
    dtype_counts = {}
    for nm in chosen:
        df = ss.datasets[nm]
        for c in df.columns:
            k = _dtype_bucket(df[c]); dtype_counts[k] = dtype_counts.get(k, 0) + 1
    dtype_df = pd.DataFrame({"dtype": list(dtype_counts.keys()), "count": list(dtype_counts.values())}).sort_values("count", ascending=False)

    # Trend highlight (active)
    trend_label = ""
    ts = pd.Series(dtype="float64")
    try:
        metric_col = _ai_choose_metric(act)
        time_col = None
        for c in act.columns:
            if pd.api.types.is_datetime64_any_dtype(act[c]):
                time_col = c; break
        ts = _ai_trend(act, time_col, metric_col)
        trend_label = metric_col or "value"
    except Exception:
        pass

    # KPIs
    kpi_row([
        ("Total records", f"{total_rows:,}"),
        ("Tables selected", f"{total_tables:,}"),
        (f"Key metric ({key_metric_label or '—'})", f"{key_metric_sum:,.0f}" if not np.isnan(key_metric_sum) else "—"),
        ("Missing overall", _pct_fmt(miss_overall)),
    ])

    with section("Data Health"):
        c1, c2 = st.columns([2, 1])
        with c1:
            miss_cols = act.isna().mean().sort_values(ascending=False).head(20)
            st.markdown("**Missing by column (active dataset)**")
            if not miss_cols.empty:
                st.bar_chart(miss_cols)
            else:
                st.caption("No missing values detected.")
        with c2:
            st.markdown("**Types**")
            if not dtype_df.empty:
                render_table(dtype_df)
                st.bar_chart(dtype_df.set_index("dtype")["count"])
            else:
                st.caption("No columns.")

    with section("Trend Highlights"):
        if ts is not None and len(ts) > 0:
            st.line_chart(ts.tail(24))
        else:
            st.caption("No obvious time+metric combination detected for the active table.")

    # Summary bullets
    bullets = []
    bullets.append(f"Data covers **{total_rows:,}** records across **{total_tables}** table(s).")
    bullets.append(f"Overall missingness (active table) is **{_pct_fmt(miss_overall)}** with **{dups:,}** duplicate row(s).")
    if key_metric_label:
        bullets.append(f"Detected **{key_metric_label}** as a salient numeric field; total observed = **{key_metric_sum:,.0f}**.")
    if ts is not None and len(ts) >= 2 and ts.iloc[-2] != 0:
        mom = (ts.iloc[-1] - ts.iloc[-2]) / ts.iloc[-2]
        bullets.append(f"Latest period changed **{_pct_fmt(mom)}** vs. prior.")
    st.markdown("**Summary**")
    for b in bullets: st.markdown(f"- {b}")

# -------------------- Page 2: Table Preview --------------------

def render_table_preview_page(ss):
    st.markdown("### Table Preview")

    chosen = _tables_multiselect(ss)
    if not chosen:
        st.info("Select at least one table to preview.")
        return

    tab_objs = st.tabs(chosen)
    for tab, nm in zip(tab_objs, chosen):
        with tab:
            df = ss.datasets[nm]
            st.markdown(f"**{nm}** — {len(df):,} rows × {df.shape[1]} columns")

            kinds = [_dtype_bucket(df[c]) for c in df.columns]
            labels = [f"{_dtype_icon(k)}  {c}" for c, k in zip(df.columns, kinds)]
            show_cols = st.multiselect(
                "Columns (filter)",
                options=list(range(len(df.columns))),
                format_func=lambda i: labels[i],
                default=list(range(min(20, len(labels))))
            )
            cols_chosen = [df.columns[i] for i in show_cols] if show_cols else list(df.columns)
            st.dataframe(df[cols_chosen].head(100), use_container_width=True)

            q_rows = []
            for c in cols_chosen[:12]:
                try:
                    nun = int(pd.Series(df[c]).nunique(dropna=True))
                except Exception:
                    nun = np.nan
                q_rows.append({
                    "column": c,
                    "type": _dtype_bucket(df[c]),
                    "non_null": int(df[c].notna().sum()),
                    "unique": nun,
                })
            if q_rows:
                st.markdown("**Quick stats (first 12 shown)**")
                render_table(pd.DataFrame(q_rows))

# -------------------- Page 3: EDA Insights --------------------

def render_eda_insights_page(ss):
    st.markdown("### EDA Insights")

    names = sorted(ss.datasets.keys())
    if not names:
        st.info("Load a dataset first.")
        return

    ds_name = st.selectbox("Dataset", names, index=names.index(ss.active_ds) if ss.active_ds in names else 0)
    df = ss.datasets[ds_name]
    if df.empty:
        st.warning("Selected dataset is empty.")
        return

    cap = _robust_slider("Rows to analyze (sampled if larger)", len(df), key="eda_suite_cap", default_cap=200_000)
    work = df if len(df) <= cap else df.sample(cap, random_state=42)

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if (_dtype_bucket(df[c]) in ("categorical", "text")) and df[c].nunique(dropna=True) <= 50]

    with section("Distributions"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Numeric (histograms)**")
            for c in num_cols[:3]:
                s = pd.to_numeric(work[c], errors="coerce").dropna()
                if s.empty: continue
                hist = pd.DataFrame({"bin": pd.cut(s, bins=20).astype(str)}).value_counts().sort_index()
                st.bar_chart(hist)
        with c2:
            st.markdown("**Categorical (top bars)**")
            for c in cat_cols[:3]:
                vc = work[c].astype(str).value_counts().head(15)
                st.bar_chart(vc)

    with section("Missing & Outliers (quick view)"):
        miss = work.isna().mean().sort_values(ascending=False).head(20)
        st.markdown("**Missing by column**")
        st.bar_chart(miss)
        out = []
        for c in num_cols:
            s = pd.to_numeric(work[c], errors="coerce").dropna()
            if s.empty: 
                out.append((c, 0)); continue
            q1, q3 = np.percentile(s, 25), np.percentile(s, 75)
            iqr = q3 - q1
            if iqr <= 0: out.append((c, 0)); continue
            lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
            out.append((c, int(((s < lo)|(s > hi)).sum())))
        out_df = pd.DataFrame(out, columns=["column","outliers"]).sort_values("outliers", ascending=False).head(20)
        st.markdown("**Outliers (count by numeric column)**")
        render_table(out_df)

    with section("Correlation View"):
        nums = [c for c in num_cols if pd.to_numeric(work[c], errors="coerce").notna().sum() > 0]
        if len(nums) >= 2:
            corr = work[nums].corr(numeric_only=True)
            st.dataframe(corr.style.format("{:.2f}"), use_container_width=True)
            top_pairs = []
            for i, a in enumerate(nums):
                for j, b in enumerate(nums):
                    if j <= i: continue
                    v = abs(corr.loc[a, b])
                    if not np.isnan(v):
                        top_pairs.append((a, b, float(v)))
            top_pairs.sort(key=lambda x: x[2], reverse=True)
            st.markdown("**Most related pairs**")
            render_table(pd.DataFrame(top_pairs[:10], columns=["col_a","col_b","|corr|"]))
        else:
            st.caption("Not enough numeric columns for correlations.")

# -------------------- Page 4: Business Insights (auto) --------------------

def render_business_insights_page(ss):
    st.markdown("### Business Insights (auto)")
    names = sorted(ss.datasets.keys())
    if not names:
        st.info("Load a dataset first.")
        return

    ds_name = st.selectbox("Dataset", names, index=names.index(ss.active_ds) if ss.active_ds in names else 0, key="biz_ds")
    df = ss.datasets[ds_name]
    if df.empty:
        st.warning("Selected dataset is empty.")
        return

    cap = _robust_slider("Rows to analyze (sampled if larger)", len(df), key="biz_cap", default_cap=200_000)

    try:
        payload, ts, metric_col, time_col, topcats_map = _ai_compute(df, cap_rows=cap)
    except Exception as e:
        st.error(f"Auto insights failed: {e}")
        return

    kpi_row([
        ("Rows", payload["kpis"]["rows"]),
        ("Missing overall", payload["kpis"]["miss_overall"]),
        ("Duplicates", payload["kpis"]["dups"]),
        ("Forecast (next period)", payload["kpis"]["forecast"]),
    ])

    st.markdown("**Top performers (segments)**")
    shown = 0
    for cat, df_top in topcats_map.items():
        if df_top is None or df_top.empty: 
            continue
        st.markdown(f"*{cat}*")
        render_table(df_top)
        try:
            st.bar_chart(df_top.set_index(df_top.columns[0])[df_top.columns[1]])
        except Exception:
            pass
        shown += 1
        if shown >= 2: break
    if shown == 0:
        st.caption("No low-cardinality categorical columns to segment.")

    st.markdown("**Trend & Forecast**")
    if ts is not None and len(ts) > 0:
        st.line_chart(ts.tail(24))
        last = ts.tail(2)
        if len(last) == 2 and last.iloc[-2] != 0:
            mom = (last.iloc[-1] - last.iloc[-2]) / last.iloc[-2]
            st.caption(f"Latest period change: {_pct_fmt(mom)} vs prior.")
    else:
        st.caption("No clear time+metric combo detected to chart trend.")

    st.markdown("**Plain-language story**")
    if payload.get("story"):
        for b in payload["story"]:
            st.markdown(f"- {b}")
    else:
        st.caption("Add a time and numeric column to enrich the narrative.")

# -------------------- Page 5: Report & Export --------------------

def render_report_export_page(ss):
    st.markdown("### Report & Export")

    names = sorted(ss.datasets.keys())
    if not names:
        st.info("Load a dataset first.")
        return

    ds_name = st.selectbox("Dataset for summary export", names, index=names.index(ss.active_ds) if ss.active_ds in names else 0, key="rep_ds")
    df = ss.datasets[ds_name]
    if df.empty:
        st.warning("Selected dataset is empty.")
        return

    cap = _robust_slider("Rows to analyze for report", len(df), key="rep_cap", default_cap=200_000)
    payload, ts, metric_col, time_col, topcats_map = _ai_compute(df, cap_rows=cap)
    payload["dataset"] = ds_name

    # CSV bundle of core tables
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        if payload["miss_cols"] is not None:
            zf.writestr(f"{ds_name}_missing.csv", payload["miss_cols"].to_csv(index=False))
        if payload["card_high"] is not None:
            zf.writestr(f"{ds_name}_high_cardinality.csv", payload["card_high"].to_csv(index=False))
        if payload["outliers"] is not None:
            zf.writestr(f"{ds_name}_outliers.csv", payload["outliers"].to_csv(index=False))
        if payload["corr"] is not None:
            zf.writestr(f"{ds_name}_correlations.csv", payload["corr"].to_csv(index=False))
        if payload["trend_tbl"] is not None:
            zf.writestr(f"{ds_name}_trend.csv", payload["trend_tbl"].to_csv(index=False))
    zip_bytes = buf.getvalue()

    col_a, col_b = st.columns(2)
    with col_a:
        # PDF summary (your existing generator)
        try:
            pdf_bytes = build_summary_pdf(df, dataset_name=ds_name)
            st.download_button(
                "⬇️ Download Executive Summary (PDF)",
                data=pdf_bytes,
                file_name=f"{ds_name}_summary.pdf",
                mime="application/pdf",
                key="rep_pdf",
            )
        except Exception:
            st.caption("PDF export uses build_summary_pdf; if not applicable, skip.")

        # HTML business insights (reuse auto_insights HTML)
        from core.auto_insights import _ai_build_html_bytes
        st.download_button(
            "⬇️ Download Business Insights (HTML)",
            data=_ai_build_html_bytes(payload),
            file_name=f"{ds_name}_business_insights.html",
            mime="text/html",
            key="rep_html",
        )

    with col_b:
        st.download_button(
            "⬇️ Download Insights Tables (ZIP of CSVs)",
            data=zip_bytes,
            file_name=f"{ds_name}_insights_tables.zip",
            mime="application/zip",
            key="rep_zip",
        )
        st.caption("Contains CSVs for missingness, cardinality, outliers, correlations, trend.")

# -------------------- Entrypoint: full 6-page suite --------------------

def render_final_summary_suite(ss):
    st.subheader("Executive Dashboard")

    tabs = st.tabs([
        "Technical Summary", "Overview", "Table Preview", "EDA Insights", "Business Insights", "Report & Export"
    ])

    with tabs[0]:
        render_technical_summary_page(ss)
    with tabs[1]:
        render_overview_page(ss)
    with tabs[2]:
        render_table_preview_page(ss)
    with tabs[3]:
        render_eda_insights_page(ss)
    with tabs[4]:
        render_business_insights_page(ss)
    with tabs[5]:
        render_report_export_page(ss)
