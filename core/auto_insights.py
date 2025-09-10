# core/auto_insights.py
from __future__ import annotations

import re
from collections import Counter
from textwrap import dedent

import numpy as np
import pandas as pd
import streamlit as st

# Consistent look & feel
from ui.components import render_table


# ------------------ INTERNAL HELPERS (dataset-agnostic) ------------------

def _ai_numeric_cols(df: pd.DataFrame) -> list[str]:
    """Return names of numeric columns (per pandas' dtype semantics)."""
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _ai_categorical_cols(df: pd.DataFrame, max_card: int = 50) -> list[str]:
    """
    Return categorical/text-like columns with bounded cardinality.
    We allow object/category/string columns whose nunique ∈ (1, max_card].
    """
    out: list[str] = []
    for c in df.columns:
        if (
            pd.api.types.is_object_dtype(df[c])
            or pd.api.types.is_categorical_dtype(df[c])
            or df[c].dtype == "string"
        ):
            try:
                k = int(pd.Series(df[c]).nunique(dropna=True))
                if 1 < k <= max_card:
                    out.append(c)
            except Exception:
                continue
    return out


def _ai_time_col(df: pd.DataFrame) -> str | None:
    """
    Heuristic: prefer a real datetime column; otherwise try common name patterns
    and ensure enough parsable values to be useful.
    """
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if dt_cols:
        return dt_cols[0]

    name_hits = sorted(
        [c for c in df.columns if any(k in str(c).lower() for k in ["date", "time", "timestamp", "created", "dt"])],
        key=lambda x: -len(str(x)),
    )
    for c in name_hits:
        coerced = pd.to_datetime(df[c], errors="coerce")
        if coerced.notna().sum() >= max(10, 0.05 * len(df)):
            return c
    return None


def _ai_choose_metric(df: pd.DataFrame) -> str | None:
    """
    Choose a numeric 'metric' to sum/trend on: prefer high variance & low missingness.
    Returns the column name or None if no numeric columns exist.
    """
    nums = _ai_numeric_cols(df)
    if not nums:
        return None
    cand: list[tuple[float, str]] = []
    for c in nums:
        s = pd.to_numeric(df[c], errors="coerce")
        nn = s.notna().sum()
        if nn == 0:
            continue
        var = float(np.nanvar(s.astype("float64")))
        null_rate = 1 - nn / len(df) if len(df) else 1
        score = var * (1 - null_rate)
        cand.append((score, c))
    cand.sort(reverse=True)
    return cand[0][1] if cand else None


def _ai_outlier_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute outlier counts by numeric column using Tukey's fences (IQR).
    Returns a table sorted by outlier_rate desc.
    """
    rows = []
    for c in _ai_numeric_cols(df):
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            rows.append((c, 0, 0.0))
            continue
        q1, q3 = np.percentile(s, 25), np.percentile(s, 75)
        iqr = q3 - q1
        if iqr <= 0:
            rows.append((c, 0, 0.0))
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        k = int(((s < lo) | (s > hi)).sum())
        rows.append((c, k, k / max(1, len(s))))
    out = pd.DataFrame(rows, columns=["column", "outliers", "outlier_rate"])
    return out.sort_values("outlier_rate", ascending=False)


def _ai_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Missing rate per column, descending."""
    pct = df.isna().mean().sort_values(ascending=False)
    # Older pandas: use to_frame(...).reset_index() and rename "index"
    return pct.to_frame("missing_rate").reset_index().rename(columns={"index": "column"})


def _ai_cardinality(df: pd.DataFrame) -> pd.DataFrame:
    """Cardinality and unique_rate per column, sorted by nunique desc."""
    rows = []
    n = len(df)
    for c in df.columns:
        try:
            k = int(pd.Series(df[c]).nunique(dropna=True))
        except Exception:
            k = np.nan
        rows.append((c, k, (k / n) if n else np.nan))
    return (
        pd.DataFrame(rows, columns=["column", "nunique", "unique_rate"])
        .sort_values("nunique", ascending=False)
    )


def _ai_correlations(df: pd.DataFrame, topk: int = 10, min_abs: float = 0.6) -> pd.DataFrame:
    """
    Top absolute correlations among numeric columns only (|r| >= min_abs).
    """
    nums = _ai_numeric_cols(df)
    if len(nums) < 2:
        return pd.DataFrame(columns=["col_a", "col_b", "corr_abs"])
    c = df[nums].corr(numeric_only=True).abs()
    rows = []
    for i, a in enumerate(nums):
        for j, b in enumerate(nums):
            if j <= i:
                continue
            v = c.loc[a, b]
            if pd.isna(v):
                continue
            if v >= min_abs:
                rows.append((a, b, float(v)))
    out = pd.DataFrame(rows, columns=["col_a", "col_b", "corr_abs"]).sort_values("corr_abs", ascending=False)
    return out.head(topk)


def _ai_top_categories(df: pd.DataFrame, metric_col: str | None, max_show: int = 10) -> dict[str, pd.DataFrame]:
    """
    For each low-cardinality categorical column, return:
      - counts table if metric_col is None, else
      - sum(metric_col) by category, both limited to top rows.
    """
    res: dict[str, pd.DataFrame] = {}
    if metric_col is None:
        for cat in _ai_categorical_cols(df):
            t = (
                df[cat]
                .value_counts(dropna=False)
                .head(max_show)
                .rename("count")
                .reset_index()
                .rename(columns={"index": cat})
            )
            res[cat] = t
        return res

    for cat in _ai_categorical_cols(df):
        g = df.groupby(cat)[metric_col].apply(lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum())
        g = g.sort_values(ascending=False).head(max_show).reset_index().rename(columns={metric_col: "metric_sum"})
        res[cat] = g
    return res


def _ai_text_glance(df: pd.DataFrame, max_cols: int = 3) -> pd.DataFrame:
    """
    For up to max_cols text-like columns, show average length and top tokens from a sample.
    """
    text_cols = [c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or df[c].dtype == "string"]
    out = []
    for c in text_cols[:max_cols]:
        s = df[c].astype(str)
        avg_len = float(s.str.len().mean())
        tokens: list[str] = []
        # Lightweight glance on a sample
        sample_n = min(len(s), 2000)
        if sample_n > 0:
            for v in s.sample(sample_n, random_state=42):
                tokens.extend(re.findall(r"[A-Za-z0-9]+", v.lower()))
        common = Counter(tokens).most_common(10) if tokens else []
        out.append({"column": c, "avg_len": round(avg_len, 1), "top_tokens": ", ".join([t for t, _ in common])})
    return pd.DataFrame(out)


def _ai_trend(df: pd.DataFrame, time_col: str | None, metric_col: str | None) -> pd.Series:
    """
    Build a coarse monthly trend: sum(metric) by month from (time_col, metric_col).
    Returns a pandas Series indexed by month (timestamp).
    """
    if time_col is None or metric_col is None:
        return pd.Series(dtype="float64")
    w = df[[time_col, metric_col]].copy()
    w[time_col] = pd.to_datetime(w[time_col], errors="coerce")
    w[metric_col] = pd.to_numeric(w[metric_col], errors="coerce")
    w = w.dropna(subset=[time_col]).copy()
    if w.empty:
        return pd.Series(dtype="float64")
    w["__period"] = w[time_col].dt.to_period("M").dt.to_timestamp()
    ts = w.groupby("__period")[metric_col].sum().sort_index()
    return ts


def _ai_forecast_next(ts: pd.Series) -> tuple[float, float]:
    """
    Naive linear projection on the most recent <= 12 points.
    Returns (next_value, pct_delta_vs_last).
    """
    if ts is None or len(ts) < 3:
        return (np.nan, np.nan)
    last = ts.tail(12) if len(ts) >= 12 else ts
    x = np.arange(len(last))
    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, last.values, rcond=None)[0]
    next_val = slope * len(x) + intercept
    delta = (next_val - last.iloc[-1]) / last.iloc[-1] if last.iloc[-1] != 0 else np.nan
    return (float(next_val), float(delta))


def _pct_fmt(x) -> str:
    """Format a fraction as a percentage string, fallback to em-dash."""
    try:
        return f"{x*100:.1f}%"
    except Exception:
        return "—"


def _ai_build_html(payload: dict) -> str:
    """Standalone HTML export (compact styling, safe defaults for empty frames)."""
    css = dedent("""
    <style>
      body{font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#111827;padding:24px}
      h1,h2,h3{margin:0 0 8px}
      .muted{color:#6b7280}
      .kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:16px 0 24px}
      .kpi{border:1px solid #e5e7eb;border-radius:10px;padding:12px}
      .kpi .label{color:#6b7280;font-size:12px}
      .kpi .value{font-size:20px;font-weight:600}
      table{border-collapse:collapse;width:100%;font-size:13px}
      th,td{border:1px solid #e5e7eb;padding:6px 8px;text-align:left}
      th{background:#f9fafb}
      .section{margin:24px 0}
      .two{display:grid;grid-template-columns:1fr 1fr;gap:16px}
    </style>
    """)
    def _tbl(df):
        if df is None or (hasattr(df, "empty") and df.empty):
            return "<p class='muted'>—</p>"
        return df.head(20).to_html(index=False, border=0)

    bullets = "".join([f"<li>{b}</li>" for b in payload.get("story", [])]) or "<li class='muted'>Provide more structure to enrich narrative.</li>"
    return f"""
    <html><head><meta charset='utf-8'/>{css}</head><body>
      <h1>Auto Insights</h1>
      <div class="muted">Dataset: {payload.get('dataset','—')}</div>

      <div class="kpis">
        <div class="kpi"><div class="label">Rows</div><div class="value">{payload['kpis'].get('rows','—')}</div></div>
        <div class="kpi"><div class="label">Missing overall</div><div class="value">{payload['kpis'].get('miss_overall','—')}</div></div>
        <div class="kpi"><div class="label">Duplicates</div><div class="value">{payload['kpis'].get('dups','—')}</div></div>
        <div class="kpi"><div class="label">Next-month forecast</div><div class="value">{payload['kpis'].get('forecast','—')} <span class="muted">({_pct_fmt(payload['kpis'].get('forecast_delta'))})</span></div></div>
      </div>

      <div class="section"><h3>Top missing columns</h3>{_tbl(payload.get('miss_cols'))}</div>
      <div class="section"><h3>High-cardinality columns</h3>{_tbl(payload.get('card_high'))}</div>
      <div class="section"><h3>Outlier-heavy numeric columns</h3>{_tbl(payload.get('outliers'))}</div>
      <div class="section"><h3>Strong correlations</h3>{_tbl(payload.get('corr'))}</div>
      <div class="section two">
        <div><h3>Top categories (by {payload.get('metric','row count')})</h3>{_tbl(payload.get('topcats_any'))}</div>
        <div><h3>Text columns (glance)</h3>{_tbl(payload.get('text_glance'))}</div>
      </div>
      <div class="section"><h3>Trend (monthly)</h3>{_tbl(payload.get('trend_tbl'))}</div>

      <div class="section"><h3>Plain-language story</h3><ul>{bullets}</ul></div>
    </body></html>
    """


def _ai_build_html_bytes(payload: dict) -> bytes:
    """Bytes wrapper for download button."""
    return _ai_build_html(payload).encode("utf-8")


@st.cache_data(show_spinner=False)
def _ai_compute(df: pd.DataFrame, cap_rows: int = 200_000):
    """
    Compute generalized insights on a copy/sampled slice of the data.
    Returns: (payload_dict, trend_series, metric_col, time_col, topcats_map)
    """
    # Sample for speed but keep behaviour stable
    work = df
    if len(work) > cap_rows:
        work = work.sample(cap_rows, random_state=42)

    # Basics
    miss_overall = float(work.isna().mean().mean())
    dups = int(work.duplicated().sum())
    miss_cols = _ai_missingness(work).head(15)

    # Cardinality
    card = _ai_cardinality(work)
    card_high = card[card["unique_rate"] >= 0.5].head(15)

    # Numeric quality
    outliers = _ai_outlier_counts(work).head(15)

    # Correlations
    corr = _ai_correlations(work, topk=12, min_abs=0.6)

    # Time & metric
    time_col = _ai_time_col(work)
    metric_col = _ai_choose_metric(work)
    ts = _ai_trend(work, time_col, metric_col)
    trend_tbl = (
        ts.tail(12).reset_index().rename(columns={"__period": "period", metric_col or "value": "value"})
        if len(ts) > 0 else pd.DataFrame()
    )

    # Forecast
    f_val, f_delta = _ai_forecast_next(ts.tail(12)) if len(ts) > 0 else (np.nan, np.nan)

    # Categories
    topcats_map = _ai_top_categories(work, metric_col)
    # pick one example for HTML preview
    topcats_any = next((v for v in topcats_map.values() if v is not None and not v.empty), None)

    # Text glance
    text_glance = _ai_text_glance(work)

    payload = {
        "kpis": {
            "rows": f"{len(df):,}",
            "miss_overall": _pct_fmt(miss_overall),
            "dups": f"{dups:,}",
            "forecast": f"{f_val:,.0f}" if not np.isnan(f_val) else "—",
            "forecast_delta": f_delta,
        },
        "miss_cols": miss_cols,
        "card_high": card_high,
        "outliers": outliers,
        "corr": corr,
        "trend_tbl": trend_tbl,
        "text_glance": text_glance,
        "topcats_any": topcats_any,
        "metric": metric_col or "row count",
        "story": _ai_story(miss_overall, outliers, corr, ts, topcats_any),
    }, ts, metric_col, time_col, topcats_map

    return payload


def _ai_story(miss_overall, outliers_df, corr_df, ts, topcats_any) -> list[str]:
    """Generate a short, plain-language narrative from key stats."""
    bullets: list[str] = []
    if miss_overall > 0.05:
        bullets.append(f"Overall missingness is {_pct_fmt(miss_overall)}; consider imputations.")
    if outliers_df is not None and not outliers_df.empty and outliers_df.iloc[0]["outlier_rate"] >= 0.05:
        bullets.append(
            f"Column **{outliers_df.iloc[0]['column']}** shows a high outlier rate "
            f"({_pct_fmt(outliers_df.iloc[0]['outlier_rate'])})."
        )
    if corr_df is not None and not corr_df.empty:
        a, b, v = corr_df.iloc[0].tolist()
        bullets.append(f"Strong correlation between **{a}** and **{b}** (|r|={v:.2f}); may indicate redundancy or a causal link.")
    if ts is not None and len(ts) >= 3:
        last12 = ts.tail(12)
        if len(last12) >= 2 and last12.iloc[-2] != 0:
            mom = (last12.iloc[-1] - last12.iloc[-2]) / last12.iloc[-2]
            bullets.append(f"Latest month changed {_pct_fmt(mom)} vs. prior month.")
        f_val, f_delta = _ai_forecast_next(last12)
        if not np.isnan(f_delta):
            bullets.append(f"Projected next month {_pct_fmt(f_delta)} based on a linear trend.")
    if topcats_any is not None and not topcats_any.empty:
        top_row = topcats_any.iloc[0]
        bullets.append(
            f"Dominant segment in **{topcats_any.columns[0]}** is **{top_row.iloc[0]}** "
            f"({float(top_row.iloc[1]):,.0f})."
        )
    return bullets


# ------------------ PUBLIC RENDERER ------------------

def render_auto_insights(ss) -> None:
    """
    Streamlit renderer for generalized Auto Insights.
    - Works with ANY tabular dataset (no business-specific columns needed)
    - Shows KPIs, missingness, outliers, correlations, segments, trend, narrative
    - Provides a Download (HTML) button
    """
    st.subheader("Auto Insights")

    # --- Compact glossary (collapsible) ---
    with st.expander("Key terms (quick guide)", expanded=False):
        st.markdown(
            """
- **Missingness** — Share of blanks/NaN values. High rates can bias analysis.
- **Duplicates** — Rows that appear more than once. Consider de-duplication.
- **High cardinality** — Many distinct labels in a column (e.g., IDs). Beware wide one-hot encoding.
- **Outliers (IQR rule)** — Values far outside the middle 50% (Tukey fences). They can skew means/variances.
- **Correlation |r|** — Strength of linear relationship between two numeric columns. ~0.1 weak, ~0.3 moderate, ≥0.5 strong (context matters).
- **Trend (monthly)** — Sum of a chosen numeric metric by month; used for a quick trajectory view.
- **Naive forecast** — Simple linear trend extrapolation on the latest data; directional, not production-grade.
- **Top categories** — Most important segments by count or by the chosen metric’s sum.
- **Text glance** — Quick look at average length and frequent tokens in text-like columns (no heavy NLP).
            """
        )

    if not ss.datasets:
        st.info("Load a dataset first.")
        return

    # Pick dataset (default to active)
    names = sorted(ss.datasets.keys())
    if ss.active_ds not in names:
        ss.active_ds = names[0]
    ds_name = st.selectbox("Dataset", names, index=names.index(ss.active_ds), key="ai_ds")
    df = ss.datasets[ds_name]
    if df.empty:
        st.warning("Selected dataset is empty.")
        return

    # Robust slider that works for tiny or huge datasets
    total = int(len(df))
    cap_max = int(min(500_000, max(1, total)))
    min_val = int(min(50_000, cap_max))  # if total < 50k, min == max == total
    default_v = int(max(min_val, min(200_000, cap_max)))
    step_val = int(max(1, min(50_000, max(1, (cap_max - min_val) // 4))))

    cap = st.slider(
        "Rows to analyze (sampled if larger)",
        min_value=min_val,
        max_value=cap_max,
        value=default_v,
        step=step_val,
        key="ai_cap",
    )

    # Compute insights (cached)
    payload_bundle = _ai_compute(df, cap_rows=cap)
    payload, ts, metric_col, time_col, topcats_map = payload_bundle

    # Inject dataset name for HTML export
    payload["dataset"] = ds_name

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows", payload["kpis"]["rows"])
    k2.metric("Missing overall", payload["kpis"]["miss_overall"])
    k3.metric("Duplicates", payload["kpis"]["dups"])
    k4.metric("Next-month forecast", payload["kpis"]["forecast"], delta=_pct_fmt(payload["kpis"]["forecast_delta"]))

    # Tables: quality + associations
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top missing columns**")
        render_table(payload["miss_cols"])
        st.markdown("**High-cardinality columns**")
        render_table(payload["card_high"])
    with c2:
        st.markdown("**Outlier-heavy numeric columns**")
        render_table(payload["outliers"])
        st.markdown("**Strong correlations**")
        render_table(payload["corr"])

    # Trend
    st.markdown("**Trend (monthly)**")
    if ts is not None and len(ts) > 0:
        render_table(payload["trend_tbl"])
        st.line_chart(ts.tail(24))
    else:
        st.caption("No clear time column + numeric metric found to build a trend.")

    # Categories (show up to two as charts for quick visual)
    st.markdown(f"**Top categories (by {payload['metric']})**")
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
        if shown >= 2:
            break
    if shown == 0:
        st.caption("No low-cardinality categorical columns to segment.")

    # Text glance
    st.markdown("**Text columns (glance)**")
    render_table(payload["text_glance"])

    # Story
    st.markdown("**Plain-language story**")
    if payload["story"]:
        for b in payload["story"]:
            st.markdown(f"- {b}")
    else:
        st.caption("Add a time and numeric column to enrich the narrative.")

    # Download
    st.download_button(
        "⬇️ Download Auto Insights (HTML)",
        data=_ai_build_html_bytes(payload),
        file_name=f"{ds_name}_auto_insights.html",
        mime="text/html",
        type="primary",
        key="ai_download_html",
    )
