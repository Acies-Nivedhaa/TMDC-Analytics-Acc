# core/summary.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List
import json as _json
from io import BytesIO

# ReportLab is optional: only needed for PDF export
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, LongTable, TableStyle
    _REPORTLAB_OK = True
except Exception:
    _REPORTLAB_OK = False


__all__ = [
    # analytics helpers
    "overview_stats",
    "infer_schema",
    "column_quick_stats",
    "numeric_correlations",
    "suggest_actions",
    "dataset_meta",
    "demo_data",
    "nunique_safe",
    # exports
    "build_summary_pdf",
    "build_summary_html",
    "summary_pdf_bytes",
    "summary_html_bytes",
]


# =========================
# KPI / Stats
# =========================

def overview_stats(df: pd.DataFrame) -> Dict[str, float]:
    n_rows, n_cols = df.shape
    missing = int(df.isna().sum().sum())
    total   = int(n_rows * n_cols)
    missing_pct = (missing / total * 100.0) if total else 0.0
    mem_mb = float(df.memory_usage(deep=True).sum()) / (1024 ** 2)
    n_dup = _n_duplicates_safe(df)
    return {
        "rows": int(n_rows),
        "cols": int(n_cols),
        "memory_mb": mem_mb,
        "missing_pct": missing_pct,
        "n_duplicates": n_dup,
    }


# =========================
# PDF helpers
# =========================

def _longtables_from_df(
    df_in: pd.DataFrame,
    title: str,
    styles,
    max_cols: int = 8
) -> list:
    """Paginate vertically and split wide frames horizontally to avoid truncation."""
    from reportlab.platypus import Paragraph, Spacer
    flow = []
    if df_in is None or df_in.empty:
        return flow

    cols = df_in.columns.tolist()
    n = len(cols)
    parts = (n + max_cols - 1) // max_cols

    for i in range(parts):
        sub = df_in.iloc[:, i * max_cols : (i + 1) * max_cols]
        part_label = f" (part {i+1})" if parts > 1 else ""
        flow.append(Paragraph(f"{title}{part_label}", styles["Heading3"]))

        data = [list(sub.columns)] + sub.astype(str).values.tolist()
        tbl = LongTable(
            data,
            repeatRows=1,
            style=TableStyle([
                ("FONT", (0,0), (-1,-1), "Helvetica", 8),
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f0f2f6")),
                ("LINEBELOW", (0,0), (-1,0), 0.5, colors.HexColor("#c5cbd3")),
                ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#e4e7ea")),
                ("ALIGN", (0,0), (-1,-1), "LEFT"),
                ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ])
        )
        flow += [tbl, Spacer(1, 8)]
    return flow


# =========================
# HTML helpers (scrollable)
# =========================

def _html_table(df_in: pd.DataFrame, *, max_height_px: int | None = 420, caption: str | None = None) -> str:
    """
    Scrollable (both directions) HTML table with sticky header.
    - Horizontal scroll: table uses width:max-content and sits inside an overflow:auto container.
    """
    if df_in is None or df_in.empty:
        return "<div class='empty'>No data</div>"

    html_tbl = df_in.to_html(
        index=False,
        border=0,
        classes="grid",
        na_rep="—",
        escape=True,
    )

    cap = f"<div class='cap'>{caption}</div>" if caption else ""
    mh = f"max-height:{max_height_px}px;" if max_height_px else ""
    return f"""
    <div class='box'>
      {cap}
      <div class='scroll' style="{mh} overflow:auto;">
        <div class='scroll-inner'>
          {html_tbl}
        </div>
      </div>
    </div>
    """


# =========================
# HTML Summary (returns str)
# =========================

def build_summary_html(
    active_name: str,
    df: pd.DataFrame,
    datasets: dict[str, pd.DataFrame] | None = None,
) -> str:
    """Self-contained HTML report with vertical & horizontal scrolling (no truncation)."""
    ov = overview_stats(df)
    meta = dataset_meta(df)

    # Datasets in session
    datasets_df = None
    if datasets:
        rows = []
        for nm, dsub in sorted(datasets.items()):
            o = overview_stats(dsub)
            rows.append({
                "Dataset": nm,
                "Rows": f"{o['rows']:,}",
                "Cols": o["cols"],
                "Memory (MB)": f"{o['memory_mb']:.2f}",
                "Duplicates": f"{o['n_duplicates']:,}",
            })
        datasets_df = pd.DataFrame(rows) if rows else None

    # Core frames
    schema_tbl = pd.DataFrame({"column": df.columns, "dtype": [str(t) for t in df.dtypes]})
    preview_df = df.head(20)  # ALL columns; only rows limited to 20
    schema_local = infer_schema(df)
    nunique_tbl = (
        schema_local[["column", "unique"]]
        .rename(columns={"unique": "nunique"})
        .sort_values("nunique", ascending=False)
        .reset_index(drop=True)
    )
    col_stats_tbl = column_quick_stats(df, schema_local)
    show_cols = [c for c in ["column","type","min","p50","max","mean","std","unique","top","true","false"] if c in col_stats_tbl.columns]
    if show_cols:
        col_stats_tbl = col_stats_tbl[show_cols]
    tips = suggest_actions(df)

    # Sections
    kpi_html = f"""
      <div class="kpi-grid">
        <div class="kpi"><div class="kpi-label">Rows</div><div class="kpi-value">{ov['rows']:,}</div></div>
        <div class="kpi"><div class="kpi-label">Columns</div><div class="kpi-value">{ov['cols']}</div></div>
        <div class="kpi"><div class="kpi-label">Missing (%)</div><div class="kpi-value">{ov['missing_pct']:.2f}</div></div>
        <div class="kpi"><div class="kpi-label">Memory (MB)</div><div class="kpi-value">{ov['memory_mb']:.2f}</div></div>
        <div class="kpi"><div class="kpi-label">Duplicate rows</div><div class="kpi-value">{ov['n_duplicates']:,}</div></div>
        <div class="kpi"><div class="kpi-label">Profile</div><div class="kpi-value">{meta['profile']}</div></div>
        <div class="kpi"><div class="kpi-label">Time start</div><div class="kpi-value">{meta['time_min'] or '—'}</div></div>
        <div class="kpi"><div class="kpi-label">Time end</div><div class="kpi-value">{meta['time_max'] or '—'}</div></div>
      </div>
    """

    datasets_html = _html_table(datasets_df, caption="Datasets in session", max_height_px=260) if (datasets_df is not None and not datasets_df.empty) else ""
    schema_html   = _html_table(schema_tbl.rename(columns={"column":"Column","dtype":"Dtype"}), caption="Schema (all columns)", max_height_px=360)
    preview_html  = _html_table(preview_df, caption="Preview (first 20 rows — all columns)", max_height_px=280)
    card_html     = _html_table(nunique_tbl.rename(columns={"column":"Column","nunique":"Unique"}), caption="Cardinality (nunique per column)", max_height_px=360)
    stats_html    = _html_table(col_stats_tbl.rename(columns={
        "column":"Column","type":"Type","min":"Min","p50":"P50","max":"Max","mean":"Mean","std":"Std","unique":"Unique","top":"Top","true":"True","false":"False"
    }), caption="Schema & Column Summary", max_height_px=360)

    tips_html = (
        "<ul class='tips'>" + "".join(f"<li>{t}</li>" for t in tips) + "</ul>"
        if tips else "<div class='muted'>No immediate issues detected.</div>"
    )

    # Full HTML
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Summary — {active_name}</title>
<style>
  :root {{
    --bg:#ffffff; --ink:#0f172a; --muted:#64748b; --line:#e2e8f0; --soft:#f8fafc; --accent:#0ea5e9;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, "Helvetica Neue", Arial, "Noto Sans", "Apple Color Emoji", "Segoe UI Emoji";
    color: var(--ink); background: var(--bg); margin: 24px;
  }}
  h1 {{ font-size: 22px; margin: 0 0 6px 0; }}
  h2 {{ font-size: 18px; margin: 22px 0 8px 0; }}
  .muted {{ color: var(--muted); }}
  .cap {{ font-size: 12px; color: var(--muted); margin: 4px 0 6px 0; }}
  .box {{ margin: 6px 0 12px 0; }}

  /* Scroll container: both axes */
  .scroll {{
    border: 1px solid var(--line); border-radius: 10px;
    background: #fff; overflow: auto; box-shadow: inset 0 0 0 1px rgba(0,0,0,0.02);
  }}
  /* Make inner width expand to content so horizontal scroll can happen */
  .scroll-inner {{ width: max-content; min-width: 100%; }}

  /* Table styling */
  table.grid {{
    border-collapse: collapse;
    table-layout: auto;
    width: max-content;   /* key for horizontal scroll */
    min-width: 100%;      /* but don't be smaller than container */
  }}
  table.grid thead th {{
    position: sticky; top: 0; z-index: 2;
    background: var(--soft); border-bottom: 1px solid var(--line);
    text-align: left; padding: 8px; font-weight: 600; font-size: 12px;
    white-space: nowrap;
  }}
  table.grid td {{
    border-bottom: 1px solid #f1f5f9; padding: 6px 8px; font-size: 12px; vertical-align: top;
    white-space: nowrap;   /* keep cells on one line to enable L↔R scroll */
  }}

  .kpi-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 10px; margin: 8px 0 12px 0;
  }}
  .kpi {{ border: 1px solid var(--line); border-radius: 10px; padding: 10px; background: #fff; }}
  .kpi-label {{ font-size: 11px; color: var(--muted); margin-bottom: 4px; }}
  .kpi-value {{ font-size: 16px; font-weight: 600; }}
  ul.tips {{ margin: 6px 0 0 18px; padding: 0; }}
  ul.tips li {{ margin: 4px 0; }}
  .header {{ display:flex; align-items:baseline; gap:8px; }}
  .brand {{ color: var(--muted); font-size: 12px; }}
  .hr {{ height:1px; background: var(--line); margin: 14px 0; }}
</style>
</head>
<body>
  <div class="header">
    <h1>Summary — {active_name}</h1>
    <div class="brand">Analytics Accelerator</div>
  </div>
  <div class="hr"></div>

  <h2>Datasets (session)</h2>
  {datasets_html or "<div class='muted'>Only one dataset loaded in session.</div>"}

  <h2>Active Dataset — KPIs</h2>
  {kpi_html}

  <h2>Profile & Time Span</h2>
  <div class="box">
    <div class="muted">Profile</div>
    <div><strong>{meta['profile']}</strong></div>
    <div style="height:6px"></div>
    <div class="muted">Time span</div>
    <div>{meta['time_min'] or '—'} — {meta['time_max'] or '—'}</div>
  </div>

  <h2>Schema (column → dtype)</h2>
  {schema_html}

  <h2>Preview</h2>
  {preview_html}

  <h2>Cardinality</h2>
  {card_html}

  <h2>Schema & Column Summary</h2>
  {stats_html}

  <h2>Suggested Actions</h2>
  {tips_html}
</body>
</html>
"""
    return html


def summary_html_bytes(
    df: pd.DataFrame,
    dataset_name: str = "dataset",
    *,
    datasets: dict[str, pd.DataFrame] | None = None,
) -> bytes:
    """Return the same HTML as bytes for a download button."""
    return build_summary_html(dataset_name, df, datasets=datasets).encode("utf-8")


# =========================
# PDF Summary
# =========================

def build_summary_pdf(
    active_name: str,
    df: pd.DataFrame,
    datasets: dict[str, pd.DataFrame] | None = None,
) -> bytes:
    """PDF version – uses LongTable + horizontal chunking."""
    if not _REPORTLAB_OK:
        raise ImportError("ReportLab is required for PDF export. Install with: pip install reportlab")

    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import Paragraph, Spacer

    styles = getSampleStyleSheet()
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    h3 = styles["Heading3"]
    body = styles["BodyText"]
    small = ParagraphStyle("small", parent=body, fontSize=9, leading=11)

    ov = overview_stats(df)
    meta = dataset_meta(df)

    # Datasets overview
    datasets_df = None
    if datasets:
        rows = []
        for nm, dsub in sorted(datasets.items()):
            o = overview_stats(dsub)
            rows.append({
                "dataset": nm,
                "rows": f"{o['rows']:,}",
                "cols": o["cols"],
                "memory_mb": f"{o['memory_mb']:.2f}",
                "duplicates": f"{o['n_duplicates']:,}",
            })
        if rows:
            datasets_df = pd.DataFrame(rows)

    # Derived frames
    schema_tbl = pd.DataFrame({"column": df.columns, "dtype": [str(t) for t in df.dtypes]})
    preview_df = df.head(20)  # UI parity (all columns, 20 rows)
    schema_local = infer_schema(df)
    nunique_tbl = (
        schema_local[["column", "unique"]]
        .rename(columns={"unique": "nunique"})
        .sort_values("nunique", ascending=False)
        .reset_index(drop=True)
    )
    col_stats_tbl = column_quick_stats(df, schema_local)
    tips = suggest_actions(df)

    buff = BytesIO()
    doc = SimpleDocTemplate(
        buff, pagesize=A4,
        leftMargin=24, rightMargin=24, topMargin=28, bottomMargin=28
    )
    story = []

    story.append(Paragraph(f"Summary — {active_name}", h1))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Analytics Accelerator", small))
    story.append(Spacer(1, 12))

    if datasets_df is not None and not datasets_df.empty:
        story.append(Paragraph("Datasets (loaded in session)", h2))
        story += _longtables_from_df(datasets_df, "Loaded datasets", styles, max_cols=6)

    story.append(Paragraph("Active Dataset — KPIs", h2))
    kpi_line = (
        f"<b>Rows:</b> {ov['rows']:,} &nbsp;&nbsp; "
        f"<b>Columns:</b> {ov['cols']} &nbsp;&nbsp; "
        f"<b>Memory (MB):</b> {ov['memory_mb']:.2f} &nbsp;&nbsp; "
        f"<b>Duplicate rows:</b> {ov['n_duplicates']:,} &nbsp;&nbsp; "
        f"<b>Missing (%):</b> {ov['missing_pct']:.2f}"
    )
    story.append(Paragraph(kpi_line, small))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Profile & Time Span", h3))
    meta_text = (
        f"<b>Profile:</b> {meta['profile']} &nbsp;&nbsp; "
        f"<b>Numeric cols:</b> {meta['n_numeric']} &nbsp;&nbsp; "
        f"<b>Categorical cols:</b> {meta['n_categorical']}<br/>"
        f"<b>Time span:</b> {meta['time_min']} — {meta['time_max']}"
    )
    story.append(Paragraph(meta_text, small))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Schema (column → dtype)", h2))
    story += _longtables_from_df(schema_tbl, "Schema", styles, max_cols=8)

    if not preview_df.empty:
        story.append(Paragraph("Preview (first 20 rows)", h2))
        story += _longtables_from_df(preview_df, "Preview", styles, max_cols=8)

    story.append(Paragraph("Cardinality (nunique per column)", h2))
    story += _longtables_from_df(nunique_tbl, "Cardinality", styles, max_cols=8)

    if not col_stats_tbl.empty:
        show_cols = []
        for c in ["column", "type", "min", "p50", "max", "mean", "std", "unique", "top", "true", "false"]:
            if c in col_stats_tbl.columns:
                show_cols.append(c)
        stats_clean = col_stats_tbl[show_cols].copy()
        story.append(Paragraph("Schema & Column Summary", h2))
        story += _longtables_from_df(stats_clean, "Column summary", styles, max_cols=8)

    story.append(Paragraph("Suggested Actions", h2))
    if tips:
        for t in tips:
            story.append(Paragraph(f"• {t}", small))
    else:
        story.append(Paragraph("No immediate issues detected.", small))

    doc.build(story)
    return buff.getvalue()


def summary_pdf_bytes(
    df: pd.DataFrame,
    dataset_name: str = "dataset",
    datasets: dict[str, pd.DataFrame] | None = None
) -> bytes:
    """Backward-compatible wrapper around `build_summary_pdf(...)`."""
    return build_summary_pdf(active_name=dataset_name, df=df, datasets=datasets)


# =========================
# Inference utilities
# =========================

def _is_bool_series(s: pd.Series) -> bool:
    try:
        if pd.api.types.is_bool_dtype(s):
            return True
    except Exception:
        pass
    try:
        if getattr(s.dtype, "kind", None) in ("i", "u") and s.dropna().isin([0, 1]).all():
            return True
    except Exception:
        pass
    return False


def _looks_like_datetime(s: pd.Series) -> bool:
    try:
        if pd.api.types.is_datetime64_any_dtype(s):
            return True
    except Exception:
        pass
    try:
        if pd.api.types.is_string_dtype(s) or s.dtype == object or pd.api.types.is_categorical_dtype(s):
            sample = s.dropna().astype(str).head(200)
            if sample.empty:
                return False
            parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
            return parsed.notna().mean() > 0.8
    except Exception:
        return False
    return False


def infer_schema(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        non_null = int(s.notna().sum())
        miss_pct = float((1 - non_null / len(df)) * 100) if len(df) else 0.0
        nunique = _nunique_safe(s)

        if _is_bool_series(s):
            ltype = "boolean"
        elif pd.api.types.is_numeric_dtype(s):
            ltype = "numeric"
        elif _looks_like_datetime(s):
            ltype = "datetime"
        elif _has_unhashable(s):
            ltype = "nested"
        else:
            try:
                median_len = s.dropna().astype(str).str.len().median()
            except Exception:
                median_len = 0
            if median_len and median_len > 40 and nunique > 30:
                ltype = "text"
            else:
                ltype = "categorical"

        sample_val = s.dropna().iloc[0] if non_null else None
        rows.append({
            "column": col,
            "type": ltype,
            "non_null": non_null,
            "missing_%": round(miss_pct, 2),
            "unique": int(nunique),
            "example": sample_val,
        })
    return pd.DataFrame(rows)


def column_quick_stats(df: pd.DataFrame, schema: pd.DataFrame | None = None) -> pd.DataFrame:
    if schema is None:
        schema = infer_schema(df)
    out = []
    type_map = dict(zip(schema["column"], schema["type"]))
    for col in df.columns:
        s = df[col]
        t = type_map.get(col, "categorical")
        rec = {"column": col, "type": t}
        if t == "numeric":
            vals = s.dropna().astype(float)
            if vals.empty:
                rec.update({"min": None, "p50": None, "max": None, "mean": None, "std": None})
            else:
                rec.update({
                    "min": float(np.nanmin(vals)),
                    "p50": float(np.nanpercentile(vals, 50)),
                    "max": float(np.nanmax(vals)),
                    "mean": float(np.nanmean(vals)),
                    "std": float(np.nanstd(vals)),
                })
        elif t == "datetime":
            raw = s.astype("string") if pd.api.types.is_categorical_dtype(s) else s
            vals = pd.to_datetime(raw, errors="coerce")
            non_na = vals[vals.notna()]
            rec.update({
                "min": str(non_na.min()) if not non_na.empty else None,
                "max": str(non_na.max()) if not non_na.empty else None,
            })
        elif t == "boolean":
            vc = s.dropna().value_counts()
            rec.update({"true": int(vc.get(1, vc.get(True, 0))), "false": int(vc.get(0, vc.get(False, 0)))})
        else:
            rec.update({
                "unique": int(_nunique_safe(s)),
                "top": (s.dropna().mode().iloc[0] if not _has_unhashable(s) and s.dropna().size else None),
            })
        out.append(rec)
    return pd.DataFrame(out)


def numeric_correlations(df: pd.DataFrame, top_k: int = 20, min_abs: float = 0.4) -> pd.DataFrame:
    num_df = df.select_dtypes(include=[np.number]).copy()
    if num_df.shape[1] < 2:
        return pd.DataFrame(columns=["col_a", "col_b", "corr"])
    corr = num_df.corr(numeric_only=True)
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if pd.notna(val) and abs(val) >= min_abs:
                pairs.append({"col_a": cols[i], "col_b": cols[j], "corr": float(val)})
    pairs.sort(key=lambda x: -abs(x["corr"]))
    return pd.DataFrame(pairs[:top_k])


def suggest_actions(df: pd.DataFrame) -> List[str]:
    tips: List[str] = []
    n = len(df)
    schema = infer_schema(df)

    for _, r in schema.iterrows():
        if r["missing_%"] >= 40:
            tips.append(f"'{r['column']}' has {r['missing_%']}% missing — consider imputation or dropping.")
        if r["type"] in ("categorical", "text") and r["unique"] > max(1000, 0.5 * n):
            tips.append(f"'{r['column']}' is high-cardinality ({r['unique']:,}) — consider hashing/target encoding or exclude.")
        if r["unique"] == 1:
            tips.append(f"'{r['column']}' is constant — drop it.")
        if r["type"] == "categorical" and r["unique"] <= 2:
            tips.append(f"'{r['column']}' is near-binary — treat as boolean if appropriate.")

    dups = _n_duplicates_safe(df)
    if dups:
        tips.append(f"{dups:,} duplicate rows detected — consider deduping.")

    num_df = df.select_dtypes(include=[np.number])
    for col in num_df.columns:
        s = num_df[col].dropna().astype(float)
        if s.size < 30:
            continue
        q1, q3 = np.percentile(s, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        frac = float(((s < lower) | (s > upper)).mean())
        if frac >= 0.03:
            tips.append(f"'{col}' shows ~{frac*100:.1f}% outliers by IQR — consider winsorizing or robust scaling.")

    for _, r in schema.iterrows():
        if r["type"] == "categorical":
            s = df[r["column"]]
            sample = s.dropna().astype(str).head(200)
            parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
            if parsed.notna().mean() > 0.9:
                tips.append(f"'{r['column']}' looks like a date/time stored as text — parse to datetime.")
    return tips


def dataset_meta(df: pd.DataFrame) -> Dict[str, str | int]:
    schema = infer_schema(df)
    n_numeric = int((schema["type"] == "numeric").sum())
    n_categorical = int((schema["type"] == "categorical").sum())

    dt_cols = [c for c in df.columns if _looks_like_datetime(df[c])]
    time_min = None
    time_max = None
    for c in dt_cols:
        raw = df[c]
        try:
            if pd.api.types.is_categorical_dtype(raw):
                raw = raw.astype("string")
            vals = pd.to_datetime(raw, errors="coerce")
            vals = vals[vals.notna()]
            if not vals.empty:
                vmin, vmax = vals.min(), vals.max()
                if pd.notna(vmin):
                    time_min = vmin if time_min is None or vmin < time_min else time_min
                if pd.notna(vmax):
                    time_max = vmax if time_max is None or vmax > time_max else time_max
        except Exception:
            continue

    lower_cols = set(c.lower() for c in df.columns)
    if {"order_id", "customer_id", "sku"} <= lower_cols:
        profile = "retail"
    elif {"device_id", "battery_health"} & lower_cols:
        profile = "telemetry"
    elif {"cost", "azure", "gcp", "snowflake"} & lower_cols:
        profile = "cloud-cost"
    else:
        profile = "generic"

    fmt = lambda x: str(x) if x is not None else ""
    return {
        "n_numeric": n_numeric,
        "n_categorical": n_categorical,
        "time_min": fmt(time_min),
        "time_max": fmt(time_max),
        "profile": profile,
    }


# =========================
# Helpers for unhashable/nested
# =========================

def nunique_safe(s: pd.Series) -> int:
    return _nunique_safe(s)

def _has_unhashable(s: pd.Series) -> bool:
    try:
        sample = s.dropna().head(200)
        return sample.map(lambda v: isinstance(v, (list, dict, set, tuple))).any()
    except Exception:
        return False

def _stable_str(v):
    try:
        if isinstance(v, (dict, list, tuple, set)):
            return _json.dumps(v, sort_keys=True)
        return str(v)
    except Exception:
        return str(v)

def _nunique_safe(s: pd.Series) -> int:
    try:
        return int(s.nunique(dropna=True))
    except TypeError:
        return int(pd.Series(s.map(_stable_str)).nunique(dropna=True))

def _n_duplicates_safe(df: pd.DataFrame) -> int:
    try:
        return int(df.duplicated().sum())
    except TypeError:
        tmp = df.copy()
        for c in tmp.columns:
            if tmp[c].dtype == object and _has_unhashable(tmp[c]):
                tmp[c] = tmp[c].map(_stable_str)
        return int(tmp.duplicated().sum())


# =========================
# Demo data
# =========================

def demo_data(n_rows: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    categories = np.array(["Electronics","Home","Beauty","Grocery","Sports","Toys","Books","Fashion","Automotive","Office","Pets","Outdoors"])
    channels = np.array(["web","app","store","email","affiliate","social"])
    regions  = np.array(["North","South","East","West"])
    devices  = np.array(["desktop","mobile","tablet"])
    skus     = np.array([f"SKU-{i:04d}" for i in range(1, 301)])

    n_rows = int(n_rows)
    order_id    = np.arange(1, n_rows + 1)
    customer_id = rng.integers(1, n_rows // 2 + 2, size=n_rows)
    sku         = rng.choice(skus, size=n_rows)
    category    = rng.choice(categories, size=n_rows)
    channel     = rng.choice(channels, size=n_rows)
    region      = rng.choice(regions,  size=n_rows)
    device      = rng.choice(devices,  size=n_rows, p=[0.55,0.38,0.07])

    start = np.datetime64("2023-01-01")
    order_ts = start + pd.to_timedelta(rng.integers(0, 730*24*3600, size=n_rows), unit="s")
    event_time_str = pd.Series(pd.to_datetime(order_ts).strftime("%Y-%m-%d %H:%M:%S"))

    base_price = np.round(np.exp(rng.normal(3.5, 0.6, size=n_rows)), 2)
    quantity   = rng.integers(1, 6, size=n_rows)
    promo_used = (rng.random(n_rows) < 0.35).astype(int)
    discount_rate = np.where(promo_used==1, np.clip(rng.normal(0.22, 0.12, size=n_rows), 0, 0.6), np.nan)
    promo_price = np.where(promo_used==1, base_price*(1-np.nan_to_num(discount_rate, nan=0.0)), base_price)
    shipping_fee = np.round(np.maximum(0, rng.normal(6, 3, size=n_rows)), 2)

    amount = np.round(promo_price * quantity + shipping_fee, 2)
    tax_amount = np.round(amount * 0.07, 2)
    amount_plus_tax = np.round(amount + tax_amount, 2)

    user_note = pd.Series(["great fast" if r<0.2 else "ok" for r in rng.random(n_rows)])

    prob = 1/(1+np.exp(-(-1.5 + 0.6*(device=="mobile").astype(int) + 0.4*(promo_used==1).astype(int) - 0.002*base_price)))
    repeat_purchase_30d = (rng.random(n_rows) < prob).astype(int)
    spend_next_30d = np.round(np.clip(prob * rng.normal(180,40,size=n_rows) + (device=="mobile").astype(int)*15, 0, None),2)

    df = pd.DataFrame({
        "order_id": order_id,
        "customer_id": customer_id,
        "sku": sku,
        "category": category,
        "channel": channel,
        "region": region,
        "device": device,
        "utm_campaign": pd.Series(rng.integers(1,81,size=n_rows)).map(lambda x: f"cmp_{int(x):03d}"),
        "order_ts": pd.to_datetime(order_ts),
        "event_time_str": event_time_str,
        "quantity": quantity,
        "base_price": base_price,
        "promo_used": promo_used,
        "discount_rate": discount_rate,
        "promo_price": np.round(promo_price,2),
        "shipping_fee": shipping_fee,
        "amount": amount,
        "tax_amount": tax_amount,
        "amount_plus_tax": amount_plus_tax,
        "user_note": user_note,
        "repeat_purchase_30d": repeat_purchase_30d,
        "spend_next_30d": spend_next_30d,
    })
    if n_rows >= 500:
        dup_idx = rng.choice(n_rows, size=int(n_rows*0.01), replace=False)
        df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)
    return df
