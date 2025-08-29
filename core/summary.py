# core/summary.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List
import json as _json
from io import BytesIO
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.units import cm
    _REPORTLAB_OK = True
except Exception:
    _REPORTLAB_OK = False




__all__ = [
    "overview_stats",
    "infer_schema",
    "column_quick_stats",
    "numeric_correlations",
    "suggest_actions",
    "dataset_meta",
    "demo_data",
    "nunique_safe",
    "build_summary_pdf",        # <-- NEW: export Summary as PDF
]

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

def build_summary_pdf(
    active_name: str,
    df: pd.DataFrame,
    datasets: dict[str, pd.DataFrame] | None = None,
) -> bytes:
    """
    Build a PDF that mirrors the Summary page:
    - Datasets overview (if provided)
    - KPIs for active dataset
    - Profile/meta
    - Schema (column + dtype)
    - Data preview (first 20 rows, first 10 cols)
    - Cardinality (nunique per column, sorted)
    - Column quick stats
    - Suggested actions
    """
    # Lazy import so users without reportlab don't break normal app usage
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
        )
    except Exception as e:
        raise ImportError(
            "reportlab is required for PDF export. Install with: pip install reportlab"
        ) from e

    from io import BytesIO

    # ---- helpers ---------------------------------------------------------
    styles = getSampleStyleSheet()
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    h3 = styles["Heading3"]
    body = styles["BodyText"]

    # tighter paragraph
    small = ParagraphStyle("small", parent=body, fontSize=9, leading=11)
    tiny  = ParagraphStyle("tiny", parent=body, fontSize=8, leading=10)

    def _df_to_table(df_in: pd.DataFrame, max_rows: int | None = None, max_cols: int | None = None):
        """Convert a DataFrame to a ReportLab Table with basic styling."""
        df_work = df_in.copy()
        note = None
        if max_cols is not None and df_work.shape[1] > max_cols:
            df_work = df_work.iloc[:, :max_cols]
            note = f"(showing first {max_cols} columns)"
        if max_rows is not None and df_work.shape[0] > max_rows:
            df_work = df_work.iloc[:max_rows, :]
            note = (note + " and " if note else "") + f"(first {max_rows} rows)"

        data = [df_work.columns.tolist()] + [[
            "" if pd.isna(v) else str(v) for v in row
        ] for _, row in df_work.iterrows()]

        tbl = Table(data, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f2f6")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c8d1e0")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        return tbl, note

    def _keyvals_table(kv: dict[str, str]):
        data = [["Metric", "Value"]] + [[k, str(v)] for k, v in kv.items()]
        tbl = Table(data, repeatRows=1, colWidths=[140, 360])
        tbl.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f2f6")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c8d1e0")),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ]))
        return tbl

    # ---- content we already compute on the Summary page -------------------
    ov = overview_stats(df)
    meta = dataset_meta(df)

    # Datasets overview (if provided)
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

    # Schema
    schema_tbl = pd.DataFrame({"column": df.columns, "dtype": [str(t) for t in df.dtypes]})

    # Preview
    preview_df = df.head(20)

    # Cardinality (show all columns with nunique)
    schema_local = infer_schema(df)
    nunique_tbl = (
        schema_local[["column", "unique"]]
        .rename(columns={"unique": "nunique"})
        .sort_values("nunique", ascending=False)
        .reset_index(drop=True)
    )

    # Column quick stats
    col_stats_tbl = column_quick_stats(df, schema_local)

    # Suggested actions
    tips = suggest_actions(df)

    # ---- Build PDF --------------------------------------------------------
    buff = BytesIO()
    doc = SimpleDocTemplate(
        buff, pagesize=A4, leftMargin=24, rightMargin=24, topMargin=28, bottomMargin=28
    )
    story = []

    # Title
    story.append(Paragraph(f"Summary — {active_name}", h1))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Analytics Accelerator", small))
    story.append(Spacer(1, 12))

    # Datasets overview
    if datasets_df is not None:
        story.append(Paragraph("Datasets (loaded in session)", h2))
        tbl, note = _df_to_table(datasets_df, max_rows=40)
        story.append(tbl)
        if note:
            story.append(Paragraph(note, tiny))
        story.append(Spacer(1, 10))

    # KPIs for active df
    story.append(Paragraph("Active Dataset — KPIs", h2))
    kpi_data = {
        "Rows": f"{ov['rows']:,}",
        "Columns": ov["cols"],
        "Memory (MB)": f"{ov['memory_mb']:.2f}",
        "Duplicate rows": f"{ov['n_duplicates']:,}",
        "Missing (%)": f"{ov['missing_pct']:.2f}",
    }
    story.append(_keyvals_table(kpi_data))
    story.append(Spacer(1, 10))

    # Meta / profile
    story.append(Paragraph("Profile & Time Span", h3))
    meta_text = (
        f"<b>Profile:</b> {meta['profile']} &nbsp;&nbsp; "
        f"<b>Numeric cols:</b> {meta['n_numeric']} &nbsp;&nbsp; "
        f"<b>Categorical cols:</b> {meta['n_categorical']}<br/>"
        f"<b>Time span:</b> {meta['time_min']} — {meta['time_max']}"
    )
    story.append(Paragraph(meta_text, small))
    story.append(Spacer(1, 10))

    # Schema
    story.append(Paragraph("Schema (column → dtype)", h2))
    sch_tbl, note = _df_to_table(schema_tbl, max_rows=120)
    story.append(sch_tbl)
    if note:
        story.append(Paragraph(note, tiny))
    story.append(Spacer(1, 10))

    # Data preview
    story.append(Paragraph("Preview (first 20 rows)", h2))
    prev_tbl, note = _df_to_table(preview_df, max_rows=20, max_cols=10)
    story.append(prev_tbl)
    story.append(Paragraph("(first 10 columns shown for width)", tiny))
    story.append(Spacer(1, 10))

    # Cardinality
    story.append(Paragraph("Cardinality (nunique per column)", h2))
    card_tbl, note = _df_to_table(nunique_tbl, max_rows=120)
    story.append(card_tbl)
    if note:
        story.append(Paragraph(note, tiny))
    story.append(Spacer(1, 10))

    # Column quick stats
    story.append(Paragraph("Schema & Column Summary", h2))
    # reorder/clean for readability
    show_cols = []
    for c in ["column", "type", "min", "p50", "max", "mean", "std", "unique", "top", "true", "false"]:
        if c in col_stats_tbl.columns:
            show_cols.append(c)
    stats_clean = col_stats_tbl[show_cols].copy()
    stats_tbl, note = _df_to_table(stats_clean, max_rows=120)
    story.append(stats_tbl)
    if note:
        story.append(Paragraph(note, tiny))
    story.append(Spacer(1, 10))

    # Suggested actions
    story.append(Paragraph("Suggested Actions", h2))
    if tips:
        for t in tips:
            story.append(Paragraph(f"• {t}", small))
    else:
        story.append(Paragraph("No immediate issues detected.", small))

    # Build PDF
    doc.build(story)
    return buff.getvalue()


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
        return pd.DataFrame(columns=["col_a","col_b","corr"])
    corr = num_df.corr(numeric_only=True)
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
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
        if r["type"] in ("categorical","text") and r["unique"] > max(1000, 0.5 * n):
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

    lower_cols = set([c.lower() for c in df.columns])
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

# ---- Helpers for unhashable/nested types ----

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

# ---- Demo dataset (compact) ----

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
    shipping_fee = np.round(np.maximum(0, rng.normal(6, 3, size=n_rows)), 2)  # NOTE: if you had a typo, fix to size=n_rows

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

# ---- NEW: Summary PDF export ----

def summary_pdf_bytes(df: pd.DataFrame, dataset_name: str = "dataset") -> bytes:
    """
    Build a compact PDF of the Summary tab (KPIs, time span, schema snapshot, top correlations, suggestions).
    Returns raw PDF bytes (ready for st.download_button).
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.units import cm
    except Exception as e:
        raise ImportError(
            "reportlab is required for PDF export. Install with `pip install reportlab`."
        ) from e

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=1.6*cm, rightMargin=1.6*cm,
        topMargin=1.2*cm, bottomMargin=1.2*cm,
        title=f"{dataset_name} — Summary",
    )

    styles = getSampleStyleSheet()
    H1 = styles["Heading1"]; H1.fontSize = 16
    H2 = styles["Heading2"]; H2.fontSize = 13
    P  = styles["BodyText"]; P.leading = 14

    flow = []
    # Title
    flow.append(Paragraph(f"Summary — {dataset_name}", H1))
    flow.append(Spacer(1, 6))

    # KPIs
    ov = overview_stats(df)
    meta = dataset_meta(df)
    kpi_data = [
        ["Rows", f"{ov['rows']:,}", "Cols", f"{ov['cols']}"],
        ["Missing (%)", f"{ov['missing_pct']:.2f}", "Memory (MB)", f"{ov['memory_mb']:.2f}"],
        ["Duplicate rows", f"{ov['n_duplicates']:,}", "Profile", meta["profile"]],
        ["Time start", meta["time_min"] or "—", "Time end", meta["time_max"] or "—"],
    ]
    kpi_tbl = Table(kpi_data, hAlign="LEFT", colWidths=[3*cm, 4.5*cm, 3*cm, 6*cm])
    kpi_tbl.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 9.5),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("BOX", (0,0), (-1,-1), 0.25, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("ALIGN", (0,0), (-1,-1), "LEFT"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    flow.append(kpi_tbl)
    flow.append(Spacer(1, 8))

    # Schema snapshot
    flow.append(Paragraph("Schema (first 25)", H2))
    schema = infer_schema(df)
    snap = schema[["column","type","missing_%","unique"]].head(25)
    schema_data = [["Column","Type","Missing %","Unique"]] + snap.values.tolist()
    schema_tbl = Table(schema_data, hAlign="LEFT", colWidths=[6*cm, 3*cm, 2.5*cm, 3*cm])
    schema_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8.8),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
        ("BOX", (0,0), (-1,-1), 0.25, colors.lightgrey),
        ("ALIGN", (2,1), (3,-1), "RIGHT"),
    ]))
    flow.append(schema_tbl)
    flow.append(Spacer(1, 8))

    # Top correlations (if any)
    corr = numeric_correlations(df, top_k=12, min_abs=0.4)
    if not corr.empty:
        flow.append(Paragraph("Top numeric correlations (|r| ≥ 0.4)", H2))
        corr_data = [["A","B","Corr (r)"]] + [[a,b,f"{r:.2f}"] for a,b,r in corr[["col_a","col_b","corr"]].values]
        corr_tbl = Table(corr_data, hAlign="LEFT", colWidths=[5*cm,5*cm,3*cm])
        corr_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 8.8),
            ("INNERGRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
            ("BOX", (0,0), (-1,-1), 0.25, colors.lightgrey),
            ("ALIGN", (2,1), (2,-1), "RIGHT"),
        ]))
        flow.append(corr_tbl)
        flow.append(Spacer(1, 8))

    # Suggested actions
    tips = suggest_actions(df)
    flow.append(Paragraph("Suggested actions", H2))
    if tips:
        for t in tips[:12]:
            flow.append(Paragraph(f"• {t}", P))
    else:
        flow.append(Paragraph("No immediate issues detected.", P))

    doc.build(flow)
    return buf.getvalue()
