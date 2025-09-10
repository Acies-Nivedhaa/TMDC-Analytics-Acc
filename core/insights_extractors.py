# core/insights_extractors.py
# ---------------------------------------------------------------------
# Collects, stores (in-memory + optional disk), renders, and exports
# dataset "insights" discovered across the app (summary/EDA/preprocess).
#
# Notes:
# - Maintains both a legacy flat store (ss["insights"]) and a newer,
#   structured store (ss["_insights"] with items and metrics).
# - Disk persistence is optional and best-effort (silent on failure).
# - Heuristic helpers focus on speed and robustness for mixed datasets.
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import os
import json
import math
import time

import numpy as np
import pandas as pd
import streamlit as st

# ---- Simple on-disk persistence for insights -------------------------
_DEFAULT_STORE = os.path.join(os.getcwd(), ".acc_insights_store.json")


def save_insights_to_disk(ss, path: str | None = None) -> None:
    """
    Persist the structured insights store to disk (best-effort).
    """
    try:
        path = path or ss.get("insights_store_path") or _DEFAULT_STORE
        data = ss.get("_insights", {})
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        ss["insights_store_path"] = path
        ss["_ins_last_saved"] = time.time()
    except Exception:
        # Silent by design—persistence is optional
        pass


def load_insights_from_disk(ss, path: str | None = None) -> bool:
    """
    Load structured insights from disk and (shallow) merge into current session.
    Current session values win on conflict.
    """
    try:
        path = path or ss.get("insights_store_path") or _DEFAULT_STORE
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return False

        cur = ss.get("_insights") or {}
        # Shallow merge so current-session values are preserved
        for k, v in data.items():
            if k not in cur:
                cur[k] = v
            else:
                if isinstance(v, dict) and isinstance(cur[k], dict):
                    for kk, vv in v.items():
                        cur[k].setdefault(kk, vv)
                else:
                    cur[k] = v

        ss["_insights"] = cur
        ss["insights_store_path"] = path
        ss["_ins_last_loaded"] = time.time()
        return True
    except Exception:
        return False


# ======================================================
# Legacy storage (kept for backward compatibility)
# ======================================================
def init_insights_store(ss) -> None:
    """
    Ensure legacy flat store exists.
    Shape: ss["insights"] = { "<dataset>": {"items": [ ... ] } }
    """
    ss.setdefault("insights", {})


def _clear_area(
    ss,
    ds_name: str,
    *,
    section: str,
    subsection: str,
    legacy_area: str | None = None,
) -> None:
    """
    Remove prior insights for an area so we don't duplicate when re-saving.
    Applies to structured store and (optionally) the legacy store.
    """
    # Structured store
    s = ss.setdefault("_insights", {}).setdefault(ds_name, {"items": []})
    s["items"] = [
        it
        for it in s["items"]
        if not (it.get("section") == section and it.get("subsection") == subsection)
    ]
    # Legacy store (optional)
    if legacy_area is not None:
        l = ss.setdefault("insights", {}).setdefault(ds_name, {"items": []})
        l["items"] = [it for it in l["items"] if it.get("area") != legacy_area]


def _ds_slot(ss, ds_name: str) -> Dict[str, Any]:
    """Return the legacy slot for a dataset, creating if needed."""
    if not ds_name:
        ds_name = "__default__"
    if ds_name not in ss["insights"]:
        ss["insights"][ds_name] = {"items": []}
    return ss["insights"][ds_name]


def clear_insights(ss, ds_name: str) -> None:
    """Clear legacy insights for a dataset."""
    _ds_slot(ss, ds_name)["items"].clear()


def add_insight(
    ss,
    ds_name: str,
    area: str,
    title: str,
    detail: str,
    score: Optional[float] = None,
    tags: Optional[List[str]] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Append a legacy insight to the flat store.
    """
    slot = _ds_slot(ss, ds_name)
    slot["items"].append(
        {
            "ts": int(time.time()),
            "area": area,
            "title": title,
            "detail": detail,
            "score": None if score is None else float(score),
            "tags": tags or [],
            "payload": payload or {},
        }
    )


def get_insights(ss, ds_name: str) -> List[Dict[str, Any]]:
    """Return a copy of legacy insights for a dataset."""
    return list(_ds_slot(ss, ds_name)["items"])


# ======================================================
# Structured storage (new, richer schema)
# ======================================================
def _append_item(ss, ds_name: str, item: dict) -> None:
    """
    Append a structured insight item and persist (best-effort).
    """
    store = ss.setdefault("_insights", {})
    ds_obj = store.setdefault(ds_name, {"items": []})
    if isinstance(ds_obj, list):  # migrate very old shape
        store[ds_name] = {"items": list(ds_obj)}
        ds_obj = store[ds_name]
    ds_obj["items"].append(item)
    # Persist on each change (best-effort)
    try:
        save_insights_to_disk(ss)
    except Exception:
        pass


def _add(
    ss,
    ds_name: str,
    *,
    section: str,
    subsection: str,
    kind: str,
    title: str = "",
    text: str = "",
    metrics: dict | None = None,
    **extra,
):
    """
    Convenience wrapper to construct and append a structured insight.
    """
    item = {
        "section": section,
        "subsection": subsection,
        "kind": kind,
        "title": title,
        "text": text,
        "metrics": (metrics or {}),
        "ts": int(time.time()),
    }
    item.update(extra or {})
    _append_item(ss, ds_name, item)


def insights_to_json_bytes(ss, ds_name: str) -> bytes:
    """
    Serialize (structured + legacy) insights to JSON bytes for download.
    """
    struct = ss.get("_insights", {}).get(ds_name, {"items": []})
    legacy = ss.get("insights", {}).get(ds_name, {"items": []})
    obj = {"dataset": ds_name, "structured": struct, "legacy": legacy}
    return json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8")


def saved_badge() -> None:
    """Small visual hint in the UI that something was captured."""
    st.caption("✅ saved to **Final Summary**")


# ======================================================
# Heuristics & helpers
# ======================================================
_NUMERIC_HINTS = (
    "amount",
    "amt",
    "price",
    "revenue",
    "sales",
    "spend",
    "cost",
    "value",
    "qty",
    "quantity",
    "score",
)
_MEASURE_HINTS = ["amount", "revenue", "sales", "spend", "price", "value", "qty", "quantity"]
_TIME_HINTS = ["date", "time", "ts", "timestamp"]


def _likely_value_columns(df: pd.DataFrame) -> List[str]:
    """
    Return numeric columns ordered with "value-like" names first
    (amount/price/qty etc.) to prioritize business measures.
    """
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    hinted = [c for c in num_cols if any(h in c.lower() for h in _NUMERIC_HINTS)]
    rest = [c for c in num_cols if c not in hinted]
    return hinted + rest


def _guess_measure_col(df: pd.DataFrame) -> Optional[str]:
    """
    Guess a natural measure column: first by common name hints,
    otherwise by highest variance numeric column.
    """
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return None
    for h in _MEASURE_HINTS:
        for c in numeric_cols:
            if h in c.lower():
                return c
    # fallback: highest variance
    return pd.Series({c: df[c].var(skipna=True) for c in numeric_cols}).sort_values(ascending=False).index[0]


def _guess_datetime_col(df: pd.DataFrame) -> Optional[str]:
    """
    Guess a datetime column via dtype, then by name, then by coercion coverage.
    """
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    for c in df.columns:
        lc = c.lower()
        if any(h in lc for h in _TIME_HINTS):
            s = pd.to_datetime(df[c], errors="coerce")
            if s.notna().any():
                return c
    for c in df.columns:
        s = pd.to_datetime(df[c], errors="coerce")
        if s.notna().mean() > 0.7:
            return c
    return None


def _iqr_outlier_share(s: pd.Series) -> float:
    """
    Share of points outside Tukey IQR fences (robust outlier proxy).
    """
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return 0.0
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return float(((s < lo) | (s > hi)).mean())


def _correlation_ratio(cat: pd.Series, num: pd.Series) -> float:
    """
    Correlation ratio η (categorical -> numeric).
    """
    c = cat.astype("category")
    z = pd.to_numeric(num, errors="coerce")
    m = ~(c.isna() | z.isna())
    c, z = c[m], z[m]
    if c.nunique() < 2 or len(z) < 3:
        return float("nan")
    groups = [z[c.cat.codes == i] for i in range(len(c.cat.categories))]
    counts = np.array([g.size for g in groups], dtype=float)
    means = np.array([g.mean() if g.size else np.nan for g in groups], dtype=float)
    means = np.nan_to_num(means, nan=np.nanmean(means))
    mu = float(z.mean())
    ss_between = float(np.sum(counts * (means - mu) ** 2))
    ss_total = float(np.sum((z - mu) ** 2))
    return float(np.sqrt(ss_between / ss_total)) if ss_total > 0 else 0.0


def _pareto_share(series_by_group: pd.Series) -> Tuple[float, int, List[Tuple[str, float]]]:
    """
    Compute Pareto-type stats:
      - share contributed by the top 20% of groups
      - number of groups needed to reach 80% of total
      - top few (group, share) pairs for display
    """
    s = series_by_group.dropna().sort_values(ascending=False)
    total = float(s.sum()) if s.size else 0.0
    if total <= 0.0 or s.empty:
        return (0.0, 0, [])
    shares = (s / total).to_list()
    groups = s.index.astype(str).to_list()
    cum = np.cumsum(shares)
    n80 = int(np.searchsorted(cum, 0.8, side="left") + 1)
    k = max(1, math.ceil(0.2 * len(shares)))  # top 20% of groups
    share20 = float(np.sum(shares[:k]))
    top_items = list(zip(groups[: min(5, len(groups))], shares[: min(5, len(shares))]))
    return (share20, n80, top_items)


# ======================================================
# Capture functions (legacy-style + structured)
# ======================================================
def capture_summary_insights(ss, ds_name: str, df: pd.DataFrame, n_tables: int) -> None:
    """
    Quick size/health notes for current dataset (both legacy + structured).
    """
    _clear_area(ss, ds_name, section="summary", subsection="overview", legacy_area="summary.overview")
    _clear_area(ss, ds_name, section="summary", subsection="health",   legacy_area="summary.health")

    # legacy (append a single fresh record)
    add_insight(
        ss, ds_name, "summary.overview", "Dataset size",
        f"{len(df):,} rows × {df.shape[1]} columns across {n_tables} table(s).",
        tags=["kpi"],
    )
    dup = int(df.duplicated().sum())
    miss = float(df.isna().mean().mean() * 100)
    add_insight(
        ss, ds_name, "summary.health", "Data health",
        f"Duplicate rows: {dup:,}. Average missing across columns ≈ {miss:.1f}%.",
        tags=["quality"],
    )

    # structured (single fresh record)
    _add(
        ss, ds_name,
        section="summary", subsection="overview", kind="kpi",
        title="Dataset size",
        text=f"{len(df):,} rows × {df.shape[1]} columns across {n_tables} table(s).",
        metrics={"rows": len(df), "cols": df.shape[1], "n_tables": n_tables},
    )
    _add(
        ss, ds_name,
        section="summary", subsection="health", kind="quality",
        title="Data health",
        text=f"Duplicate rows: {dup:,}. Average missing across columns ≈ {miss:.1f}%.",
        metrics={"duplicates": dup, "avg_missing_pct": miss / 100.0},
    )

def capture_univariate_insight(ss, ds_name: str, col: str, s_full: pd.Series) -> None:
    """
    Save a univariate snapshot for a single column (type-aware).
    """
    # numeric
    if pd.api.types.is_numeric_dtype(s_full):
        s = pd.to_numeric(s_full, errors="coerce")
        miss = float(s.isna().mean() * 100)
        zero_share = float((s == 0).mean() * 100)
        out_share = 0.0
        try:
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                out_share = float(((s < lo) | (s > hi)).mean() * 100)
        except Exception:
            pass

        detail = f"Missing {miss:.1f}%. Zeros {zero_share:.1f}%. Potential outliers ~{out_share:.1f}%."
        # legacy
        add_insight(
            ss,
            ds_name,
            "eda.univariate",
            f"Univariate • {col}",
            detail,
            score=min(1.0, out_share / 50.0),
            tags=["numeric", "distribution"],
            payload={
                "missing_pct": miss,
                "zero_pct": zero_share,
                "outlier_pct": out_share,
            },
        )
        # structured
        _add(
            ss,
            ds_name,
            section="eda",
            subsection="univariate",
            kind="numeric",
            title=col,
            text=detail,
            metrics={
                "col": col,
                "missing_pct": miss / 100.0,
                "zeros_pct": zero_share / 100.0,
                "outlier_pct": out_share / 100.0,
            },
        )
        return

    # datetime
    if pd.api.types.is_datetime64_any_dtype(s_full):
        s = pd.to_datetime(s_full, errors="coerce").dropna()
        if s.empty:
            return
        span = (s.max() - s.min()).days
        by_dow = s.dt.day_name().value_counts(normalize=True)
        topdow = f"{by_dow.index[0]} ({by_dow.iloc[0]*100:.1f}%)" if not by_dow.empty else "—"
        detail = f"Time span ≈ {span} days. Busiest day: {topdow}."
        add_insight(
            ss,
            ds_name,
            "eda.univariate",
            f"Univariate • {col}",
            detail,
            tags=["datetime"],
            payload={"span_days": span, "busiest_day": topdow},
        )
        _add(
            ss,
            ds_name,
            section="eda",
            subsection="univariate",
            kind="datetime",
            title=col,
            text=detail,
            metrics={
                "col": col,
                "span_days": span,
                "busiest_day": by_dow.index[0] if not by_dow.empty else None,
                "busiest_share": float(by_dow.iloc[0]) if not by_dow.empty else None,
            },
        )
        return

    # categorical
    s = s_full.astype("string")
    vc = s.fillna("__NA__").value_counts(normalize=True)
    if vc.empty:
        return
    top_cat, top_share = str(vc.index[0]), float(vc.iloc[0] * 100)
    detail = f"Top category **{top_cat}** covers **{top_share:.1f}%**."
    add_insight(
        ss,
        ds_name,
        "eda.univariate",
        f"Univariate • {col}",
        detail,
        score=min(1.0, top_share / 100.0),
        tags=["categorical", "dominance"],
        payload={"top_category": top_cat, "top_share_pct": top_share},
    )
    _add(
        ss,
        ds_name,
        section="eda",
        subsection="univariate",
        kind="categorical",
        title=col,
        text=detail,
        metrics={"col": col, "top_category": top_cat, "top_share_pct": top_share},
    )


# ===== Bulk univariate snapshot (ALL columns for the active dataset) =====
def save_univariate_all_to_store(ss, ds_name: str, df: pd.DataFrame) -> int:
    """
    Compute and save univariate stats for every column into:
    ss['insights_store'][ds_name]['univariate'].
    Returns the number of columns processed.
    """
    store = ss.setdefault("insights_store", {})
    ds_store = store.setdefault(ds_name, {})
    uni: dict[str, dict] = {}

    for col in df.columns:
        s = df[col]
        # Numeric
        if pd.api.types.is_numeric_dtype(s):
            sn = pd.to_numeric(s, errors="coerce")
            miss = float(sn.isna().mean() * 100)
            zeros = float((sn == 0).mean() * 100)
            # IQR outlier share
            out = 0.0
            try:
                q1, q3 = sn.quantile([0.25, 0.75])
                iqr = q3 - q1
                if iqr > 0:
                    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    out = float(((sn < lo) | (sn > hi)).mean() * 100)
            except Exception:
                pass
            uni[col] = {
                "column": col,
                "type": "numeric",
                "missing_pct": round(miss, 1),
                "zeros_pct": round(zeros, 1),
                "outliers_pct": round(out, 1),
                "top_value": None,
                "top_share_pct": None,
            }

        # Datetime
        elif pd.api.types.is_datetime64_any_dtype(s):
            sdt = pd.to_datetime(s, errors="coerce")
            miss = float(sdt.isna().mean() * 100)
            span = None
            if sdt.notna().any():
                span = int((sdt.max() - sdt.min()).days)
            uni[col] = {
                "column": col,
                "type": "datetime",
                "missing_pct": round(miss, 1),
                "zeros_pct": None,
                "outliers_pct": None,
                "top_value": None,
                "top_share_pct": None,
                "span_days": span,
            }

        # Categorical / other
        else:
            sc = s.astype("string")
            miss = float(sc.isna().mean() * 100)
            vc = sc.fillna("<NA>").value_counts(dropna=False)
            if vc.empty:
                top_val, top_share = None, None
            else:
                top_val = str(vc.index[0])
                top_share = float(vc.iloc[0] / max(1, len(sc)) * 100)
            uni[col] = {
                "column": col,
                "type": "categorical",
                "missing_pct": round(miss, 1),
                "zeros_pct": None,
                "outliers_pct": None,
                "top_value": top_val,
                "top_share_pct": None if top_share is None else round(top_share, 1),
            }

    ds_store["univariate"] = uni
    return len(uni)


# ---------- Save-all helpers for Bivariate / Correlation ----------
def save_bivariate_all(
    ss,
    ds_name: str,
    df: pd.DataFrame,
    *,
    min_abs_corr: float = 0.30,  # numeric-numeric threshold
    max_pairs_num_num: int = 50,  # cap saved num-num pairs
    max_pairs_cat_num: int = 50,  # cap saved cat-num pairs
    max_cat_cardinality: int = 50,  # ignore super-high-cardinality cats
) -> int:
    """
    Capture bivariate insights for the whole table:
      - numeric↔numeric (Pearson) above |r| threshold
      - categorical→numeric Pareto (top 20% share)
    Returns count of insights added.
    """
    init_insights_store(ss)
    before = len(get_insights(ss, ds_name))

    dfx = df.copy()

    # ----- numeric ↔ numeric: strongest pairs by |r|
    nums = [c for c in dfx.columns if pd.api.types.is_numeric_dtype(dfx[c])]
    if len(nums) >= 2:
        C = dfx[nums].corr()
        pairs = []
        for i, a in enumerate(nums):
            for j in range(i + 1, len(nums)):
                b = nums[j]
                r = C.loc[a, b]
                if pd.notna(r) and abs(float(r)) >= min_abs_corr:
                    pairs.append((a, b, abs(float(r))))
        pairs.sort(key=lambda t: t[2], reverse=True)
        for a, b, _ in pairs[:max_pairs_num_num]:
            capture_bivariate_insight(ss, ds_name, a, b, dfx)

    # ----- categorical → numeric: Pareto
    cats = [c for c in dfx.columns if not pd.api.types.is_numeric_dtype(dfx[c])]
    cats = [c for c in cats if 2 <= dfx[c].nunique(dropna=False) <= max_cat_cardinality]
    values = _likely_value_columns(dfx)  # prioritizes revenue/amount/qty names

    taken = 0
    for cat in cats:
        for val in values:
            capture_bivariate_insight(ss, ds_name, cat, val, dfx)
            taken += 1
            if taken >= max_pairs_cat_num:
                break
        if taken >= max_pairs_cat_num:
            break

    after = len(get_insights(ss, ds_name))
    return after - before


def save_correlation_overview(
    ss,
    ds_name: str,
    df: pd.DataFrame,
    *,
    min_abs: float = 0.60,
) -> int:
    """
    Save a compact 'Top strong numeric relationships' entry (like the heatmap).
    Returns count of insights added.
    """
    init_insights_store(ss)
    before = len(get_insights(ss, ds_name))
    capture_correlation_matrix_insights(ss, ds_name, df, min_abs=min_abs)
    after = len(get_insights(ss, ds_name))
    return after - before


def _capture_cat_num(ss, ds_name: str, cat: str, val: str, df: pd.DataFrame) -> None:
    """
    Helper for cat→num Pareto (top 20% share & 80% coverage).
    """
    g = df.groupby(cat, dropna=False)[val].sum(numeric_only=True).sort_values(ascending=False)
    if g.empty or float(g.sum()) == 0.0:
        return
    share20, n80, top_items = _pareto_share(g)
    detail = (
        f"Top 20% of **{cat}** groups contribute **{share20*100:.1f}%** of total **{val}**. "
        f"~**{n80}** groups cover 80%."
    )
    add_insight(
        ss,
        ds_name,
        "eda.bivariate",
        f"Pareto (80/20) • {cat} → {val}",
        detail,
        score=share20,
        tags=["category-driver", "pareto"],
        payload={"top_20pct_share": share20, "n_groups_for_80pct": n80, "top": top_items},
    )
    _add(
        ss,
        ds_name,
        section="business",
        subsection="top_performers",
        kind="pareto",
        title=f"Pareto • {cat} → {val}",
        text=f"Top 20% {cat} contribute {share20*100:.1f}% of {val}. ~{n80} groups cover 80%.",
        metrics={"by": cat, "measure": val, "top_20pct_share": share20, "n80": n80, "top": top_items},
    )


def capture_bivariate_insight(ss, ds_name: str, x: str, y: str, df: pd.DataFrame) -> None:
    """
    Save a single bivariate relationship:
      - numeric-numeric: Pearson r
      - cat-numeric: Pareto statement (top share & coverage)
      - cat-cat: brief dominance (optional)
    """
    sx, sy = df[x], df[y]

    # numeric-numeric -> Pearson
    if pd.api.types.is_numeric_dtype(sx) and pd.api.types.is_numeric_dtype(sy):
        cx = pd.to_numeric(sx, errors="coerce")
        cy = pd.to_numeric(sy, errors="coerce")
        if cx.notna().any() and cy.notna().any():
            corr = float(pd.concat([cx, cy], axis=1).corr().iloc[0, 1])
            strength = "strong" if abs(corr) >= 0.6 else "weak/moderate"
            detail = f"Pearson r = {corr:.2f} ({strength})."
            add_insight(
                ss,
                ds_name,
                "eda.bivariate",
                f"Correlation • {x} ↔ {y}",
                detail,
                score=min(1.0, abs(corr)),
                tags=["numeric-numeric", "correlation"],
                payload={"pearson_r": corr},
            )
            _add(
                ss,
                ds_name,
                section="eda",
                subsection="bivariate",
                kind="correlation",
                title=f"{x} ↔ {y}",
                text=detail,
                metrics={"a": x, "b": y, "r": corr, "method": "pearson"},
            )
        return

    # cat-num -> Pareto statement
    # NOTE: there's also a top-level `_capture_cat_num` helper with the same logic;
    # this inner function keeps existing behavior exactly as-is.
    def _capture_cat_num(cat: str, val: str):
        g = df.groupby(cat, dropna=False)[val].sum(numeric_only=True).sort_values(ascending=False)
        if g.empty or float(g.sum()) == 0.0:
            return
        share20, n80, top_items = _pareto_share(g)
        detail = (
            f"Top 20% of **{cat}** groups contribute **{share20*100:.1f}%** of total **{val}**. "
            f"~**{n80}** groups cover 80%."
        )
        add_insight(
            ss,
            ds_name,
            "eda.bivariate",
            f"Pareto (80/20) • {cat} → {val}",
            detail,
            score=share20,
            tags=["category-driver", "pareto"],
            payload={"top_20pct_share": share20, "n_groups_for_80pct": n80, "top": top_items},
        )
        _add(
            ss,
            ds_name,
            section="eda",
            subsection="bivariate",
            kind="pareto",
            title=f"{cat} → {val}",
            text=detail,
            metrics={
                "cat": cat,
                "val": val,
                "share20": share20,
                "n_groups_80": n80,
                "top": top_items,
            },
        )

    if pd.api.types.is_numeric_dtype(sx) and not pd.api.types.is_numeric_dtype(sy):
        _capture_cat_num(y, x)
        return
    if not pd.api.types.is_numeric_dtype(sx) and pd.api.types.is_numeric_dtype(sy):
        _capture_cat_num(x, y)
        return

    # cat-cat (optional, brief dominance)
    ct = pd.crosstab(sx.astype("string"), sy.astype("string"))
    if ct.size:
        max_cell = float((ct / ct.values.sum()).max().max() * 100)
        detail = f"Largest category pair ≈ {max_cell:.1f}% of records."
        add_insight(
            ss,
            ds_name,
            "eda.bivariate",
            f"Association • {x} × {y}",
            detail,
            tags=["categorical-categorical", "crosstab"],
            payload={"max_pair_pct": max_cell},
        )
        _add(
            ss,
            ds_name,
            section="eda",
            subsection="bivariate",
            kind="crosstab",
            title=f"{x} × {y}",
            text=detail,
            metrics={"max_pair_pct": max_cell},
        )


def capture_correlation_matrix_insights(ss, ds_name: str, df: pd.DataFrame, min_abs: float = 0.6) -> None:
    """
    Capture a compact list of strong numeric-numeric correlations (by |r|).
    """
    _clear_area(ss, ds_name, section="eda", subsection="correlation", legacy_area="eda.correlation")

    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(nums) < 2:
        return
    C = df[nums].corr().abs().unstack().dropna()
    C = C[C.index.get_level_values(0) < C.index.get_level_values(1)]
    top = C[C >= min_abs].sort_values(ascending=False).head(5)
    if top.empty:
        return
    lines = [f"{a}–{b}: {v:.2f}" for (a, b), v in top.items()]
    add_insight(
        ss, ds_name, "eda.correlation", "Strong numeric relationships",
        "; ".join(lines), score=float(min(1.0, top.iloc[0])),
        tags=["correlation","heatmap"],
        payload={"pairs": [{"a": a, "b": b, "r": float(v)} for (a, b), v in top.items()]},
    )
    _add(
        ss, ds_name, section="eda", subsection="correlation", kind="heatmap",
        title="Top correlations", text="; ".join(lines),
        metrics={"top_pairs": [{"a": a, "b": b, "r": float(v)} for (a, b), v in top.items()],
                 "n_numeric": len(nums), "method": "pearson"},
    )


def capture_missingness_insight(ss, ds_name: str, df: pd.DataFrame) -> None:
    """
    Capture columns with missing values (top few by %).
    """
    _clear_area(ss, ds_name, section="preprocess", subsection="missing", legacy_area="preprocess.missing")

    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    if miss.empty:
        return
    top = miss.head(5)
    lines = [f"{c}: {p*100:.1f}%" for c, p in top.items()]
    add_insight(
        ss, ds_name, "preprocess.missing",
        "Columns with missing values", "; ".join(lines),
        tags=["quality", "missing"],
        payload={"top_missing": [{"col": c, "pct": float(p)} for c, p in top.items()]},
    )
    _add(
        ss, ds_name,
        section="preprocess", subsection="missing", kind="missing",
        title="Columns with missing values",
        text=", ".join([f"{c}: {p*100:.1f}%" for c, p in top.items()]),
        metrics={"missing_pct": top.to_dict()},
    )


def capture_outlier_action(ss, ds_name: str, cols: List[str], method: str, action: str) -> None:
    """
    Record an outlier-handling action summary (method/action/columns).
    """
    add_insight(
        ss,
        ds_name,
        "preprocess.outliers",
        "Outlier handling applied",
        f"Method: {method}. Action: {action}. Columns: {', '.join(map(str, cols))[:120]}",
        tags=["outliers", "preprocess"],
        payload={"method": method, "action": action, "columns": cols},
    )
    _add(
        ss,
        ds_name,
        section="preprocess",
        subsection="outliers",
        kind="action",
        title="Outlier handling applied",
        text=f"Method: {method}. Action: {action}. Columns: {', '.join(map(str, cols))[:120]}",
        metrics={"method": method, "action": action, "columns": cols},
    )


# --------- NEW business-style capture functions ----------
def capture_missing_outlier_overview(ss, ds_name: str, df: pd.DataFrame, top_n: int = 15) -> int:
    """
    Add a compact health+missing+outliers overview in structured store.
    """
    added = 0

    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0].head(top_n)
    if not miss.empty:
        _add(
            ss, ds_name,
            section="preprocess", subsection="missing", kind="missing",
            title="Columns with missing values",
            text=", ".join([f"{c}: {p*100:.1f}%" for c, p in miss.items()]),
            metrics={"missing_pct": miss.to_dict()},
        ); added += 1

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    flagged = {}
    for c in num_cols:
        share = _iqr_outlier_share(df[c])
        if share >= 0.05:
            flagged[c] = float(share)
    if flagged:
        top = dict(sorted(flagged.items(), key=lambda kv: kv[1], reverse=True)[:top_n])
        _add(
            ss, ds_name,
            section="eda", subsection="univariate", kind="outliers",
            title="Columns with potential outliers (IQR)",
            text=", ".join([f"{c}: {p*100:.1f}%" for c, p in top.items()]),
            metrics={"outlier_share": top},
        ); added += 1

    return added


def capture_key_drivers(
    ss, ds_name: str, df: pd.DataFrame, target: Optional[str] = None, top_k: int = 8
) -> int:
    """
    Rank likely key drivers for a target measure using simple signals:
      - numeric predictors by |Pearson r|
      - categorical predictors by correlation ratio η
    """
    target = target or _guess_measure_col(df)
    if not target or target not in df.columns:
        return 0

    y = pd.to_numeric(df[target], errors="coerce")
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != target]
    cat_cols = [
        c
        for c in df.columns
        if (not pd.api.types.is_numeric_dtype(df[c])) and df[c].nunique(dropna=True) <= 50
    ]

    scores: List[Dict[str, Any]] = []

    # numeric predictors -> abs Pearson r
    for c in num_cols:
        x = pd.to_numeric(df[c], errors="coerce")
        m = ~(x.isna() | y.isna())
        if m.sum() < 3:
            continue
        r = float(pd.Series(x[m]).corr(pd.Series(y[m]), method="pearson"))
        if pd.notna(r):
            scores.append({"feature": c, "type": "numeric", "score": abs(r)})

    # categorical predictors -> correlation ratio η
    for c in cat_cols:
        eta = _correlation_ratio(df[c], y)
        if pd.notna(eta):
            scores.append({"feature": c, "type": "categorical", "score": float(eta)})

    if not scores:
        return 0

    scores = sorted(scores, key=lambda d: d["score"], reverse=True)[:top_k]
    _add(
        ss,
        ds_name,
        section="business",
        subsection="key_drivers",
        kind="drivers",
        title=f"Top drivers for {target}",
        text="; ".join([f"{s['feature']} ({s['type']}): {s['score']:.2f}" for s in scores]),
        metrics={"target": target, "drivers": scores},
    )
    return 1


def capture_top_performers(
    ss,
    ds_name: str,
    df: pd.DataFrame,
    measure: Optional[str] = None,
    max_cats: int = 30,
    top_k: int = 3,
) -> int:
    """
    For each reasonable categorical column, rank top groups by share of a measure.
    """
    measure = measure or _guess_measure_col(df)
    if not measure or measure not in df.columns:
        return 0
    y = pd.to_numeric(df[measure], errors="coerce")

    added = 0
    for c in df.columns:
        if c == measure:
            continue
        nunique = df[c].nunique(dropna=True)
        if nunique < 2 or nunique > max_cats:
            continue
        g = pd.DataFrame({"grp": df[c].astype("string"), "y": y}).dropna()
        if g.empty:
            continue
        agg = g.groupby("grp")["y"].sum().sort_values(ascending=False)
        total = float(agg.sum())
        if total <= 0:
            continue
        top = (agg / total).head(top_k)
        _add(
            ss,
            ds_name,
            section="business",
            subsection="top_performers",
            kind="ranking",
            title=f"Top performers — by {c} on {measure}",
            text=", ".join([f"{idx} ({pct*100:.1f}%)" for idx, pct in top.items()]),
            metrics={
                "by": c,
                "measure": measure,
                "top_share": {k: float(v) for k, v in top.items()},
                "total": total,
            },
        )
        added += 1
    return added


def capture_segments(ss, ds_name: str, df: pd.DataFrame, max_cats: int = 15, top_k: int = 5) -> int:
    """
    Summarize dominant segments (categorical columns with manageable cardinality).
    """
    added = 0
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        vc = df[c].astype("string").value_counts(dropna=False, normalize=True)
        if 2 <= len(vc) <= max_cats:
            top = vc.head(top_k)
            _add(
                ss,
                ds_name,
                section="business",
                subsection="segments",
                kind="distribution",
                title=f"Dominant segments — {c}",
                text=", ".join([f"{idx} ({pct*100:.1f}%)" for idx, pct in top.items()]),
                metrics={"column": c, "share": {k: float(v) for k, v in top.items()}},
            )
            added += 1
    return added


def capture_trend_forecast(
    ss,
    ds_name: str,
    df: pd.DataFrame,
    date_col: Optional[str] = None,
    measure: Optional[str] = None,
    rule: str = "M",
    lookback_points: int = 6,
) -> int:
    """
    Simple trend and one-step-ahead forecast using linear fit over recent points.
    """
    date_col = date_col or _guess_datetime_col(df)
    measure = measure or _guess_measure_col(df)
    if not date_col or not measure or date_col not in df.columns or measure not in df.columns:
        return 0

    t = pd.to_datetime(df[date_col], errors="coerce")
    y = pd.to_numeric(df[measure], errors="coerce")
    g = pd.DataFrame({"t": t, "y": y}).dropna().sort_values("t")
    if g.empty:
        return 0

    s = g.set_index("t")["y"].resample(rule).sum(min_count=1).dropna()
    if len(s) < 3:
        return 0

    last = float(s.iloc[-1])
    prev = float(s.iloc[-2]) if len(s) >= 2 else np.nan
    growth = float(((last / prev) - 1) * 100) if prev and prev != 0 else np.nan

    y_tail = s.tail(lookback_points)
    x = np.arange(len(y_tail))
    coeff = np.polyfit(x, y_tail.values, deg=1)
    next_val = float(np.polyval(coeff, len(y_tail)))

    _add(
        ss,
        ds_name,
        section="business",
        subsection="trend",
        kind="forecast",
        title=f"Trend & forecast — {measure} ({rule})",
        text=(
            f"Last period {measure}: {last:,.0f}. MoM change: {growth:.1f}% (if defined). "
            f"Next period forecast: {next_val:,.0f}."
        ),
        metrics={
            "measure": measure,
            "rule": rule,
            "last": last,
            "prev": prev,
            "mom_pct": growth,
            "forecast_next": next_val,
        },
    )
    return 1


# ======================================================
# AUTO-SCAN (efficient batch extraction)
# ======================================================
def scan_dataset_for_insights(
    ss,
    ds_name: str,
    df: pd.DataFrame,
    sample_rows: int = 5000,
    max_num_num_pairs: int = 30,
    max_cat_num_pairs: int = 30,
    min_abs_corr: float = 0.6,
    max_cat_cardinality: int = 50,
    include_cat_cat: bool = False,
) -> int:
    """
    Fast, bounded pass over the dataset to populate a rich set of insights.
    Returns total number of (legacy + structured) items added this call.
    """
    init_insights_store(ss)
    before = len(ss.get("_insights", {}).get(ds_name, {}).get("items", [])) + len(
        get_insights(ss, ds_name)
    )

    # Sample for speed when data is large
    dfx = df.sample(sample_rows, random_state=42).copy() if len(df) > sample_rows else df.copy()

    # Overview / health / missing & outliers
    capture_summary_insights(ss, ds_name, dfx, n_tables=len(ss.get("datasets", {})))
    capture_missingness_insight(ss, ds_name, dfx)
    capture_missing_outlier_overview(ss, ds_name, dfx)

    # Univariate — all columns
    for c in dfx.columns:
        capture_univariate_insight(ss, ds_name, c, dfx[c])

    # Top numeric-numeric pairs
    nums = [c for c in dfx.columns if pd.api.types.is_numeric_dtype(dfx[c])]
    if len(nums) >= 2:
        C = dfx[nums].corr()
        pairs = []
        for i, a in enumerate(nums):
            for j, b in enumerate(nums):
                if j <= i:
                    continue
                r = C.loc[a, b]
                if pd.notna(r):
                    pairs.append((a, b, float(abs(r))))
        for a, b, _ in sorted(pairs, key=lambda t: t[2], reverse=True)[:max_num_num_pairs]:
            capture_bivariate_insight(ss, ds_name, a, b, dfx)

    # Cat×num Pareto (bounded)
    cats = [c for c in dfx.columns if not pd.api.types.is_numeric_dtype(dfx[c])]
    cats = [c for c in cats if 2 <= dfx[c].nunique(dropna=False) <= max_cat_cardinality]
    value_cols = _likely_value_columns(dfx)
    for (cat, val) in list((c, v) for c in cats for v in value_cols)[:max_cat_num_pairs]:
        capture_bivariate_insight(ss, ds_name, cat, val, dfx)

    # Optional cat×cat
    if include_cat_cat and len(cats) >= 2:
        done = 0
        for i, a in enumerate(cats):
            for j, b in enumerate(cats):
                if j <= i:
                    continue
                capture_bivariate_insight(ss, ds_name, a, b, dfx)
                done += 1
                if done >= 20:
                    break

    capture_correlation_matrix_insights(ss, ds_name, dfx, min_abs=min_abs_corr)

    after = len(ss.get("_insights", {}).get(ds_name, {}).get("items", [])) + len(
        get_insights(ss, ds_name)
    )
    return after - before


# ======================================================
# Markdown / PDF exports
# ======================================================
def generate_markdown_report(ss, ds_name: str) -> str:
    """
    Build a human-readable Markdown report from structured insights,
    falling back to legacy if none collected yet.

    - De-duplicates items: keep the latest per (section, subsection, kind, title).
      For 'summary/overview' specifically, prefer the item with the largest `rows`
      metric so we keep the full-dataset summary over sampled ones.
    - Emits each section header once.
    """
    items = ss.get("_insights", {}).get(ds_name, {}).get("items", [])
    if not items:
        # Fallback to legacy
        legacy = get_insights(ss, ds_name)
        if not legacy:
            return f"# Insights Report\n\nDataset: **{ds_name}**\n\n_No insights collected yet._\n"
        lines = [f"# Insights Report", f"Dataset: **{ds_name}**", "", "## Appendix — Legacy Insights"]
        by = {}
        for it in legacy:
            by.setdefault(it["area"], []).append(it)
        for area, rows in sorted(by.items()):
            lines.append(f"### {area} ({len(rows)})")
            for it in rows:
                tags = f" _({' • '.join(it['tags'])})_" if it.get("tags") else ""
                lines.append(f"- **{it['title']}** — {it['detail']}{tags}")
        return "\n".join(lines)

    # --- De-duplicate: keep latest by (section, subsection, kind, title).
    #     Special case: for 'summary/overview', prefer the one with larger `metrics.rows`.
    best = {}
    for it in items:
        key = (it.get("section"), it.get("subsection"), it.get("kind"), it.get("title"))
        cur = best.get(key)
        if cur is None:
            best[key] = it
            continue

        if key[:2] == ("summary", "overview"):
            # Prefer the item with the largest row count (full dataset over sampled)
            r_new = (it.get("metrics") or {}).get("rows") or 0
            r_old = (cur.get("metrics") or {}).get("rows") or 0
            if r_new > r_old:
                best[key] = it
            elif r_new == r_old and (it.get("ts", 0) >= cur.get("ts", 0)):
                best[key] = it
        else:
            # Otherwise, prefer the most recent
            if it.get("ts", 0) >= cur.get("ts", 0):
                best[key] = it

    items = list(best.values())

    def group(section, subsection=None):
        return [
            it for it in items
            if (it.get("section") == section and (subsection is None or it.get("subsection") == subsection))
        ]

    out = [f"# Insights Report", f"Dataset: **{ds_name}**", ""]

    # Overview (single header; may contain multiple concise bullets)
    ov = group("summary", "overview")
    if ov:
        out.append("## Overview")
        for it in ov:
            out.append(f"- {it.get('text')}")

    # Data Health
    hl = group("summary", "health")
    if hl:
        out.append("## Data Health")
        for it in hl:
            out.append(f"- {it.get('text')}")

    # Univariate
    uni = group("eda", "univariate")
    if uni:
        out.append("## EDA — Univariate")
        for it in uni:
            out.append(f"- **{it.get('title')}** — {it.get('text')}")

    # Bivariate (Correlations + Pareto)
    biv = group("eda", "bivariate")

    corr_rows = [it for it in biv if it.get("kind") == "correlation"]
    if corr_rows:
        out.append("## EDA — Correlations (top)")
        corr_rows = sorted(
            corr_rows,
            key=lambda it: abs((it.get("metrics") or {}).get("r", 0.0)),
            reverse=True,
        )
        for it in corr_rows[:30]:
            m = it.get("metrics") or {}
            out.append(f"- **{it.get('title')}** — r={m.get('r', float('nan')):.2f}")

    pareto = [it for it in biv if it.get("kind") == "pareto"]
    if pareto:
        out.append("## EDA — Key Drivers (Pareto)")
        for it in pareto[:30]:
            out.append(f"- **{it.get('title')}** — {it.get('text')}")

    # Correlation overview (single compact block)
    cor_over = group("eda", "correlation")
    if cor_over:
        out.append("## Correlation Overview")
        for it in cor_over:
            out.append(f"- {it.get('text')}")

    # Append legacy (if any)
    legacy = get_insights(ss, ds_name)
    if legacy:
        out.append("")
        out.append("## Appendix — Legacy Insights")
        by = {}
        for it in legacy:
            by.setdefault(it["area"], []).append(it)
        for area, rows in sorted(by.items()):
            out.append(f"### {area} ({len(rows)})")
            for it in rows:
                tags = f" _({' • '.join(it['tags'])})_" if it.get("tags") else ""
                out.append(f"- **{it['title']}** — {it['detail']}{tags}")

    return "\n".join(out)



def insights_pdf_bytes(md_text: str) -> Optional[bytes]:
    """
    Minimal Markdown→PDF (plain text) converter using reportlab, if installed.
    Returns PDF bytes or None on failure/missing dependency.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.units import cm
        from reportlab.lib.utils import simpleSplit

        # Very light "markdown": strip headings/bold markers to keep it readable.
        plain = (
            md_text.replace("**", "")
            .replace("# ", "")
            .replace("## ", "")
            .replace("### ", "")
            .replace("_(", "(")
            .replace(")_", ")")
            .replace("|---|---|---:|", "")
        )

        buf = bytearray()

        class _BAIO:
            def write(self, b):
                buf.extend(b)

            def getvalue(self):
                return bytes(buf)

        target = _BAIO()

        doc = SimpleDocTemplate(
            target, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm
        )
        styles = getSampleStyleSheet()
        body = styles["BodyText"]
        story = []
        for block in plain.split("\n"):
            if not block.strip():
                story.append(Spacer(1, 6))
                continue
            for ln in simpleSplit(block, body.fontName, body.fontSize, A4[0] - 4 * cm):
                story.append(Paragraph(ln, body))
            story.append(Spacer(1, 4))
        doc.build(story)
        return target.getvalue()
    except Exception:
        return None


# ======================================================
# Renderer for Final Summary page (tabular views)
# ======================================================
def render_collected_insights(ss, ds_name: str, use_expanders: bool = False) -> None:
    """
    Compact renderer of collected insights (grouped) for the UI.
    """
    sitems = ss.get("_insights", {}).get(ds_name, {}).get("items", [])
    if not sitems:
        st.caption("No insights captured yet.")
        return

    def block(title: str):
        if use_expanders:
            return st.expander(title, expanded=False)
        st.markdown(f"### {title}")
        return st.container()

    # Overview & Health
    with block("Overview & Health"):
        for it in sitems:
            if it.get("section") != "summary":
                continue
            st.markdown(f"- {it.get('title')}: {it.get('text')}")

    # Univariate
    uni = [it for it in sitems if it.get("section") == "eda" and it.get("subsection") == "univariate"]
    if uni:
        rows = []
        for it in uni:
            m = it.get("metrics") or {}
            rows.append(
                {
                    "column": it.get("title"),
                    "type": it.get("kind"),
                    "missing%": round((m.get("missing_pct") or 0) * 100, 1) if "missing_pct" in m else None,
                    "zeros%": round((m.get("zeros_pct") or 0) * 100, 1) if "zeros_pct" in m else None,
                    "outliers%": round((m.get("outlier_pct") or 0) * 100, 1) if "outlier_pct" in m else None,
                    "top_value": m.get("top_category"),
                    "top_share%": round(m.get("top_share_pct", 0.0), 1) if "top_share_pct" in m else None,
                }
            )
        with block("EDA — Univariate"):
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Bivariate: correlations & Pareto
    biv = [it for it in sitems if it.get("section") == "eda" and it.get("subsection") == "bivariate"]
    corr = [it for it in biv if it.get("kind") == "correlation"]
    pare = [it for it in biv if it.get("kind") == "pareto"]

    if corr:
        dfc = pd.DataFrame(
            [{"x": it["metrics"]["a"], "y": it["metrics"]["b"], "r": it["metrics"]["r"]} for it in corr]
        ).sort_values("r", key=lambda s: s.abs(), ascending=False)
        with block("EDA — Correlations (top)"):
            st.dataframe(dfc.head(30), use_container_width=True)

    if pare:
        def _safe_pct(x):
            try:
                return round(float(x) * 100, 1)
            except Exception:
                return None

        dfp = pd.DataFrame(
            [
                {
                    "group": it.get("metrics", {}).get("cat"),
                    "value": it.get("metrics", {}).get("val"),
                    "top20_share%": _safe_pct(it.get("metrics", {}).get("share20")),
                    "groups_to_80%": it.get("metrics", {}).get("n_groups_80"),
                }
                for it in pare
            ]
        ).sort_values("top20_share%", ascending=False)
        with block("EDA — Key Drivers (Pareto)"):
            st.dataframe(dfp.head(30), use_container_width=True)


# ===== Pretty Collected Insights (scrollable tables) =====
def _dedupe_items(items: list[dict]) -> list[dict]:
    """
    Keep the latest item per (section, subsection, kind, title).
    """
    out = {}
    for it in items or []:
        key = (it.get("section"), it.get("subsection"), it.get("kind"), it.get("title"))
        old = out.get(key)
        if (old is None) or (it.get("ts", 0) >= old.get("ts", 0)):
            out[key] = it
    return list(out.values())


def build_insights_tables(ss, ds_name: str) -> dict[str, pd.DataFrame]:
    """
    Produce deduped DataFrames for univariate, correlations, and pareto tables.
    """
    store = ss.get("_insights", {})
    items = store.get(ds_name, {}).get("items", [])
    items = _dedupe_items(items)
    out: dict[str, pd.DataFrame] = {}

    # Univariate
    uni = [it for it in items if it.get("section") == "eda" and it.get("subsection") == "univariate"]
    if uni:
        rows = []
        for it in uni:
            m = it.get("metrics") or {}
            rows.append(
                {
                    "Column": it.get("title"),
                    "Summary": it.get("text"),
                    "Missing%": round((m.get("missing_pct") or 0.0) * 100, 2) if "missing_pct" in m else None,
                    "Zero%": round((m.get("zeros_pct") or 0.0) * 100, 2) if "zeros_pct" in m else None,
                    "Outlier% (IQR)": round((m.get("outlier_pct") or 0.0) * 100, 2) if "outlier_pct" in m else None,
                    "Top Category": m.get("top_category"),
                    "Top Share%": round(float(m.get("top_share_pct")), 2) if m.get("top_share_pct") is not None else None,
                }
            )
        dfu = pd.DataFrame(rows)
        if not dfu.empty:
            dfu = dfu.sort_values(["Outlier% (IQR)", "Missing%"], ascending=[False, False], na_position="last")
        out["univariate"] = dfu

    # Correlations (bivariate)
    corr = [
        it
        for it in items
        if it.get("section") == "eda" and it.get("subsection") == "bivariate" and it.get("kind") == "correlation"
    ]
    if corr:
        best = {}
        for it in corr:
            m = it.get("metrics") or {}
            a, b, r = m.get("a"), m.get("b"), m.get("r")
            if a is None or b is None or r is None:
                continue
            key = tuple(sorted([str(a), str(b)]))
            cur = best.get(key)
            if (cur is None) or (abs(float(r)) > abs(cur["r"])):
                best[key] = {"A": a, "B": b, "r": float(r)}
        out["correlations"] = pd.DataFrame(best.values()).sort_values("r", key=lambda s: s.abs(), ascending=False)

    # Pareto (cat → num)
    pare = [
        it
        for it in items
        if it.get("section") == "eda" and it.get("subsection") == "bivariate" and it.get("kind") == "pareto"
    ]
    if pare:

        def _pct(x):
            try:
                return round(float(x) * 100, 1)
            except Exception:
                return None

        rows = []
        for it in pare:
            m = it.get("metrics") or {}
            rows.append(
                {
                    "Group (Category)": m.get("cat"),
                    "Target (Numeric)": m.get("val"),
                    "Top20% Share": _pct(m.get("share20")),
                    "Groups to 80%": m.get("n_groups_80"),
                    "Headline": it.get("title"),
                }
            )
        out["pareto"] = pd.DataFrame(rows).sort_values("Top20% Share", ascending=False, na_position="last")

    return out


def render_collected_insights_pretty(ss, ds_name: str) -> None:
    """
    UI renderer: show scrollable, pretty tables for the collected insights.
    """
    tables = build_insights_tables(ss, ds_name)
    if not tables:
        st.caption("No insights captured yet.")
        return

    # Univariate
    if "univariate" in tables:
        with st.expander("EDA — Univariate (scrollable)", expanded=True):
            df = tables["univariate"]
            try:
                st.dataframe(
                    df,
                    use_container_width=True,
                    height=420,
                    column_config={
                        "Column": st.column_config.TextColumn(width="large"),
                        "Summary": st.column_config.TextColumn(width=600),
                    },
                )
            except Exception:
                st.dataframe(df, use_container_width=True, height=420)

    # Correlations
    if "correlations" in tables:
        with st.expander("EDA — Correlations", expanded=True):
            df = tables["correlations"]
            try:
                st.dataframe(
                    df,
                    use_container_width=True,
                    height=360,
                    column_config={
                        "A": st.column_config.TextColumn(width="large"),
                        "B": st.column_config.TextColumn(width="large"),
                    },
                )
            except Exception:
                st.dataframe(df, use_container_width=True, height=360)

    # Pareto
    if "pareto" in tables:
        with st.expander("EDA — Key Drivers (Pareto)", expanded=False):
            df = tables["pareto"]
            try:
                st.dataframe(
                    df,
                    use_container_width=True,
                    height=360,
                    column_config={
                        "Headline": st.column_config.TextColumn(width=520),
                        "Group (Category)": st.column_config.TextColumn(width="large"),
                        "Target (Numeric)": st.column_config.TextColumn(width="large"),
                    },
                )
            except Exception:
                st.dataframe(df, use_container_width=True, height=360)
