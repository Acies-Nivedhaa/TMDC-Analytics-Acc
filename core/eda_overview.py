# core/eda_overview.py
from __future__ import annotations
import json
import streamlit as st
import pandas as pd
import pandas.api.types as ptypes
from ui.components import kpi_row, render_table


# ---------------- helpers ----------------

def _type_buckets(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Return (numeric_cols, datetime_cols, categorical_cols) names."""
    num_cols = [c for c in df.columns if ptypes.is_numeric_dtype(df[c])]
    dt_cols = [c for c in df.columns if ptypes.is_datetime64_any_dtype(df[c])]
    cat_cols = [
        c
        for c in df.columns
        if (ptypes.is_string_dtype(df[c]) or ptypes.is_categorical_dtype(df[c]) or ptypes.is_bool_dtype(df[c]))
        and c not in dt_cols
    ]
    return num_cols, dt_cols, cat_cols


def _make_hashable(x):
    """Convert unhashable nested objects (list/dict/set) into stable, hashable forms."""
    try:
        hash(x)
        return x
    except TypeError:
        # Normalize common container types
        if isinstance(x, dict):
            return tuple(sorted((k, _make_hashable(v)) for k, v in x.items()))
        if isinstance(x, (list, tuple)):
            return tuple(_make_hashable(v) for v in x)
        if isinstance(x, set):
            return tuple(sorted(_make_hashable(v) for v in x))
        # Fallback to a stable string
        try:
            return json.dumps(x, sort_keys=True, default=str)
        except Exception:
            return str(x)


def _safe_duplicated_count(df: pd.DataFrame) -> int:
    try:
        return int(df.duplicated().sum())
    except TypeError:
        sig = df.applymap(_cell_key).apply(tuple, axis=1)
        return int(sig.duplicated().sum())
    except Exception:
        return 0


def _kpis(df: pd.DataFrame) -> dict[str, float | int]:
    rows = int(len(df)); cols = int(df.shape[1])
    dups = _safe_duplicated_count(df) if rows else 0
    total_cells = rows * cols
    miss_pct = float((df.isna().sum().sum() / total_cells) * 100) if total_cells else 0.0
    return {"rows": rows, "cols": cols, "dups": dups, "miss_pct": miss_pct}

def _cell_key(v):
    try:
        hash(v); return v
    except TypeError:
        pass
    try:
        if isinstance(v, (dict, list, tuple, set)):
            import json
            return json.dumps(v, sort_keys=True, default=str)
    except Exception:
        pass
    return str(v)


# ---------------- render ----------------

def render_overview(df: pd.DataFrame) -> None:
    """
    Overview subtab: KPI tiles + Dtypes table. Works with a single dataset.
    Handles unhashable cells (lists/dicts) safely for duplicate counting.
    """
    k = _kpis(df)
    num_cols, _, cat_cols = _type_buckets(df)

    # KPI row (matches your minimal style)
    kpi_row([
        ("Rows", f"{k['rows']:,}"),
        ("Columns", f"{k['cols']:,}"),
        ("Numeric cols", f"{len(num_cols):,}"),
        ("Categorical cols", f"{len(cat_cols):,}"),
    ])

    st.markdown("**Dtypes**")
    dtypes_tbl = pd.DataFrame({"column": df.columns, "dtype": [str(t) for t in df.dtypes]})
    render_table(dtypes_tbl, height=360)
