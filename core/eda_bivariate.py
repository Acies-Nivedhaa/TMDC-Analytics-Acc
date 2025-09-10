# core/eda_bivariate.py
from __future__ import annotations

import math
import json as _json
import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import streamlit as st
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_timedelta64_dtype,
    is_numeric_dtype,
)
from core.insights_extractors import (
    init_insights_store,
    capture_bivariate_insight,
    saved_badge,
    save_bivariate_all,
)


# =========================
# Small utilities (robust to nested/mixed dtypes)
# =========================

def _make_hashable(x):
    """Convert unhashable nested objects (list/dict/set) into stable, hashable labels."""
    try:
        hash(x)
        return x
    except TypeError:
        try:
            if isinstance(x, dict):
                return tuple(sorted((k, _make_hashable(v)) for k, v in x.items()))
            if isinstance(x, (list, tuple, set)):
                return tuple(_make_hashable(v) for v in x)
            # Fallback to a stable, deterministic string
            return _json.dumps(x, sort_keys=True, default=str)
        except Exception:
            return str(x)


def _as_numeric(s) -> pd.Series:
    """
    Best-effort numeric coercion that won't crash on Categoricals/Datetime/Timedelta.
    Returns float Series with NaN where conversion isn't possible.
    """
    s = s if isinstance(s, pd.Series) else pd.Series(s)

    # Categorical -> go through object first (avoid pandas astype ValueError)
    if is_categorical_dtype(s):
        s = s.astype("object")

    # Datetime -> epoch seconds (float)
    if is_datetime64_any_dtype(s):
        dt = pd.to_datetime(s, errors="coerce")
        return (dt.astype("int64") / 1e9).astype("float64")  # ns -> seconds

    # Timedelta -> seconds (float)
    if is_timedelta64_dtype(s):
        td = pd.to_timedelta(s, errors="coerce")
        return (td.view("int64") / 1e9).astype("float64")

    # Numeric or object -> numeric coercion
    if is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype("float64")

    return pd.to_numeric(s.astype("object"), errors="coerce").astype("float64")


def _as_category(s: pd.Series) -> pd.Series:
    """Cast to category, handling unhashable cells safely."""
    if ptypes.is_categorical_dtype(s):
        return s
    try:
        return s.astype("category")
    except Exception:
        # Robust path: map unhashables -> stable labels, then cast
        return s.map(_make_hashable).astype("category")


def _is_binary(s: pd.Series) -> bool:
    """Return True if a series behaves as binary after dropping NaNs."""
    cats = _as_category(pd.Series(s)).dropna().astype("category").cat.categories
    return len(cats) == 2


def _co_dropna(x: pd.Series, y: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Drop rows where either x or y is NaN; return aligned series."""
    m = ~(x.isna() | y.isna())
    return x[m], y[m]


def _discretize_numeric(s: pd.Series, q: int = 10) -> pd.Series:
    """Discretize a numeric series to quantile bins; fallback to categorical if needed."""
    s = _as_numeric(s)
    try:
        return pd.qcut(s, q=min(q, s.nunique(dropna=True)), duplicates="drop")
    except Exception:
        return _as_category(s)


def _entropy(p: np.ndarray) -> float:
    """Shannon entropy for a probability vector."""
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _to_str_for_crosstab(s) -> pd.Series:
    """
    Convert input (incl. pandas.Categorical) into a string Series
    that works with pd.crosstab across pandas versions.
    """
    s = s if isinstance(s, pd.Series) else pd.Series(s)
    if is_categorical_dtype(s):
        s = s.astype("object")  # pandas quirk
    return s.astype("string").fillna("<NA>")


# =========================
# Association measures
# =========================

def pearson_r(x: pd.Series, y: pd.Series) -> float:
    x, y = _co_dropna(_as_numeric(x), _as_numeric(y))
    return float(x.corr(y, method="pearson"))


def spearman_rho(x: pd.Series, y: pd.Series) -> float:
    x, y = _co_dropna(_as_numeric(x), _as_numeric(y))
    return float(x.corr(y, method="spearman"))


def kendall_tau(x: pd.Series, y: pd.Series) -> float:
    x, y = _co_dropna(_as_numeric(x), _as_numeric(y))
    return float(x.corr(y, method="kendall"))


def point_biserial(x: pd.Series, y: pd.Series) -> float:
    """
    Point-biserial correlation between a numeric column and a binary category.
    Returns NaN if the categorical side is not binary.
    """
    if _is_binary(x) and ptypes.is_numeric_dtype(y):
        b = _as_category(x)
        z = _as_numeric(y)
    elif _is_binary(y) and ptypes.is_numeric_dtype(x):
        b = _as_category(y)
        z = _as_numeric(x)
    else:
        return float("nan")

    b, z = _co_dropna(b, z)
    if len(z) < 3:
        return float("nan")

    codes = b.cat.codes  # 0/1
    z0 = z[codes == 0]
    z1 = z[codes == 1]
    n0, n1 = len(z0), len(z1)
    if n0 == 0 or n1 == 0:
        return float("nan")

    mean0, mean1 = float(z0.mean()), float(z1.mean())
    s = float(z.std(ddof=1))
    if s == 0:
        return 0.0
    r = (mean1 - mean0) / s * math.sqrt((n0 * n1) / (len(z) ** 2))
    return float(r)


def correlation_ratio(categories: pd.Series, measurements: pd.Series) -> float:
    """
    Correlation ratio η (0..1): share of numeric variance explained by a categorical split.
    """
    c = _as_category(categories)
    z = _as_numeric(measurements)
    c, z = _co_dropna(c, z)
    if len(z) < 2 or c.nunique() < 2:
        return float("nan")

    groups = [z[c.cat.codes == i] for i in range(len(c.cat.categories))]
    counts = np.array([g.size for g in groups], dtype=float)
    means = np.array([g.mean() if g.size else np.nan for g in groups], dtype=float)
    means = np.nan_to_num(means, nan=np.nanmean(means))
    overall_mean = float(z.mean())

    ss_between = float(np.sum(counts * (means - overall_mean) ** 2))
    ss_total = float(np.sum((z - overall_mean) ** 2))
    return float(np.sqrt(ss_between / ss_total)) if ss_total > 0 else 0.0


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Bias-corrected Cramér’s V (0..1) for two categorical columns.
    """
    a = _as_category(x)
    b = _as_category(y)
    a, b = _co_dropna(a, b)
    if a.nunique() < 2 or b.nunique() < 2:
        return float("nan")

    ct = pd.crosstab(a, b)
    n = ct.values.sum()
    row_sums = ct.sum(axis=1).values[:, None]
    col_sums = ct.sum(axis=0).values[None, :]
    expected = row_sums @ col_sums / n
    chi2 = float(((ct.values - expected) ** 2 / np.where(expected == 0, 1, expected)).sum())

    r, k = ct.shape
    phi2 = chi2 / n
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / max(1, n - 1))
    rcorr = r - ((r - 1) ** 2) / max(1, n - 1)
    kcorr = k - ((k - 1) ** 2) / max(1, n - 1)
    denom = max(1e-12, min(rcorr - 1, kcorr - 1))
    return float(np.sqrt(phi2corr / denom))


def normalized_mutual_info(x: pd.Series, y: pd.Series) -> float:
    """
    Normalized Mutual Information (0..1). Numerics are discretized into quantiles first.
    """
    xs = _discretize_numeric(x) if ptypes.is_numeric_dtype(x) else _as_category(x)
    ys = _discretize_numeric(y) if ptypes.is_numeric_dtype(y) else _as_category(y)
    xs, ys = _co_dropna(xs, ys)
    if xs.nunique() < 2 or ys.nunique() < 2:
        return float("nan")

    ct = pd.crosstab(xs, ys).values.astype(float)
    pxy = ct / ct.sum()
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        mi = np.nansum(pxy * (np.log(pxy + 1e-12) - np.log(px + 1e-12) - np.log(py + 1e-12)))

    hx = _entropy(px.flatten())
    hy = _entropy(py.flatten())
    hmax = max(hx, hy)
    return float(mi / hmax) if hmax > 0 else 0.0


# =========================
# Auto routing
# =========================

def _kind(s: pd.Series) -> str:
    """Return 'num' or 'cat' to guide method selection."""
    if ptypes.is_numeric_dtype(s):
        return "num"
    return "cat"


def compute_association(x: pd.Series, y: pd.Series, method: str = "Auto") -> tuple[str, float]:
    """
    Compute a sensible association metric for (x, y).
    If method == "Auto", pick based on dtypes; otherwise use the requested metric.
    Returns (method_used, value).
    """
    kx, ky = _kind(x), _kind(y)

    # Choose default method
    if method == "Auto":
        if kx == "num" and ky == "num":
            method = "Pearson"
        elif (kx == "num" and ky == "cat") or (kx == "cat" and ky == "num"):
            # numeric-binary -> point biserial; else η
            num, cat = (x, y) if kx == "num" else (y, x)
            method = "Point-biserial" if _is_binary(cat) else "Correlation ratio (η)"
        else:
            method = "Cramér’s V"

    # Compute
    if method == "Pearson":
        val = pearson_r(x, y)
    elif method == "Spearman":
        val = spearman_rho(x, y)
    elif method == "Kendall":
        val = kendall_tau(x, y)
    elif method == "Point-biserial":
        val = point_biserial(x, y)
    elif method == "Correlation ratio (η)":
        if _kind(x) == "cat" and _kind(y) == "num":
            val = correlation_ratio(x, y)
        elif _kind(y) == "cat" and _kind(x) == "num":
            val = correlation_ratio(y, x)
        else:
            val = float("nan")
    elif method == "Cramér’s V":
        val = cramers_v(x, y)
    elif method == "Normalized MI":
        val = normalized_mutual_info(x, y)
    else:
        val = float("nan")

    return method, float(val)


# =========================
# UI
# =========================

def render_bivariate(df: pd.DataFrame, sample_n: int | None = None) -> None:
    """
    Bivariate subtab:
      - Pick any X,Y columns
      - Auto-select a suitable association metric and show its value
      - Visualize (scatter, mean-by-category, or crosstab)
      - Save *all* pairwise insights or the *current* pair to Final Summary
    """
    st.markdown("**Bivariate**")

    # Compact, plain-English glossary for this tab
    with st.expander("Key terms (quick guide)", expanded=False):
        st.markdown(
            """
- **Bivariate Analysis**: Examining relationships between two variables
- **Scatter plot**: Shows relationship between two numeric variables
- **Cross-tabulation**: Frequency table for two categorical variables
- **Box plot by group**: Numeric variable grouped by categorical
- **Association** — One number describing how two columns move together.
  - Signed metrics (Pearson/Spearman/Kendall) range **−1…+1**.
  - Others (η, Cramér’s V, NMI) range **0…1** (higher = stronger).
- **Auto method** — Picked from data types:
  - **num ↔ num:** *Pearson r* (linear).
  - **cat ↔ num:** *Point-biserial* if the category is **binary**, else *Correlation ratio (η)*.
  - **cat ↔ cat:** *Cramér’s V* (bias-corrected).
- **Mean by category** — Average of a numeric measure for each category (quick driver view).
- **Counts heatmap** — Category×category table; concentrated cells hint at strong links.
- **Rules of thumb** — For |r|: ~0.1 weak, ~0.3 moderate, ≥0.5 strong.
  For 0–1 metrics: ~0.2 weak, ~0.4 moderate, ≥0.6 strong (context matters).
- **Note** — All metrics use rows where **both** columns are non-missing.
            """
        )

    init_insights_store(st.session_state)

    cols = df.columns.tolist()
    if len(cols) < 2:
        st.info("Need at least two columns.")
        return

    # Pick X / Y
    c1, c2 = st.columns(2)
    with c1:
        x_col = st.selectbox("X", cols, index=0, key="biv_x")
    with c2:
        y_col = st.selectbox("Y", cols, index=1, key="biv_y")

    # Optional sampling for speed
    x_full, y_full = df[x_col], df[y_col]
    if sample_n:
        n = min(sample_n, len(df))
        idx = df.sample(n, random_state=0).index if n < len(df) else df.index
        x, y = x_full.loc[idx], y_full.loc[idx]
    else:
        x, y = x_full, y_full

    # Compute association in Auto mode
    used, value = compute_association(x, y, method="Auto")
    if np.isnan(value):
        st.caption(f"Association: **{used}** (not available for these columns)")
    else:
        st.caption(f"Association (**{used}**): **{value:.3f}**")

    # Visualization chosen by type combo
    kx, ky = _kind(x), _kind(y)

    if kx == "num" and ky == "num":
        st.markdown("**Scatter (numeric vs numeric)**")
        xs = _as_numeric(x)
        ys = _as_numeric(y)
        df_xy = pd.DataFrame({x_col: xs, y_col: ys}).dropna()
        if df_xy.empty:
            st.caption("No numeric data to plot for this pair.")
        else:
            st.scatter_chart(df_xy)

    elif kx == "cat" and ky == "num":
        st.markdown("**Mean by category**")
        temp = pd.DataFrame({x_col: _as_category(x).astype(str), y_col: _as_numeric(y)})
        means = temp.groupby(x_col)[y_col].mean().sort_values(ascending=False)
        st.bar_chart(means)

    elif kx == "num" and ky == "cat":
        st.markdown("**Mean by category**")
        temp = pd.DataFrame({y_col: _as_category(y).astype(str), x_col: _as_numeric(x)})
        means = temp.groupby(y_col)[x_col].mean().sort_values(ascending=False)
        st.bar_chart(means)

    else:
        st.markdown("**Counts heatmap (category vs category)**")
        xs = _to_str_for_crosstab(_as_category(x))
        ys = _to_str_for_crosstab(_as_category(y))
        ct = pd.crosstab(xs, ys)
        st.dataframe(ct, use_container_width=True)

    # Save-all control
    st.markdown("---")
    if st.button("💾 Save insights for ALL pairs in this table", key="biv_save_all"):
        ds_name = st.session_state.get("active_ds") or "dataset"
        n = save_bivariate_all(
            st.session_state,
            ds_name,
            df,
            min_abs_corr=0.30,      # adjust if you need fewer/more correlations
            max_pairs_num_num=50,
            max_pairs_cat_num=50,
            max_cat_cardinality=50,
        )
        st.success(f"Saved {n} bivariate insight(s) for '{ds_name}'.")
        saved_badge()

    # Save current pair (optional)
    ds_name = st.session_state.get("active_ds") or "dataset"
    if st.checkbox("Save this insight to Final Summary", key=f"biv_save_{x_col}_{y_col}"):
        try:
            # Pass the full df so the extractor can compute robust stats
            capture_bivariate_insight(st.session_state, ds_name, x_col, y_col, df)
            saved_badge()
        except Exception as e:
            st.warning(f"Could not save bivariate insight: {e}")
