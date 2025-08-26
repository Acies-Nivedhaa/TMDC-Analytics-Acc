# core/eda_bivariate.py
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import streamlit as st


# -------------------- small utils --------------------

def _as_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _as_category(s: pd.Series) -> pd.Series:
    if ptypes.is_categorical_dtype(s):
        return s
    return s.astype("category")

def _is_binary(s: pd.Series) -> bool:
    # binary after dropping NaNs and casting to category
    vals = pd.Series(s).dropna().astype("category").cat.categories
    return len(vals) == 2

def _co_dropna(x: pd.Series, y: pd.Series) -> tuple[pd.Series, pd.Series]:
    m = ~(x.isna() | y.isna())
    return x[m], y[m]

def _discretize_numeric(s: pd.Series, q: int = 10) -> pd.Series:
    s = _as_numeric(s)
    try:
        return pd.qcut(s, q=min(q, s.nunique(dropna=True)), duplicates="drop")
    except Exception:
        # if qcut fails (few uniques), fall back to category on value
        return s.astype("category")

def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())

# -------------------- association measures --------------------

def pearson_r(x: pd.Series, y: pd.Series) -> float:
    x, y = _co_dropna(_as_numeric(x), _as_numeric(y))
    return float(x.corr(y, method="pearson"))

def spearman_rho(x: pd.Series, y: pd.Series) -> float:
    # ranks then Pearson, works without SciPy
    x, y = _co_dropna(_as_numeric(x), _as_numeric(y))
    return float(x.corr(y, method="spearman"))

def kendall_tau(x: pd.Series, y: pd.Series) -> float:
    x, y = _co_dropna(_as_numeric(x), _as_numeric(y))
    return float(x.corr(y, method="kendall"))

def point_biserial(x: pd.Series, y: pd.Series) -> float:
    # numeric <-> binary
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
    z0 = z[codes == 0]; z1 = z[codes == 1]
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
    # η (cat -> num)
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
    # bias-corrected Cramér's V
    a = _as_category(x); b = _as_category(y)
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
    # discretize numeric to quantiles; NMI using max(Hx, Hy)
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

# -------------------- auto routing --------------------

def _kind(s: pd.Series) -> str:
    if ptypes.is_numeric_dtype(s):
        return "num"
    return "cat"

def compute_association(x: pd.Series, y: pd.Series, method: str = "Auto") -> tuple[str, float]:
    kx, ky = _kind(x), _kind(y)

    # choose default method
    if method == "Auto":
        if kx == "num" and ky == "num":
            method = "Pearson"
        elif (kx == "num" and ky == "cat") or (kx == "cat" and ky == "num"):
            # numeric-binary -> point biserial; else η
            num, cat = (x, y) if kx == "num" else (y, x)
            method = "Point-biserial" if _is_binary(cat) else "Correlation ratio (η)"
        else:
            method = "Cramér’s V"

    # compute
    if method == "Pearson":
        val = pearson_r(x, y)
    elif method == "Spearman":
        val = spearman_rho(x, y)
    elif method == "Kendall":
        val = kendall_tau(x, y)
    elif method == "Point-biserial":
        val = point_biserial(x, y)
    elif method == "Correlation ratio (η)":
        # direction cat->num if possible
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

# -------------------- UI --------------------

def render_bivariate(df: pd.DataFrame, sample_n: int | None = None) -> None:
    import streamlit as st
    st.markdown("**Bivariate**")

    cols = df.columns.tolist()
    if len(cols) < 2:
        st.info("Need at least two columns.")
        return

    # pick X / Y
    c1, c2 = st.columns(2)
    with c1:
        x_col = st.selectbox("X", cols, index=0, key="biv_x")
    with c2:
        y_col = st.selectbox("Y", cols, index=1, key="biv_y")

    # sample (for speed) and get series
    x_full, y_full = df[x_col], df[y_col]
    if sample_n:
        n = min(sample_n, len(df))
        idx = df.sample(n, random_state=0).index if n < len(df) else df.index
        x, y = x_full.loc[idx], y_full.loc[idx]
    else:
        x, y = x_full, y_full

    # compute association in Auto mode (no picker)
    used, value = compute_association(x, y, method="Auto")
    if np.isnan(value):
        st.caption(f"Association: **{used}** (not available for these columns)")
    else:
        st.caption(f"Association (**{used}**): **{value:.3f}**")

    # visualization
    kx, ky = _kind(x), _kind(y)
    if kx == "num" and ky == "num":
        st.markdown("**Scatter (numeric vs numeric)**")
        st.scatter_chart(pd.DataFrame({x_col: _as_numeric(x), y_col: _as_numeric(y)}))
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
        ct = pd.crosstab(_as_category(x).astype(str), _as_category(y).astype(str))
        st.dataframe(ct, use_container_width=True)
