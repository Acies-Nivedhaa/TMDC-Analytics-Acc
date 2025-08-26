import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- small helpers ----------
def _kind(s: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(s):
        return "num"
    return "cat"

def _as_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _as_cat(s: pd.Series) -> pd.Series:
    if pd.api.types.is_categorical_dtype(s):
        return s
    return s.astype("category")

def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    # bias-corrected Cramér's V without SciPy
    table = pd.crosstab(x, y)
    if table.size == 0:
        return np.nan
    n = table.values.sum()
    if n == 0:
        return np.nan
    row_sums = table.sum(axis=1).values[:, None]
    col_sums = table.sum(axis=0).values[None, :]
    expected = (row_sums @ col_sums) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((table.values - expected) ** 2 / expected)
    phi2 = max(chi2 / n, 0.0)
    r, k = table.shape
    if n == 1:
        return np.nan
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(rcorr - 1, kcorr - 1)
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else np.nan

def _corr_ratio(categories: pd.Series, values: pd.Series) -> float:
    # correlation ratio (eta): num ~ cat
    df = pd.DataFrame({"cat": categories, "val": values})
    df = df.dropna()
    if df.empty:
        return np.nan
    groups = df.groupby("cat")["val"]
    n = len(df)
    mu = df["val"].mean()
    ss_between = sum(g.size * (g.mean() - mu) ** 2 for _, g in groups)
    ss_total = float(((df["val"] - mu) ** 2).sum())
    return float(np.sqrt(ss_between / ss_total)) if ss_total > 0 else np.nan

def _point_biserial(num: pd.Series, cat: pd.Series) -> float:
    # If cat is binary -> Pearson(num, encoded cat)
    c = _as_cat(cat)
    if c.cat.categories.size != 2:
        return np.nan
    codes = c.cat.codes.replace(-1, np.nan)  # -1 are NaN
    a = _as_num(num).astype(float)
    mask = ~(a.isna() | codes.isna())
    if mask.sum() < 3:
        return np.nan
    return float(np.corrcoef(a[mask], codes[mask])[0, 1])

def _mixed_pair(x: pd.Series, y: pd.Series) -> float:
    kx, ky = _kind(x), _kind(y)
    if kx == "num" and ky == "num":
        a, b = _as_num(x), _as_num(y)
        mask = ~(a.isna() | b.isna())
        if mask.sum() < 3:
            return np.nan
        return float(np.corrcoef(a[mask], b[mask])[0, 1])
    if kx == "cat" and ky == "cat":
        return _cramers_v(_as_cat(x), _as_cat(y))
    # mixed
    if kx == "num":
        num, cat = x, y
    else:
        num, cat = y, x
    # try point-biserial if binary; else correlation ratio
    pv = _point_biserial(_as_num(num), _as_cat(cat))
    if not np.isnan(pv):
        return pv
    return _corr_ratio(_as_cat(cat), _as_num(num))

def _plot_heatmap(corr: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(min(0.45 * corr.shape[1] + 6, 18),
                                    min(0.45 * corr.shape[0] + 6, 18)))
    sns.heatmap(
        corr, ax=ax, annot=True, fmt=".2f", cmap="viridis",
        vmin=-1, vmax=1, linewidths=0.5, cbar_kws={"label": "corr"}
    )
    ax.set_title(title)
    st.pyplot(fig, clear_figure=True)

# ---------- main renderer ----------
def render_correlation(df: pd.DataFrame, sample_n: int | None = None) -> None:
    st.markdown("**Correlation**")

    # choose method
    method = st.selectbox(
        "Correlation method",
        [
            "Pearson (numeric)",
            "Spearman (numeric)",
            "Kendall (numeric)",
            "Cramér’s V (categorical)",
            "Auto (mixed types)",
        ],
        index=0,
        key="corr_method",
        help="Choose how correlations are computed and which columns are included."
    )

    # sample for speed
    work = df
    if sample_n:
        n = min(sample_n, len(df))
        if n < len(df):
            work = df.sample(n, random_state=0)

    # decide columns and compute
    if method in ("Pearson (numeric)", "Spearman (numeric)", "Kendall (numeric)"):
        num_df = work.select_dtypes(include=[np.number])
        if num_df.shape[1] < 2:
            st.info("Need at least two numeric columns.")
            return
        alg = {"Pearson (numeric)": "pearson",
               "Spearman (numeric)": "spearman",
               "Kendall (numeric)": "kendall"}[method]
        corr = num_df.corr(method=alg, numeric_only=True)
        _plot_heatmap(corr, f"Correlation heatmap ({method.split()[0]})")
        return

    if method == "Cramér’s V (categorical)":
        cat_cols = [c for c in work.columns if _kind(work[c]) == "cat" or not pd.api.types.is_numeric_dtype(work[c])]
        if len(cat_cols) < 2:
            st.info("Need at least two categorical columns.")
            return
        # optional column selector
        sel = st.multiselect("Columns", cat_cols, default=cat_cols, key="corr_cat_cols")
        if len(sel) < 2:
            st.info("Pick two or more categorical columns.")
            return
        cols = sel
        m = np.zeros((len(cols), len(cols)), dtype=float)
        for i, c1 in enumerate(cols):
            for j, c2 in enumerate(cols[i:], start=i):
                v = _cramers_v(_as_cat(work[c1]), _as_cat(work[c2]))
                m[i, j] = m[j, i] = v
        corr = pd.DataFrame(m, index=cols, columns=cols)
        _plot_heatmap(corr, "Cramér’s V heatmap (categorical)")
        return

    # Auto (mixed types)
    cols_all = list(work.columns)
    # optional selector to keep the matrix manageable
    default_cols = cols_all[: min(20, len(cols_all))]
    sel = st.multiselect("Columns", cols_all, default=default_cols, key="corr_auto_cols")
    if len(sel) < 2:
        st.info("Pick two or more columns.")
        return
    cols = sel
    m = np.zeros((len(cols), len(cols)), dtype=float)
    for i, c1 in enumerate(cols):
        s1 = work[c1]
        for j, c2 in enumerate(cols[i:], start=i):
            s2 = work[c2]
            v = _mixed_pair(s1, s2)
            # clamp to [-1, 1] for color scale
            if not np.isnan(v):
                v = float(max(-1.0, min(1.0, v)))
            m[i, j] = m[j, i] = v
    corr = pd.DataFrame(m, index=cols, columns=cols)
    _plot_heatmap(corr, "Mixed-type association heatmap (Auto)")
