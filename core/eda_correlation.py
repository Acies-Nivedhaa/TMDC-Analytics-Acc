# core/eda_correlation.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from core.insights_extractors import save_correlation_overview, saved_badge


# =========================
# Small helpers (robust to mixed dtypes)
# =========================

def _kind(s: pd.Series) -> str:
    """Return 'num' if numeric-like, else 'cat' (used for routing)."""
    if pd.api.types.is_numeric_dtype(s):
        return "num"
    return "cat"


def _as_num(s: pd.Series) -> pd.Series:
    """Best-effort numeric coercion; invalid parses become NaN."""
    return pd.to_numeric(s, errors="coerce")


def _as_cat(s: pd.Series) -> pd.Series:
    """Cast to pandas 'category' (no-op if already categorical)."""
    if pd.api.types.is_categorical_dtype(s):
        return s
    return s.astype("category")


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Bias-corrected Cramér’s V (0..1) measuring association between two categorical columns.
    Implementation avoids SciPy dependency.
    """
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

    # Bias correction
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(rcorr - 1, kcorr - 1)
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else np.nan


def _corr_ratio(categories: pd.Series, values: pd.Series) -> float:
    """
    Correlation ratio (η, 0..1): how much of numeric variance is explained by a categorical split.
    """
    df = pd.DataFrame({"cat": categories, "val": values}).dropna()
    if df.empty:
        return np.nan

    groups = df.groupby("cat")["val"]
    mu = df["val"].mean()
    ss_between = sum(g.size * (g.mean() - mu) ** 2 for _, g in groups)
    ss_total = float(((df["val"] - mu) ** 2).sum())
    return float(np.sqrt(ss_between / ss_total)) if ss_total > 0 else np.nan


def _point_biserial(num: pd.Series, cat: pd.Series) -> float:
    """
    Point-biserial correlation (−1..+1): numeric vs binary category.
    Falls back to NaN if category isn’t binary or data is insufficient.
    """
    c = _as_cat(cat)
    if c.cat.categories.size != 2:
        return np.nan

    # Encode categories to 0/1; −1 indicates NaN
    codes = c.cat.codes.replace(-1, np.nan)
    a = _as_num(num).astype(float)
    mask = ~(a.isna() | codes.isna())
    if mask.sum() < 3:
        return np.nan

    return float(np.corrcoef(a[mask], codes[mask])[0, 1])


def _mixed_pair(x: pd.Series, y: pd.Series) -> float:
    """
    Unified association for a pair of columns:
    - num↔num: Pearson r
    - cat↔cat: Cramér’s V
    - mixed : point-biserial if binary; else correlation ratio (η)
    """
    kx, ky = _kind(x), _kind(y)

    if kx == "num" and ky == "num":
        a, b = _as_num(x), _as_num(y)
        mask = ~(a.isna() | b.isna())
        if mask.sum() < 3:
            return np.nan
        return float(np.corrcoef(a[mask], b[mask])[0, 1])

    if kx == "cat" and ky == "cat":
        return _cramers_v(_as_cat(x), _as_cat(y))

    # Mixed: choose point-biserial if binary; else η
    num, cat = (x, y) if kx == "num" else (y, x)
    pv = _point_biserial(_as_num(num), _as_cat(cat))
    if not np.isnan(pv):
        return pv
    return _corr_ratio(_as_cat(cat), _as_num(num))


def _plot_heatmap(corr: pd.DataFrame, title: str) -> None:
    """
    Render a heatmap with dynamic sizing to reduce clipping on wider matrices.
    """
    fig, ax = plt.subplots(
        figsize=(
            min(0.45 * corr.shape[1] + 6, 18),
            min(0.45 * corr.shape[0] + 6, 18),
        )
    )
    sns.heatmap(
        corr,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"label": "corr"},
    )
    ax.set_title(title)
    st.pyplot(fig, clear_figure=True)


# =========================
# Main renderer
# =========================

def render_correlation(df: pd.DataFrame, sample_n: int | None = None) -> None:
    """
    Correlation subtab:
      - Lets users pick a method (Pearson/Spearman/Kendall/Cramér’s V/Auto)
      - Computes correlation/association matrix accordingly
      - Renders a heatmap
      - Offers a control to save a “top relationships” summary to Final Summary
    """
    st.markdown("**Correlation**")

    # --- Compact glossary (collapsible) ---
    with st.expander("Key terms (quick guide)", expanded=False):
        st.markdown(
            """
- **Correlation / association** — One number summarizing how two columns vary together.
  - Signed measures (e.g., Pearson/Spearman/Kendall) range **−1 to +1**; the sign shows direction.
  - Others (e.g., Cramér’s V, correlation ratio η) range **0 to 1**; higher = stronger link.
- **Pearson r (−1…+1)** — Linear correlation between two numeric columns. Fast, but sensitive to outliers.
- **Spearman ρ (−1…+1)** — Rank correlation; captures monotonic (not strictly linear) trends. More robust to outliers.
- **Kendall τ (−1…+1)** — Rank concordance; similar goal to Spearman, often more conservative on small samples.
- **Cramér’s V (0…1)** — Strength of association between **two categorical** columns via a contingency table.
- **Point-biserial (−1…+1)** — Correlation between a **binary** category (two labels) and a numeric column.
- **Correlation ratio, η (0…1)** — Share of numeric variation explained by a categorical split (cat → num).
- **Auto (mixed types)** — Chooses a sensible metric based on dtypes:
  - **num ↔ num** → Pearson
  - **binary cat ↔ num** → Point-biserial
  - **multi-level cat ↔ num** → Correlation ratio (η)
- **Heatmap** — Color grid of pairwise associations; brighter = stronger. Diagonal is always 1.0 by definition.
- **Min |r| to save** — Threshold when saving “top relationships” to the final summary (numeric↔numeric only).
- **Rules of thumb** — For |r|: ~0.1 weak, ~0.3 moderate, ≥0.5 strong. For 0–1 metrics: ~0.2 weak, ~0.4 moderate, ≥0.6 strong (context matters).
- **Note** — Stats use rows where **both** columns are non-missing; outliers/rare categories can affect results.
            """
        )

    # Method picker
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
    )

    # Optional sampling for speed on large data
    work = df.sample(min(sample_n, len(df)), random_state=0) if sample_n else df

    corr = None
    title = ""

    # --- Numeric-only: Pearson / Spearman / Kendall ---
    if method in ("Pearson (numeric)", "Spearman (numeric)", "Kendall (numeric)"):
        num_df = work.select_dtypes(include=[np.number])
        if num_df.shape[1] < 2:
            st.info("Need at least two numeric columns.")
        else:
            alg = {
                "Pearson (numeric)": "pearson",
                "Spearman (numeric)": "spearman",
                "Kendall (numeric)": "kendall",
            }[method]
            corr = num_df.corr(method=alg, numeric_only=True)
            title = f"Correlation heatmap ({method.split()[0]})"

    # --- Categorical-only: Cramér’s V ---
    elif method == "Cramér’s V (categorical)":
        cat_cols = [c for c in work.columns if not pd.api.types.is_numeric_dtype(work[c])]
        if len(cat_cols) < 2:
            st.info("Need at least two categorical columns.")
        else:
            sel = st.multiselect("Columns", cat_cols, default=cat_cols, key="corr_cat_cols")
            if len(sel) >= 2:
                cols = sel
                m = np.zeros((len(cols), len(cols)), dtype=float)
                for i, c1 in enumerate(cols):
                    for j, c2 in enumerate(cols[i:], start=i):
                        v = _cramers_v(_as_cat(work[c1]), _as_cat(work[c2]))
                        m[i, j] = m[j, i] = v
                corr = pd.DataFrame(m, index=cols, columns=cols)
                title = "Cramér’s V heatmap (categorical)"

    # --- Mixed: Auto routing ---
    else:  # Auto (mixed types)
        cols_all = list(work.columns)
        default_cols = cols_all[: min(20, len(cols_all))]
        sel = st.multiselect("Columns", cols_all, default=default_cols, key="corr_auto_cols")
        if len(sel) >= 2:
            cols = sel
            m = np.zeros((len(cols), len(cols)), dtype=float)
            for i, c1 in enumerate(cols):
                s1 = work[c1]
                for j, c2 in enumerate(cols[i:], start=i):
                    s2 = work[c2]
                    v = _mixed_pair(s1, s2)
                    if not np.isnan(v):
                        v = float(max(-1.0, min(1.0, v)))  # clip to [-1,1] for display
                    m[i, j] = m[j, i] = v
            corr = pd.DataFrame(m, index=cols, columns=cols)
            title = "Mixed-type association heatmap (Auto)"

    # Render heatmap if we produced a matrix
    if corr is not None and corr.shape[1] >= 2:
        _plot_heatmap(corr, title)

    # Save “top correlations” overview to Final Summary
    st.markdown("---")
    min_abs = st.slider(
        "Min |r| to save (for numeric pairs)",
        0.0, 1.0, 0.60, 0.05,
        key="corr_min_abs_to_save"
    )
    if st.button("💾 Save correlation insights for this table", key="corr_save_all"):
        ds_name = st.session_state.get("active_ds") or "dataset"
        n = save_correlation_overview(
            st.session_state, ds_name, df, min_abs=float(min_abs)
        )
        st.success(f"Saved {n} correlation insight(s) for '{ds_name}'.")
        saved_badge()
