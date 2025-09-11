# core/eda_bivariate.py
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from pandas.api.types import is_numeric_dtype, is_bool_dtype, is_datetime64_any_dtype
from ui.components import section, render_table, kpi_row

# -------- type guards --------
def _is_cat(s: pd.Series) -> bool:
    return (str(s.dtype) in ("object", "string", "category")) or is_bool_dtype(s)

def _is_time(s: pd.Series) -> bool:
    if is_datetime64_any_dtype(s):
        return True
    try:
        p = pd.to_datetime(s, errors="coerce")
        return p.notna().mean() >= 0.5  # at least half parse as datetime
    except Exception:
        return False

def _to_time(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _safe_str(s: pd.Series) -> pd.Series:
    return s.astype("string")

# ---------------- time helpers ----------------
_FREQ_LABELS = {
    "D": "Daily", "W": "Weekly", "M": "Monthly", "Q": "Quarterly"
}

def _infer_freq(dt: pd.Series) -> str:
    idx = pd.DatetimeIndex(dt.dropna().sort_values().unique())
    if len(idx) < 3:
        return "D"
    try:
        f = pd.infer_freq(idx)
    except Exception:
        f = None
    # map pandas granular codes to our menu
    if f and f.startswith("Q"):
        return "Q"
    if f and f.startswith("M"):
        return "M"
    if f and f.startswith("W"):
        return "W"
    return "D"

# -------------- charts: category × category --------------
def _cat_cat_bars(df: pd.DataFrame, x: str, series: str, *, top_x: int, top_series: int,
                  stacked: bool, normalize_pct: bool, sort_desc: bool):
    d = (
        df.assign(_x=_safe_str(df[x]), _s=_safe_str(df[series]))
          .dropna(subset=["_x", "_s"])
    )
    if d.empty:
        st.caption("No data to plot."); return
    topx = d["_x"].value_counts().head(top_x).index
    tops = d["_s"].value_counts().head(top_series).index
    d = d[d["_x"].isin(topx) & d["_s"].isin(tops)]

    agg = d.groupby(["_x", "_s"]).size().rename("count").reset_index()
    tot = agg.groupby("_x")["count"].transform("sum")
    agg["pct"] = agg["count"] / tot.replace(0, np.nan)
    x_sort = agg.groupby("_x")["count"].sum().sort_values(ascending=not sort_desc).index.tolist()

    with section("Counts (category vs category)", expandable=False):
        wide = agg.pivot(index="_x", columns="_s", values="count").fillna(0).astype(int)
        wide.index.name = x; wide.columns.name = series
        render_table(wide.reset_index(), height=300)

        y_field, y_title, y_kwargs = ("pct:Q", "Share within X", dict(scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"))) \
            if normalize_pct else ("count:Q", "Count", {})
        chart = (
            alt.Chart(agg).mark_bar()
            .encode(
                x=alt.X("_x:N", sort=x_sort, title=x),
                y=alt.Y(y_field, title=y_title, stack=("zero" if stacked else None), **y_kwargs),
                color=alt.Color("_s:N", title=series),
                tooltip=[alt.Tooltip("_x:N", title=x),
                         alt.Tooltip("_s:N", title=series),
                         alt.Tooltip("count:Q", title="Count"),
                         alt.Tooltip("pct:Q", title="Share", format=".1%")],
            )
            .properties(height=380, title=f"{series} distribution across top {len(x_sort)} {x} levels")
        ).interactive()
        st.altair_chart(chart, use_container_width=True)

# -------------- charts: category × numeric --------------
def _cat_num_bars(df: pd.DataFrame, cat: str, num: str, *, top_k: int, sort_by: str):
    d = pd.DataFrame({cat: _safe_str(df[cat]), num: _safe_num(df[num])}).dropna()
    if d.empty:
        st.caption("No data to plot."); return
    top = d[cat].value_counts().head(top_k).index
    d = d[d[cat].isin(top)]

    agg = d.groupby(cat)[num].agg(["mean", "count", "std"]).reset_index()
    z = 1.96
    agg["ci"] = z * (agg["std"] / np.sqrt(agg["count"].clip(lower=1)))
    agg["lo"] = agg["mean"] - agg["ci"]
    agg["hi"] = agg["mean"] + agg["ci"]

    order = "mean" if sort_by == "Mean" else "count"
    x_sort = agg.sort_values(order, ascending=False)[cat].tolist()

    with section("Group mean (with 95% CI)", expandable=False):
        kpi_row([("Groups", len(agg)), ("Rows (non-null)", f"{len(d):,}")])
        base = alt.Chart(agg).encode(x=alt.X(f"{cat}:N", sort=x_sort, title=cat))
        bars = base.mark_bar().encode(
            y=alt.Y("mean:Q", title=f"mean({num})"),
            tooltip=[cat, alt.Tooltip("mean:Q", title=f"mean({num})"), alt.Tooltip("count:Q", title="n")]
        )
        err = base.mark_errorbar().encode(y="lo:Q", y2="hi:Q")
        st.altair_chart((bars + err).properties(height=380), use_container_width=True)

# -------------- charts: numeric × numeric --------------
def _num_num_scatter(df: pd.DataFrame, x: str, y: str, *, sample_n: int):
    d = pd.DataFrame({x: _safe_num(df[x]), y: _safe_num(df[y])}).dropna()
    if len(d) > sample_n:
        d = d.sample(sample_n, random_state=7)
    with section("Scatter with trend", expandable=False):
        if d.empty:
            st.caption("No data to plot."); return
        r = float(d[x].corr(d[y], method="pearson"))
        rho = float(d[x].corr(d[y], method="spearman"))
        st.caption(f"Pearson r: **{r:.3f}** • Spearman ρ: **{rho:.3f}**")
        base = alt.Chart(d)
        pts = base.mark_circle(opacity=0.55, size=36).encode(
            x=alt.X(f"{x}:Q", title=x),
            y=alt.Y(f"{y}:Q", title=y),
            tooltip=[alt.Tooltip(x, type="quantitative"), alt.Tooltip(y, type="quantitative")],
        )
        trend = base.transform_regression(x, y, method="linear").mark_line()
        st.altair_chart((pts + trend).properties(height=380), use_container_width=True)

# -------------- charts: time × numeric --------------
def _time_num_line(df: pd.DataFrame, tcol: str, num: str, *, freq: str, agg: str,
                   last_n: int, roll_w: int):
    d = pd.DataFrame({"_t": _to_time(df[tcol]), "_y": _safe_num(df[num])}).dropna(subset=["_t"])
    if d.empty:
        st.caption("No data to plot."); return

    # resample
    s = d.set_index("_t")["_y"].resample(freq).agg(agg)
    if last_n > 0:
        s = s.tail(last_n)
    out = s.reset_index().rename(columns={"_t": tcol, "_y": num, 0: agg})
    out.columns = ["period", "value"]  # simpler for Altair
    if roll_w >= 2:
        out["rolling_mean"] = out["value"].rolling(roll_w, min_periods=max(2, roll_w // 2)).mean()

    with section(f"{_FREQ_LABELS.get(freq, freq)} {agg} over time", expandable=False):
        kpi_row([
            ("Start", out["period"].min().strftime("%Y-%m-%d") if len(out) else "—"),
            ("End", out["period"].max().strftime("%Y-%m-%d") if len(out) else "—"),
            ("Points", len(out)),
        ])
        base = alt.Chart(out)
        line = base.mark_line(point=True).encode(
            x=alt.X("period:T", title=tcol),
            y=alt.Y("value:Q", title=f"{agg}({num})"),
            tooltip=[alt.Tooltip("period:T", title=tcol),
                     alt.Tooltip("value:Q", title=f"{agg}({num})", format=",.0f")]
        )
        if "rolling_mean" in out:
            roll = base.mark_line(strokeDash=[4, 3]).encode(
                x="period:T", y=alt.Y("rolling_mean:Q", title=f"{agg}({num})"),
                tooltip=[alt.Tooltip("rolling_mean:Q", title="Rolling mean", format=",.0f")]
            )
            chart = line + roll
        else:
            chart = line
        st.altair_chart(chart.properties(height=380, title=f"{tcol} ({_FREQ_LABELS.get(freq,freq)}) vs {agg} of {num}"),
                        use_container_width=True)

# -------------- charts: time × categorical --------------
def _time_cat_area(df: pd.DataFrame, tcol: str, cat: str, *, freq: str,
                   top_k: int, normalize: bool, last_n: int):
    d = pd.DataFrame({"_t": _to_time(df[tcol]), "_c": _safe_str(df[cat])}).dropna(subset=["_t"])
    if d.empty:
        st.caption("No data to plot."); return

    top = d["_c"].value_counts().head(top_k).index
    d = d[d["_c"].isin(top)]
    g = (d.set_index("_t")
           .groupby([pd.Grouper(freq=freq), "_c"])
           .size()
           .rename("count")
           .reset_index())
    if last_n > 0:
        # keep last N periods per whole frame
        last_periods = g["_t"].drop_duplicates().sort_values().tail(last_n)
        g = g[g["_t"].isin(last_periods)]
    g = g.rename(columns={"_t": "period", "_c": cat})
    # normalise within each period if requested
    if normalize:
        g["share"] = g["count"] / g.groupby("period")["count"].transform("sum").replace(0, np.nan)

    with section(f"{_FREQ_LABELS.get(freq, freq)} distribution of {cat}", expandable=False):
        field = "share:Q" if normalize else "count:Q"
        y_title = "% within period" if normalize else "Count"
        base = alt.Chart(g)
        area = base.mark_area().encode(
            x=alt.X("period:T", title=tcol),
            y=alt.Y(field, stack="normalize" if normalize else "zero", title=y_title,
                    axis=alt.Axis(format="%" if normalize else None)),
            color=alt.Color(f"{cat}:N", title=cat),
            tooltip=[
                alt.Tooltip("period:T", title=tcol),
                alt.Tooltip(f"{cat}:N", title=cat),
                alt.Tooltip("count:Q", title="Count", format=",.0f"),
                alt.Tooltip("share:Q", title="Share", format=".1%")
            ],
        )
        st.altair_chart(area.properties(height=380, title=f"{tcol} → {cat} (top {top_k})"),
                        use_container_width=True)

# ---------------- public ----------------
def render_bivariate(df: pd.DataFrame, *, sample_n: int = 5000) -> None:
    """
    EDA ▸ Bivariate
    - time×numeric: resampled line (sum/mean/…)
    - time×categorical: stacked area (counts or %)
    - cat×cat: grouped/stacked bars (counts; optional % within X)
    - cat×num: group mean bars + 95% CI
    - num×num: scatter with linear trend
    """
    if df is None or df.empty:
        st.info("No data.")
        return

    # tiny glossary
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

    cols = list(df.columns)
    c1, c2 = st.columns(2)
    x = c1.selectbox("X", options=cols, index=0 if cols else 0, key="bi_x")
    y = c2.selectbox("Y", options=[c for c in cols if c != x] or cols, index=0, key="bi_y")
    if not x or not y: 
        return

    sx, sy = df[x], df[y]
    x_is_time = _is_time(sx)
    x_is_num, y_is_num = is_numeric_dtype(sx), is_numeric_dtype(sy)
    x_is_cat, y_is_cat = _is_cat(sx), _is_cat(sy)

    # ---- time on X ----
    if x_is_time and y_is_num:
        # Options
        with section("Options", expandable=False):
            freq_default = _infer_freq(_to_time(sx))
            cA, cB, cC, cD = st.columns(4)
            freq = cA.selectbox("Frequency", ["D", "W", "M", "Q"], index=["D","W","M","Q"].index(freq_default))
            agg = cB.selectbox("Aggregation", ["sum", "mean", "median", "min", "max"], index=0)
            last_n = cC.number_input("Last N periods (0 = all)", min_value=0, value=0, step=1)
            roll_w = cD.number_input("Rolling mean window", min_value=0, value=0, step=1, help="0 = disabled")
        _time_num_line(df, x, y, freq=freq, agg=agg, last_n=int(last_n), roll_w=int(roll_w))
        return

    if x_is_time and y_is_cat:
        with section("Options", expandable=False):
            freq_default = _infer_freq(_to_time(sx))
            cA, cB, cC, cD = st.columns(4)
            freq = cA.selectbox("Frequency", ["D", "W", "M", "Q"], index=["D","W","M","Q"].index(freq_default), key="tcat_freq")
            top_k = cB.number_input("Top categories", min_value=2, max_value=30, value=8, step=1)
            normalize = cC.checkbox("Normalize to % per period", value=False)
            last_n = cD.number_input("Last N periods (0 = all)", min_value=0, value=0, step=1, key="tcat_lastn")
        _time_cat_area(df, x, y, freq=freq, top_k=int(top_k), normalize=bool(normalize), last_n=int(last_n))
        return

    # ---- non-time cases (previous functionality) ----
    if x_is_cat and y_is_cat:
        with section("Options", expandable=False):
            colA, colB, colC, colD, colE = st.columns(5)
            top_x = colA.number_input("Top X levels", min_value=3, max_value=50, value=10, step=1)
            top_series = colB.number_input("Top series levels", min_value=2, max_value=30, value=8, step=1)
            stacked = colC.checkbox("Stacked", value=False)
            normalize_pct = colD.checkbox("% within X", value=False)
            sort_desc = colE.checkbox("Sort X by total desc", value=True)
        _cat_cat_bars(df, x, y, top_x=int(top_x), top_series=int(top_series),
                      stacked=bool(stacked), normalize_pct=bool(normalize_pct), sort_desc=bool(sort_desc))
        return

    if x_is_cat and y_is_num:
        with section("Options", expandable=False):
            colA, colB = st.columns(2)
            top_k = colA.number_input("Top categories", min_value=3, max_value=50, value=10, step=1)
            sort_by = colB.radio("Sort by", ["Mean", "Count"], horizontal=True)
        _cat_num_bars(df, x, y, top_k=int(top_k), sort_by=sort_by)
        return

    if x_is_num and y_is_cat:
        with section("Options", expandable=False):
            colA, colB = st.columns(2)
            top_k = colA.number_input("Top categories", min_value=3, max_value=50, value=10, step=1)
            sort_by = colB.radio("Sort by", ["Mean", "Count"], horizontal=True)
        _cat_num_bars(df, y, x, top_k=int(top_k), sort_by=sort_by)  # flip: cat first
        return

    if x_is_num and y_is_num:
        with section("Options", expandable=False):
            sample = st.number_input("Max points", min_value=500, max_value=200_000, value=int(sample_n), step=500)
        _num_num_scatter(df, x, y, sample_n=int(sample))
        return

    st.info("Selected columns aren’t plottable in a bivariate chart.")
