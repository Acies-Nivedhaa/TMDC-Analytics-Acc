# core/preprocess_timeseries.py
from __future__ import annotations
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from ui.components import section, render_table, kpi_row


def _infer_freq_safe(ts: pd.Series) -> str | None:
    try:
        f = pd.infer_freq(ts.sort_values().dropna().unique())
        return f
    except Exception:
        return None


def _trend_slope(y: pd.Series) -> float:
    y = pd.to_numeric(y, errors="coerce").astype(float)
    y = y.dropna()
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=float)
    try:
        m, b = np.polyfit(x, y.values, 1)
        return float(m)
    except Exception:
        return 0.0


def _cov(y: pd.Series) -> float:
    y = pd.to_numeric(y, errors="coerce")
    if y.count() == 0:
        return 0.0
    mu = y.mean()
    if mu == 0 or np.isnan(mu):
        return 0.0
    return float(y.std(ddof=0) / mu)


def _gap_report(dt: pd.Series) -> pd.DataFrame:
    s = pd.to_datetime(dt, errors="coerce").sort_values().dropna()
    if len(s) < 2:
        return pd.DataFrame(columns=["from", "to", "delta"])
    d = s.diff()
    med = d.median()
    if pd.isna(med) or med == pd.Timedelta(0):
        thresh = d.max()
    else:
        thresh = 3 * med
    gaps = d[d > thresh]
    if gaps.empty:
        return pd.DataFrame(columns=["from", "to", "delta"])
    out = pd.DataFrame({
        "from": s.loc[gaps.index - 1].values,
        "to": s.loc[gaps.index].values,
        "delta": gaps.values
    })
    return out.reset_index(drop=True)


def _line_with_rolling(df: pd.DataFrame, dt_col: str, val_col: str, window: int = 30):
    tmp = df[[dt_col, val_col]].copy()
    tmp = tmp.dropna(subset=[dt_col]).sort_values(dt_col)
    tmp["rolling_mean"] = pd.to_numeric(tmp[val_col], errors="coerce").rolling(
        window, min_periods=max(3, window // 3)
    ).mean()

    y = pd.to_numeric(tmp[val_col], errors="coerce")
    idx = np.arange(len(y))
    m, b = (0.0, 0.0)
    if y.notna().sum() >= 2:
        try:
            m, b = np.polyfit(idx[y.notna()], y[y.notna()], 1)
            tmp["trend"] = m * idx + b
        except Exception:
            tmp["trend"] = np.nan
    else:
        tmp["trend"] = np.nan

    base = alt.Chart(tmp).mark_line().encode(
        x=alt.X(dt_col, title="Date"),
        y=alt.Y(val_col, title=val_col)
    )
    roll = alt.Chart(tmp).mark_line(strokeDash=[4, 3]).encode(
        x=dt_col, y="rolling_mean", color=alt.value("#6baed6")
    )
    trend = alt.Chart(tmp).mark_line(color="#e34f4f").encode(
        x=dt_col, y="trend"
    )
    return (base + roll + trend).properties(height=280, title="Value with Rolling Mean and Trend")


def _seasonality_bars(df: pd.DataFrame, dt_col: str, val_col: str | None, how: str = "count"):
    sdt = pd.to_datetime(df[dt_col], errors="coerce")
    tmp = pd.DataFrame({"dt": sdt})
    tmp["v"] = pd.to_numeric(df[val_col], errors="coerce") if (how == "sum" and val_col) else 1.0

    by_month = tmp.dropna().assign(month=lambda d: d["dt"].dt.month).groupby("month")["v"].agg("sum").reset_index()
    by_wday = tmp.dropna().assign(weekday=lambda d: d["dt"].dt.dayofweek).groupby("weekday")["v"].agg("sum").reset_index()
    by_dom = tmp.dropna().assign(dom=lambda d: d["dt"].dt.day).groupby("dom")["v"].agg("sum").reset_index()

    c1 = alt.Chart(by_month).mark_bar().encode(
        x=alt.X("month:O", title="Month (1–12)"),
        y=alt.Y("v:Q", title="Coverage / Sum")
    ).properties(height=200, title="Coverage by Month")

    c2 = alt.Chart(by_wday).mark_bar().encode(
        x=alt.X("weekday:O", title="Weekday (0=Mon … 6=Sun)"),
        y=alt.Y("v:Q", title="Coverage / Sum")
    ).properties(height=200, title="Coverage by Weekday")

    c3 = alt.Chart(by_dom).mark_bar().encode(
        x=alt.X("dom:O", title="Day of Month"),
        y=alt.Y("v:Q", title="Coverage / Sum")
    ).properties(height=200, title="Coverage by Day of Month")

    return c1, c2, c3


# core/preprocess_timeseries.py  (only function shown below)
def render_preprocess_timeseries(ss) -> None:
    """Preprocess ▸ Time Series — diagnostics, parsing, calendar, rolling."""
    if not ss.active_ds or ss.active_ds not in ss.datasets:
        st.info("Pick a dataset to begin.")
        st.stop()

    df = ss.datasets[ss.active_ds]

    # ---------- helpers ----------
    def _safe_index(options: list[str], preferred: str | None) -> int:
        try:
            return options.index(preferred) if preferred in options else 0
        except Exception:
            return 0

    def _numeric_cols(d: pd.DataFrame) -> list[str]:
        return [c for c in d.columns if is_numeric_dtype(d[c])]

    def _datetime_cols(d: pd.DataFrame) -> list[str]:
        return [c for c in d.columns if is_datetime64_any_dtype(d[c])]

    def _ensure_dt(series: pd.Series) -> pd.Series:
        return pd.to_datetime(series, errors="coerce")

    def _fmt_dt(x) -> str:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "—"
        try:
            return pd.to_datetime(x).strftime("%Y-%m-%d")
        except Exception:
            return str(x)

    # ---------- picks & quick stats ----------
    cols_dt = _datetime_cols(df)
    cols_num = _numeric_cols(df)

    prev_dt = st.session_state.get("ts_last_dt_col")
    prev_val = st.session_state.get("ts_last_val_col")

    dt_options = cols_dt or df.columns.tolist()
    dt_col = st.selectbox("Datetime column", dt_options, index=_safe_index(dt_options, prev_dt))
    st.session_state["ts_last_dt_col"] = dt_col

    val_options = cols_num or df.columns.tolist()
    value_col = st.selectbox("Value column for diagnostics", val_options, index=_safe_index(val_options, prev_val))
    st.session_state["ts_last_val_col"] = value_col

    s_dt = _ensure_dt(df[dt_col])
    s_val = pd.to_numeric(df[value_col], errors="coerce")

    ts = (
        pd.DataFrame({dt_col: s_dt, value_col: s_val})
        .dropna(subset=[dt_col])
        .sort_values(dt_col)
        .set_index(dt_col)[value_col]
    )

    if not ts.empty:
        start, end = ts.index.min(), ts.index.max()
        try:
            inferred = pd.infer_freq(ts.index)
        except Exception:
            inferred = None
        mu = ts.mean(skipna=True)
        cov = float(ts.std(skipna=True) / mu) if pd.notna(mu) and mu != 0 else np.nan
        y = ts.values.astype("float64")
        msk = ~np.isnan(y)
        slope = float(np.polyfit(np.arange(len(ts))[msk], y[msk], 1)[0]) if msk.sum() > 1 else np.nan
    else:
        start = end = inferred = cov = slope = None

    kpi_row([
        ("Start", _fmt_dt(start)),
        ("End", _fmt_dt(end)),
        ("Inferred freq", inferred or "—"),
        ("CoV", f"{cov:.2f}" if cov is not None and pd.notna(cov) else "—"),
        ("Trend slope", f"{slope:.5f}" if slope is not None and pd.notna(slope) else "—"),
    ])

    # ---------- visuals ----------
    with section("Visuals: seasonality & trend", expandable=True):
        if ts.empty:
            st.info("No valid time series after parsing/cleaning.")
        else:
            w = max(5, min(30, int(len(ts) * 0.05)))
            roll = ts.rolling(w, min_periods=max(3, w // 3)).mean()
            idx = np.arange(len(ts))
            coefs = np.polyfit(idx, ts.to_numpy(dtype=float), 1)
            trend_line = pd.Series(coefs[0] * idx + coefs[1], index=ts.index)

            chart_df = pd.DataFrame({"value": ts, "rolling_mean": roll, "trend": trend_line})
            st.line_chart(chart_df)

            st.caption("Seasonality coverage (share of months & weekdays present)")
            c1, c2 = st.columns(2)
            with c1:
                by_month = ts.groupby(ts.index.month).size()
                st.bar_chart((by_month / by_month.max()).rename(index=lambda m: str(m)))
            with c2:
                by_wday = ts.groupby(ts.index.dayofweek).size()
                st.bar_chart((by_wday / by_wday.max()).rename(index=lambda d: str(d)))

    # =========================
    # 1) PARSE TO DATETIME
    # =========================
    with section("1) Parse to datetime", expandable=True):
        src_options = df.columns.tolist()
        prev_src = st.session_state.get("ts_parse_src")
        src_col = st.selectbox("Source column", src_options, index=_safe_index(src_options, prev_src))
        st.session_state["ts_parse_src"] = src_col

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            day_first = st.checkbox("Interpret day first (e.g., 02/01 = 2 Jan)", value=False, key="ts_dayfirst")
        with c2:
            as_utc = st.checkbox("Parse as UTC", value=False, key="ts_as_utc")
        with c3:
            replace_src = st.checkbox("Replace original column", value=False, key="ts_replace_src")

        out_name_default = f"{src_col}_dt"
        out_name = st.text_input("Parsed column name (if not replacing)", value=out_name_default, disabled=replace_src)

        cprev, capply = st.columns([1, 1])
        with cprev:
            if st.button("Preview parsing", key="ts_preview_parse"):
                prev = pd.to_datetime(df[src_col], errors="coerce", dayfirst=day_first, utc=as_utc)
                st.dataframe(pd.DataFrame({"raw": df[src_col].head(20), "parsed": prev.head(20)}), use_container_width=True)
        with capply:
            if st.button("Apply parsing", key="ts_apply_parse"):
                ss.df_history.append(df.copy())
                parsed = pd.to_datetime(df[src_col], errors="coerce", dayfirst=day_first, utc=as_utc)
                if replace_src:
                    df[src_col] = parsed
                else:
                    df[out_name or out_name_default] = parsed
                ss.datasets[ss.active_ds] = df
                st.success("Datetime parsing applied.")

    # =========================
    # 2) CALENDAR FEATURES
    # =========================
    with section("2) Calendar features", expandable=True):
        cal_dt_options = _datetime_cols(df) or [dt_col]
        cal_dt = st.selectbox("Datetime column for features", cal_dt_options, index=0, key="ts_feat_dt")

        feats_all = ["year", "quarter", "month", "week", "dayofweek", "day", "is_month_end", "is_month_start"]
        pick = st.multiselect("Features to add", feats_all, default=["year", "quarter", "month", "dayofweek"])

        def _make_feats(idx: pd.DatetimeIndex) -> pd.DataFrame:
            out = {}
            if "year" in pick: out["year"] = idx.year
            if "quarter" in pick: out["quarter"] = idx.quarter
            if "month" in pick: out["month"] = idx.month
            if "week" in pick:
                try:
                    out["week"] = idx.isocalendar().week.values
                except Exception:
                    out["week"] = idx.week
            if "dayofweek" in pick: out["dayofweek"] = idx.dayofweek
            if "day" in pick: out["day"] = idx.day
            if "is_month_end" in pick: out["is_month_end"] = idx.is_month_end.astype(int)
            if "is_month_start" in pick: out["is_month_start"] = idx.is_month_start.astype(int)
            return pd.DataFrame(out)

        cprev, capply = st.columns([1, 1])
        with cprev:
            if st.button("Preview features", key="ts_feat_preview"):
                idx = _ensure_dt(df[cal_dt])
                st.dataframe(_make_feats(idx).head(20), use_container_width=True)
        with capply:
            if st.button("Add selected features", key="ts_feat_apply"):
                ss.df_history.append(df.copy())
                idx = _ensure_dt(df[cal_dt])
                add = _make_feats(idx)
                for c in add.columns:
                    df[f"{cal_dt}_{c}"] = add[c].values
                ss.datasets[ss.active_ds] = df
                st.success(f"Added {len(add.columns)} feature(s).")

    # =========================
    # 3) ROLLING FEATURES
    # =========================
    with section("3) Rolling features (row-based windows)", expandable=True):
        roll_dt_options = [dt_col] + [c for c in _datetime_cols(df) if c != dt_col] if _datetime_cols(df) else [dt_col]
        order_col = st.selectbox("Time column (used for ordering)", roll_dt_options, index=0)
        num_cols = st.multiselect("Numeric columns", _numeric_cols(df), default=_numeric_cols(df)[:1])
        group_by = st.selectbox("Group by (optional roll-up level)", ["—"] + df.columns.tolist(), index=0)
        windows = st.multiselect("Windows (rows)", [3, 7, 14, 30, 60, 90], default=[3, 7, 14])
        funcs = st.multiselect("Functions", ["mean", "std", "min", "max", "sum"], default=["mean", "std"])
        overwrite = st.checkbox("Overwrite if output columns already exist", value=False, key="ts_roll_overwrite")

        def _make_rolling(dfin: pd.DataFrame) -> pd.DataFrame:
            d = dfin.copy()
            d[order_col] = _ensure_dt(d[order_col])
            d = d.sort_values(order_col)
            if group_by != "—":
                gb = d.groupby(group_by, group_keys=False)
                for w in windows:
                    for f in funcs:
                        rolled = gb[num_cols].rolling(window=w, min_periods=max(2, w // 3)).agg(f).reset_index(level=0, drop=True)
                        for c in num_cols:
                            d[f"{c}_roll{w}_{f}"] = rolled[c].values
            else:
                for w in windows:
                    for f in funcs:
                        rolled = d[num_cols].rolling(window=w, min_periods=max(2, w // 3)).agg(f)
                        for c in num_cols:
                            d[f"{c}_roll{w}_{f}"] = rolled[c].values
            return d

        cprev, capply = st.columns([1, 1])
        with cprev:
            if st.button("Preview rolling", key="ts_roll_preview"):
                prev = _make_rolling(df).head(20)
                st.dataframe(prev, use_container_width=True)
        with capply:
            if st.button("Add rolling features", key="ts_roll_apply"):
                out = _make_rolling(df)
                ss.df_history.append(df.copy())
                if overwrite:
                    ss.datasets[ss.active_ds] = out
                else:
                    for c in out.columns:
                        if c not in df.columns:
                            df[c] = out[c]
                    ss.datasets[ss.active_ds] = df
                st.success("Rolling features added.")
