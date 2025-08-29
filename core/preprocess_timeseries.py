from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
from ui.components import section, kpi_row

def render_preprocess_timeseries(ss) -> None:
    if not ss.active_ds or ss.active_ds not in ss.datasets:
        st.info("Pick a dataset to begin."); st.stop()
    df = ss.datasets[ss.active_ds]

    def _numeric_cols(d):  return [c for c in d.columns if is_numeric_dtype(d[c])]
    def _datetime_cols(d): return [c for c in d.columns if is_datetime64_any_dtype(d[c])]
    def _ensure_dt(s):     return pd.to_datetime(s, errors="coerce")

    # ---- picks (robust to renamed/missing columns) ----
    cols_dt  = _datetime_cols(df)
    cols_num = _numeric_cols(df)

    prev_dt  = ss.get("ts_last_dt_col")
    prev_val = ss.get("ts_last_val_col")

    dt_opts  = cols_dt or df.columns.tolist()
    dt_idx   = dt_opts.index(prev_dt) if prev_dt in dt_opts else 0
    dt_col   = st.selectbox("Datetime column", dt_opts, index=dt_idx, key="ts_pick_dt")

    val_opts = cols_num or df.columns.tolist()
    val_idx  = val_opts.index(prev_val) if prev_val in val_opts else 0
    value_col = st.selectbox("Value column for diagnostics", val_opts, index=val_idx, key="ts_pick_val")

    ss["ts_last_dt_col"]  = dt_col
    ss["ts_last_val_col"] = value_col

    # ---- build safe series (never KeyError) ----
    s_dt  = _ensure_dt(df[dt_col]) if dt_col in df.columns else pd.to_datetime(pd.Series([], dtype="float64"))
    s_raw = df[value_col] if value_col in df.columns else pd.Series([], dtype="float64")
    s_val = pd.to_numeric(s_raw, errors="coerce")

    tmp = pd.DataFrame({dt_col: s_dt, value_col: s_val})
    tmp = tmp.dropna(subset=[dt_col]).sort_values(dt_col)

    try:
        ts = tmp.set_index(dt_col)[value_col]
    except Exception:
        ts = pd.Series([], dtype="float64")

    # ---- KPIs ----
    if not ts.empty:
        start, end = ts.index.min(), ts.index.max()
        try: inferred = pd.infer_freq(ts.index)
        except Exception: inferred = None
        mu = ts.mean(skipna=True)
        cov = float(ts.std(skipna=True) / mu) if pd.notna(mu) and mu != 0 else np.nan
        y = ts.to_numpy(dtype=float)
        mask = ~np.isnan(y)
        slope = float(np.polyfit(np.arange(len(ts))[mask], y[mask], 1)[0]) if mask.sum() > 1 else np.nan
    else:
        start = end = inferred = cov = slope = None

    def _fmt_dt(x):
        try:
            return pd.to_datetime(x).strftime("%Y-%m-%d")
        except Exception:
            return "—"

    kpi_row([
        ("Start", _fmt_dt(start)), ("End", _fmt_dt(end)),
        ("Inferred freq", inferred or "—"),
        ("CoV", f"{cov:.2f}" if pd.notna(cov) else "—"),
        ("Trend slope", f"{slope:.5f}" if pd.notna(slope) else "—"),
    ])

    # ---- visuals ----
    with section("Visuals: seasonality & trend", expandable=True):
        if ts.empty:
            st.info("No valid time series after parsing/cleaning.")
        else:
            w = max(5, min(30, int(len(ts) * 0.05)))
            roll = ts.rolling(w, min_periods=max(3, w // 3)).mean()
            idx = np.arange(len(ts))
            coefs = np.polyfit(idx, ts.to_numpy(float), 1)
            trend_line = pd.Series(coefs[0] * idx + coefs[1], index=ts.index)
            st.line_chart(pd.DataFrame({"value": ts, "rolling_mean": roll, "trend": trend_line}))
            st.caption("Seasonality coverage (share of months & weekdays present)")
            c1, c2 = st.columns(2)
            with c1:
                by_month = ts.groupby(ts.index.month).size()
                st.bar_chart((by_month / by_month.max()).rename(index=str))
            with c2:
                by_wday = ts.groupby(ts.index.dayofweek).size()
                st.bar_chart((by_wday / by_wday.max()).rename(index=str))

    # ---- 1) Parse to datetime ----
    with section("1) Parse to datetime", expandable=True):
        src_opts = df.columns.tolist()
        src_idx  = src_opts.index(ss.get("ts_parse_src")) if ss.get("ts_parse_src") in src_opts else 0
        src_col  = st.selectbox("Source column", src_opts, index=src_idx, key="ts_src_col")
        ss["ts_parse_src"] = src_col

        c1, c2, c3 = st.columns([1,1,1])
        with c1: day_first   = st.checkbox("Interpret day first (e.g., 02/01 = 2 Jan)", value=False, key="ts_dayfirst")
        with c2: as_utc      = st.checkbox("Parse as UTC", value=False, key="ts_asutc")
        with c3: replace_src = st.checkbox("Replace original column", value=False, key="ts_replacesrc")

        out_name_default = f"{src_col}_dt"
        out_name = st.text_input("Parsed column name (if not replacing)", value=out_name_default, disabled=replace_src, key="ts_parse_outname")

        cprev, capply = st.columns([1,1])
        with cprev:
            if st.button("Preview parsing", key="ts_preview_parse"):
                prev = pd.to_datetime(df[src_col], errors="coerce", dayfirst=day_first, utc=as_utc)
                st.dataframe(pd.DataFrame({"raw": df[src_col].head(20), "parsed": prev.head(20)}), use_container_width=True)
        with capply:
            if st.button("Apply parsing", key="ts_apply_parse"):
                ss.df_history.append(df.copy())
                parsed = pd.to_datetime(df[src_col], errors="coerce", dayfirst=day_first, utc=as_utc)
                if replace_src: df[src_col] = parsed
                else:           df[out_name or out_name_default] = parsed
                ss.datasets[ss.active_ds] = df
                st.success("Datetime parsing applied.")

    # ---- 2) Calendar features ----
    with section("2) Calendar features", expandable=True):
        cal_dt_opts = _datetime_cols(df) or [dt_col]
        cal_dt = st.selectbox("Datetime column for features", cal_dt_opts, index=0, key="ts_feat_dt")
        feats_all = ["year","quarter","month","week","dayofweek","day","is_month_end","is_month_start"]
        pick = st.multiselect("Features to add", feats_all, default=["year","quarter","month","dayofweek"], key="ts_feat_pick")

        def _make_feats(idx: pd.DatetimeIndex) -> pd.DataFrame:
            out = {}
            if "year" in pick: out["year"] = idx.year
            if "quarter" in pick: out["quarter"] = idx.quarter
            if "month" in pick: out["month"] = idx.month
            if "week" in pick:
                try: out["week"] = idx.isocalendar().week.values
                except Exception: out["week"] = idx.week
            if "dayofweek" in pick: out["dayofweek"] = idx.dayofweek
            if "day" in pick: out["day"] = idx.day
            if "is_month_end" in pick: out["is_month_end"] = idx.is_month_end.astype(int)
            if "is_month_start" in pick: out["is_month_start"] = idx.is_month_start.astype(int)
            return pd.DataFrame(out)

        cprev, capply = st.columns([1,1])
        with cprev:
            if st.button("Preview features", key="ts_feat_preview"):
                idx = _ensure_dt(df[cal_dt]); st.dataframe(_make_feats(idx).head(20), use_container_width=True)
        with capply:
            if st.button("Add selected features", key="ts_feat_apply"):
                ss.df_history.append(df.copy())
                idx = _ensure_dt(df[cal_dt]); add = _make_feats(idx)
                for c in add.columns: df[f"{cal_dt}_{c}"] = add[c].values
                ss.datasets[ss.active_ds] = df
                st.success(f"Added {len(add.columns)} feature(s).")

    # ---- 3) Rolling features ----
    with section("3) Rolling features (row-based windows)", expandable=True):
        roll_dt_opts = _datetime_cols(df) or [dt_col]
        order_col = st.selectbox("Time column (used for ordering)", roll_dt_opts, index=0, key="ts_roll_order")
        num_cols = st.multiselect("Numeric columns", _numeric_cols(df), default=_numeric_cols(df)[:1], key="ts_roll_nums")
        group_by = st.selectbox("Group by (optional roll-up level)", ["—"] + df.columns.tolist(), index=0, key="ts_roll_group")
        windows = st.multiselect("Windows (rows)", [3,7,14,30,60,90], default=[3,7,14], key="ts_roll_windows")
        funcs = st.multiselect("Functions", ["mean","std","min","max","sum"], default=["mean","std"], key="ts_roll_funcs")
        overwrite = st.checkbox("Overwrite if output columns already exist", value=False, key="ts_roll_overwrite")

        def _make_rolling(dfin: pd.DataFrame) -> pd.DataFrame:
            d = dfin.copy()
            d[order_col] = pd.to_datetime(d[order_col], errors="coerce")
            d = d.sort_values(order_col)
            if group_by != "—":
                gb = d.groupby(group_by, group_keys=False)
                for w in windows:
                    for f in funcs:
                        rolled = gb[num_cols].rolling(window=w, min_periods=max(2, w//3)).agg(f).reset_index(level=0, drop=True)
                        for c in num_cols: d[f"{c}_roll{w}_{f}"] = rolled[c].values
            else:
                for w in windows:
                    for f in funcs:
                        rolled = d[num_cols].rolling(window=w, min_periods=max(2, w//3)).agg(f)
                        for c in num_cols: d[f"{c}_roll{w}_{f}"] = rolled[c].values
            return d

        cprev, capply = st.columns([1,1])
        with cprev:
            if st.button("Preview rolling", key="ts_roll_preview"):
                prev = _make_rolling(df).head(20); st.dataframe(prev, use_container_width=True)
        with capply:
            if st.button("Add rolling features", key="ts_roll_apply"):
                out = _make_rolling(df); ss.df_history.append(df.copy())
                if overwrite: ss.datasets[ss.active_ds] = out
                else:
                    for c in out.columns:
                        if c not in df.columns: df[c] = out[c]
                    ss.datasets[ss.active_ds] = df
                st.success("Rolling features added.")
