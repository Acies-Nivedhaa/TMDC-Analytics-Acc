# core/eda_types.py
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
from pandas.api.types import (
    is_numeric_dtype, is_datetime64_any_dtype,
    is_bool_dtype, is_categorical_dtype, is_integer_dtype, is_float_dtype
)
from ui.components import section, kpi_row

# ---------- sampling helpers ----------
def _sample(s: pd.Series, n: int = 4000) -> pd.Series:
    return s.sample(n, random_state=0) if len(s) > n else s

# ---------- detection heuristics (string/object friendly) ----------
def _boolean_rate(s: pd.Series) -> float:
    s = _sample(s.astype("string")).str.strip().str.lower()
    truthy = {"true","t","yes","y","1"}
    falsy  = {"false","f","no","n","0"}
    denom = s.notna().sum()
    return float(s.isin(truthy | falsy).sum()) / max(1, denom)

def _numeric_rate(s: pd.Series) -> tuple[float, bool]:
    x = pd.to_numeric(_sample(s), errors="coerce")
    base = _sample(s)
    rate = float(x.notna().sum()) / max(1, base.notna().sum())
    intish = False
    if x.notna().sum() > 0:
        frac = np.abs(x.dropna() - np.round(x.dropna()))
        intish = bool((frac < 1e-9).mean() > 0.98)
    return rate, intish

def _datetime_rate(s: pd.Series) -> float:
    x = pd.to_datetime(_sample(s), errors="coerce", infer_datetime_format=True)
    base = _sample(s)
    return float(x.notna().sum()) / max(1, base.notna().sum())

# ---------- dtype naming (no 'category' in UI) ----------
def _canonical_current_dtype(s: pd.Series) -> str:
    """Map pandas dtype to technical labels WITHOUT surfacing 'category'."""
    if is_datetime64_any_dtype(s): return "datetime"
    if is_bool_dtype(s):           return "boolean"
    if is_integer_dtype(s):        return "integer"
    if is_float_dtype(s):          return "float"
    # treat pandas 'category' and 'object' as string in this UI
    return "string"

def _suggest_for(s: pd.Series) -> tuple[str, str]:
    """
    Recommend one of: {'string','integer','float','datetime','boolean'} + reason.
    Never return 'category' (point users to Encoding tab instead).
    """
    # already strongly typed? keep it
    if is_datetime64_any_dtype(s): return "datetime", "Already parsed as datetime"
    if is_bool_dtype(s):           return "boolean",  "Already boolean"
    if is_integer_dtype(s):        return "integer",  "Already integer"
    if is_float_dtype(s):          return "float",    "Already float"

    # current is object/category -> infer
    br = _boolean_rate(s)
    if br >= 0.95:
        return "boolean", f"≈{br:.0%} values look like yes/no/true/false"

    dr = _datetime_rate(s)
    if dr >= 0.85:
        return "datetime", f"≈{dr:.0%} values parse as dates/times"

    nr, intish = _numeric_rate(s)
    if nr >= 0.90:
        return ("integer" if intish else "float",
                f"≈{nr:.0%} values are numeric{' (mostly whole numbers)' if intish else ''}")

    # low variety: keep as string but hint encoding later
    nun = s.nunique(dropna=True)
    nonnull = s.notna().sum()
    if nun <= min(50, max(3, int(0.2 * max(1, nonnull)))):
        return "string", f"Low variety ({nun} distinct values) — consider one-hot in Encoding tab"

    return "string", "Mixed/free-text or high variety"

# ---------- conversion helpers ----------
_TRUTHY = {"true","t","yes","y","1"}
_FALSY  = {"false","f","no","n","0"}

def _to_bool_series(s: pd.Series) -> pd.Series:
    s_str = s.astype("string").str.strip().str.lower()
    out = pd.Series(pd.NA, index=s.index, dtype="boolean")
    out = out.mask(s_str.isin(_TRUTHY), True)
    out = out.mask(s_str.isin(_FALSY),  False)
    return out

def _convert(series: pd.Series, target: str, *, dt_opts: dict) -> tuple[pd.Series, pd.Series]:
    """
    Convert a Series to target dtype; returns (converted, invalid_mask).
    """
    target = target.lower()
    if target in ("no change","keep"):
        return series, pd.Series(False, index=series.index)

    if target == "string":
        return series.astype("string"), pd.Series(False, index=series.index)

    if target == "datetime":
        fmt = dt_opts.get("fmt", "").strip()
        dayfirst = dt_opts.get("dayfirst", False)
        as_utc   = dt_opts.get("as_utc", False)
        if fmt:
            parsed = pd.to_datetime(series, format=fmt, errors="coerce", dayfirst=dayfirst, utc=as_utc)
        else:
            parsed = pd.to_datetime(series, errors="coerce", dayfirst=dayfirst, utc=as_utc)
        invalid = series.notna() & parsed.isna()
        return parsed, invalid

    if target == "float":
        parsed = pd.to_numeric(series, errors="coerce").astype("Float64")
        invalid = series.notna() & parsed.isna()
        return parsed, invalid

    if target == "integer":
        parsed = pd.to_numeric(series, errors="coerce").round().astype("Int64")
        invalid = series.notna() & parsed.isna()
        return parsed, invalid

    if target == "boolean":
        parsed = _to_bool_series(series)
        invalid = series.notna() & parsed.isna()
        return parsed, invalid

    # no 'category' in this UI
    return series, pd.Series(False, index=series.index)

# ---------- main UI ----------
def render_eda_types(ss) -> None:
    """EDA ▸ Types (no 'category' dtype shown or recommended)."""
    if not ss.active_ds or ss.active_ds not in ss.datasets:
        st.info("Pick a dataset to begin.")
        st.stop()

    df = ss.datasets[ss.active_ds]
    rows, cols = df.shape

    ss.setdefault("types_selected", set())
    ss.setdefault("types_target", {})
    ss.setdefault("types_dt_opts", {})
    ss.setdefault("types_policy", {})
    ss.setdefault("types_fill", {})

    kpi_row([("Rows", f"{rows:,}"), ("Cols", cols),
             ("Text-like cols", int((df.dtypes == "object").sum() + sum(is_categorical_dtype(df[c]) for c in df.columns)))])

    # build recs (map categorical -> string in 'current dtype')
    recs = []
    flagged = set()
    for c in df.columns:
        s = df[c]
        rec, why = _suggest_for(s)
        cur = _canonical_current_dtype(s)
        need_change = (cur != rec)
        if need_change:
            flagged.add(c)
        recs.append({
            "column": c,
            "current dtype": cur,
            "recommended": rec,
            "reason": why,
            "needs_change": need_change,
            "status": "⚠️ change" if need_change else "OK",
        })
    rec_df = pd.DataFrame(recs)

    with section("Columns & recommendations", expandable=False):
        q = st.text_input("Search", key="types_search", placeholder="Type to filter columns…")
        view = rec_df
        if q:
            ql = q.strip().lower()
            mask = (
                rec_df["column"].str.lower().str.contains(ql) |
                rec_df["current dtype"].str.lower().str.contains(ql) |
                rec_df["recommended"].str.lower().str.contains(ql) |
                rec_df["reason"].str.lower().str.contains(ql)
            )
            view = rec_df[mask].reset_index(drop=True)

        view = view.sort_values(["needs_change", "column"], ascending=[False, True]).reset_index(drop=True)
        view = view.copy()
        view["✓ Select"] = view["column"].apply(lambda c: (c in ss.types_selected) or (c in flagged))

        edited = st.data_editor(
            view[["status","✓ Select","column","current dtype","recommended","reason"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "status": st.column_config.TextColumn("Status"),
                "✓ Select": st.column_config.CheckboxColumn("✓ Select", help="Select columns to convert", default=False),
                "column": st.column_config.TextColumn("Column", disabled=True),
                "current dtype": st.column_config.TextColumn("Current dtype", disabled=True),
                "recommended": st.column_config.TextColumn("Recommended dtype", disabled=True),
                "reason": st.column_config.TextColumn("Reason", disabled=True),
            },
            key="types_editor",
        )
        ss.types_selected = set(edited.loc[edited["✓ Select"] == True, "column"].tolist())

        c1, c2 = st.columns([1,1])
        with c1:
            if st.button("Select all (filtered)"):
                ss.types_selected = set(edited["column"].tolist()); st.rerun()
        with c2:
            if st.button("Clear selection"):
                ss.types_selected = set(); st.rerun()

    if not ss.types_selected:
        st.info("Select one or more columns above to set conversions, preview, and apply.")
        return

    with section("Selected columns — conversions & preview", expandable=False):
        rec_map = {r["column"]: r["recommended"] for r in recs}
        for col in [c for c in df.columns if c in ss.types_selected]:
            s = df[col]
            cur_dtype = _canonical_current_dtype(s)
            rec = rec_map.get(col, cur_dtype)

            st.divider()
            st.markdown(f"#### {col}")
            st.caption(f"Current: `{cur_dtype}` • Recommended: **{rec}**")

            options = ["no change", "string", "datetime", "float", "integer", "boolean"]
            default_label = ss.types_target.get(col, rec if cur_dtype != rec else "no change")
            default_idx = options.index(default_label) if default_label in options else 0
            target = st.selectbox("Convert to", options=options, index=default_idx, key=f"types_target_select_{col}")
            ss.types_target[col] = target

            # datetime options
            dt_opts = ss.types_dt_opts.get(col, {"fmt": "", "dayfirst": False, "as_utc": False})
            if target == "datetime":
                c1, c2, c3 = st.columns([1,1,1])
                with c1:
                    dt_opts["fmt"] = st.text_input("Datetime format (optional)", value=dt_opts.get("fmt",""),
                                                   placeholder="%d/%m/%Y", key=f"types_dt_fmt_{col}")
                with c2:
                    dt_opts["dayfirst"] = st.checkbox("Day first", value=dt_opts.get("dayfirst", False),
                                                      key=f"types_dt_dayfirst_{col}")
                with c3:
                    dt_opts["as_utc"] = st.checkbox("Parse as UTC", value=dt_opts.get("as_utc", False),
                                                    key=f"types_dt_utc_{col}")
            else:
                dt_opts = {"fmt": "", "dayfirst": False, "as_utc": False}
            ss.types_dt_opts[col] = dt_opts

            # invalid policy
            pol_map = {"Set as NaN/NaT": "nan", "Drop affected rows": "drop", "Fill with constant": "fill"}
            inv_lbls = list(pol_map.keys())
            curr_policy = ss.types_policy.get(col, "nan")
            policy_choice = st.selectbox("If a value cannot be parsed", inv_lbls,
                                         index=inv_lbls.index({"nan":inv_lbls[0],"drop":inv_lbls[1],"fill":inv_lbls[2]}[curr_policy]),
                                         key=f"types_bad_{col}")
            policy = pol_map[policy_choice]
            ss.types_policy[col] = policy

            fill_val = ss.types_fill.get(col, "")
            if policy == "fill" and target not in ("no change",):
                fill_val = st.text_input("Fill constant", value=fill_val, key=f"types_fill_{col}",
                                         placeholder="e.g., 0 / 2024-01-01 / Unknown")
                ss.types_fill[col] = fill_val

            if st.button("Preview", key=f"types_prev_btn_{col}"):
                converted, invalid = _convert(s, target, dt_opts=dt_opts)
                prev_series = converted.copy()
                if invalid.any() and target not in ("no change",):
                    if policy == "drop":
                        prev_df = pd.concat([s.rename("original"), prev_series.rename("converted")], axis=1)
                        prev_df = prev_df.loc[~invalid].head(20)
                    elif policy == "fill":
                        if target == "datetime":
                            filler = pd.to_datetime(fill_val, errors="coerce")
                        elif target in ("float","integer"):
                            filler = pd.to_numeric(fill_val, errors="coerce")
                        elif target == "boolean":
                            fv = (fill_val or "").strip().lower()
                            if fv in _TRUTHY: filler = True
                            elif fv in _FALSY: filler = False
                            else: filler = pd.NA
                        else:
                            filler = fill_val
                        prev_series = prev_series.mask(invalid, filler)
                        prev_df = pd.concat([s.rename("original"), prev_series.rename("converted")], axis=1).head(20)
                    else:
                        prev_df = pd.concat([s.rename("original"), prev_series.rename("converted")], axis=1).head(20)
                else:
                    prev_df = pd.concat([s.rename("original"), prev_series.rename("converted")], axis=1).head(20)

                st.caption(f"Invalid to fix: **{int(invalid.sum())}**")
                st.dataframe(prev_df, use_container_width=True)

    with section("Apply all", expandable=False):
        if st.button("Apply conversions", type="primary"):
            out = df.copy()
            drop_mask_global = pd.Series(False, index=out.index)
            for col in [c for c in df.columns if c in ss.types_selected]:
                target = ss.types_target.get(col, "no change")
                dt_opts = ss.types_dt_opts.get(col, {"fmt": "", "dayfirst": False, "as_utc": False})
                policy  = ss.types_policy.get(col, "nan")
                fill_val = ss.types_fill.get(col, "")

                converted, invalid = _convert(out[col], target, dt_opts=dt_opts)
                if target in ("no change",):
                    continue

                if invalid.any():
                    if policy == "drop":
                        drop_mask_global |= invalid
                    elif policy == "fill":
                        if target == "datetime":
                            filler = pd.to_datetime(fill_val, errors="coerce")
                        elif target in ("float","integer"):
                            filler = pd.to_numeric(fill_val, errors="coerce")
                        elif target == "boolean":
                            fv = (fill_val or "").strip().lower()
                            if fv in _TRUTHY: filler = True
                            elif fv in _FALSY: filler = False
                            else: filler = pd.NA
                        else:
                            filler = fill_val
                        converted = converted.mask(invalid, filler)

                out[col] = converted

            if drop_mask_global.any():
                out = out.loc[~drop_mask_global].reset_index(drop=True)

            ss.df_history.append(df.copy())
            ss.datasets[ss.active_ds] = out
            st.success(f"Applied. New shape: **{out.shape[0]:,} × {out.shape[1]:,}**")
