from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import json as _json
from pandas.api.types import is_bool_dtype, is_categorical_dtype, is_object_dtype
from ui.components import section, kpi_row, render_table

def _is_categorical(s: pd.Series) -> bool:
    return is_object_dtype(s) or is_categorical_dtype(s) or is_bool_dtype(s)

def _to_hashable(s: pd.Series) -> pd.Series:
    def _h(v):
        if isinstance(v, (list, dict, set, tuple)):
            try: return _json.dumps(v, sort_keys=True)
            except Exception: return str(v)
        return v
    return s.map(_h)

def _topk_map(s: pd.Series, k: int) -> pd.Series:
    s_h = _to_hashable(s.astype(object))
    vc = s_h.value_counts(dropna=False)
    top = set(vc.head(k).index)
    return s_h.where(s_h.isin(top), "__other__")

def _plan_reco(n_unique: int) -> str:
    if n_unique <= 2: return "Binary (0/1) or one-hot"
    if n_unique <= 10: return "One-hot (recommended)"
    if n_unique <= 50: return "Frequency or Ordinal by popularity"
    return "Target mean / Hash buckets (avoid wide one-hot)"

def render_preprocess_encoding(ss) -> None:
    if not ss.active_ds or ss.active_ds not in ss.datasets:
        st.info("Pick a dataset to begin."); st.stop()
    df = ss.datasets[ss.active_ds].copy()

    cat_cols = [c for c in df.columns if _is_categorical(df[c])]
    kpi_row([("Rows", f"{len(df):,}"), ("Cols", df.shape[1]), ("Categorical cols", len(cat_cols))])
    if not cat_cols:
        st.success("No categorical columns detected 🎉"); return

    with section("Setup", expandable=False):
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            st.caption("One-hot uses top **10** levels by default.")
            topk = 10
        numeric_targets = df.select_dtypes(include=[np.number]).columns.tolist()
        with c2:
            target_col = st.selectbox("Target column (for target-mean encoding)", ["—"] + numeric_targets, index=0)
        with c3:
            drop_first = st.checkbox("One-hot: drop first level (avoid dummy trap)", value=False)

    ss.setdefault("enc_choices", {})
    ss.setdefault("enc_preview_rows", 15)

    methods_all = [
        "None",
        "One-hot (top-K)",
        "Ordinal (by frequency)",
        "Frequency count",
        "Target mean" if target_col != "—" else "Target mean (select target first)",
        "Drop",
    ]

    with section("Encoding categorical variables", expandable=False):
        grid = [st.columns(2) for _ in range((len(cat_cols)+1)//2)]
        idx = 0; plans = []
        for row in grid:
            for col in row:
                if idx >= len(cat_cols): break
                c = cat_cols[idx]; idx += 1
                s = df[c]
                n_unique = int(_to_hashable(s).nunique(dropna=False))
                hint = _plan_reco(n_unique)
                with col:
                    st.caption(f"**{c}** — {n_unique} levels • _{hint}_")
                    opts = methods_all.copy()
                    if target_col == "—": opts[-2] = "Target mean (select target first)"
                    chosen = ss.enc_choices.get(c, "None")
                    choice = st.selectbox("", opts, key=f"enc_choice_{c}",
                                          index=opts.index(chosen) if chosen in opts else 0)
                    ss.enc_choices[c] = choice
                    plans.append((c, choice, n_unique))

    with section("Planned changes (estimate)", expandable=True):
        est_new_cols = 0; rows = []
        for c, m, nun in plans:
            if m == "One-hot (top-K)":
                add = min(10, nun) + (1 if nun > 10 else 0)
                if drop_first and add > 0: add -= 1
                rows.append({"column": c, "method": m, "new_cols": add})
                est_new_cols += add - 1
            elif m in ("Ordinal (by frequency)", "Frequency count", "Target mean"):
                rows.append({"column": c, "method": m, "new_cols": 1})
            else:
                rows.append({"column": c, "method": m, "new_cols": 0})
        render_table(pd.DataFrame(rows))
        st.caption(f"**Net new columns (approx):** {est_new_cols:+d}")

    def _encode_column(frame: pd.DataFrame, col: str, method: str) -> pd.DataFrame:
        s = frame[col]
        if method == "None": return frame
        if method == "Drop": return frame.drop(columns=[col])
        if is_bool_dtype(s): s = s.astype(object)
        s_h = _to_hashable(s.astype(object))

        if method == "One-hot (top-K)":
            mapped = _topk_map(s, 10)
            dummies = pd.get_dummies(mapped, prefix=col, dummy_na=False)
            if drop_first and dummies.shape[1] > 0: dummies = dummies.iloc[:, 1:]
            frame = frame.drop(columns=[col])
            return pd.concat([frame, dummies], axis=1)

        if method == "Ordinal (by frequency)":
            order = s_h.value_counts(dropna=False).index.tolist()
            mapping = {k: i for i, k in enumerate(order, start=1)}
            frame[col] = s_h.map(mapping).fillna(0).astype(int); return frame

        if method == "Frequency count":
            counts = s_h.value_counts(dropna=False)
            frame[col] = s_h.map(counts).fillna(0).astype(int); return frame

        if method == "Target mean":
            if target_col == "—": return frame
            means = frame.groupby(s_h, dropna=False)[target_col].mean()
            frame[col] = s_h.map(means).fillna(frame[target_col].mean()); return frame

        return frame

    with section("Preview & Apply", expandable=False):
        c1, c2 = st.columns([1,1])
        with c1:
            if st.button("Preview", key="enc_preview"):
                prev = df.copy()
                for c, m, _ in plans: prev = _encode_column(prev, c, m)
                st.caption(f"Transformed: **{prev.shape[0]:,} × {prev.shape[1]:,}**")
                changed = [c for c, m, _ in plans if m != "None"]
                sample_cols = [c for c in prev.columns if any(c.startswith(f"{b}_") or c == b for b in changed)]
                show = prev[(sample_cols[:30] or prev.columns[:30])].head(ss.enc_preview_rows)
                st.dataframe(show, use_container_width=True)
        with c2:
            if st.button("Apply Encoding", type="primary", key="enc_apply"):
                out = df.copy()
                for c, m, _ in plans: out = _encode_column(out, c, m)
                ss.df_history.append(df.copy()); ss.datasets[ss.active_ds] = out
                st.success(f"Applied. New shape: **{out.shape[0]:,} × {out.shape[1]:,}**")
