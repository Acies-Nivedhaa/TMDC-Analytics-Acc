# core/preprocess_encoding.py
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
from pandas.api.types import is_bool_dtype, is_categorical_dtype, is_object_dtype

from ui.components import section, kpi_row, render_table

# Fixed cap for one-hot width; extra levels are grouped into "__other__"
TOPK_DEFAULT = 15


def _is_categorical(s: pd.Series) -> bool:
    """Treat object, category, and boolean as categorical for encoding."""
    return is_object_dtype(s) or is_categorical_dtype(s) or is_bool_dtype(s)


def _topk_map(s: pd.Series, k: int) -> pd.Series:
    vc = s.value_counts(dropna=False)
    top = set(vc.head(k).index)
    return s.where(s.isin(top), "__other__")


def _plan_reco(n_unique: int) -> str:
    if n_unique <= 2:
        return "Binary (0/1) or one-hot"
    if n_unique <= 10:
        return "One-hot (recommended)"
    if n_unique <= 50:
        return "Frequency or Ordinal by popularity"
    return "Target mean / Hash buckets (avoid wide one-hot)"


def render_preprocess_encoding(ss) -> None:
    """Preprocess ▸ Encoding (categorical features)."""
    if not ss.active_ds or ss.active_ds not in ss.datasets:
        st.info("Pick a dataset to begin.")
        st.stop()

    df = ss.datasets[ss.active_ds].copy()

    # figure out candidate categorical columns
    cat_cols = [c for c in df.columns if _is_categorical(df[c])]
    kpi_row([
        ("Rows", f"{len(df):,}"),
        ("Cols", df.shape[1]),
        ("Categorical cols", len(cat_cols)),
    ])

    # ---- lightweight CSS for clearer per-column cards ----
    st.markdown("""
    <style>
      .enc-card{border:1px solid rgba(0,0,0,.08); border-radius:12px;
                padding:12px; background:#fbfbfb; margin-bottom:12px;}
      .enc-title{font-weight:700; margin-bottom:6px;}
      .enc-badge{font-size:.80rem; background:#eef2ff; color:#3730a3;
                 padding:2px 8px; border-radius:999px; margin-left:8px;}
      .enc-hint{color:#666; font-size:.85rem; margin-top:4px;}
    </style>
    """, unsafe_allow_html=True)

    if not cat_cols:
        st.success("No categorical columns detected 🎉")
        return

    # persist per-column choices
    ss.setdefault("enc_choices", {})        # col -> method
    ss.setdefault("enc_preview_rows", 15)

    # ---- global options (no Top-K slider) ----
    with section("Setup", expandable=False):
        numeric_targets = df.select_dtypes(include=[np.number]).columns.tolist()
        c1, c2 = st.columns([1, 1])
        with c1:
            target_col = st.selectbox(
                "Target column (for target-mean encoding)",
                ["—"] + numeric_targets, index=0,
                help="Required only if you choose 'Target mean' for any column.",
            )
        with c2:
            drop_first = st.checkbox(
                "One-hot: drop first level (avoid dummy trap)",
                value=False,
                help="Drops one dummy per encoded column to avoid multicollinearity.",
            )
        st.caption(f"Fixed one-hot width: **top-K = {TOPK_DEFAULT}** (extra levels → `__other__`).")

    one_hot_label = f"One-hot (top-K={TOPK_DEFAULT})"
    methods_all = [
        "None",
        one_hot_label,
        "Ordinal (by frequency)",
        "Frequency count",
        "Target mean" if target_col != "—" else "Target mean (select target first)",
        "Drop",
    ]

    # ---- per-column controls (card layout, obvious ownership) ----
    plans: list[tuple[str, str, int]] = []

    with section("Encoding categorical variables", expandable=False):
        # 2-column grid of cards
        cols_per_row = 2
        n_rows = (len(cat_cols) + cols_per_row - 1) // cols_per_row
        idx = 0
        for _ in range(n_rows):
            row = st.columns(cols_per_row)
            for area in row:
                if idx >= len(cat_cols):
                    break
                c = cat_cols[idx]; idx += 1
                s = df[c]
                n_unique = int(s.nunique(dropna=False))
                hint = _plan_reco(n_unique)

                with area:
                    st.markdown('<div class="enc-card">', unsafe_allow_html=True)
                    # Title + levels badge
                    st.markdown(
                        f'<div class="enc-title">{c}'
                        f'<span class="enc-badge">{n_unique} levels</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # Quick top values peek
                    try:
                        top_vals = [str(v) for v in s.astype(str).value_counts(dropna=False).head(5).index]
                        st.caption("Top values: " + ", ".join(top_vals))
                    except Exception:
                        pass

                    # Method picker, clearly scoped to this column
                    opts = methods_all.copy()
                    if target_col == "—":
                        # visually signal dependency
                        opts[-2] = "Target mean (select target first)"
                    choice = st.selectbox(
                        f"Method for {c}",
                        opts,
                        key=f"enc_choice_{c}",
                        index=opts.index(ss.enc_choices.get(c, "None"))
                        if ss.enc_choices.get(c) in opts else 0,
                        label_visibility="visible",
                        help=f"Recommendation: {hint}",
                    )
                    ss.enc_choices[c] = choice

                    st.markdown(f'<div class="enc-hint">{hint}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)  # end card

                plans.append((c, choice, n_unique))

    # ---- Planned impact summary ----
    with section("Planned changes (estimate)", expandable=True):
        est_new_cols = 0
        rows = []
        for c, m, nun in plans:
            if m == one_hot_label:
                add = min(TOPK_DEFAULT, nun) + (1 if nun > TOPK_DEFAULT else 0)  # +1 for "__other__"
                if drop_first and add > 0:
                    add -= 1
                rows.append({"column": c, "method": m, "new_cols": add})
                # net effect: drop original col (+ add new dummies)
                est_new_cols += add - 1
            elif m in ("Ordinal (by frequency)", "Frequency count", "Target mean"):
                rows.append({"column": c, "method": m, "new_cols": 1})
                # replace original → net 0
            elif m == "Drop":
                rows.append({"column": c, "method": m, "new_cols": 0})
            else:
                rows.append({"column": c, "method": m, "new_cols": 0})
        render_table(pd.DataFrame(rows))
        st.caption(f"**Net new columns (approx):** {est_new_cols:+d}")

    # ---- transform helpers ----
    def _encode_column(frame: pd.DataFrame, col: str, method: str) -> pd.DataFrame:
        s = frame[col]
        if method == "None":
            return frame
        if method == "Drop":
            return frame.drop(columns=[col])

        # Normalize types
        if is_bool_dtype(s):
            s = s.astype(object)

        if method == one_hot_label:
            mapped = _topk_map(s.astype(object), TOPK_DEFAULT)
            dummies = pd.get_dummies(mapped, prefix=col, dummy_na=False)
            if drop_first and dummies.shape[1] > 0:
                dummies = dummies.iloc[:, 1:]
            frame = frame.drop(columns=[col])
            return pd.concat([frame, dummies], axis=1)

        if method == "Ordinal (by frequency)":
            order = s.value_counts(dropna=False).index.tolist()
            mapping = {k: i for i, k in enumerate(order, start=1)}
            out = s.map(mapping).fillna(0).astype(int)
            frame[col] = out
            return frame

        if method == "Frequency count":
            counts = s.value_counts(dropna=False)
            frame[col] = s.map(counts).fillna(0).astype(int)
            return frame

        if method == "Target mean":
            if target_col == "—":
                return frame  # guard (UI should prevent this)
            tgt = frame[target_col]
            means = frame.groupby(col, dropna=False)[target_col].mean()
            global_mean = tgt.mean()
            frame[col] = s.map(means).fillna(global_mean)
            return frame

        return frame  # fallback

    # ---- Preview & Apply ----
    with section("Preview & Apply", expandable=False):
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Preview", key="enc_preview"):
                prev = df.copy()
                for c, m, _ in plans:
                    prev = _encode_column(prev, c, m)
                st.caption(f"Transformed: **{prev.shape[0]:,} × {prev.shape[1]:,}**")

                # Show only columns that changed/expanded (first 30 for brevity)
                changed_bases = [c for c, m, _ in plans if m != "None"]
                sample_cols = [
                    col for col in prev.columns
                    if any(col == base or col.startswith(f"{base}_") for base in changed_bases)
                ]
                show = prev[(sample_cols[:30] or prev.columns[:30])].head(ss.enc_preview_rows)
                st.dataframe(show, use_container_width=True)

        with c2:
            if st.button("Apply Encoding", type="primary", key="enc_apply"):
                out = df.copy()
                for c, m, _ in plans:
                    out = _encode_column(out, c, m)
                # push history then replace dataset in place
                ss.df_history.append(df.copy())
                ss.datasets[ss.active_ds] = out
                st.success(f"Applied. New shape: **{out.shape[0]:,} × {out.shape[1]:,}**")
