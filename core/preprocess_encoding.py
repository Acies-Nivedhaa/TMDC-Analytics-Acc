# core/preprocess_encoding.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import json as _json
from pandas.api.types import is_bool_dtype, is_categorical_dtype, is_object_dtype

from ui.components import section, kpi_row, render_table


# ----------------------------- helpers -----------------------------

def _is_categorical(s: pd.Series) -> bool:
    """Heuristic: treat object, category, and boolean dtypes as categorical."""
    return is_object_dtype(s) or is_categorical_dtype(s) or is_bool_dtype(s)


def _to_hashable(s: pd.Series) -> pd.Series:
    """Map unhashable values (list/dict/set/tuple) to a stable JSON/string key."""
    def _h(v):
        if isinstance(v, (list, dict, set, tuple)):
            try:
                return _json.dumps(v, sort_keys=True)
            except Exception:
                return str(v)
        return v
    return s.map(_h)


def _topk_map(s: pd.Series, k: int) -> pd.Series:
    """
    Keep top-k most frequent categories; map the rest to '__other__'.
    Works with nested/unhashable values via _to_hashable.
    """
    s_h = _to_hashable(s.astype(object))
    vc = s_h.value_counts(dropna=False)
    top = set(vc.head(k).index)
    return s_h.where(s_h.isin(top), "__other__")


def _plan_reco(n_unique: int) -> str:
    """Lightweight recommendation string by cardinality."""
    if n_unique <= 2:
        return "Binary (0/1) or one-hot"
    if n_unique <= 10:
        return "One-hot (recommended)"
    if n_unique <= 50:
        return "Frequency or Ordinal by popularity"
    return "Target mean / Hash buckets (avoid wide one-hot)"


# ------------------------------ UI -------------------------------

def render_preprocess_encoding(ss) -> None:
    """
    Preprocess ▸ Encoding
    - Detects categorical columns and lets you choose an encoding per column.
    - Methods: One-hot (top-K), Ordinal (by frequency), Frequency count, Target mean, Drop, or None.
    - Safe with nested/unhashable cells (lists/dicts) by using a stable hashable view.
    - Preview does not mutate data; Apply replaces/drops as needed and pushes to session state.
    """
    if not ss.active_ds or ss.active_ds not in ss.datasets:
        st.info("Pick a dataset to begin.")
        st.stop()

    df = ss.datasets[ss.active_ds].copy()

    # --- Compact glossary (collapsible) ---
    with st.expander("Key terms (quick guide)", expanded=False):
        st.markdown(
            """
- **Categorical column** — Non-numeric labels (strings, categories, booleans).
- **One-hot encoding** — One column → many 0/1 columns (one per level). We limit to **top 10** + an **__other__** bucket to avoid explosion.
- **Drop first level** — Removes one dummy to reduce multicollinearity (“dummy trap”).
- **Ordinal (by frequency)** — Replace labels with rank by popularity (most frequent → 1, …).
- **Frequency count** — Replace labels with how often they appear (counts).
- **Target mean** — Replace each label with the target column’s mean for that label (supervised; beware leakage on train/test splits).
            """
        )

    # ------ detect categorical columns & show KPIs ------
    cat_cols = [c for c in df.columns if _is_categorical(df[c])]
    kpi_row([("Rows", f"{len(df):,}"), ("Cols", df.shape[1]), ("Categorical cols", len(cat_cols))])

    if not cat_cols:
        st.success("No categorical columns detected 🎉")
        return

    # ------ setup controls ------
    with section("Setup", expandable=False):
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            # Fixed policy per original implementation; easy to make this a slider if desired.
            st.caption("One-hot uses top **10** levels by default.")
            topk = 10
        # numeric targets for target-mean encoding
        numeric_targets = df.select_dtypes(include=[np.number]).columns.tolist()
        with c2:
            target_col = st.selectbox("Target column (for target-mean encoding)", ["—"] + numeric_targets, index=0)
        with c3:
            drop_first = st.checkbox("One-hot: drop first level (avoid dummy trap)", value=False)

    # state
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

    # ------ per-column choices ------
    with section("Encoding categorical variables", expandable=False):
        grid = [st.columns(2) for _ in range((len(cat_cols) + 1) // 2)]
        idx = 0
        plans: list[tuple[str, str, int]] = []  # (column, method, nunique)
        for row in grid:
            for col in row:
                if idx >= len(cat_cols):
                    break
                c = cat_cols[idx]
                idx += 1
                s = df[c]
                n_unique = int(_to_hashable(s).nunique(dropna=False))
                hint = _plan_reco(n_unique)
                with col:
                    st.caption(f"**{c}** — {n_unique} levels • _{hint}_")
                    opts = methods_all.copy()
                    if target_col == "—":
                        # Disable target-mean in UI text if target not selected
                        opts[-2] = "Target mean (select target first)"
                    chosen = ss.enc_choices.get(c, "None")
                    choice = st.selectbox(
                        "",
                        opts,
                        key=f"enc_choice_{c}",
                        index=opts.index(chosen) if chosen in opts else 0,
                    )
                    ss.enc_choices[c] = choice
                    plans.append((c, choice, n_unique))

    # ------ estimate net column changes ------
    with section("Planned changes (estimate)", expandable=True):
        rows_est = []
        net_new = 0  # positive adds columns; negative removes
        for c, m, nun in plans:
            if m == "One-hot (top-K)":
                # top-k + "__other__" if spillover
                add = min(topk, nun) + (1 if nun > topk else 0)
                if drop_first and add > 0:
                    add -= 1
                # Replace original column with 'add' columns => net = add - 1
                rows_est.append({"column": c, "method": m, "new_cols": add})
                net_new += (add - 1)
            elif m in ("Ordinal (by frequency)", "Frequency count"):
                # Replace in place => net = 0 (1 in, 1 out)
                rows_est.append({"column": c, "method": m, "new_cols": 1})
            elif m.startswith("Target mean"):
                # Only valid if target chosen; otherwise no change
                if target_col == "—":
                    rows_est.append({"column": c, "method": m, "new_cols": 0})
                else:
                    rows_est.append({"column": c, "method": "Target mean", "new_cols": 1})
            elif m == "Drop":
                # Drop original => net = -1
                rows_est.append({"column": c, "method": m, "new_cols": 0})
                net_new -= 1
            else:
                rows_est.append({"column": c, "method": m, "new_cols": 0})

        render_table(pd.DataFrame(rows_est))
        st.caption(f"**Net new columns (approx):** {net_new:+d}")

    # ------ encoding core ------
    def _encode_column(frame: pd.DataFrame, col: str, method: str) -> pd.DataFrame:
        """
        Apply encoding for one column and return the updated frame.
        - One-hot (top-K): expands into K (+__other__) dummies; drops original.
        - Ordinal/Frequency/Target mean: replaces the original column.
        - Drop: removes the column.
        """
        if method == "None":
            return frame
        if method == "Drop":
            return frame.drop(columns=[col])

        s = frame[col]
        # Keep booleans as objects to avoid get_dummies interpreting them as numeric
        if is_bool_dtype(s):
            s = s.astype(object)

        s_h = _to_hashable(s.astype(object))

        if method == "One-hot (top-K)":
            mapped = _topk_map(s, topk)
            dummies = pd.get_dummies(mapped, prefix=col, dummy_na=False)
            if drop_first and dummies.shape[1] > 0:
                dummies = dummies.iloc[:, 1:]
            frame = frame.drop(columns=[col])
            return pd.concat([frame, dummies], axis=1)

        if method == "Ordinal (by frequency)":
            order = s_h.value_counts(dropna=False).index.tolist()
            mapping = {k: i for i, k in enumerate(order, start=1)}
            frame[col] = s_h.map(mapping).fillna(0).astype(int)
            return frame

        if method == "Frequency count":
            counts = s_h.value_counts(dropna=False)
            frame[col] = s_h.map(counts).fillna(0).astype(int)
            return frame

        if method.startswith("Target mean"):
            # Require a numeric target column
            if target_col == "—":
                return frame
            means = frame.groupby(s_h, dropna=False)[target_col].mean()
            # Fill unknowns with global mean (simple fallback; for modeling, consider CV smoothing)
            frame[col] = s_h.map(means).fillna(frame[target_col].mean())
            return frame

        return frame  # fallback no-op

    # ------ Preview & Apply ------
    with section("Preview & Apply", expandable=False):
        c1, c2 = st.columns([1, 1])

        with c1:
            if st.button("Preview", key="enc_preview"):
                prev = df.copy()
                for c, m, _ in plans:
                    prev = _encode_column(prev, c, m)

                st.caption(f"Transformed: **{prev.shape[0]:,} × {prev.shape[1]:,}**")

                # Show only columns that changed (or first 30 as fallback)
                changed_bases = [c for c, m, _ in plans if m != "None"]
                sample_cols = [
                    c for c in prev.columns
                    if any(c == base or c.startswith(f"{base}_") for base in changed_bases)
                ]
                show_cols = (sample_cols[:30] or prev.columns[:30])
                st.dataframe(prev[show_cols].head(ss.enc_preview_rows), use_container_width=True)

        with c2:
            if st.button("Apply Encoding", type="primary", key="enc_apply"):
                out = df.copy()
                for c, m, _ in plans:
                    out = _encode_column(out, c, m)
                ss.df_history.append(df.copy())
                ss.datasets[ss.active_ds] = out
                st.success(f"Applied. New shape: **{out.shape[0]:,} × {out.shape[1]:,}**")
