# core/eda_combine.py
from __future__ import annotations

"""
EDA ▸ Combine datasets (Join/Merge)
-----------------------------------
UI to merge two in-memory datasets with pandas.merge:
- Pick left/right datasets
- Choose join type and key columns (supports multi-column keys)
- Preview result and save as a new dataset (deduped name)
- Compact glossary expander for non-experts

Notes:
- Append/union is intentionally not provided here (by request).
"""

import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, is_bool_dtype

from ui.components import section, kpi_row, render_table


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def ensure_unique_name(existing: set[str], base: str) -> str:
    """Return a dataset name that doesn't collide with existing names."""
    if base not in existing:
        return base
    i = 2
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


def _common_cols(df_left: pd.DataFrame, df_right: pd.DataFrame) -> list[str]:
    """Intersection of column names (sorted)."""
    return sorted(list(set(df_left.columns) & set(df_right.columns)))


def _dtype_group(s: pd.Series) -> str:
    """
    Coarse dtype grouping used only for user warnings.
    Groups: numeric / datetime / bool / other
    """
    if is_numeric_dtype(s):
        return "numeric"
    if is_datetime64_any_dtype(s):
        return "datetime"
    if is_bool_dtype(s):
        return "bool"
    return "other"


# ---------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------

def render_combine(ss) -> None:
    """EDA ▸ Combine datasets — Join/Merge only (append/union removed)."""
    if not ss.datasets or len(ss.datasets) < 2:
        st.info("Add at least two datasets in **Overview** to use Join/Merge.")
        return

    names = sorted(ss.datasets.keys())

    # --- Quick glossary (collapsible) ---
    with st.expander("Key terms (quick guide)", expanded=False):
        st.markdown(
            """
            - **Left / Right dataset** — The two tables you’re joining.
            - **Join keys** — Column(s) that should match across both tables (e.g., `order_id`, `customer_id`).
            - **Join type**  
              - **Inner**: only matching rows.  
              - **Left**: all rows from the left table, matches from right.  
              - **Right**: all rows from the right table, matches from left.  
              - **Outer**: all rows from both (non-matches show as missing).  
            - **Composite key** — Use multiple columns as the key when one column isn’t unique enough.
            - **Suffixes** — If both tables share a non-key column name, suffixes (like `_x`, `_y`) avoid clashes.
            - **Watch out** — If key dtypes differ (e.g., number vs text), join results may be empty or duplicated.
            """
        )

    # --- Defaults / state ---
    ss.setdefault("cmb_left", names[0])
    ss.setdefault("cmb_right", names[1] if len(names) > 1 else names[0])
    if ss.cmb_left == ss.cmb_right and len(names) > 1:
        # keep left/right distinct by default
        ss.cmb_right = next(n for n in names if n != ss.cmb_left)

    # ------------------ Dataset selection ------------------
    with section("Pick datasets", expandable=False):
        c1, c2 = st.columns(2)
        with c1:
            left_name = st.selectbox("Left dataset", names, index=names.index(ss.cmb_left), key="cmb_left")
        with c2:
            right_options = [n for n in names if n != left_name] or names
            right_name = st.selectbox("Right dataset", right_options, index=0, key="cmb_right")

        df_left = ss.datasets[left_name]
        df_right = ss.datasets[right_name]

        kpi_row([
            (f"{left_name} rows", f"{len(df_left):,}"),  (f"{left_name} cols", df_left.shape[1]),
            (f"{right_name} rows", f"{len(df_right):,}"), (f"{right_name} cols", df_right.shape[1]),
        ])

    # ------------------ Join settings ------------------
    with section("Join settings", expandable=False):
        common = _common_cols(df_left, df_right)
        default_keys = common[:1] if common else []

        # Persist user choices
        ss.setdefault("cmb_join_type", "inner")
        ss.setdefault("cmb_left_on", default_keys)
        ss.setdefault("cmb_right_on", default_keys)
        ss.setdefault("cmb_suffixes_l", "_x")
        ss.setdefault("cmb_suffixes_r", "_y")

        ctop1, ctop2, ctop3 = st.columns([1, 1, 1])
        with ctop1:
            join_type = st.selectbox(
                "Join type",
                ["inner", "left", "right", "outer"],
                index=["inner", "left", "right", "outer"].index(ss.cmb_join_type),
                key="cmb_join_type",
                help=(
                    "Inner = matching rows only; Left/Right = keep all rows from one side; "
                    "Outer = keep all rows from both."
                ),
            )
        with ctop2:
            left_on = st.multiselect(
                f"Keys in '{left_name}'",
                options=list(df_left.columns),
                default=ss.cmb_left_on if ss.cmb_left_on else default_keys,
                key="cmb_left_on",
            )
        with ctop3:
            right_on = st.multiselect(
                f"Keys in '{right_name}'",
                options=list(df_right.columns),
                default=ss.cmb_right_on if ss.cmb_right_on else default_keys,
                key="cmb_right_on",
            )

        # Basic validations & helpful warnings
        if len(left_on) != len(right_on):
            st.warning("Pick the **same number of key columns** on each side.")
        if not left_on or not right_on:
            st.info("Select at least one key column on each side to enable preview.")

        # Warn if key dtype groups differ (common reason for empty joins)
        if left_on and right_on and len(left_on) == len(right_on):
            mismatches = []
            for lcol, rcol in zip(left_on, right_on):
                lg = _dtype_group(df_left[lcol])
                rg = _dtype_group(df_right[rcol])
                if lg != rg:
                    mismatches.append((lcol, rg, rcol, lg))
            if mismatches:
                lines = "\n".join(
                    f"- `{l}` ({_dtype_group(df_left[l])}) vs `{r}` ({_dtype_group(df_right[r])})"
                    for (l, _, r, __) in mismatches
                )
                st.warning(
                    "Selected key columns have different data types. Consider casting to match:\n" + lines
                )

        csuf1, csuf2 = st.columns(2)
        with csuf1:
            sfx_l = st.text_input("Suffix for left duplicates", value=ss.cmb_suffixes_l, key="cmb_suffixes_l")
        with csuf2:
            sfx_r = st.text_input("Suffix for right duplicates", value=ss.cmb_suffixes_r, key="cmb_suffixes_r")

    # ------------------ Merge helpers ------------------

    def _merge_preview() -> tuple[pd.DataFrame | None, str | None]:
        """
        Try the merge with current settings. Returns (df_or_none, error_message_or_None).
        """
        if not left_on or not right_on:
            return None, "Select join keys on both sides."
        if len(left_on) != len(right_on):
            return None, "The number of keys must match (left vs right)."
        try:
            out = pd.merge(
                df_left,
                df_right,
                how=join_type,
                left_on=left_on,
                right_on=right_on,
                suffixes=(sfx_l, sfx_r),
            )
            return out, None
        except Exception as e:
            return None, f"Merge failed: {e}"

    # ------------------ Preview & save ------------------
    with section("Preview & save", expandable=False):
        cprev, csave = st.columns([1, 1])

        with cprev:
            if st.button("Preview join", key="cmb_preview"):
                merged, err = _merge_preview()
                if err:
                    st.error(err)
                elif merged is None or merged.empty:
                    st.warning("Join produced no rows.")
                else:
                    st.caption(f"Result: **{merged.shape[0]:,} × {merged.shape[1]:,}**")
                    # show a tidy preview (first 50 rows)
                    render_table(merged.head(50))

        with csave:
            new_base = f"join_{left_name}_{right_name}"
            new_name = st.text_input("Save as (dataset name)", value=new_base, key="cmb_new_name")
            if st.button("Save dataset", type="primary", key="cmb_save"):
                merged, err = _merge_preview()
                if err:
                    st.error(err)
                elif merged is None:
                    st.warning("Nothing to save — please preview first.")
                else:
                    safe_name = ensure_unique_name(set(ss.datasets.keys()), new_name.strip() or new_base)
                    # Store into datasets; keep a copy in raw_datasets if present
                    ss.datasets[safe_name] = merged
                    if hasattr(ss, "raw_datasets"):
                        ss.raw_datasets[safe_name] = merged.copy()
                    ss.active_ds = safe_name
                    st.success(f"Saved merged dataset as **{safe_name}**.")
                    st.rerun()

    # ------------------ Notes ------------------
    with section("Notes", expandable=True):
        st.markdown(
            "- Use matching key columns (e.g., `order_id`, `customer_id`, or a composite of multiple fields).\n"
            "- If both tables have a non-key column with the same name, suffixes (e.g., `_x`, `_y`) will be added."
        )
