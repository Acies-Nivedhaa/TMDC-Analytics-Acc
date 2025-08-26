# core/eda_combine.py
from __future__ import annotations
import streamlit as st
import pandas as pd
from typing import Iterable, List, Tuple
import re

# ---------- Helpers ----------

def ensure_unique_name(existing: Iterable[str], base: str) -> str:
    existing = set(existing)
    if base not in existing:
        return base
    i = 2
    while f"{base} ({i})" in existing:
        i += 1
    return f"{base} ({i})"

def _abbr(name: str, n: int = 8) -> str:
    """Short, filesystem-safe token from a dataset name."""
    tok = re.sub(r"[^A-Za-z0-9]+", "", str(name))
    return tok[:n] if tok else "ds"

def suggest_join_keys(df_left: pd.DataFrame, df_right: pd.DataFrame, max_keys: int = 2) -> List[str]:
    common = [c for c in df_left.columns if c in df_right.columns]
    if not common:
        return []
    nL, nR = len(df_left), len(df_right)
    scores = []
    for c in common:
        try:
            uL = df_left[c].nunique(dropna=True)
        except TypeError:
            uL = pd.Series(df_left[c].astype(str)).nunique(dropna=True)
        try:
            uR = df_right[c].nunique(dropna=True)
        except TypeError:
            uR = pd.Series(df_right[c].astype(str)).nunique(dropna=True)
        score = (uL / max(1, nL)) * (uR / max(1, nR))
        if any(k in c.lower() for k in ("id", "key", "uuid")):
            score += 0.25
        scores.append((score, c))
    scores.sort(reverse=True)
    picks = [c for s, c in scores[:max_keys] if s > 0.3]
    return picks

# core/<where-this-lives>.py
from typing import List, Tuple
import pandas as pd
import pandas.api.types as ptypes

def merge_datasets(
    left: pd.DataFrame,
    right: pd.DataFrame,
    how: str,
    left_on: List[str],
    right_on: List[str],
    suffixes: Tuple[str, str] = ("_x", "_y"),
) -> pd.DataFrame:
    """Robust merge:
    - normalizes inputs to lists
    - auto-falls back to common column names if lengths differ/are empty
    - aligns dtypes for each key (datetime -> datetime, else safe string)
    - never mutates the original DataFrames
    """
    # normalize → lists and drop keys not present
    lkeys = [left_on] if isinstance(left_on, str) else list(left_on or [])
    rkeys = [right_on] if isinstance(right_on, str) else list(right_on or [])
    lkeys = [c for c in lkeys if c in left.columns]
    rkeys = [c for c in rkeys if c in right.columns]

    # auto-match if mismatch/empty
    if (not lkeys and not rkeys) or (len(lkeys) != len(rkeys)):
        commons = [c for c in left.columns if c in right.columns]
        if commons:
            lkeys = rkeys = commons
        else:
            raise ValueError(
                f"Please select the same number of key columns on both sides. "
                f"Left keys={lkeys} ({len(lkeys)}), Right keys={rkeys} ({len(rkeys)})."
            )

    # work on copies, not originals
    L, R = left.copy(), right.copy()

    # align dtypes for each key pair
    for lk, rk in zip(lkeys, rkeys):
        a, b = L[lk], R[rk]
        if ptypes.is_datetime64_any_dtype(a) or ptypes.is_datetime64_any_dtype(b):
            L[lk] = pd.to_datetime(a, errors="coerce")
            R[rk] = pd.to_datetime(b, errors="coerce")
        elif ptypes.is_numeric_dtype(a) and ptypes.is_numeric_dtype(b):
            # both numeric -> OK
            pass
        else:
            # safe fallback to string for heterogeneous types
            L[lk] = a.astype(str)
            R[rk] = b.astype(str)

    return pd.merge(L, R, how=how, left_on=lkeys, right_on=rkeys, suffixes=suffixes)


def union_concat(
    datasets: List[Tuple[str, pd.DataFrame]],
    mode: str = "union",  # "union" | "intersection"
    add_source: bool = True,
) -> pd.DataFrame:
    if not datasets:
        return pd.DataFrame()
    if mode == "intersection":
        cols = set(datasets[0][1].columns)
        for _, df in datasets[1:]:
            cols &= set(df.columns)
        cols = list(cols)
        frames = []
        for name, df in datasets:
            part = df[cols].copy()
            if add_source:
                part["source_dataset"] = name
            frames.append(part)
        return pd.concat(frames, ignore_index=True)
    # union mode
    frames = []
    for name, df in datasets:
        part = df.copy()
        if add_source:
            part["source_dataset"] = name
        frames.append(part)
    return pd.concat(frames, ignore_index=True, sort=False)

# ---------- UI ----------

def render_combine(ss) -> None:
    """
    Join wizard (3 steps) + Append/Union with clearer naming and before/after column counts.
    """
    import streamlit as st
    import pandas as pd

    # ---------- guard ----------
    if len(ss.datasets) < 2:
        st.info("Add at least two datasets to perform a join or append.")
        return

    names_all = sorted(ss.datasets.keys())

    # ---------- state ----------
    ss.setdefault("combine_step", 1)
    ss.setdefault("combine_primary", ss.active_ds if ss.active_ds in names_all else names_all[0])
    ss.setdefault("combine_secondary",
                  next((n for n in names_all if n != ss.combine_primary), names_all[0]))

    st.markdown(
        """
        <style>
          .join-box{border:1px solid #e5e7eb;border-radius:12px;padding:14px 18px;text-align:center;background:#fff;}
          .join-name{font-weight:600;font-size:15px;margin-bottom:6px;}
          .join-rows{color:#6b7280;font-size:13px;}
          .join-arrow{font-size:28px; text-align:center; padding-top:24px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    def _box(label: str, rows: int, cols: int):
        st.markdown(
            f"""<div class="join-box">
                  <div class="join-name">{label}</div>
                  <div class="join-rows">{rows:,} rows • {cols:,} cols</div>
                </div>""",
            unsafe_allow_html=True,
        )

    step = int(ss.combine_step)
    st.markdown(f"**Join Tables – Step {step} of 3**")

    # ---------- STEP 1 ----------
    if step == 1:
        st.write("Which tables do you want to combine?")

        c1, c2 = st.columns(2)
        with c1:
            ss.combine_primary = st.selectbox("Primary table", names_all,
                                              index=names_all.index(ss.combine_primary),
                                              key="combine_primary_select")
        with c2:
            choices = [n for n in names_all if n != ss.combine_primary] or names_all
            if ss.combine_secondary == ss.combine_primary or ss.combine_secondary not in choices:
                ss.combine_secondary = choices[0]
            ss.combine_secondary = st.selectbox("Join with", choices,
                                                index=choices.index(ss.combine_secondary),
                                                key="combine_secondary_select")

        # visual cards with rows+cols
        left_df = ss.datasets[ss.combine_primary]
        right_df = ss.datasets[ss.combine_secondary]
        colA, colArrow, colB = st.columns([1, 0.2, 1])
        with colA:
            _box(ss.combine_primary, len(left_df), left_df.shape[1])
        with colArrow:
            st.markdown('<div class="join-arrow">➜</div>', unsafe_allow_html=True)
        with colB:
            _box(ss.combine_secondary, len(right_df), right_df.shape[1])

        st.caption("Tip: After saving, select the new dataset as **Primary** to join another table.")

        if st.button("Next: Choose columns →", key="combine_next1"):
            ss.combine_step = 2
            st.rerun()
        return

    # ---------- STEP 2 ----------
    left_nm, right_nm = ss.combine_primary, ss.combine_secondary
    left_df, right_df = ss.datasets[left_nm], ss.datasets[right_nm]
    suggested = suggest_join_keys(left_df, right_df)

    c_l, c_r = st.columns(2)
    with c_l:
        left_keys = st.multiselect(
            f"Primary key column(s) — {left_nm}",
            options=left_df.columns.tolist(),
            default=ss.get("combine_left_keys", suggested),
            key="combine_left_keys",
        )
    with c_r:
        right_keys = st.multiselect(
            f"Join key column(s) — {right_nm}",
            options=right_df.columns.tolist(),
            default=ss.get("combine_right_keys", suggested),
            key="combine_right_keys",
        )

    # flattened (no nested columns)
    c_how, c_sfx_l, c_sfx_r = st.columns([0.4, 0.3, 0.3])
    with c_how:
        how = st.selectbox("Join type", ["inner", "left", "right", "outer"],
                           index=0, key="combine_join_how")
    with c_sfx_l:
        sfx_l = st.text_input("Left suffix", value=ss.get("combine_sfx_l", "_x"), key="combine_sfx_l")
    with c_sfx_r:
        sfx_r = st.text_input("Right suffix", value=ss.get("combine_sfx_r", "_y"), key="combine_sfx_r")

    valid = bool(left_keys) and len(left_keys) == len(right_keys)

    # quick preview: shape and column counts before/after
    before_cols_l = left_df.shape[1]
    before_cols_r = right_df.shape[1]
    if valid:
        try:
            prev = merge_datasets(
                left_df, right_df, how=how,
                left_on=left_keys, right_on=right_keys,
                suffixes=(sfx_l, sfx_r)
            )
            st.caption(
                f"Preview: **{left_nm}** {before_cols_l:,} cols + **{right_nm}** {before_cols_r:,} cols "
                f"→ result **{prev.shape[1]:,} cols**, **{prev.shape[0]:,} rows**"
            )
        except Exception as e:
            st.error(f"Join preview failed: {e}")
    else:
        st.warning("Select the same number of key columns on both sides.")

    c_back, c_next = st.columns([0.25, 0.75])
    with c_back:
        if st.button("← Back", key="combine_back2"):
            ss.combine_step = 1
            st.rerun()
    with c_next:
        if valid and st.button("Next: Preview & save →", key="combine_next2"):
            ss.combine_step = 3
            st.rerun()
    if step == 2:
        return

    # ---------- STEP 3 ----------
    left_keys = ss.combine_left_keys
    right_keys = ss.combine_right_keys
    how, sfx_l, sfx_r = ss.combine_join_how, ss.combine_sfx_l, ss.combine_sfx_r

    merged = merge_datasets(
        left_df, right_df, how=how,
        left_on=left_keys, right_on=right_keys,
        suffixes=(sfx_l, sfx_r)
    )
    st.write("**Preview**")
    st.write(
        f"Result: **{merged.shape[0]:,} rows × {merged.shape[1]:,} cols** "
        f"from **{left_nm}** ({len(left_df):,}×{before_cols_l:,}) and "
        f"**{right_nm}** ({len(right_df):,}×{before_cols_r:,})."
    )
    st.dataframe(merged.head(20), use_container_width=True)

    # short, friendly default name
    base_short = f"{_abbr(left_nm)}_{_abbr(right_nm)}_{how[0].lower()}"  # e.g. ord_cust_i
    save_name = st.text_input(
        "Save as dataset name",
        value=ensure_unique_name(set(ss.datasets.keys()), base_short),
        key="combine_save_name"
    )
    c_back3, c_save = st.columns([0.25, 0.75])
    with c_back3:
        if st.button("← Back", key="combine_back3"):
            ss.combine_step = 2
            st.rerun()
    with c_save:
        if st.button("Save dataset", key="combine_apply"):
            try:
                merged = merge_datasets(
                    left_df, right_df, how=how,
                    left_on=left_keys, right_on=right_keys,
                    suffixes=(sfx_l, sfx_r),
                )
            except Exception as e:
                st.error(f"Join failed: {e}")
                st.stop()

            ss.datasets[save_name] = merged
            ss.raw_datasets[save_name] = merged.copy()
            ss.active_ds = save_name
            ss.df_history.clear()
            ss.combine_step = 1
            st.success(f"Saved as **{save_name}**")
            st.rerun()


    # ---------- Append / Union ----------
    st.markdown("---")
    st.markdown("#### Append / Union")

    picks = st.multiselect(
        "Datasets to append",
        names_all,
        default=[ss.active_ds] if ss.active_ds else [],
        key="eda_union_picks",
    )
    mode = st.selectbox(
        "Column alignment",
        ["Union (all columns)", "Intersection (common columns)"],
        key="eda_union_mode",
    )
    add_src = st.checkbox("Add source_dataset column", value=True, key="eda_union_addsrc")

    if len(picks) >= 2:
        # --- live before/after counts ---
        rows_before = {nm: len(ss.datasets[nm]) for nm in picks}
        colsets = {nm: set(ss.datasets[nm].columns) for nm in picks}

        if mode.startswith("Union"):
            cols_after = len(set().union(*colsets.values()))
        else:
            cols_after = len(set.intersection(*colsets.values()))

        rows_after = sum(rows_before.values())

        st.caption(
            "Rows before: " + ", ".join([f"**{nm}** {r:,}" for nm, r in rows_before.items()]) +
            f" → result **{rows_after:,} rows**"
        )
        st.caption(
            "Cols before: " + ", ".join([f"**{nm}** {len(colsets[nm]):,}" for nm in picks]) +
            f" → result **{cols_after:,} cols**"
        )

        # --- lightweight preview (samples a few rows per dataset) ---
        if st.button("Preview append", key="eda_preview_append"):
            per_ds = 10  # small, just for a peek
            combo_small = []
            for nm in picks:
                df = ss.datasets[nm]
                take = df.sample(min(per_ds, len(df)), random_state=0) if len(df) > per_ds else df
                combo_small.append((nm, take))

            prev = union_concat(
                combo_small,
                mode="union" if mode.startswith("Union") else "intersection",
                add_source=add_src,
            )
            st.write(
                f"Preview shape: {prev.shape[0]:,} rows × {prev.shape[1]:,} cols "
                f"(full result would be **{rows_after:,} × {cols_after:,}**)"
            )
            st.dataframe(prev.head(20), use_container_width=True)

    if len(picks) >= 2 and st.button("Apply append", key="eda_apply_append"):
        combo = [(nm, ss.datasets[nm]) for nm in picks]
        out = union_concat(
            combo,
            mode="union" if mode.startswith("Union") else "intersection",
            add_source=add_src,
        )
        # short, friendly name (keeps your earlier short-name helper)
        short_tokens = [_abbr(nm) for nm in picks[:3]]
        base_app = "app_" + "_".join(short_tokens)
        if len(picks) > 3:
            base_app += f"+{len(picks)-3}"
        new_name = ensure_unique_name(set(ss.datasets.keys()), base_app)

        ss.datasets[new_name] = out
        ss.raw_datasets[new_name] = out.copy()
        ss.active_ds = new_name
        ss.df_history.clear()
        st.success(f"Saved as **{new_name}**")
        st.rerun()



