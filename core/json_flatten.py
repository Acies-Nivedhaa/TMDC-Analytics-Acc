# core/json_flatten.py
from __future__ import annotations

import ast
import json
from typing import List, Tuple

import numpy as np
import pandas as pd


# ---------- Lightweight JSON-ish detection & parsing ----------

def _looks_jsonish(s: str) -> bool:
    """Fast check: string trimmed starts/ends with {} or []."""
    s = s.strip()
    return (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]"))


def _try_parse_json(v):
    """
    Best-effort parse:
      - pass through dict/list
      - parse strict JSON (json.loads)
      - fallback to ast.literal_eval for Python-literal-like strings
    Returns parsed object or None if not parseable.
    """
    if isinstance(v, (dict, list)):
        return v
    if not isinstance(v, str):
        return None

    s = v.strip()
    if not s or not _looks_jsonish(s):
        return None

    try:
        return json.loads(s)  # strict JSON
    except Exception:
        pass

    try:
        lit = ast.literal_eval(s)  # tolerates single quotes, etc.
        if isinstance(lit, (dict, list)):
            return lit
    except Exception:
        pass

    return None


def detect_jsonlike_columns(df: pd.DataFrame, sample: int = 200, min_hits_ratio: float = 0.15) -> List[str]:
    """
    Heuristic detector for JSON-like columns.

    Counts a "hit" if any sampled value:
      - is already a dict/list
      - is a string beginning with '{' or '['
      - becomes a dict/list when parsed (_try_parse_json)

    A column is flagged JSON-like if:
      hits/total >= min_hits_ratio  OR  hits >= 3

    Parameters
    ----------
    df : DataFrame
    sample : int
        Max number of non-null rows inspected per column (head).
    min_hits_ratio : float
        Minimum fraction of hits to flag.

    Returns
    -------
    List[str] : column names likely containing JSON-ish content
    """
    cols: List[str] = []
    if len(df) == 0:
        return cols

    for c in df.columns:
        s = df[c].dropna()
        if s.empty:
            continue

        samp = s.head(sample)
        total = len(samp)
        hits = 0

        for v in samp:
            if isinstance(v, (dict, list)):
                hits += 1
                continue
            if isinstance(v, str):
                sv = v.strip()
                if sv.startswith("{") or sv.startswith("["):
                    hits += 1
                    continue
                if _try_parse_json(v) is not None:
                    hits += 1

        if total and (hits / total >= min_hits_ratio or hits >= 3):
            cols.append(c)

    return cols


def flatten_json_columns(
    df: pd.DataFrame,
    columns: List[str] | None = None,
    max_level: int = 2,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Flatten JSON-like columns into a wider table.

    Behavior
    --------
    - Dict rows: expand into `<col>.<key>` using pandas.json_normalize
    - List rows:
        * add `<col>._len` (list length)
        * if the first element is a dict for any row, expand it into `<col>[0].*`
          (keeps width bounded and avoids exploding into many rows)

    Parameters
    ----------
    df : DataFrame
    columns : list[str] | None
        Columns to flatten. If None, auto-detects with detect_jsonlike_columns().
    max_level : int
        Max nesting level for json_normalize (limits very deep structures).

    Returns
    -------
    (df_out, flattened_cols, notes)
      df_out          : original data with new flattened columns appended
      flattened_cols  : list of columns that were processed (deduped/sorted)
      notes           : non-fatal warnings/errors per column (strings)
    """
    if columns is None:
        columns = detect_jsonlike_columns(df)

    out = df.copy()
    flattened: List[str] = []
    notes: List[str] = []

    for c in columns:
        try:
            # Parse each cell (best-effort). Non-JSON stays as None.
            parsed = df[c].apply(_try_parse_json)

            # ---- Dict rows -> columns: <col>.<key>
            dict_mask = parsed.apply(lambda x: isinstance(x, dict))
            if dict_mask.any():
                sub = pd.json_normalize(parsed.where(dict_mask, None), max_level=max_level)
                sub.columns = [f"{c}.{k}" for k in sub.columns]
                out = pd.concat([out, sub], axis=1)
                flattened.append(c)

            # ---- List rows -> metadata + first element (if dict)
            list_mask = parsed.apply(lambda x: isinstance(x, list))
            if list_mask.any():
                # Length of list per row
                out[f"{c}._len"] = parsed.apply(lambda x: len(x) if isinstance(x, list) else np.nan)

                # If list-of-dicts, expand the first element only to keep width bounded
                first_dict = parsed.apply(
                    lambda x: (x[0] if isinstance(x, list) and x and isinstance(x[0], dict) else None)
                )
                if first_dict.notna().any():
                    sub0 = pd.json_normalize(first_dict, max_level=max_level)
                    sub0.columns = [f"{c}[0].{k}" for k in sub0.columns]
                    out = pd.concat([out, sub0], axis=1)

                flattened.append(c)

        except Exception as e:
            # Do not fail the whole process—record the issue and continue
            notes.append(f"{c}: failed to flatten ({e})")

    return out, sorted(set(flattened)), notes
