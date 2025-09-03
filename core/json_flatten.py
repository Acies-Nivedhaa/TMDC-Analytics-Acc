# core/json_flatten.py
from __future__ import annotations
import ast
import json
from typing import List, Tuple
import numpy as np
import pandas as pd

def _looks_jsonish(s: str) -> bool:
    s = s.strip()
    return (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]"))

def _try_parse_json(v):
    if isinstance(v, (dict, list)):
        return v
    if not isinstance(v, str):
        return None
    s = v.strip()
    if not s or not _looks_jsonish(s):
        return None
    try:
        return json.loads(s)        # strict JSON
    except Exception:
        pass
    try:
        lit = ast.literal_eval(s)   # single-quoted / python-literal
        if isinstance(lit, (dict, list)):
            return lit
    except Exception:
        pass
    return None

# core/json_flatten.py  (replace just this function)
def detect_jsonlike_columns(df: pd.DataFrame, sample: int = 200, min_hits_ratio: float = 0.15) -> List[str]:
    """
    More permissive detector:
    - counts a hit if value is a dict/list
    - OR if a string trimmed starts with '{' or '[' (JSON-ish)
    - OR if it can be parsed by json.loads/ast.literal_eval
    Flags column JSON-like if hits/total >= min_hits_ratio OR >=3 hits.
    """
    cols = []
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
    Returns (df_flattened, flattened_columns, notes)
    - Dicts -> expand keys into <col>.<key>
    - Lists -> add <col>._len and, if list of dicts, expand first element into <col>[0].*
    """
    if columns is None:
        columns = detect_jsonlike_columns(df)

    out = df.copy()
    flattened, notes = [], []

    for c in columns:
        try:
            parsed = df[c].apply(_try_parse_json)

            # Dict rows
            dict_mask = parsed.apply(lambda x: isinstance(x, dict))
            if dict_mask.any():
                sub = pd.json_normalize(parsed.where(dict_mask, None), max_level=max_level)
                sub.columns = [f"{c}.{k}" for k in sub.columns]
                out = pd.concat([out, sub], axis=1)
                flattened.append(c)

            # List rows
            list_mask = parsed.apply(lambda x: isinstance(x, list))
            if list_mask.any():
                out[f"{c}._len"] = parsed.apply(lambda x: len(x) if isinstance(x, list) else np.nan)
                first_dict = parsed.apply(lambda x: (x[0] if isinstance(x, list) and x and isinstance(x[0], dict) else None))
                if first_dict.notna().any():
                    sub0 = pd.json_normalize(first_dict, max_level=max_level)
                    sub0.columns = [f"{c}[0].{k}" for k in sub0.columns]
                    out = pd.concat([out, sub0], axis=1)
                flattened.append(c)

        except Exception as e:
            notes.append(f"{c}: failed to flatten ({e})")

    return out, sorted(set(flattened)), notes
