from __future__ import annotations
from typing import Optional, Any, Iterable
import io, json, csv, gzip, zipfile, math
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "read_any",           # main entry
    "read_any_preview",
    "read_zip_all",   # capped rows for summary/preview
]

# -------- Tunables (safe defaults) --------
MAX_PREVIEW_ROWS = 250_000          # cap for preview reads (CSV/JSONL) to keep memory safe
CSV_CHUNK_ROWS   = 100_000          # chunk size when streaming CSV


# ============== Public API ==============

def read_any(upload, *, max_rows: Optional[int] = None) -> Optional[pd.DataFrame]:
    """
    Read CSV/TSV/TXT, Excel, JSON/JSONL/NDJSON (flattened), Parquet, and ZIP/GZ wrappers
    into a pandas DataFrame. Designed to be robust for large files:
      - CSV: streamed in chunks (respecting max_rows if provided)
      - JSONL/NDJSON: line-by-line incremental parse
      - JSON (deeply nested): auto-flatten; if top-level is a large array of dicts and
        ijson is available, stream without loading the whole file
      - Excel: first non-empty sheet; honors max_rows when supported
      - Parquet: uses PyArrow if available, can slice to first max_rows

    Returns None on failure.
    """
    if upload is None:
        return None

    name = _get_name(upload).lower()
    try:
        if name.endswith(".gz"):
            inner_bytes = gzip.decompress(_to_bytes(upload))
            inner_name  = name[:-3] or "decompressed"
            df = _read_from_bytes(inner_bytes, inner_name, max_rows=max_rows)
            return _optimize_dtypes(df) if isinstance(df, pd.DataFrame) else df

        if name.endswith(".zip"):
            return _read_from_zip(_to_bytes(upload), max_rows=max_rows)

        df = _read_from_bytes(_to_bytes(upload), name, max_rows=max_rows)
        return _optimize_dtypes(df) if isinstance(df, pd.DataFrame) else df

    except MemoryError:
        # Fallback: try again with preview cap
        try:
            df = _read_from_bytes(_to_bytes(upload), name, max_rows=MAX_PREVIEW_ROWS)
            return _optimize_dtypes(df) if isinstance(df, pd.DataFrame) else df
        except Exception:
            return None
    except Exception:
        return None


def read_any_preview(upload) -> Optional[pd.DataFrame]:
    """Convenience: read with a safe row cap for previews of very large files."""
    return read_any(upload, max_rows=MAX_PREVIEW_ROWS)

def read_zip_all(upload, *, max_rows: Optional[int] = None) -> list[tuple[str, pd.DataFrame]]:
    """
    Read all supported tabular files inside a ZIP and return a list of
    (inner_filename, DataFrame). Uses the same logic as read_any/_read_from_bytes.
    """
    b = _to_bytes(upload)
    return _read_zip_all_from_bytes(b, max_rows=max_rows)


# ============== Internals ==============

def _get_name(upload) -> str:
    if hasattr(upload, "name") and isinstance(upload.name, str):
        return upload.name
    if isinstance(upload, (str, Path)):
        return str(upload)
    return "uploaded"


def _to_bytes(upload) -> bytes:
    if isinstance(upload, (str, Path)):
        with open(upload, "rb") as f:
            return f.read()
    if hasattr(upload, "getvalue"):
        return upload.getvalue()
    if hasattr(upload, "read"):
        pos = getattr(upload, "tell", lambda: None)()
        try:
            return upload.read()
        finally:
            try:
                upload.seek(pos or 0)
            except Exception:
                pass
    if isinstance(upload, (bytes, bytearray)):
        return bytes(upload)
    raise TypeError("Unsupported upload type")


def _read_zip_all_from_bytes(b: bytes, *, max_rows: Optional[int] = None) -> list[tuple[str, pd.DataFrame]]:
    out: list[tuple[str, pd.DataFrame]] = []
    with zipfile.ZipFile(io.BytesIO(b)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            inner_name = info.filename
            # only attempt supported extensions (we also handle .gz inside the zip)
            if not (_is_supported(inner_name.lower()) or inner_name.lower().endswith(".gz")):
                continue
            try:
                raw = zf.read(info)
            except Exception:
                continue

            # If the inner file is .gz, decompress and strip the suffix before dispatch
            name_for_reader = inner_name.lower()
            if name_for_reader.endswith(".gz"):
                try:
                    raw = gzip.decompress(raw)
                    name_for_reader = name_for_reader[:-3] or "decompressed"
                except Exception:
                    continue

            df = _read_from_bytes(raw, name_for_reader, max_rows=max_rows)
            if isinstance(df, pd.DataFrame) and not df.empty and df.shape[1] > 0:
                # keep only the base filename (no folders)
                base = inner_name.rsplit("/", 1)[-1]
                out.append((base, _optimize_dtypes(df)))
    return out


def _read_from_zip(b: bytes, *, max_rows: Optional[int] = None) -> Optional[pd.DataFrame]:
    """
    Back-compat: keep returning a single table (the first supported one).
    Use read_zip_all() in the app to ingest ALL tables.
    """
    tables = _read_zip_all_from_bytes(b, max_rows=max_rows)
    return tables[0][1] if tables else None



def _is_supported(name: str) -> bool:
    return name.endswith((".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json", ".jsonl", ".ndjson", ".parquet"))


def _read_from_bytes(b: bytes, name_lower: str, *, max_rows: Optional[int] = None) -> Optional[pd.DataFrame]:
    bio = io.BytesIO(b)

    # JSON family first (by extension or sniff)
    if name_lower.endswith((".json", ".jsonl", ".ndjson")) or _looks_like_json(b):
        try:
            return _read_json_like(b, max_rows=max_rows)
        except Exception:
            pass

    # Excel
    if name_lower.endswith((".xlsx", ".xls")):
        try:
            # nrows supported in pandas for Excel (openpyxl engine)
            df_dict = pd.read_excel(bio, sheet_name=None, nrows=max_rows)
            for _, df in df_dict.items():
                if isinstance(df, pd.DataFrame) and not df.empty and df.shape[1] > 0:
                    return df
            return next(iter(df_dict.values())) if df_dict else None
        except Exception:
            pass

    # Parquet
    if name_lower.endswith(".parquet"):
        try:
            try:
                import pyarrow.parquet as pq  # optional for fast slicing
                table = pq.read_table(bio)
                if max_rows is not None:
                    table = table.slice(0, max_rows)
                return table.to_pandas(types_mapper=_arrow_types_mapper)
            except Exception:
                # Fallback to pandas reader
                df = pd.read_parquet(bio)
                if max_rows is not None and len(df) > max_rows:
                    df = df.iloc[:max_rows]
                return df
        except Exception:
            pass

    # CSV/TSV/TXT (try multiple encodings + delimiter sniff)
    if name_lower.endswith((".csv", ".tsv", ".txt")) or not name_lower.split(".")[-1] in {"xlsx","xls","parquet"}:
        df = _stream_csv(bio, max_rows=max_rows)
        if df is not None:
            return df

    return None


# ------------- CSV streaming -------------

def _stream_csv(buf: io.BytesIO, *, max_rows: Optional[int]) -> Optional[pd.DataFrame]:
    # Read a sample to sniff dialect/encoding
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            buf.seek(0)
            text = io.TextIOWrapper(buf, encoding=enc, newline="")
            sample = text.read(64 * 1024)
        except Exception:
            continue
        finally:
            buf.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            sep = dialect.delimiter
        except Exception:
            sep = None

        # Stream by chunks
        chunks = []
        rows_read = 0
        try:
            for chunk in pd.read_csv(
                buf,
                sep=sep,
                engine="python",
                encoding=enc,
                chunksize=CSV_CHUNK_ROWS,
                iterator=True,
                on_bad_lines="skip",
            ):
                chunks.append(chunk)
                rows_read += len(chunk)
                if max_rows is not None and rows_read >= max_rows:
                    break
        except Exception:
            # Fallback: single shot read
            try:
                buf.seek(0)
                df = pd.read_csv(buf, sep=sep, engine="python", encoding=enc, nrows=max_rows)
                return df if df.shape[1] > 0 else None
            except Exception:
                continue

        if chunks:
            df = pd.concat(chunks, ignore_index=True)
            if max_rows is not None and len(df) > max_rows:
                df = df.iloc[:max_rows]
            return df if df.shape[1] > 0 else None

    return None


# ------------- JSON readers -------------

def _looks_like_json(b: bytes) -> bool:
    # Strip UTF-8 BOM + whitespace/newlines/tabs from the beginning before checking
    s = b.lstrip(b"\xef\xbb\xbf\r\n\t ")
    return s[:1] in (b"{", b"[")


def _read_json_like(b: bytes, *, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Robust JSON: handles JSONL/NDJSON and nested JSON. Streams when possible.
    This version doesn't rely on newlines in the first 1KB (long first lines are ok).
    It first attempts JSON-lines by reading line-by-line; if the first non-empty line
    isn't valid JSON, it falls back to standard JSON (object/array).
    """
    # Try JSON Lines / NDJSON first (line-by-line). Works even if the first newline is far into the file.
    bio = io.BytesIO(b)
    rows: list[Any] | None = []
    try:
        for raw in io.TextIOWrapper(bio, encoding="utf-8-sig", newline=""):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                # If we fail on the very first non-empty line, it's likely not JSONL; abort JL path
                if rows == []:
                    rows = None
                    break
                # Otherwise ignore the bad line (e.g., comments/trailers)
                continue
            rows.append(obj)
            if max_rows is not None and len(rows) >= max_rows:
                break
        if rows:
            if all(isinstance(x, dict) for x in rows):
                return pd.json_normalize(rows, sep="__")
            return pd.DataFrame({"value": rows})
    except Exception:
        pass

    # Fallback: standard JSON (object or array). If ijson is available and top-level is array, stream it.
    try:
        import ijson  # optional
        bio2 = io.BytesIO(b)
        items = ijson.items(io.TextIOWrapper(bio2, encoding="utf-8-sig"), "item")
        rows2 = []
        for i, obj in enumerate(items):
            rows2.append(obj)
            if max_rows is not None and len(rows2) >= max_rows:
                break
        if rows2:
            if all(isinstance(x, dict) for x in rows2):
                return pd.json_normalize(rows2, sep="__")
            return pd.DataFrame({"value": rows2})
    except Exception:
        pass

    # Last resort: full parse
    txt = b.decode("utf-8-sig", errors="ignore").strip()
    obj = json.loads(txt)

    if isinstance(obj, list):
        if obj and all(isinstance(x, dict) for x in obj):
            if max_rows is not None:
                obj = obj[:max_rows]
            return pd.json_normalize(obj, sep="__")
        return pd.DataFrame({"value": obj[: (max_rows or len(obj))]})

    if isinstance(obj, dict):
        path = _find_best_record_path(obj)
        if path:
            records = _extract_nested_list(obj, path)
            if max_rows is not None:
                records = records[:max_rows]
            return pd.json_normalize(records, sep="__")
        return pd.json_normalize(obj, sep="__")

    return pd.DataFrame({"value": [obj]})


def _find_best_record_path(obj: Any) -> list[str] | None:
    best: tuple[list[str], int] | None = None
    def walk(o: Any, path: list[str]):
        nonlocal best
        if isinstance(o, list) and o and all(isinstance(x, dict) for x in o):
            cand = (path, len(o))
            if best is None or cand[1] > best[1]:
                best = cand
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(v, path + [str(k)])
    walk(obj, [])
    return best[0] if best else None


def _extract_nested_list(obj: Any, path: Iterable[str]) -> list[dict]:
    cur = obj
    for p in path:
        if isinstance(cur, dict):
            cur = cur.get(p, [])
        else:
            return []
    return cur if isinstance(cur, list) else []


# ------------- Memory helpers -------------

def _optimize_dtypes(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None:
        return None
    try:
        df = df.convert_dtypes()  # pandas nullable types
        for col in df.select_dtypes(include=["integer", "floating"]).columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if pd.api.types.is_integer_dtype(s):
                df[col] = pd.to_numeric(s, downcast="integer")
            else:
                df[col] = pd.to_numeric(s, downcast="float")
        # Optional: convert low-cardinality object/string to category
        for col in df.columns:
            s = df[col]
            if s.dtype == "string" or s.dtype == object:
                nunique = s.nunique(dropna=True)
                if nunique > 0 and nunique <= min(1000, 0.5 * len(df)):
                    # Avoid categorizing date-like strings to prevent ordered-categorical min/max errors
                    try:
                        sample = s.dropna().astype(str).head(200)
                        parse_rate = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True).notna().mean() if not sample.empty else 0
                    except Exception:
                        parse_rate = 0
                    if parse_rate < 0.7:
                        df[col] = s.astype("category")
        return df
    except Exception:
        return df


def _arrow_types_mapper(pa_type):
    # Keep integers as pandas nullable Int*, otherwise default mapping
    import pyarrow as pa
    if pa.types.is_integer(pa_type):
        width = pa_type.bit_width
        return getattr(pd, f"Int{width}Dtype")()
    return None
