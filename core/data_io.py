# core/data_io.py
from __future__ import annotations
"""
Robust, dependency-light data readers for common tabular formats.

Supports:
- CSV/TSV/TXT  (streamed in chunks, encoding & delimiter sniffing)
- Excel        (first non-empty sheet)
- JSONL/NDJSON (line-by-line)
- JSON         (auto-flatten; streams large top-level arrays if ijson is present)
- Parquet      (PyArrow preferred; falls back to pandas)
- Wrappers     (.gz and .zip, including .gz inside .zip)

Also includes mild dtype optimization (nullable ints/floats, safe category downcast).
"""

from typing import Optional, Any, Iterable
import io
import json
import csv
import gzip
import zipfile
from pathlib import Path

import pandas as pd

__all__ = [
    "read_any",          # main entry
    "read_any_preview",  # capped read for previews
    "read_zip_all",      # read ALL tables from a ZIP (list of (name, df))
]

# -------- Tunables (safe defaults) --------
MAX_PREVIEW_ROWS = 250_000   # cap for preview reads (CSV/JSONL) to keep memory safe
CSV_CHUNK_ROWS   = 100_000   # chunk size when streaming CSV


# ===================== Public API =====================

def read_any(upload, *, max_rows: Optional[int] = None) -> Optional[pd.DataFrame]:
    """
    Read a wide range of file types into a pandas DataFrame.

    Args:
        upload: File-like object, bytes, path, or object with a .name attribute.
        max_rows: Optional row cap (streamed/limited when possible).

    Returns:
        DataFrame or None if reading fails.
    """
    if upload is None:
        return None

    name = _get_name(upload).lower()
    try:
        # --- gzip wrapper ---
        if name.endswith(".gz"):
            inner_bytes = gzip.decompress(_to_bytes(upload))
            inner_name  = name[:-3] or "decompressed"
            df = _read_from_bytes(inner_bytes, inner_name, max_rows=max_rows)
            return _optimize_dtypes(df) if isinstance(df, pd.DataFrame) else df

        # --- zip wrapper (may contain multiple files) ---
        if name.endswith(".zip"):
            return _read_from_zip(_to_bytes(upload), max_rows=max_rows)

        # --- direct read ---
        df = _read_from_bytes(_to_bytes(upload), name, max_rows=max_rows)
        return _optimize_dtypes(df) if isinstance(df, pd.DataFrame) else df

    except MemoryError:
        # Fallback: try again with a safe preview cap
        try:
            df = _read_from_bytes(_to_bytes(upload), name, max_rows=MAX_PREVIEW_ROWS)
            return _optimize_dtypes(df) if isinstance(df, pd.DataFrame) else df
        except Exception:
            return None
    except Exception:
        # Swallow and return None to keep the app resilient
        return None


def read_any_preview(upload) -> Optional[pd.DataFrame]:
    """Convenience wrapper that applies a safe row cap for very large files."""
    return read_any(upload, max_rows=MAX_PREVIEW_ROWS)


def read_zip_all(upload, *, max_rows: Optional[int] = None) -> list[tuple[str, pd.DataFrame]]:
    """
    Read ALL supported tabular files inside a ZIP and return:
        [(inner_filename, DataFrame), ...]

    Uses the same logic as read_any/_read_from_bytes for each inner file.
    """
    b = _to_bytes(upload)
    return _read_zip_all_from_bytes(b, max_rows=max_rows)


# ===================== Internals =====================

def _get_name(upload) -> str:
    """Best-effort filename extraction for heterogeneous upload objects."""
    if hasattr(upload, "name") and isinstance(upload.name, str):
        return upload.name
    if isinstance(upload, (str, Path)):
        return str(upload)
    return "uploaded"


def _to_bytes(upload) -> bytes:
    """
    Normalize input into raw bytes.
    Supports file-like objects, bytes/bytearray, paths, and objects with .getvalue()/.read().
    """
    if isinstance(upload, (str, Path)):
        with open(upload, "rb") as f:
            return f.read()

    if hasattr(upload, "getvalue"):
        return upload.getvalue()  # e.g., Streamlit UploadedFile

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
    """
    Iterate files inside a ZIP and read supported ones.
    Returns non-empty DataFrames only.
    """
    out: list[tuple[str, pd.DataFrame]] = []
    with zipfile.ZipFile(io.BytesIO(b)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue

            inner_name = info.filename
            lower = inner_name.lower()

            # Only attempt supported extensions (we also handle .gz inside the zip)
            if not (_is_supported(lower) or lower.endswith(".gz")):
                continue

            try:
                raw = zf.read(info)
            except Exception:
                continue

            # If the inner file is .gz, decompress and strip the suffix before dispatch
            name_for_reader = lower
            if name_for_reader.endswith(".gz"):
                try:
                    raw = gzip.decompress(raw)
                    name_for_reader = name_for_reader[:-3] or "decompressed"
                except Exception:
                    continue

            df = _read_from_bytes(raw, name_for_reader, max_rows=max_rows)
            if isinstance(df, pd.DataFrame) and not df.empty and df.shape[1] > 0:
                base = inner_name.rsplit("/", 1)[-1]  # strip folder path for display
                out.append((base, _optimize_dtypes(df)))
    return out


def _read_from_zip(b: bytes, *, max_rows: Optional[int] = None) -> Optional[pd.DataFrame]:
    """
    Back-compat convenience: return only the FIRST supported table in the ZIP.
    Prefer read_zip_all() upstream to ingest ALL tables.
    """
    tables = _read_zip_all_from_bytes(b, max_rows=max_rows)
    return tables[0][1] if tables else None


def _is_supported(name: str) -> bool:
    """Check if a filename’s extension looks like a supported tabular format."""
    return name.endswith((
        ".csv", ".tsv", ".txt",
        ".xlsx", ".xls",
        ".json", ".jsonl", ".ndjson",
        ".parquet",
    ))


def _read_from_bytes(b: bytes, name_lower: str, *, max_rows: Optional[int] = None) -> Optional[pd.DataFrame]:
    """
    Dispatch reader by extension (with JSON sniffing fallback for unknown names).
    """
    bio = io.BytesIO(b)

    # --- JSON family first (by extension or sniff) ---
    if name_lower.endswith((".json", ".jsonl", ".ndjson")) or _looks_like_json(b):
        try:
            return _read_json_like(b, max_rows=max_rows)
        except Exception:
            pass

    # --- Excel ---
    if name_lower.endswith((".xlsx", ".xls")):
        try:
            # pandas supports nrows for Excel (openpyxl engine)
            df_dict = pd.read_excel(bio, sheet_name=None, nrows=max_rows)
            for _, df in df_dict.items():
                if isinstance(df, pd.DataFrame) and not df.empty and df.shape[1] > 0:
                    return df
            return next(iter(df_dict.values())) if df_dict else None
        except Exception:
            pass

    # --- Parquet ---
    if name_lower.endswith(".parquet"):
        try:
            try:
                # Fast path with PyArrow (slice before materializing)
                import pyarrow.parquet as pq  # optional
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

    # --- CSV/TSV/TXT or unknown (try CSV last) ---
    if name_lower.endswith((".csv", ".tsv", ".txt")) or name_lower.split(".")[-1] not in {"xlsx", "xls", "parquet"}:
        df = _stream_csv(bio, max_rows=max_rows)
        if df is not None:
            return df

    return None


# --------------------- CSV streaming ---------------------

def _stream_csv(buf: io.BytesIO, *, max_rows: Optional[int]) -> Optional[pd.DataFrame]:
    """
    Robust CSV/TSV reader:
      - tries multiple encodings (utf-8-sig, utf-8, latin1)
      - sniffs delimiter when possible
      - streams in chunks, respects max_rows
      - falls back to single-shot read if streaming fails
    """
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            buf.seek(0)
            text = io.TextIOWrapper(buf, encoding=enc, newline="")
            sample = text.read(64 * 1024)  # small sniff sample
        except Exception:
            continue
        finally:
            buf.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            sep = dialect.delimiter
        except Exception:
            sep = None  # let pandas guess

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
            # Fallback: single-shot read
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


# --------------------- JSON readers ---------------------

def _looks_like_json(b: bytes) -> bool:
    """Cheap sniff: strip BOM/whitespace and check the first byte for { or [."""
    s = b.lstrip(b"\xef\xbb\xbf\r\n\t ")
    return s[:1] in (b"{", b"[")


def _read_json_like(b: bytes, *, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Robust JSON reader:
      1) Try JSON Lines / NDJSON by scanning line-by-line (works even if the first newline is far).
      2) If not JSONL, try streaming a top-level array via ijson (optional).
      3) Fallback to full parse and auto-flatten nested objects.
    """
    # --- (1) JSON Lines / NDJSON ---
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

    # --- (2) Streaming top-level array via ijson (if available) ---
    try:
        import ijson  # optional dependency
        bio2 = io.BytesIO(b)
        items = ijson.items(io.TextIOWrapper(bio2, encoding="utf-8-sig"), "item")
        rows2 = []
        for _, obj in zip(range(max_rows or 10**12), items):
            rows2.append(obj)
        if rows2:
            if all(isinstance(x, dict) for x in rows2):
                return pd.json_normalize(rows2, sep="__")
            return pd.DataFrame({"value": rows2})
    except Exception:
        pass

    # --- (3) Full parse fallback ---
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
    """
    Heuristically find the “largest list of dicts” within a nested JSON document.
    Returns a path (list of keys) or None.
    """
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
    """Follow a path inside a dict to extract a list[dict] if present."""
    cur = obj
    for p in path:
        if isinstance(cur, dict):
            cur = cur.get(p, [])
        else:
            return []
    return cur if isinstance(cur, list) else []


# --------------------- Memory helpers ---------------------

def _optimize_dtypes(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Light, safe dtype optimization:
      - convert_dtypes() → pandas' nullable types
      - downcast ints/floats where possible
      - optional: low-cardinality object/string → category
        (skips date-like strings to avoid surprises)
    """
    if df is None:
        return None
    try:
        df = df.convert_dtypes()  # pandas nullable types

        # Downcast numeric columns
        for col in df.select_dtypes(include=["integer", "floating"]).columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if pd.api.types.is_integer_dtype(s):
                df[col] = pd.to_numeric(s, downcast="integer")
            else:
                df[col] = pd.to_numeric(s, downcast="float")

        # Convert some object/string to 'category' when safe
        for col in df.columns:
            s = df[col]
            if s.dtype == "string" or s.dtype == object:
                nunique = s.nunique(dropna=True)
                # Only if reasonably low cardinality (and non-empty)
                if nunique > 0 and nunique <= min(1000, 0.5 * len(df)):
                    # Avoid categorizing date-like strings (can act oddly in ops)
                    try:
                        sample = s.dropna().astype(str).head(200)
                        parse_rate = (
                            pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
                              .notna()
                              .mean()
                            if not sample.empty else 0
                        )
                    except Exception:
                        parse_rate = 0
                    if parse_rate < 0.7:
                        df[col] = s.astype("category")

        return df
    except Exception:
        # If anything goes wrong, return the original df untouched
        return df


def _arrow_types_mapper(pa_type):
    """
    Map PyArrow integer types to pandas' nullable Int*Dtype to preserve nulls.
    Return None for default mapping on other types.
    """
    import pyarrow as pa  # imported lazily here to keep top-level deps light
    if pa.types.is_integer(pa_type):
        width = pa_type.bit_width
        return getattr(pd, f"Int{width}Dtype")()
    return None


# ===================== Optional UI helper =====================

def render_data_io_glossary() -> None:
    """
    Optional Streamlit expander with short explanations of I/O terms.
    Call from your Upload/Import tab. No effect unless called.
    """
    try:
        import streamlit as st
    except Exception:
        return  # silently no-op if Streamlit isn't available

    with st.expander("What these file/format terms mean (quick guide)", expanded=False):
        st.markdown(
            """
            - **CSV/TSV/TXT** — Text tables where values are separated by commas, tabs, etc.
            - **Delimiter sniffing** — We auto-detect whether it’s comma, tab, semicolon, or pipe.
            - **Encoding** — Character set (e.g., UTF-8). We try common ones automatically.
            - **JSONL / NDJSON** — One JSON object **per line**; great for streaming big files.
            - **Nested JSON (auto-flatten)** — We expand nested objects into columns like `parent__child`.
            - **ijson streaming** — If installed, we stream large top-level JSON arrays without loading everything.
            - **Parquet (PyArrow)** — A fast, columnar format; supports slicing the first *N* rows efficiently.
            - **GZIP / ZIP** — Compressed files; we decompress and then parse the inner file(s).
            - **BOM (byte-order mark)** — A small header some files include; we strip it automatically.
            - **Downcasting / nullable dtypes** — We shrink numeric memory use and preserve missing values (e.g., `Int64`).
            - **Category dtype** — Efficient for low-variety strings; we avoid date-like strings to prevent surprises.
            """
        )
