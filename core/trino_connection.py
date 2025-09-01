# core/trino_connection.py
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Optional, Dict, Any, List
import pandas as pd

@dataclass
class TrinoConfig:
    host: str
    port: int = 443
    user: str = ""
    password: str = ""
    http_scheme: str = "https"                  # "http" if no TLS
    http_headers: Optional[Dict[str, str]] = None  # e.g. {"cluster-name": "minerva"}
    catalog: str = "lakehouse"
    schema: str = "default"

def _get_trino_modules():
    try:
        from trino.dbapi import connect as trino_connect
        from trino.auth import BasicAuthentication
        return trino_connect, BasicAuthentication
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Python package 'trino' not installed. Run: pip install trino"
        ) from e

def get_connection(cfg: TrinoConfig):
    trino_connect, BasicAuthentication = _get_trino_modules()
    return trino_connect(
        host=cfg.host,
        port=cfg.port,
        http_scheme=cfg.http_scheme,
        auth=BasicAuthentication(cfg.user, cfg.password),
        http_headers=cfg.http_headers or {},
        catalog=cfg.catalog,
        schema=cfg.schema,
    )

def _looks_like_tcp_gateway(cfg: TrinoConfig) -> bool:
    return cfg.host.startswith("tcp.") or cfg.port in (6432, 5432)

def _build_hints(cfg: TrinoConfig) -> List[str]:
    hints: List[str] = []
    if _looks_like_tcp_gateway(cfg):
        hints.append(
            "Host/port looks like a TCP/Postgres gateway (e.g., 6432). "
            "Use the Trino coordinator host (e.g., your-app.dataos.app) with HTTPS:443."
        )
    if cfg.http_scheme == "https" and cfg.port not in (443, 8443):
        hints.append("For HTTPS, use port 443 (or 8443).")
    if cfg.http_scheme == "http" and cfg.port not in (8080, 80):
        hints.append("For HTTP (no TLS), use port 8080 (or 80).")
    if not (cfg.http_headers or {}).get("cluster-name"):
        hints.append("Set http header 'cluster-name' to your cluster (e.g., 'minerva').")
    return hints

def get_connection_smart(cfg: TrinoConfig):
    """
    Try the given config; on SSL version errors, try sensible fallbacks automatically.
    """
    tried = [f"{cfg.http_scheme}://{cfg.host}:{cfg.port}"]
    try:
        return get_connection(cfg)
    except Exception as e:
        msg = str(e)

        # Auto-fallback on typical TLS/port mismatch
        fallbacks: list[TrinoConfig] = []
        if "WRONG_VERSION_NUMBER" in msg or "HTTPSConnectionPool" in msg:
            if cfg.http_scheme == "https":
                # try HTTP on 8080
                fallbacks.append(replace(cfg, http_scheme="http", port=8080))
            else:
                # try HTTPS on 443
                fallbacks.append(replace(cfg, http_scheme="https", port=443))

        for alt in fallbacks:
            tried.append(f"{alt.http_scheme}://{alt.host}:{alt.port}")
            try:
                return get_connection(alt)
            except Exception:
                pass  # try next fallback

        hints = _build_hints(cfg)
        hint_text = ("\nHints:\n- " + "\n- ".join(hints)) if hints else ""
        raise RuntimeError(
            f"Failed to connect to Trino.\n"
            f"Tried: {', '.join(tried)}\n"
            f"Underlying error: {msg}{hint_text}"
        ) from e

def query_df(sql: str, cfg: TrinoConfig, params: Optional[Any] = None) -> pd.DataFrame:
    with get_connection_smart(cfg) as conn:
        return pd.read_sql(sql, conn, params=params)

def q_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'

def build_select_sql(
    catalog: str, schema: str, table: str, limit: int | None = None, *, quote_table: bool = False
) -> str:
    parts = [q_ident(catalog), q_ident(schema)]
    t = (table or "").strip()
    needs_quote = quote_table or (not t) or any(ch in t for ch in [' ', '"', '.'])
    parts.append(q_ident(t) if needs_quote else t)
    fq = ".".join(parts)
    sql = f"SELECT * FROM {fq}"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return sql
