# core/trino_connection.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Iterable
import pandas as pd


@dataclass
class TrinoConfig:
    host: str
    port: int = 443
    user: str = ""
    password: str = ""
    http_scheme: str = "https"                     # "http" if no TLS
    http_headers: Optional[Dict[str, str]] = None  # e.g. {"cluster-name": "minerva"}
    catalog: str = "icebase"
    schema: str = "telemetry"


def _get_trino_modules():
    try:
        from trino.dbapi import connect as trino_connect
        from trino.auth import BasicAuthentication
        return trino_connect, BasicAuthentication
    except ModuleNotFoundError as e:
        # Raised only when a connection is attempted
        raise RuntimeError(
            "Python package 'trino' is not installed. Install with: pip install trino"
        ) from e


def get_connection(cfg: TrinoConfig):
    trino_connect, BasicAuthentication = _get_trino_modules()
    return trino_connect(
        host=cfg.host,
        port=int(cfg.port),
        http_scheme=cfg.http_scheme,
        auth=BasicAuthentication(cfg.user, cfg.password),
        http_headers=cfg.http_headers or {},
        catalog=cfg.catalog,
        schema=cfg.schema,
    )


def query_df(sql: str, cfg: TrinoConfig, params: Optional[Any] = None) -> pd.DataFrame:
    with get_connection(cfg) as conn:
        return pd.read_sql(sql, conn, params=params)


# --------- SQL helpers (safe identifier quoting) ---------

def q_ident(name: str) -> str:
    """
    Quote an identifier for Trino/ANSI SQL.
    foo -> "foo"; a"b -> "a""b"
    """
    name = (name or "").strip()
    return '"' + name.replace('"', '""') + '"'


def _join_qualified(parts: Iterable[str]) -> str:
    """Join non-empty identifier parts with dots, each safely quoted."""
    parts = [p for p in parts if p]  # drop empty
    return ".".join(q_ident(p) for p in parts)


def build_select_sql(catalog: str, schema: str, table: str, limit: int | None = None) -> str:
    """
    Build a portable SELECT with fully-qualified table:
      SELECT * FROM "catalog"."schema"."table" [LIMIT n]
    """
    fq = _join_qualified([catalog, schema, table])
    sql = f"SELECT * FROM {fq}"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return sql


def fetch_table_preview(cfg: TrinoConfig, table: str, limit: int = 1000) -> pd.DataFrame:
    """
    Convenience: SELECT * from cfg.catalog.cfg.schema.table LIMIT n
    """
    sql = build_select_sql(cfg.catalog, cfg.schema, table, limit)
    return query_df(sql, cfg)
