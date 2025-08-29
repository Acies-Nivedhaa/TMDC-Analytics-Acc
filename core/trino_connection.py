# core/trino_connection.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd

@dataclass
class TrinoConfig:
    host: str
    port: int = 443
    user: str = ""
    password: str = ""
    http_scheme: str = "https"                # "http" if no TLS
    http_headers: Optional[Dict[str, str]] = None  # e.g. {"cluster-name": "minervac"}
    catalog: str = "icebase"
    schema: str = "telemetry"

def _get_trino_modules():
    try:
        from trino.dbapi import connect as trino_connect
        from trino.auth import BasicAuthentication
        return trino_connect, BasicAuthentication
    except ModuleNotFoundError as e:
        # Raise a friendly error only when we actually try to connect
        raise RuntimeError(
            "Python package 'trino' not installed in this environment. "
            "Install with: pip install trino"
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

def query_df(sql: str, cfg: TrinoConfig, params: Optional[Any] = None) -> pd.DataFrame:
    with get_connection(cfg) as conn:
        return pd.read_sql(sql, conn, params=params)

def q_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'
