# core/trino_connection.py
from __future__ import annotations

"""
Thin Trino connector:
- Small dataclass to hold connection settings
- Safe connect() with smart fallbacks (scheme/port) + human hints
- Convenience helpers: query_df(), identifier quoting, simple SELECT builder
- Optional Streamlit glossary expander for end-users
"""

from dataclasses import dataclass, replace
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd


# ---------------------------------------------------------------------
# Configuration model
# ---------------------------------------------------------------------

@dataclass
class TrinoConfig:
    """
    Minimal Trino connection settings.

    Notes
    -----
    - `http_scheme`: "https" for TLS (recommended), "http" for local/non-TLS.
    - `http_headers`: Some deployments require custom headers like
      {"cluster-name": "minerva"} to route to the right cluster.
    - `catalog` / `schema`: Default catalog & schema used by the session.
    """
    host: str
    port: int = 443
    user: str = ""
    password: str = ""
    http_scheme: str = "https"                       # "http" if no TLS
    http_headers: Optional[Dict[str, str]] = None    # e.g. {"cluster-name": "minerva"}
    catalog: str = "lakehouse"
    schema: str = "default"


# ---------------------------------------------------------------------
# Imports (lazy), raw connect, and helpers
# ---------------------------------------------------------------------

def _get_trino_modules():
    """
    Import the Trino DB-API client lazily so importing this module
    doesn't hard-require the dependency unless we actually connect.
    """
    try:
        from trino.dbapi import connect as trino_connect
        from trino.auth import BasicAuthentication
        return trino_connect, BasicAuthentication
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Python package 'trino' not installed. Install with: pip install trino"
        ) from e


def get_connection(cfg: TrinoConfig):
    """
    Build a raw Trino DB-API connection from the config.
    Raises any underlying errors directly.

    Returns
    -------
    DB-API connection object (context-manager compatible)
    """
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


# ---------------------------------------------------------------------
# Diagnostics / fallbacks / hints
# ---------------------------------------------------------------------

def _looks_like_tcp_gateway(cfg: TrinoConfig) -> bool:
    """
    Heuristic: user pointed at a TCP/Postgres gateway rather than the Trino coordinator.
    """
    return cfg.host.startswith("tcp.") or cfg.port in (6432, 5432)


def _build_hints(cfg: TrinoConfig, last_error: str | None = None) -> List[str]:
    """
    Produce human-friendly configuration hints based on the provided config
    and (optionally) the last error message we saw.
    """
    hints: List[str] = []

    if _looks_like_tcp_gateway(cfg):
        hints.append(
            "Host/port looks like a TCP/Postgres gateway (e.g., 6432). "
            "Use the Trino coordinator host (e.g., your-app.dataos.app) with HTTPS:443."
        )

    if cfg.http_scheme == "https" and cfg.port not in (443, 8443):
        hints.append("For HTTPS, try port 443 (or 8443 if your cluster uses it).")

    if cfg.http_scheme == "http" and cfg.port not in (8080, 80):
        hints.append("For HTTP (no TLS), try port 8080 (or 80).")

    if not (cfg.http_headers or {}).get("cluster-name"):
        hints.append("Add http header 'cluster-name' (e.g., 'minerva') if your platform requires routing.")

    if last_error:
        msg = last_error.lower()
        if "connection refused" in msg:
            hints.append("Connection refused: verify host/port reachability (VPC, VPN, security groups).")
        if "name or service not known" in msg or "temporary failure in name resolution" in msg:
            hints.append("DNS failure: check the hostname spelling or your DNS/VPN setup.")
        if "unauthorized" in msg or "authentication" in msg:
            hints.append("Auth failed: verify username/password and any SSO/proxy requirements.")

    return hints


def _fallback_matrix(cfg: TrinoConfig) -> List[TrinoConfig]:
    """
    Generate a small list of sensible scheme/port fallbacks when the first attempt fails.
    """
    alts: List[TrinoConfig] = []
    if cfg.http_scheme == "https":
        # Try common secure ports + a plain HTTP option
        for p in (443, 8443):
            if p != cfg.port:
                alts.append(replace(cfg, port=p))
        alts.append(replace(cfg, http_scheme="http", port=8080))
    else:
        # Try typical HTTP ports + a secure option
        for p in (8080, 80):
            if p != cfg.port:
                alts.append(replace(cfg, port=p))
        alts.append(replace(cfg, http_scheme="https", port=443))
    return alts


def get_connection_smart(cfg: TrinoConfig):
    """
    Connect to Trino with light resilience:
      1) Try the given configuration.
      2) If it fails with a likely scheme/port mismatch, try a few sensible fallbacks.
      3) If all fail, raise a single RuntimeError with the URLs tried + clear hints.

    This does NOT swallow genuine auth/query errors once a connection is established.
    """
    tried: List[str] = [f"{cfg.http_scheme}://{cfg.host}:{cfg.port}"]
    try:
        return get_connection(cfg)
    except Exception as e:
        last_error = str(e)

        # Detect common TLS/HTTP mismatches or transport errors and try alternates
        mismatch_signals = (
            "WRONG_VERSION_NUMBER",
            "HTTPSConnectionPool",
            "HTTPConnectionPool",
            "SSLError",
            "EOF occurred in violation of protocol",
            "bad handshake",
            "Connection refused",
        )
        fallbacks: List[TrinoConfig] = _fallback_matrix(cfg) if any(s in last_error for s in mismatch_signals) else []

        for alt in fallbacks:
            tried.append(f"{alt.http_scheme}://{alt.host}:{alt.port}")
            try:
                return get_connection(alt)
            except Exception:
                # keep looping; we’ll surface aggregate info below
                pass

        hints = _build_hints(cfg, last_error=last_error)
        hint_text = ("\nHints:\n- " + "\n- ".join(hints)) if hints else ""
        raise RuntimeError(
            "Failed to connect to Trino.\n"
            f"Tried: {', '.join(tried)}\n"
            f"Underlying error: {last_error}{hint_text}"
        ) from e


# ---------------------------------------------------------------------
# Public convenience functions
# ---------------------------------------------------------------------

def query_df(sql: str, cfg: TrinoConfig, params: Optional[Any] = None) -> pd.DataFrame:
    """
    Run a SQL statement and return a pandas DataFrame.

    Parameters
    ----------
    sql : str
        SQL text. Use `?` placeholders with `params` for safety when applicable.
    cfg : TrinoConfig
        Connection settings.
    params : Any, optional
        DB-API param style supported by python-trino.

    Returns
    -------
    pandas.DataFrame
    """
    with get_connection_smart(cfg) as conn:
        return pd.read_sql(sql, conn, params=params)


def q_ident(name: str) -> str:
    """
    Quote an identifier with ANSI double quotes and escape embedded quotes.
    Useful for catalog/schema/table that contain uppercase, spaces, or reserved words.
    """
    return '"' + (name or "").replace('"', '""') + '"'


def build_select_sql(
    catalog: str,
    schema: str,
    table: str,
    limit: int | None = None,
    *,
    quote_table: bool = False
) -> str:
    """
    Construct a simple SELECT * with a fully-qualified table name.

    Parameters
    ----------
    catalog, schema, table : str
        Pieces of the fully-qualified name.
    limit : int | None
        Optional LIMIT clause.
    quote_table : bool
        Force quoting the table part (identifier); catalog/schema are always quoted.

    Returns
    -------
    str : SELECT statement
    """
    parts = [q_ident(catalog), q_ident(schema)]
    t = (table or "").strip()
    needs_quote = quote_table or (not t) or any(ch in t for ch in [' ', '"', '.'])
    parts.append(q_ident(t) if needs_quote else t)
    fq = ".".join(parts)
    sql = f"SELECT * FROM {fq}"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return sql


# ---------------------------------------------------------------------
# Optional: tiny glossary expander for the Trino connection UI
# Call this from the Streamlit page where users configure Trino.
# ---------------------------------------------------------------------

def render_trino_glossary() -> None:
    """
    Show a compact, non-technical glossary for connection fields (Streamlit).
    Safe to import even if Streamlit isn't installed; we import lazily.
    """
    try:
        import streamlit as st
    except Exception:
        return  # no-op if Streamlit not available

    with st.expander("Key terms (Trino connection quick guide)", expanded=False):
        st.markdown(
            """
            - **Coordinator host** — The HTTPS endpoint of your Trino cluster (not a TCP/DB gateway).
            - **Port** — Usually **443** for HTTPS (or **8443**); **8080/80** for HTTP (no TLS).
            - **HTTP scheme** — **https** (encrypted) or **http** (plain). Prefer **https**.
            - **User / Password** — Credentials for Basic authentication (set by your platform/IdP).
            - **HTTP headers** — Extra headers like `cluster-name: minerva` used by some platforms for routing.
            - **Catalog / Schema** — Default namespace for your session (like database/schema).
            - **Common fixes**
              - “Wrong SSL version” → flip **http/https** or use the standard port for that scheme.
              - “Connection refused” → wrong port, security group, or VPN/VPC route.
              - DNS errors → check the hostname spelling and VPN/DNS configuration.
            """
        )
