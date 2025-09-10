# app.py - Analytics Accelerator Main Application
"""
Analytics Accelerator: A comprehensive data analysis and preprocessing tool
Supports file uploads, DataOS (Trino) connections, EDA, preprocessing, and reporting
"""

# Standard library imports
import hashlib
import os
from pathlib import Path

# Third-party imports
import pandas as pd
import streamlit as st
import base64, pathlib

# Core module imports - Data I/O and processing
from core.data_io import read_any, read_zip_all
from core.overview import (
    overview_stats, infer_schema, column_quick_stats,
    suggest_actions, dataset_meta, demo_data, nunique_safe,
    build_summary_pdf, build_summary_html, summary_html_bytes, _REPORTLAB_OK
)
from core.insights_extractors import (
    init_insights_store,
    capture_summary_insights,
    insights_to_json_bytes,
)

# EDA (Exploratory Data Analysis) modules
from core.eda_overview import render_overview
from core.eda_combine import render_combine, ensure_unique_name
from core.eda_missingness import render_missingness
from core.eda_univariate import render_univariate
from core.eda_bivariate import render_bivariate
from core.eda_correlation import render_correlation
from core.eda_types import render_eda_types

# Preprocessing modules
from core.preprocess_overview import render_preprocess_overview
from core.preprocess_missing import render_preprocess_missing
from core.preprocess_outliers import render_preprocess_outliers
from core.preprocess_text import render_preprocess_text
from core.preprocess_timeseries import render_preprocess_timeseries
from core.preprocess_encoding import render_preprocess_encoding

# UI and utility modules
from ui.components import header_bar, kpi_row, section, render_table, control_bar
from core.trino_connection import TrinoConfig, query_df, q_ident
from core.auto_insights import render_auto_insights
from core.json_flatten import detect_jsonlike_columns, flatten_json_columns
from core.final_summary import render_final_summary

# Configure Streamlit page
st.set_page_config(page_title="Analytics Accelerator — Overview", layout="wide")

# =============================================================================
# LOGO AND BRANDING UTILITIES
# =============================================================================

def _svg_data_uri(path: str) -> str:
    """
    Convert SVG file to data URI for inline display
    
    Args:
        path: File path to SVG
        
    Returns:
        Data URI string or empty string if conversion fails
    """
    try:
        data = pathlib.Path(path).read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:image/svg+xml;base64,{b64}"
    except Exception:
        return ""

def _data_uri_from_file(p: Path) -> str:
    """
    Convert image file to data URI with proper MIME type detection
    
    Args:
        p: Path object to image file
        
    Returns:
        Data URI string with appropriate MIME type
    """
    try:
        data = p.read_bytes()
        suf = p.suffix.lower()
        
        # Determine MIME type based on file extension
        if suf == ".svg":
            mime = "image/svg+xml"
        elif suf in (".png", ".apng"):
            mime = "image/png"
        elif suf in (".jpg", ".jpeg"):
            mime = "image/jpeg"
        else:
            return ""
            
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return ""

def _resolve_logo_uri() -> str:
    """
    Resolve logo file from multiple possible locations
    Priority: Environment variable > Repository assets > Current directory assets
    
    Returns:
        Data URI for logo or empty string if not found
    """
    # 1) Check environment variable override
    env = os.getenv("APP_LOGO_PATH")
    if env:
        uri = _data_uri_from_file(Path(env).expanduser())
        if uri:
            return uri

    # 2) Check repository assets directory
    here = Path(__file__).parent
    for name in ("logo.svg", "logo.png", "logo.jpg", "logo.jpeg"):
        uri = _data_uri_from_file(here / "assets" / name)
        if uri:
            return uri

    # 3) Check current working directory assets
    for name in ("logo.svg", "logo.png", "logo.jpg", "logo.jpeg"):
        uri = _data_uri_from_file(Path.cwd() / "assets" / name)
        if uri:
            return uri

    return ""

def _fallback_logo_svg() -> str:
    """
    Generate inline SVG fallback logo when no logo file is found
    
    Returns:
        SVG markup string for fallback "AA" logo
    """
    return """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128"
         width="22" height="22" role="img" aria-label="Logo">
      <rect x="0" y="0" width="128" height="128" rx="24" fill="#111827"/>
      <text x="50%" y="56%" text-anchor="middle" dominant-baseline="middle"
            font-family="Inter,system-ui,Arial,sans-serif" font-size="64" fill="#ffffff">AA</text>
    </svg>
    """

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

# Initialize session state with default values
ss = st.session_state

# Core data storage
ss.setdefault("datasets", {})          # name -> DataFrame (current state)
ss.setdefault("raw_datasets", {})      # name -> original DataFrame (for Restore RAW)
ss.setdefault("active_ds", None)       # currently selected dataset name
ss.setdefault("df_history", [])        # undo stack for active dataset
ss.setdefault("activity_log", [])      # user activity log
ss.setdefault("step", "Overview")      # current application step
ss.setdefault("_step_changed", False)  # flag to track step changes

# File upload management
ss.setdefault("loaded_hashes", set())   # prevent duplicate file uploads
ss.setdefault("hash_to_name", {})      # hash -> dataset name mapping
ss.setdefault("file_meta", {})         # file metadata storage
ss.setdefault("uploader_key", 0)       # force uploader widget refresh
ss.setdefault("json_flat_sigs", {})    # JSON flattening signatures

# Trino/DataOS connection management
ss.setdefault("tr_loaded_map", {})     # dataset_name -> "catalog.schema.table"
ss.setdefault("trino_cache", {         # Trino metadata cache
    "sig": "", 
    "catalogs": [], 
    "schemas": {}, 
    "tables": {}
})
ss.setdefault("tr_auto_kick_done", False)  # prevent auto-catalog loading loops

# Initialize insights storage system
init_insights_store(ss)

# =============================================================================
# UI SETUP - LOGO AND TITLE
# =============================================================================

# Resolve logo URI (file-based or fallback)
LOGO_URI = _resolve_logo_uri()

# Fixed position logo at top-left corner
if LOGO_URI:
    st.markdown(
        f'''
        <div style="position:fixed;top:8px;left:12px;z-index:2147483647;">
          <img src="{LOGO_URI}" alt="Logo" style="height:22px;display:block"/>
        </div>
        ''',
        unsafe_allow_html=True,
    )
else:
    # Use fallback SVG logo
    st.markdown(
        f'''
        <div style="position:fixed;top:8px;left:12px;z-index:2147483647;">
          {_fallback_logo_svg()}
        </div>
        ''',
        unsafe_allow_html=True,
    )

# Main application title - centered and prominent
st.markdown(
    """
    <style>
      .aa-brandbar{
        width:100%;
        text-align:center;
        margin:36px 0 12px;
      }
      .aa-brandbar .aa-title{
        margin:0;
        font-weight:900;
        font-size:44px;         /* Larger than page section titles */
        line-height:1.2;
        letter-spacing:.3px;
      }
    </style>
    <div class="aa-brandbar"><div class="aa-title">Analytics Accelerator</div></div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def log(msg: str):
    """Add message to activity log"""
    ss.activity_log.append(msg)

def push_history(df: pd.DataFrame):
    """Save current DataFrame state to history stack for undo functionality"""
    ss.df_history.append(df.copy())

def pop_history() -> pd.DataFrame | None:
    """Retrieve last DataFrame state from history stack"""
    if ss.df_history:
        return ss.df_history.pop()
    return None

def delete_dataset(ds_name: str):
    """
    Remove dataset from all storage locations and clean up references
    
    Args:
        ds_name: Name of dataset to delete
    """
    # Clean up Trino connection mapping
    ss.tr_loaded_map.pop(ds_name, None)
    
    # Remove from main storage
    ss.datasets.pop(ds_name, None)
    ss.raw_datasets.pop(ds_name, None)
    
    # Clean up file metadata references
    for key in list(ss.file_meta.keys()):
        if ss.file_meta[key].get("name") == ds_name:
            ss.file_meta.pop(key, None)
    
    # Update active dataset if we just deleted it
    if ss.active_ds == ds_name:
        ss.active_ds = next(iter(ss.datasets.keys()), None)
    
    log(f"Removed dataset '{ds_name}'.")

def dataset_combo(label: str, key_prefix: str):
    """
    Render dataset selection dropdown and update active dataset
    
    Args:
        label: Label for the selectbox
        key_prefix: Unique prefix for widget key
        
    Returns:
        Selected DataFrame or None if no datasets available
    """
    names = sorted(ss.datasets.keys())
    if not names:
        return None
        
    # Ensure active dataset is valid
    if ss.active_ds not in names:
        ss.active_ds = names[0]
        
    sel = st.selectbox(
        label, names,
        index=names.index(ss.active_ds),
        key=f"{key_prefix}_dataset",
        placeholder="Type to search…",
        label_visibility="visible",
    )
    ss.active_ds = sel
    return ss.datasets[ss.active_ds]

def render_left_steps():
    """Render left sidebar navigation with step pills"""
    nav_labels = ["Overview", "EDA", "Preprocess", "Final Summary"]
    
    # Custom CSS for pill-style navigation
    st.markdown("""
    <style>
      .left-pills button {
        border-radius: 999px !important; 
        border: 1px solid rgba(0,0,0,.08);
        margin: 4px 0; 
        padding: 6px 12px !important;
      }
      .left-pills .active button {
        background: rgb(59,130,246) !important; 
        color: white !important;
        border-color: rgb(59,130,246) !important;
      }
      .left-pills button:hover {
        border-color: rgba(0,0,0,.18);
      }
    </style>
    """, unsafe_allow_html=True)
    
    current = st.session_state.get("step", "Overview")
    for label in nav_labels:
        active = (label == current)
        st.markdown(f'<div class="left-pills {"active" if active else ""}">', unsafe_allow_html=True)
        if st.button(label, key=f"nav_{label}", use_container_width=True):
            if label != current:
                st.session_state["_step_changed"] = True
                st.session_state["step"] = label
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# MAIN LAYOUT
# =============================================================================

# Create two-column layout: sidebar (22%) and main content (78%)
left, right = st.columns([0.22, 0.78], gap="large")

# LEFT SIDEBAR - Navigation and Activity Log
with left:
    render_left_steps()
    
    st.markdown("**Activity log**")
    if ss.activity_log:
        # Show last 12 activity log entries
        for line in ss.activity_log[-12:]:
            st.write("• ", line)
    else:
        st.caption("—")

# RIGHT MAIN CONTENT AREA
with right:
    # Header bar and control buttons
    header_bar(ss.step)
    clicked = control_bar()

    # =============================================================================
    # GLOBAL ACTIONS (Clear, Undo, Restore)
    # =============================================================================

    # CLEAR - Reset entire application state
    if clicked.get("clear"):
        ss.datasets.clear()
        ss.raw_datasets.clear()
        ss.active_ds = None
        ss.df_history.clear()
        ss.activity_log.clear()
        ss.loaded_hashes = set()
        ss.hash_to_name = {}
        ss.file_meta = {}
        ss.tr_loaded_map = {}
        ss.pop("eda_sample_n", None)
        ss.uploader_key += 1
        ss.step = "Overview"
        st.rerun()

    # UNDO - Revert last change to active dataset
    if clicked.get("undo") and ss.active_ds and ss.df_history:
        prev = pop_history()
        if prev is not None:
            ss.datasets[ss.active_ds] = prev
            log(f"Undid last change on '{ss.active_ds}'.")

    # RESTORE RAW - Revert active dataset to original state
    if clicked.get("restore") and ss.active_ds:
        base = ss.raw_datasets.get(ss.active_ds)
        if isinstance(base, pd.DataFrame):
            ss.datasets[ss.active_ds] = base.copy()
            ss.df_history.clear()
            log(f"Restored RAW for '{ss.active_ds}'.")

    # =============================================================================
    # OVERVIEW STEP - Data Loading and Initial Analysis
    # =============================================================================
    
    if ss.step == "Overview":

        # Data Source Selection Section
        with section("Select Files", expandable=False):
            st.caption("Limit ~200MB per file • CSV, XLSX, XLS, JSON/JSONL, PARQUET, ZIP/GZ")
            
            # Technical Terms Collapsible
            with st.expander("📖 Technical Terms"):
                st.markdown("""
                **CSV**: Comma-Separated Values - Plain text format for tabular data  
                **XLSX/XLS**: Microsoft Excel file formats  
                **JSON/JSONL**: JavaScript Object Notation - Structured data format  
                **PARQUET**: Columnar storage format optimized for analytics  
                **ZIP/GZ**: Compressed archive formats containing multiple files  
                **DataOS (Trino)**: Distributed SQL query engine for big data analytics  
                **Schema**: Structure definition of data including column names and types  
                **Cardinality**: Number of unique values in a column  
                """)
            
            source = st.radio("Data source", ["Upload files", "DataOS (Trino)"], horizontal=True)

            # FILE UPLOAD BRANCH
            if source == "Upload files":
                col_up, col_demo = st.columns([5, 1])
                
                # File uploader widget
                with col_up:
                    uploads = st.file_uploader(
                        "Drag and drop or browse",
                        type=["csv", "tsv", "txt", "xlsx", "xls", "json", "jsonl", "ndjson", "parquet", "zip", "gz"],
                        accept_multiple_files=True,
                        key=f"uploader_{ss.uploader_key}",
                    )
                
                # Demo data button
                with col_demo:
                    st.markdown("<div style='height:44px'></div>", unsafe_allow_html=True)
                    if st.button("Use Demo Data", use_container_width=True):
                        df_demo = demo_data(n_rows=8040)
                        base = ensure_unique_name(set(ss.datasets.keys()), "demo")
                        ss.datasets[base] = df_demo
                        ss.raw_datasets[base] = df_demo.copy()
                        ss.active_ds = base
                        ss.df_history.clear()
                        log(f"Loaded demo into '{base}'.")

                # Process uploaded files
                processed_any = False
                if uploads:
                    for up in uploads:
                        content = up.getvalue()
                        h = hashlib.sha1(content).hexdigest()
                        
                        # Skip if already processed (prevents duplicates)
                        if h in ss.loaded_hashes:
                            continue
                            
                        base_name = getattr(up, "name", "uploaded").rsplit("/", 1)[-1]
                        lower_name = base_name.lower()

                        # Handle ZIP archives containing multiple files
                        if lower_name.endswith(".zip"):
                            tables = read_zip_all(up)
                            if not tables:
                                st.warning(f"'{base_name}' contains no supported tabular files.")
                                continue
                                
                            added = 0
                            for i, (inner_name, df_i) in enumerate(tables):
                                if df_i is None or df_i.empty:
                                    continue
                                    
                                ds_name = ensure_unique_name(set(ss.datasets.keys()), inner_name.rsplit("/", 1)[-1])
                                ss.datasets[ds_name] = df_i
                                ss.raw_datasets[ds_name] = df_i.copy()
                                
                                if ss.active_ds is None:
                                    ss.active_ds = ds_name
                                    
                                # Track metadata for each file in archive
                                meta_key = f"{h}:{i}"
                                ss.file_meta[meta_key] = {"name": ds_name, "filename": f"{base_name} › {inner_name}"}
                                added += 1
                                
                            if added:
                                ss.loaded_hashes.add(h)
                                log(f"Loaded {added} tables from '{base_name}'.")
                                processed_any = True
                            continue

                        # Handle individual files
                        df = read_any(up)
                        if df is None or df.empty:
                            st.warning(f"Skipped '{getattr(up, 'name', 'file')}' — no tabular data detected.")
                            continue
                            
                        name = ensure_unique_name(set(ss.datasets.keys()), base_name)
                        ss.datasets[name] = df
                        ss.raw_datasets[name] = df.copy()
                        ss.active_ds = name
                        ss.loaded_hashes.add(h)
                        ss.hash_to_name[h] = name
                        ss.file_meta[h] = {"name": name, "filename": base_name}
                        log(f"Loaded '{name}'.")
                        processed_any = True

                    # Refresh UI after processing files
                    if processed_any:
                        ss.uploader_key += 1
                        if not ss.get("_step_changed", False):
                            st.rerun()

                # Display loaded files summary
                if ss.file_meta:
                    st.markdown("**Files added**")
                    for key, meta in ss.file_meta.items():
                        ds_name = meta.get("name")
                        fname = meta.get("filename", ds_name)
                        col1, col2 = st.columns([6, 6])
                        col1.write(f"**{fname}**")
                        col2.caption(f"Dataset: {ds_name}")

            # DATAOS (TRINO) BRANCH
            else:
                st.caption("Connect and pull tables from DataOS (Trino)")
                
                # Connection configuration
                c1, c2 = st.columns(2)
                host = c1.text_input("Host", value=ss.get("tr_host", ""), placeholder="your.trino.host", key="tr_host")
                port = c2.number_input("Port", min_value=1, value=int(ss.get("tr_port", 7432)), step=1, key="tr_port")
                user = c1.text_input("Username", value=ss.get("tr_user", ""), key="tr_user")
                # Password not persisted for security
                password = c2.text_input("Password", type="password", value="", key="tr_pass_input")

                http_scheme = c1.selectbox(
                    "HTTP scheme", ["https", "http"],
                    index=0 if ss.get("tr_scheme", "https") == "https" else 1,
                    key="tr_scheme",
                )
                cluster_name = c2.text_input(
                    "Cluster name (HTTP header: cluster-name)",
                    value=ss.get("tr_cluster", "minervac"),
                    key="tr_cluster",
                )

                def _required_ok() -> bool:
                    """Check if required connection fields are filled"""
                    return bool(host.strip() and user.strip() and http_scheme.strip() and cluster_name.strip())

                # Cache management based on connection signature
                sig = f"{host}|{port}|{user}|{http_scheme}|{cluster_name}"
                if ss.trino_cache.get("sig") != sig:
                    ss.trino_cache = {"sig": sig, "catalogs": [], "schemas": {}, "tables": {}}
                    ss.tr_auto_kick_done = False

                def _cfg(catalog_hint: str | None = None, schema_hint: str | None = None):
                    """Create Trino configuration object"""
                    try:
                        import trino  # noqa: F401
                    except Exception:
                        st.error("The 'trino' package isn't installed. Run: `pip install trino`")
                        return None
                    return TrinoConfig(
                        host=host.strip(),
                        port=int(port),
                        user=user.strip(),
                        password=password,
                        http_scheme=http_scheme,
                        http_headers={"cluster-name": (cluster_name.strip() or "minervac")},
                        catalog=(catalog_hint or "system"),
                        schema=(schema_hint or "jdbc"),
                    )

                # Trino metadata discovery functions
                def _list_catalogs(cfg) -> list[str]:
                    """Try multiple queries to discover available catalogs"""
                    last_err = None
                    queries = [
                        ("SELECT catalog_name FROM system.metadata.catalogs", "catalog_name"),
                        ("SELECT catalog_name FROM system.jdbc.catalogs", "catalog_name"),
                        ("SELECT table_cat AS catalog_name FROM system.jdbc.catalogs", "catalog_name"),
                        ("SHOW CATALOGS", None),
                    ]
                    for sql, col in queries:
                        try:
                            df = query_df(sql, cfg)
                            if df is None or df.empty:
                                continue
                            use_col = col or next((c for c in df.columns if c.lower() in ("catalog", "catalog_name")), df.columns[0])
                            return sorted(df[use_col].astype(str).tolist())
                        except Exception as e:
                            last_err = e
                            continue
                    raise RuntimeError(f"Unable to fetch catalogs; last error: {last_err}")

                def _list_schemas(cfg, catalog: str) -> list[str]:
                    """Try multiple queries to discover schemas in a catalog"""
                    esc_catalog = catalog.replace("'", "''")  # SQL injection protection
                    candidates = [
                        (f"SELECT schema_name FROM {q_ident(catalog)}.information_schema.schemata ORDER BY schema_name", "schema_name"),
                        (f"SELECT schema_name FROM system.jdbc.schemata WHERE catalog_name = '{esc_catalog}' ORDER BY schema_name", "schema_name"),
                        (f"SELECT table_schem AS schema_name FROM system.jdbc.schemata WHERE table_cat = '{esc_catalog}' ORDER BY schema_name", "schema_name"),
                        (f"SHOW SCHEMAS FROM {q_ident(catalog)}", None),
                        (f"SELECT schema_name FROM system.metadata.schemata WHERE catalog_name = '{esc_catalog}' ORDER BY schema_name", "schema_name"),
                    ]
                    last_err = None
                    for sql, expected_col in candidates:
                        try:
                            df = query_df(sql, cfg)
                            if df is None or df.empty:
                                continue
                            if expected_col is None:
                                col = next((c for c in df.columns if c.lower() in ("schema", "schema_name")), df.columns[0])
                            else:
                                col = expected_col if expected_col in df.columns else next(
                                    (c for c in df.columns if c.lower() == expected_col.lower()), df.columns[0]
                                )
                            return sorted(df[col].astype(str).tolist())
                        except Exception as e:
                            last_err = e
                            continue
                    raise RuntimeError(f"Unable to fetch schemas for catalog {catalog}; last error: {last_err}")

                def _list_tables(cfg, catalog: str, schema: str) -> list[str]:
                    """Try multiple queries to discover tables in a schema"""
                    esc_schema = schema.replace("'", "''")
                    esc_catalog = catalog.replace("'", "''")
                    candidates = [
                        (f"SELECT table_name FROM {q_ident(catalog)}.information_schema.tables WHERE table_schema = '{esc_schema}' ORDER BY table_name", "table_name"),
                        (f"SHOW TABLES FROM {q_ident(catalog)}.{q_ident(schema)}", None),
                        (f"SELECT table_name FROM system.jdbc.tables WHERE catalog_name = '{esc_catalog}' AND schema_name = '{esc_schema}' ORDER BY table_name", "table_name"),
                        (f"SELECT table_name FROM system.jdbc.tables WHERE table_cat = '{esc_catalog}' AND table_schem = '{esc_schema}' ORDER BY table_name", "table_name"),
                    ]
                    last_err = None
                    for sql, expected_col in candidates:
                        try:
                            df = query_df(sql, cfg)
                            if df is None or df.empty:
                                continue
                            if expected_col is None:
                                col = next((c for c in df.columns if c.lower() in ("table", "table_name")), df.columns[0])
                            else:
                                col = expected_col if expected_col in df.columns else next(
                                    (c for c in df.columns if c.lower() == expected_col.lower()), df.columns[0]
                                )
                            return sorted(df[col].astype(str).tolist())
                        except Exception as e:
                            last_err = e
                            continue
                    raise RuntimeError(f"Unable to fetch tables for {catalog}.{schema}; last error: {last_err}")

                # Auto-load catalogs on first connection attempt
                ss.setdefault("tr_auto_kick_done", False)
                trigger_catalogs = False
                if _required_ok() and not ss.trino_cache["catalogs"] and not ss.tr_auto_kick_done:
                    trigger_catalogs = True
                    ss.tr_auto_kick_done = True
                    
                if st.button("Load catalogs", disabled=not _required_ok(), key="tr_btn_load_catalogs"):
                    trigger_catalogs = True

                # Execute catalog loading
                if trigger_catalogs:
                    cfg_meta = _cfg("system", "jdbc")
                    if cfg_meta:
                        try:
                            with st.spinner("Fetching catalogs…"):
                                ss.trino_cache["catalogs"] = _list_catalogs(cfg_meta)
                            if not ss.trino_cache["catalogs"]:
                                st.info("No catalogs found for these connection details.")
                        except Exception as e:
                            st.error(f"Failed to fetch catalogs: {e}")

                # Catalog selection
                selected_catalog = None
                if ss.trino_cache["catalogs"]:
                    selected_catalog = st.selectbox(
                        "Catalog",
                        options=ss.trino_cache["catalogs"],
                        key="tr_sel_catalog",
                        placeholder="Type to search catalogs…",
                    )
                else:
                    st.caption("Fill connection fields and click **Load catalogs** to continue.")

                # Schema discovery and selection
                selected_schema = None
                if selected_catalog:
                    if selected_catalog not in ss.trino_cache["schemas"]:
                        cfg_s = _cfg(selected_catalog, "information_schema")
                        if cfg_s:
                            try:
                                with st.spinner("Fetching schemas…"):
                                    ss.trino_cache["schemas"][selected_catalog] = _list_schemas(cfg_s, selected_catalog)
                                if not ss.trino_cache["schemas"][selected_catalog]:
                                    st.info("No schemas in this catalog.")
                            except Exception as e:
                                ss.trino_cache["schemas"][selected_catalog] = []
                                st.warning("Schema list unavailable for this catalog. Enter a schema manually below.")
                                st.caption(f"Details: {e}")

                    if ss.trino_cache["schemas"].get(selected_catalog):
                        selected_schema = st.selectbox(
                            "Schema",
                            options=ss.trino_cache["schemas"][selected_catalog],
                            key="tr_sel_schema",
                            placeholder="Type to search schemas…",
                        )
                    else:
                        selected_schema = st.text_input("Schema (manual)", key="tr_schema_manual", placeholder="e.g. telemetry")

                # Table discovery and selection
                if selected_catalog and selected_schema:
                    key_ts = (selected_catalog, selected_schema)
                    if key_ts not in ss.trino_cache["tables"] and selected_schema.strip():
                        cfg_t = _cfg(selected_catalog, "information_schema")
                        if cfg_t:
                            try:
                                with st.spinner("Fetching tables…"):
                                    ss.trino_cache["tables"][key_ts] = _list_tables(cfg_t, selected_catalog, selected_schema)
                                if not ss.trino_cache["tables"][key_ts]:
                                    st.info("No tables in this schema.")
                            except Exception as e:
                                ss.trino_cache["tables"][key_ts] = []
                                st.warning("Table list unavailable. Enter table names manually below.")
                                st.caption(f"Details: {e}")

                    all_tables = ss.trino_cache["tables"].get(key_ts, [])

                    # Filter out already-loaded tables to prevent duplicates
                    loaded_labels = set(ss.tr_loaded_map.values())  # {"catalog.schema.table", ...}
                    available = [t for t in all_tables if f"{selected_catalog}.{selected_schema}.{t}" not in loaded_labels]

                    c7, c8 = st.columns([2, 1])
                    if available:
                        selected_tables = c7.multiselect(
                            "Tables (type to search, multi-select)",
                            options=available,
                            default=[],
                            placeholder="Start typing to search tables…",
                            key="tr_sel_tables",
                        )
                    else:
                        # Manual table entry when auto-discovery fails
                        manual = c7.text_area(
                            "Tables (comma/space separated)",
                            value="",
                            height=80,
                            key="tr_tables_manual",
                            placeholder="table_a, table_b, table_c",
                        )
                        # Parse manual input and exclude duplicates
                        candidate_manual = [t.strip() for t in manual.replace("\n", ",").replace("\t", ",").replace(" ", ",").split(",") if t.strip()]
                        selected_tables = [t for t in candidate_manual if f"{selected_catalog}.{selected_schema}.{t}" not in loaded_labels]

                    # Row limit configuration
                    limit_each = c8.number_input("Row limit per table", min_value=1, value=10, step=100, key="tr_limit_each")

                    # Load selected tables button
                    if st.button(
                        f"Load {len(selected_tables) if selected_tables else 0} table(s)",
                        type="primary",
                        disabled=not selected_tables,
                        key="tr_btn_load_tables",
                    ):
                        cfg_q = _cfg(selected_catalog, selected_schema)
                        if cfg_q:
                            loaded = 0
                            errors = 0
                            for tname in selected_tables:
                                label = f"{selected_catalog}.{selected_schema}.{tname}"
                                # Extra guard against duplicates
                                if label in loaded_labels:
                                    continue
                                try:
                                    sql = (
                                        f"SELECT * FROM "
                                        f"{q_ident(selected_catalog)}.{q_ident(selected_schema)}.{q_ident(tname)} "
                                        f"LIMIT {int(limit_each)}"
                                    )
                                    df_t = query_df(sql, cfg_q)
                                    new_ds_name = ensure_unique_name(set(ss.datasets.keys()), label)
                                    ss.datasets[new_ds_name] = df_t
                                    ss.raw_datasets[new_ds_name] = df_t.copy()
                                    ss.active_ds = new_ds_name
                                    ss.file_meta[f"trino:{new_ds_name}:{len(ss.file_meta)}"] = {"name": new_ds_name, "filename": f"trino › {label}"}
                                    # Record mapping to prevent re-loading
                                    ss.tr_loaded_map[new_ds_name] = label
                                    loaded_labels.add(label)
                                    log(f"Loaded '{label}' into '{new_ds_name}' ({len(df_t):,} rows).")
                                    loaded += 1
                                except Exception as e:
                                    st.error(f"Failed to load '{tname}': {e}")
                                    errors += 1
                            if loaded:
                                st.success(f"Loaded {loaded} table(s).")
                            if errors:
                                st.warning(f"{errors} table(s) failed.")
                            if loaded and ss.active_ds in ss.datasets:
                                st.dataframe(ss.datasets[ss.active_ds].head(50), use_container_width=True)

        # Stop here if no datasets are loaded
        if not ss.datasets:
            st.info("Upload one or more files or use **DataOS (Trino)** to begin.")
            st.stop()

        # =============================================================================
        # DATASET MANAGEMENT SECTION
        # =============================================================================
        
        with section("Datasets", expandable=False):
            st.markdown("**Active dataset**")
            dataset_combo("Dataset", "summary")

            # Dataset overview table header
            names = sorted(ss.datasets.keys())
            h1, h2, h3, h4, h5, h6 = st.columns([6, 1.2, 1, 1.2, 1.2, 0.8])
            h1.markdown("**dataset**")
            h2.markdown("**rows**")
            h3.markdown("**cols**")
            h4.markdown("**memory_mb**")
            h5.markdown("**duplicates**")
            h6.markdown("** **")
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

            # Display dataset overview information
            for i, nm in enumerate(names):
                ov = overview_stats(ss.datasets[nm])
                c1, c2, c3, c4, c5, c6 = st.columns([6, 1.2, 1, 1.2, 1.2, 0.8])
                c1.write(nm)
                c2.write(f"{ov['rows']:,}")
                c3.write(f"{ov['cols']}")
                c4.write(f"{ov['memory_mb']:.2f}")
                c5.write(f"{ov['n_duplicates']:,}")
                if c6.button("🗑️", key=f"del_{i}_{nm}", help=f"Delete '{nm}'"):
                    delete_dataset(nm)  # Also frees Trino mapping
                    st.rerun()

            # Dataset renaming interface
            rn_src = st.selectbox("Rename dataset", ["—"] + names, index=0, key="rn_src")
            new_name = st.text_input("New name", value="", key="rn_new")
            col_rn_apply, _ = st.columns([1, 5])
            with col_rn_apply:
                if rn_src != "—" and new_name and st.button("Apply rename", key="rn_btn"):
                    if new_name in ss.datasets:
                        st.error("Name already exists.")
                    else:
                        # Transfer dataset and all associated metadata
                        ss.datasets[new_name] = ss.datasets.pop(rn_src)
                        ss.raw_datasets[new_name] = ss.raw_datasets.pop(rn_src)
                        if ss.active_ds == rn_src:
                            ss.active_ds = new_name
                        # Update hash mappings
                        for h, nm2 in list(ss.hash_to_name.items()):
                            if nm2 == rn_src:
                                ss.hash_to_name[h] = new_name
                        # Update file metadata
                        for k, meta in list(ss.file_meta.items()):
                            if meta.get("name") == rn_src:
                                meta["name"] = new_name
                        # Transfer Trino mapping if exists
                        if rn_src in ss.tr_loaded_map:
                            ss.tr_loaded_map[new_name] = ss.tr_loaded_map.pop(rn_src)
                        log(f"Renamed '{rn_src}' → '{new_name}'.")
                        st.rerun()

        # =============================================================================
        # ACTIVE DATASET ANALYSIS SECTION
        # =============================================================================
        
        df = ss.datasets[ss.active_ds]

        st.subheader("Overview")
        st.caption("Dataset Overview — Quick stats, schema, and distincts")

        # Key Performance Indicators (KPI) row
        ov = overview_stats(df)
        kpi_row([
            ("Rows", f"{ov['rows']:,}"),
            ("Cols", f"{ov['cols']:,}"),
            ("Memory (MB)", f"{ov['memory_mb']:.2f}"),
            ("Duplicate rows", f"{ov['n_duplicates']:,}"),
        ])
        
        # Capture insights for later analysis
        capture_summary_insights(ss, ss.active_ds, df, n_tables=len(ss.datasets))

        # Dataset preview and JSON flattening section
        with section("Table preview"):
            meta = dataset_meta(df)
            st.markdown(f"**Profile:** {meta['profile']}")
            st.markdown(f"**Active:** {ss.active_ds}")
            subtitle = (
                f"{ov['rows']:,} rows × {ov['cols']:,} columns"
                + (f", time span {meta['time_min']} – {meta['time_max']}" if meta['time_min'] else "")
                + f". Includes {meta['n_numeric']} numeric and {meta['n_categorical']} categorical features."
            )
            st.caption(subtitle)

            # JSON flattening functionality
            df_preview = df
            auto_json_cols = detect_jsonlike_columns(df)
            cols_to_flatten = st.multiselect(
                "JSON-like columns to flatten (auto-detected preselected)",
                options=list(df.columns),
                default=auto_json_cols,
                placeholder="Pick columns such as 'item_data' …",
                key="json_cols_select",
            )
            
            # JSON flattening controls
            c_flat, c_lvl, c_hide = st.columns([1, 1, 1])
            with c_flat:
                do_flat = st.checkbox("Flatten (auto-save to current dataset)", value=bool(cols_to_flatten), key="preview_flatten_json")
            with c_lvl:
                max_level = st.number_input("Max nested level", 1, 5, value=2, step=1, key="preview_json_maxlvl")
            with c_hide:
                hide_originals = st.checkbox("Hide original JSON cols (when saving)", value=True, key="preview_hide_json_src")

            # Execute JSON flattening if requested
            if do_flat and cols_to_flatten:
                sig = f"cols={tuple(sorted(map(str, cols_to_flatten)))}|lvl={int(max_level)}|hide={bool(hide_originals)}"
                prev_sig = ss.json_flat_sigs.get(ss.active_ds)
                df_flat_preview, flattened_cols, notes = flatten_json_columns(
                    df, columns=cols_to_flatten, max_level=int(max_level)
                )
                if hide_originals:
                    df_flat_preview = df_flat_preview.drop(columns=list(cols_to_flatten), errors="ignore")
                    
                # Apply flattening if configuration changed
                if sig != prev_sig:
                    push_history(df)
                    ss.datasets[ss.active_ds] = df_flat_preview.copy()
                    ss.json_flat_sigs[ss.active_ds] = sig
                    log(f"Flattened {len(flattened_cols)} JSON column(s) into '{ss.active_ds}' (lvl={max_level}, hide_src={hide_originals}).")
                    st.success("Flattened columns saved to the current dataset. Use Undo or Restore RAW to revert.")
                    st.rerun()
                    
                df_preview = ss.datasets[ss.active_ds]
                if flattened_cols:
                    st.caption(f"Flattened: {', '.join(flattened_cols)}")
                for n in notes:
                    st.caption(f"⚠️ {n}")
            elif cols_to_flatten:
                st.caption("Tick **Flatten** to apply and auto-save into the current dataset.")
            else:
                st.caption("No JSON-like columns detected. Use the multiselect to force columns if needed.")

            # Schema table display
            schema_tbl = pd.DataFrame({"column": df_preview.columns, "dtype": [str(t) for t in df_preview.dtypes]})
            render_table(schema_tbl, height=260)
            st.markdown("**Preview (first 20 rows)**")
            st.dataframe(df_preview.head(20), use_container_width=True)

        # Cardinality analysis section
        with section("Cardinality", expandable=False):
            with st.expander("📖 Understanding Cardinality"):
                st.markdown("""
                **Cardinality** refers to the number of unique values in a column:
                - **High cardinality**: Many unique values (e.g., customer IDs, timestamps)
                - **Low cardinality**: Few unique values (e.g., gender, status codes)
                - **Missing count**: Number of null/empty values
                - **Non-null count**: Number of valid (non-missing) values
                """)
                
            schema_local = infer_schema(df)
            default_cols = df.columns[:10].tolist()
            pick_cols = st.multiselect(
                "Pick columns to show",
                options=df.columns.tolist(),
                default=default_cols,
                key="sum_card_pick_cols",
            )
            metric = st.selectbox("Y-axis", ["Distinct count", "Missing count", "Non-null count"], index=0)
            
            # Generate cardinality visualization
            if pick_cols:
                if metric == "Distinct count":
                    vals = {c: int(nunique_safe(df[c])) for c in pick_cols}
                elif metric == "Missing count":
                    vals = {c: int(df[c].isna().sum()) for c in pick_cols}
                else:
                    vals = {c: int(df[c].notna().sum()) for c in pick_cols}
                st.bar_chart(pd.Series(vals).sort_index())
            else:
                st.caption("Select columns to visualize cardinality.")
                
            # Cardinality summary table
            nunique_tbl = (
                schema_local[["column", "unique"]]
                .rename(columns={"unique": "nunique"})
                .sort_values("nunique", ascending=False)
                .reset_index(drop=True)
            )
            render_table(nunique_tbl, height=300)

        # Schema and column statistics
        with section("Schema & Column Summary", expandable=False):
            with st.expander("📖 Column Statistics Explained"):
                st.markdown("""
                **Data Types**: 
                - **Numeric**: int64, float64 - Numbers for calculations
                - **Object**: Usually text/strings or mixed types
                - **Datetime**: Date and time values
                - **Boolean**: True/False values
                
                **Statistics**:
                - **Count**: Number of non-null values
                - **Mean/Std**: Average and standard deviation for numeric columns
                - **Min/Max**: Smallest and largest values
                - **Unique**: Number of distinct values
                """)
                
            cols = column_quick_stats(df, schema_local)
            render_table(cols)

        # Data quality suggestions
        with section("Suggested Actions"):
            tips = suggest_actions(df)
            if not tips:
                st.write("No immediate issues detected.")
            else:
                for t in tips:
                    st.markdown(f"- {t}")

        # HTML Report Generation
        st.markdown("---")
        with section("Scrollable report (HTML)", expandable=False):
            html = build_summary_html(ss.active_ds, df, datasets=ss.datasets)
            st.components.v1.html(html, height=900, scrolling=True)
            st.download_button(
                "⬇️ Download Summary (HTML)",
                data=summary_html_bytes(df, dataset_name=ss.active_ds, datasets=ss.datasets),
                file_name=f"{ss.active_ds}_summary.html",
                mime="text/html",
                type="primary",
                key=f"download_summary_html_{ss.active_ds}"
            )

        # Insights export
        st.download_button(
            "⬇️ Download insights JSON",
            data=insights_to_json_bytes(ss, ss.active_ds),
            file_name=f"{ss.active_ds}_insights.json",
            mime="application/json",
            key=f"download_insights_{ss.active_ds}",
        )

    # =============================================================================
    # EDA (EXPLORATORY DATA ANALYSIS) STEP
    # =============================================================================
    
    if ss.step == "EDA":
        if not ss.datasets:
            st.info("Upload one or more files in **Overview** to begin.")
            st.stop()
            
        names_all = sorted(ss.datasets.keys())
        if ss.active_ds not in names_all:
            ss.active_ds = names_all[0]
            
        # Dataset selection and sampling controls
        with st.container():
            c_ds, c_sample = st.columns([0.55, 0.45])
            with c_ds:
                dataset_combo("Dataset", "eda")
            with c_sample:
                cur_df = ss.datasets[ss.active_ds]
                nmax = max(1, len(cur_df))
                ss.setdefault("eda_sample_n", min(5000, nmax))
                ss.eda_sample_n = st.slider("Sample rows for charts (for speed)", 1, nmax, value=min(ss.eda_sample_n, nmax))
                
            df = ss.datasets[ss.active_ds]
            
            # EDA tabs with technical explanations
            tab_overview, tab_types, tab_combine, tab_missing, tab_univariate, tab_bivariate, tab_correlation = st.tabs(
                ["Overview", "Types", "Combine datasets", "Missingness", "Univariate", "Bivariate", "Correlation"]
            )
            
            with tab_overview:  
                with st.expander("📖 EDA Overview Terms"):
                    st.markdown("""
                    **Exploratory Data Analysis (EDA)**: Process of analyzing datasets to summarize main characteristics
                    - **Distribution**: How values are spread across a variable
                    - **Outliers**: Data points significantly different from others
                    - **Patterns**: Trends, cycles, or relationships in the data
                    """)
                render_overview(df)
                
            with tab_types:     
                with st.expander("📖 Data Types"):
                    st.markdown("""
                    **Data Type Conversion**: Changing how data is interpreted
                    - **Categorical**: Limited set of values (e.g., colors, categories)
                    - **Numerical**: Numbers for mathematical operations
                    - **Datetime**: Date and time information
                    - **Text**: String/text data for analysis
                    """)
                render_eda_types(st.session_state)
                
            with tab_missing:   
                with st.expander("📖 Missing Data"):
                    st.markdown("""
                    **Missing Data Patterns**:
                    - **MCAR**: Missing Completely At Random
                    - **MAR**: Missing At Random (depends on other variables)
                    - **MNAR**: Missing Not At Random (systematic missing pattern)
                    """)
                render_missingness(df)
                
            with tab_univariate:
                render_univariate(df, sample_n=ss.eda_sample_n)
                
            with tab_bivariate: 
                render_bivariate(df, sample_n=ss.eda_sample_n)
                
            with tab_correlation: 
                render_correlation(df, sample_n=ss.eda_sample_n)
                
            with tab_combine:   
                render_combine(ss)

    # =============================================================================
    # PREPROCESS STEP - Data Cleaning and Transformation
    # =============================================================================
    
    if ss.step == "Preprocess":
        if not ss.datasets:
            st.info("Upload one or more files in **Overview** to begin.")
            st.stop()
            
        c_ds, _ = st.columns([0.6, 0.4])
        with c_ds:
            dataset_combo("Dataset", "prep")
            
        # Preprocessing tabs with explanations
        tab_overview, tab_missing, tab_outliers, tab_text, tab_ts, tab_encoding = st.tabs(
            ["Overview", "Missing values", "Outliers", "Text", "Time Series", "Encoding"]
        )
        
        with tab_overview:  
            render_preprocess_overview(ss)
            
        with tab_missing:   
            render_preprocess_missing(ss)
            
        with tab_outliers:  
            render_preprocess_outliers(ss)
            
        with tab_text:      
            render_preprocess_text(ss)
            
        with tab_ts:        
            render_preprocess_timeseries(st.session_state)
            
        with tab_encoding:  
            render_preprocess_encoding(st.session_state)

    # =============================================================================
    # FINAL SUMMARY STEP - Comprehensive Report
    # =============================================================================
    
    if ss.step == "Final Summary":
        if not ss.datasets:
            st.info("Upload one or more files in **Overview** to begin.")
            st.stop()
            
        names_all = sorted(ss.datasets.keys())
        if ss.active_ds not in names_all:
            ss.active_ds = names_all[0]
        df = ss.datasets[ss.active_ds]
        
        with st.expander("📖 Final Summary Purpose"):
            st.markdown("""
            **Final Summary**: Comprehensive analysis report including:
            - **Data Quality Assessment**: Completeness, consistency, validity
            - **Statistical Summary**: Key metrics and distributions
            - **Data Profiling**: Column-by-column analysis
            - **Recommendations**: Suggested next steps for analysis
            """)
            
        render_final_summary(df, st.session_state)