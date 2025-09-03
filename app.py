# app.py
import hashlib
import pandas as pd
import streamlit as st

from core.data_io import read_any, read_zip_all
from core.summary import (
    overview_stats, infer_schema, column_quick_stats,
    suggest_actions, dataset_meta, demo_data, nunique_safe,
    build_summary_pdf, build_summary_html, summary_html_bytes
)

from core.eda_overview import render_overview
from core.eda_combine import render_combine, ensure_unique_name
from core.eda_missingness import render_missingness
from core.eda_univariate import render_univariate
from core.eda_bivariate import render_bivariate
from core.eda_correlation import render_correlation
from core.preprocess_overview import render_preprocess_overview
from core.preprocess_missing import render_preprocess_missing
from core.preprocess_outliers import render_preprocess_outliers
from core.preprocess_text import render_preprocess_text
from core.preprocess_timeseries import render_preprocess_timeseries
from core.preprocess_encoding import render_preprocess_encoding
from core.eda_types import render_eda_types
from ui.components import header_bar, kpi_row, section, render_table, control_bar
from core.trino_connection import TrinoConfig, query_df, q_ident
from core.auto_insights import render_auto_insights
from core.final_summary_pages import render_final_summary_suite
from core.json_flatten import detect_jsonlike_columns, flatten_json_columns



st.set_page_config(page_title="Analytics Accelerator — Summary", layout="wide")

# -----------------
# Session bootstrap
# -----------------
ss = st.session_state
ss.setdefault("datasets", {})          # name -> DataFrame
ss.setdefault("raw_datasets", {})      # name -> original DataFrame (for Restore RAW)
ss.setdefault("active_ds", None)       # current dataset name
ss.setdefault("df_history", [])        # undo stack for active_ds
ss.setdefault("activity_log", [])
ss.setdefault("step", "Summary")
ss.setdefault("_step_changed", False)

# For stable multi-file ingest + list/remove
ss.setdefault("loaded_hashes", set())  # set[str] content hashes already ingested
ss.setdefault("hash_to_name", {})      # sha1 -> dataset name (for single-file uploads)
ss.setdefault("file_meta", {})         # key -> {"name": dataset_name, "filename": label shown in Files added}
ss.setdefault("uploader_key", 0)       # forces file_uploader to reset

# Trino metadata caches (scoped by connection signature)
ss.setdefault("trino_cache", {"sig": "", "catalogs": [], "schemas": {}, "tables": {}})
ss.setdefault("tr_auto_kick_done", False)

# -----------------
# Utilities
# -----------------
def log(msg: str):
    ss.activity_log.append(msg)

def push_history(df: pd.DataFrame):
    ss.df_history.append(df.copy())

def pop_history() -> pd.DataFrame | None:
    if ss.df_history:
        return ss.df_history.pop()
    return None

def delete_dataset(ds_name: str):
    """Remove a dataset everywhere (datasets, raw, file_meta, active_ds)."""
    ss.datasets.pop(ds_name, None)
    ss.raw_datasets.pop(ds_name, None)
    # drop any file_meta entries that reference this dataset
    for key in list(ss.file_meta.keys()):
        if ss.file_meta[key].get("name") == ds_name:
            ss.file_meta.pop(key, None)
    # if we just removed the active dataset, pick another or clear
    if ss.active_ds == ds_name:
        ss.active_ds = next(iter(ss.datasets.keys()), None)
    log(f"Removed dataset '{ds_name}'.")



def dataset_combo(label: str, key_prefix: str):
    """Passive selectbox (type-to-search) that keeps ss.active_ds in sync."""
    names = sorted(ss.datasets.keys())
    if not names:
        return None
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
    """Vertical pill-style step navigation in the left column (no emojis/icons)."""
    nav_labels = ["Summary", "EDA", "Preprocess", "Final Summary"]

    st.markdown("""
    <style>
      .left-pills button {border-radius: 999px !important; border: 1px solid rgba(0,0,0,.08);
                          margin: 4px 0; padding: 6px 12px !important;}
      .left-pills .active button {background: rgb(59,130,246) !important; color: white !important;
                                  border-color: rgb(59,130,246) !important;}
      .left-pills button:hover {border-color: rgba(0,0,0,.18);}
    </style>
    """, unsafe_allow_html=True)

    current = st.session_state.get("step", "Summary")

    for label in nav_labels:
        active = (label == current)
        st.markdown(f'<div class="left-pills {"active" if active else ""}">', unsafe_allow_html=True)
        if st.button(label, key=f"nav_{label}", use_container_width=True):
            if label != current:
                st.session_state["_step_changed"] = True
                st.session_state["step"] = label
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# ------------- LAYOUT -------------
left, right = st.columns([0.22, 0.78], gap="large")

# LEFT: vertical step nav + Activity log
with left:
    render_left_steps()

    st.markdown("**Activity log**")
    if ss.activity_log:
        for line in ss.activity_log[-12:]:
            st.write("• ", line)
    else:
        st.caption("—")

with right:
    # Header + global control bar
    header_bar(ss.step)
    clicked = control_bar()

    # ----- CLEAR: full reset back to blank page -----
    if clicked.get("clear"):
        ss.datasets.clear()
        ss.raw_datasets.clear()
        ss.active_ds = None
        ss.df_history.clear()
        ss.activity_log.clear()
        ss.loaded_hashes = set()
        ss.hash_to_name = {}
        ss.file_meta = {}
        ss.pop("eda_sample_n", None)   # reset sampling slider
        ss.uploader_key += 1           # reset uploader widget
        ss.step = "Summary"
        st.rerun()

    # ----- UNDO on active dataset -----
    if clicked.get("undo") and ss.active_ds and ss.df_history:
        prev = pop_history()
        if prev is not None:
            ss.datasets[ss.active_ds] = prev
            log(f"Undid last change on '{ss.active_ds}'.")

    # ----- RESTORE RAW for active dataset -----
    if clicked.get("restore") and ss.active_ds:
        base = ss.raw_datasets.get(ss.active_ds)
        if isinstance(base, pd.DataFrame):
            ss.datasets[ss.active_ds] = base.copy()
            ss.df_history.clear()
            log(f"Restored RAW for '{ss.active_ds}'.")

    # =========================================================
    # SUMMARY STEP
    # =========================================================
    if ss.step == "Summary":

        # --------- Uploader (multi-file) OR DataOS (Trino) ---------
        with section("Select Files"):
            st.caption("Limit ~200MB per file • CSV, XLSX, XLS, JSON/JSONL, PARQUET, ZIP/GZ")

            # NEW: source switch
            source = st.radio("Data source", ["Upload files", "DataOS (Trino)"], horizontal=True)

            # ---------------------------
            # A) Upload files (existing)
            # ---------------------------
            if source == "Upload files":
                col_up, col_demo = st.columns([5, 1])

                with col_up:
                    uploads = st.file_uploader(
                        "Drag and drop or browse",
                        type=["csv", "tsv", "txt", "xlsx", "xls", "json", "jsonl", "ndjson", "parquet", "zip", "gz"],
                        accept_multiple_files=True,
                        key=f"uploader_{ss.uploader_key}",
                    )

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

                # Ingest *new* files only. Never auto-remove based on uploader state.
                processed_any = False

                if uploads:
                    for up in uploads:
                        content = up.getvalue()
                        h = hashlib.sha1(content).hexdigest()
                        if h in ss.loaded_hashes:
                            continue

                        base_name = getattr(up, "name", "uploaded").rsplit("/", 1)[-1]
                        lower_name = base_name.lower()

                        # ZIP: ingest ALL supported inner files using same parsers as read_any
                        if lower_name.endswith(".zip"):
                            tables = read_zip_all(up)  # list[(inner_filename, df)]
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
                                meta_key = f"{h}:{i}"  # unique key per inner file
                                ss.file_meta[meta_key] = {
                                    "name": ds_name,
                                    "filename": f"{base_name} › {inner_name}",
                                }
                                added += 1

                            if added:
                                ss.loaded_hashes.add(h)
                                log(f"Loaded {added} tables from '{base_name}'.")
                                processed_any = True
                            continue

                        # Non-zip: single file path
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

                    # Clear uploader chip after successful ingest (avoid interfering with step change)
                    if processed_any:
                        ss.uploader_key += 1
                        if not ss.get("_step_changed", False):
                            st.rerun()

                # --- Files added (per-file removal) ---
                # --- Files added (display only; deletions happen via bin in Datasets table) ---
                if ss.file_meta:
                    st.markdown("**Files added**")
                    for key, meta in ss.file_meta.items():
                        ds_name = meta.get("name")
                        fname = meta.get("filename", ds_name)
                        col1, col2 = st.columns([6, 6])
                        col1.write(f"**{fname}**")
                        col2.caption(f"Dataset: {ds_name}")


            # ---------------------------
            # B) DataOS (Trino) — catalogs → schemas → tables (searchable) with multi-table load
            # ---------------------------
            else:
                st.caption("Connect and pull tables from DataOS (Trino)")

                # --- Base connection fields ---
                c1, c2 = st.columns(2)
                host = c1.text_input("Host", value=ss.get("tr_host", ""), placeholder="your.trino.host", key="tr_host")
                port = c2.number_input("Port", min_value=1, value=int(ss.get("tr_port", 7432)), step=1, key="tr_port")

                user = c1.text_input("Username", value=ss.get("tr_user", ""), key="tr_user")
                password = c2.text_input("Password", type="password", value=ss.get("tr_pass", ""), key="tr_pass")

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

                # --- Simple validity check ---
                def _required_ok() -> bool:
                    return bool(host.strip() and user.strip() and password.strip() and http_scheme.strip() and cluster_name.strip())

                # ---- Internal cache keyed by connection signature ----
                sig = f"{host}|{port}|{user}|{http_scheme}|{cluster_name}"
                if ss.trino_cache.get("sig") != sig:
                    ss.trino_cache = {"sig": sig, "catalogs": [], "schemas": {}, "tables": {}}
                    ss.tr_auto_kick_done = False

                # ---- Trino config factory ----
                def _cfg(catalog_hint: str | None = None, schema_hint: str | None = None):
                    try:
                        import trino  # noqa: F401
                    except Exception:
                        st.error("The 'trino' package isn’t installed. Run: `pip install trino`")
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

                # ---- Robust metadata helpers ----
                def _list_catalogs(cfg) -> list[str]:
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
                    """Robust schema listing with fallbacks."""
                    esc_catalog = catalog.replace("'", "''")
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
                    """Robust table listing with fallbacks."""
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

                # ---- Auto/Manual trigger to fetch catalogs ----
                ss.setdefault("tr_auto_kick_done", False)
                trigger_catalogs = False
                if _required_ok() and not ss.trino_cache["catalogs"] and not ss.tr_auto_kick_done:
                    trigger_catalogs = True
                    ss.tr_auto_kick_done = True
                if st.button("Load catalogs", disabled=not _required_ok(), key="tr_btn_load_catalogs"):
                    trigger_catalogs = True

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

                # ---- Catalog selector ----
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

                # ---- Schema selector (with manual fallback) ----
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

                # ---- Tables multiselect (with manual fallback) + loader ----
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

                    available = ss.trino_cache["tables"].get(key_ts, [])

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
                        manual = c7.text_area(
                            "Tables (comma/space separated)",
                            value="",
                            height=80,
                            key="tr_tables_manual",
                            placeholder="table_a, table_b, table_c",
                        )
                        selected_tables = [t.strip() for t in manual.replace("\n", ",").replace("\t", ",").replace(" ", ",").split(",") if t.strip()]

                    limit_each = c8.number_input("Row limit per table", min_value=1, value=10, step=100, key="tr_limit_each")

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
                                try:
                                    sql = (
                                        f"SELECT * FROM "
                                        f"{q_ident(selected_catalog)}.{q_ident(selected_schema)}.{q_ident(tname)} "
                                        f"LIMIT {int(limit_each)}"
                                    )
                                    df_t = query_df(sql, cfg_q)
                                    label = f"{selected_catalog}.{selected_schema}.{tname}"
                                    ds_name = ensure_unique_name(set(ss.datasets.keys()), label)
                                    ss.datasets[ds_name] = df_t
                                    ss.raw_datasets[ds_name] = df_t.copy()
                                    ss.active_ds = ds_name
                                    ss.file_meta[f"trino:{ds_name}:{len(ss.file_meta)}"] = {"name": ds_name, "filename": f"trino › {label}"}
                                    log(f"Loaded '{label}' into '{ds_name}' ({len(df_t):,} rows).")
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

        # If nothing loaded yet, stop here
        if not ss.datasets:
            st.info("Upload one or more files or use **DataOS (Trino)** to begin.")
            st.stop()

        # --------- Datasets manager (no Delete UI) ---------
        with section("Datasets", expandable=False):
            st.markdown("**Active dataset**")
            dataset_combo("Dataset", "summary")

            # Stats table
            stats_rows = []
            names = sorted(ss.datasets.keys())
            for nm in names:
                ov = overview_stats(ss.datasets[nm])
                stats_rows.append({
                    "dataset": nm,
                    "rows": f"{ov['rows']:,}",
                    "cols": ov['cols'],
                    "memory_mb": f"{ov['memory_mb']:.2f}",
                    "duplicates": f"{ov['n_duplicates']:,}",
                })
            # Stats table WITH delete bin per row (keeps the same columns you had)
            names = sorted(ss.datasets.keys())

            # Header row
            h1, h2, h3, h4, h5, h6 = st.columns([6, 1.2, 1, 1.2, 1.2, 0.8])
            h1.markdown("**dataset**")
            h2.markdown("**rows**") 
            h3.markdown("**cols**")
            h4.markdown("**memory_mb**")
            h5.markdown("**duplicates**")
            h6.markdown("** **")  # bin column

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

            # Data rows with per-row bin button
            for i, nm in enumerate(names):
                ov = overview_stats(ss.datasets[nm])
                c1, c2, c3, c4, c5, c6 = st.columns([6, 1.2, 1, 1.2, 1.2, 0.8])

                c1.write(nm)
                c2.write(f"{ov['rows']:,}")
                c3.write(f"{ov['cols']}")
                c4.write(f"{ov['memory_mb']:.2f}")
                c5.write(f"{ov['n_duplicates']:,}")

                if c6.button("🗑️", key=f"del_{i}_{nm}", help=f"Delete '{nm}'"):
                    delete_dataset(nm)
                    st.rerun()


            # Rename (existing)
            rn_src = st.selectbox("Rename dataset", ["—"] + names, index=0, key="rn_src")
            new_name = st.text_input("New name", value="", key="rn_new")
            col_rn_apply, col_spacer = st.columns([1, 5])
            with col_rn_apply:
                if rn_src != "—" and new_name and st.button("Apply rename", key="rn_btn"):
                    if new_name in ss.datasets:
                        st.error("Name already exists.")
                    else:
                        # Move in datasets + raw
                        ss.datasets[new_name] = ss.datasets.pop(rn_src)
                        ss.raw_datasets[new_name] = ss.raw_datasets.pop(rn_src)
                        # Update active selection
                        if ss.active_ds == rn_src:
                            ss.active_ds = new_name
                        # Update hash->name mapping
                        for h, nm in list(ss.hash_to_name.items()):
                            if nm == rn_src:
                                ss.hash_to_name[h] = new_name
                        # Update file_meta mapping so list shows new name
                        for k, meta in list(ss.file_meta.items()):
                            if meta.get("name") == rn_src:
                                meta["name"] = new_name
                        log(f"Renamed '{rn_src}' → '{new_name}'.")
                        st.rerun()

            st.markdown("---")

            # NEW: Delete dataset(s) — works for Trino-loaded tables too
            st.markdown("**Delete dataset(s)**")
            to_delete = st.multiselect(
                "Choose one or more datasets to remove",
                options=names,
                default=[],
                placeholder="Type to search datasets…",
                key="ds_delete_choices",
            )
            cols_del = st.columns([1, 5])
            with cols_del[0]:
                if st.button(
                    f"Delete {len(to_delete)} selected" if to_delete else "Delete",
                    type="secondary",
                    disabled=not to_delete,
                    key="btn_delete_datasets",
                    help="Removes selected datasets from this app session",
                ):
                    for nm in to_delete:
                        delete_dataset(nm)
                    st.success(f"Deleted {len(to_delete)} dataset(s).")
                    st.rerun()


        # ----- Active dataset content (Summary visuals) -----
        df = ss.datasets[ss.active_ds]

        st.subheader("Summary")
        st.caption("📊  Dataset Overview — Quick stats, schema, and distincts")

        ov = overview_stats(df)
        kpi_row([
            ("Rows", f"{ov['rows']:,}"),
            ("Cols", f"{ov['cols']:,}"),
            ("Memory (MB)", f"{ov['memory_mb']:.2f}"),
            ("Duplicate rows", f"{ov['n_duplicates']:,}"),
        ])

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

            # --- JSON flatten preview/save ---
            from core.json_flatten import detect_jsonlike_columns, flatten_json_columns

            df_preview = df

            # Auto-detect JSON-like columns (permissive)
            auto_json_cols = detect_jsonlike_columns(df)

            # Manual override: user can force columns to flatten
            cols_to_flatten = st.multiselect(
                "JSON-like columns to flatten (auto-detected preselected)",
                options=list(df.columns),
                default=auto_json_cols,
                placeholder="Pick columns such as 'item_data' …",
                key="json_cols_select",
            )

            c_flat, c_lvl, c_hide = st.columns([1, 1, 1])
            with c_flat:
                do_flat = st.checkbox("Flatten in preview", value=bool(cols_to_flatten), key="preview_flatten_json")
            with c_lvl:
                max_level = st.number_input("Max nested level", 1, 5, value=2, step=1, key="preview_json_maxlvl")
            with c_hide:
                hide_originals = st.checkbox("Hide original JSON cols", value=True, key="preview_hide_json_src")

            if do_flat and cols_to_flatten:
                df_preview, flattened_cols, notes = flatten_json_columns(
                    df, columns=cols_to_flatten, max_level=int(max_level)
                )
                if hide_originals:
                    # Hide the original nested columns in the preview only
                    df_preview = df_preview.drop(columns=list(cols_to_flatten), errors="ignore")

                if flattened_cols:
                    st.caption(f"Flattened in preview: {', '.join(flattened_cols)}")
                for n in notes:
                    st.caption(f"⚠️ {n}")

                # Save flattened as a new dataset for downstream steps
                if st.button("Save flattened as NEW dataset", key="btn_save_flattened"):
                    new_name = ensure_unique_name(set(ss.datasets.keys()), f"{ss.active_ds}_flat")
                    ss.datasets[new_name] = df_preview.copy()
                    ss.raw_datasets[new_name] = df_preview.copy()
                    ss.active_ds = new_name
                    log(f"Created flattened dataset '{new_name}'.")
                    st.success(f"Saved as **{new_name}**.")
                    st.rerun()
            elif cols_to_flatten:
                st.caption("Tick **Flatten in preview** to see expanded columns.")
            else:
                st.caption("No JSON-like columns detected. Use the multiselect to force columns if needed.")

            # Show schema/preview for the (possibly) flattened frame
            schema_tbl = pd.DataFrame({"column": df_preview.columns, "dtype": [str(t) for t in df_preview.dtypes]})
            render_table(schema_tbl, height=260)

            st.markdown("**Preview (first 20 rows)**")
            st.dataframe(df_preview.head(20), use_container_width=True)



        with section("Cardinality", expandable=False):
            schema_local = infer_schema(df)
            default_cols = df.columns[:10].tolist()
            pick_cols = st.multiselect("Pick columns to show", options=df.columns.tolist(), default=default_cols)
            metric = st.selectbox("Y-axis", ["Distinct count", "Missing count", "Non-null count"], index=0)

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

            nunique_tbl = (
                schema_local[["column", "unique"]]
                .rename(columns={"unique": "nunique"})
                .sort_values("nunique", ascending=False)
                .reset_index(drop=True)
            )
            render_table(nunique_tbl, height=300)

        with section("Schema & Column Summary"):
            cols = column_quick_stats(df, schema_local)
            render_table(cols)

        with section("Suggested Actions"):
            tips = suggest_actions(df)
            if not tips:
                st.write("No immediate issues detected.")
            else:
                for t in tips:
                    st.markdown(f"- {t}")

        # --- Export: Scrollable HTML report (inline + download) ---
        st.markdown("---")
        with section("Scrollable report (HTML)", expandable=False):
            html = build_summary_html(
                ss.active_ds,
                df,
                datasets=ss.datasets
            )
            # show inline (fully scrollable, sticky headers handled in HTML)
            st.components.v1.html(html, height=900, scrolling=True)

            # download same HTML
            st.download_button(
                "⬇️ Download Summary (HTML)",
                data=summary_html_bytes(
                    df,
                    dataset_name=ss.active_ds,
                    datasets=ss.datasets
                ),
                file_name=f"{ss.active_ds}_summary.html",
                mime="text/html",
                type="primary",
                key=f"download_summary_html_{ss.active_ds}"
            )

    # =========================================================
    # EDA STEP
    # =========================================================
    if ss.step == "EDA":
        if not ss.datasets:
            st.info("Upload one or more files in **Summary** to begin.")
            st.stop()

        # keep active_ds valid
        names_all = sorted(ss.datasets.keys())
        if ss.active_ds not in names_all:
            ss.active_ds = names_all[0]

        with st.container():
            # ---- one dataset picker for ALL subtabs ----
            c_ds, c_sample = st.columns([0.55, 0.45])
            with c_ds:
                dataset_combo("Dataset", "eda")

            # ---- one global sample slider (used by all charts) ----
            with c_sample:
                cur_df = ss.datasets[ss.active_ds]
                nmax = max(1, len(cur_df))
                ss.setdefault("eda_sample_n", min(5000, nmax))
                ss.eda_sample_n = st.slider(
                    "Sample rows for charts (for speed)",
                    min_value=1, max_value=nmax,
                    value=min(ss.eda_sample_n, nmax)
                )

            # working dataframe for every subtab below
            df = ss.datasets[ss.active_ds]

            # ---- subtabs ----
            tab_overview, tab_types, tab_combine, tab_missing, tab_univariate, tab_bivariate, tab_correlation = st.tabs(
                ["Overview", "Types", "Combine datasets", "Missingness", "Univariate", "Bivariate", "Correlation"]
            )

            with tab_overview:
                render_overview(df)

            with tab_types:
                render_eda_types(st.session_state)

            with tab_missing:
                render_missingness(df)

            with tab_univariate:
                render_univariate(df, sample_n=ss.eda_sample_n)

            with tab_bivariate:
                render_bivariate(df, sample_n=ss.eda_sample_n)

            with tab_correlation:
                render_correlation(df, sample_n=ss.eda_sample_n)

            # Combine uses ss; when a new dataset is saved inside, it should set
            # ss.active_ds = new_name and st.rerun() internally, so it will show up everywhere.
            with tab_combine:
                render_combine(ss)

    # =========================================================
    # PREPROCESS STEP
    # =========================================================
    if ss.step == "Preprocess":
        if not ss.datasets:
            st.info("Upload one or more files in **Summary** to begin.")
            st.stop()

        # keep your dataset picker (reuse the same combo you use in EDA)
        c_ds, _ = st.columns([0.6, 0.4])
        with c_ds:
            dataset_combo("Dataset", "prep")

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

    # =========================================================
    # FINAL SUMMARY STEP
    # =========================================================
    if ss.step == "Final Summary":
        if not ss.datasets:
            st.info("Upload one or more files in **Summary** to begin.")
            st.stop()

        names_all = sorted(ss.datasets.keys())
        if ss.active_ds not in names_all:
            ss.active_ds = names_all[0]

        render_final_summary_suite(st.session_state)

