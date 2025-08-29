# app.py
import hashlib
import pandas as pd
import streamlit as st

from core.data_io import read_any, read_zip_all
from core.summary import (
    overview_stats, infer_schema, column_quick_stats,
    suggest_actions, dataset_meta, demo_data, nunique_safe, build_summary_pdf,
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
from core.final_summary import render_final_summary
from ui.components import header_bar, kpi_row, section, render_table, control_bar

# NEW: Trino helpers
from core.trino_connection import TrinoConfig, query_df, q_ident

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
                            for i, (inner_name, df) in enumerate(tables):
                                if df is None or df.empty:
                                    continue
                                ds_name = ensure_unique_name(set(ss.datasets.keys()), inner_name.rsplit("/", 1)[-1])
                                ss.datasets[ds_name] = df
                                ss.raw_datasets[ds_name] = df.copy()
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
                if ss.file_meta:
                    st.markdown("**Files added**")
                    for key, meta in list(ss.file_meta.items()):
                        ds_name = meta.get("name")
                        fname = meta.get("filename", ds_name)

                        c1, c2, c3 = st.columns([6, 5, 1])
                        with c1:
                            st.write(f"**{fname}**")
                        with c2:
                            st.caption(f"Dataset: {ds_name}")
                        with c3:
                            if st.button("✖", key=f"rm_{key}", help=f"Remove '{ds_name}' from analysis"):
                                ss.datasets.pop(ds_name, None)
                                ss.raw_datasets.pop(ds_name, None)
                                # Only remove the specific mapping we displayed
                                ss.file_meta.pop(key, None)
                                if ss.active_ds == ds_name:
                                    ss.active_ds = next(iter(ss.datasets.keys()), None)
                                log(f"Removed dataset '{ds_name}' via file list.")
                                st.rerun()


# ---------------------------
# B) DataOS (Trino) (NEW) — prompts for ALL fields, no secrets required
# ---------------------------
            else:
                st.caption("Connect and pull a table from DataOS (Trino)")

                c1, c2 = st.columns(2)
                host = c1.text_input("Host", placeholder="your.trino.host")
                port = c2.number_input("Port", min_value=1, value=443, step=1)

                user = c1.text_input("Username")
                password = c2.text_input("Password", type="password")

                http_scheme = c1.selectbox("HTTP scheme", ["https", "http"], index=0)
                cluster_name = c2.text_input('HTTP header: cluster-name', value="minervac")

                c3, c4 = st.columns(2)
                catalog = c3.text_input("Catalog", value="icebase")
                schema  = c4.text_input("Schema",  value="telemetry")

                c5, c6 = st.columns(2)
                table = c5.text_input("Table", value="device")
                limit = c6.number_input("Row limit", min_value=1, value=1000, step=100)

                sql = f"SELECT * FROM {q_ident(catalog)}.{q_ident(schema)}.{q_ident(table)} LIMIT {int(limit)}"
                st.code(sql, language="sql")

                # Basic validation: disable until required fields are filled
                required_ok = all([host.strip(), user.strip(), password.strip(), catalog.strip(), schema.strip(), table.strip()])

                if st.button("Connect & Fetch", type="primary", disabled=not required_ok):
                    try:
                        import trino  # ensure package exists in this venv
                    except Exception:
                        st.error("The 'trino' package isn’t installed in this environment. Run: `pip install trino`")
                    else:
                        cfg = TrinoConfig(
                            host=host.strip(),
                            port=int(port),
                            user=user.strip(),
                            password=password,
                            http_scheme=http_scheme,
                            http_headers={"cluster-name": cluster_name.strip() or "minervac"},
                            catalog=catalog.strip(),
                            schema=schema.strip(),
                        )
                        try:
                            df = query_df(sql, cfg)
                            # Save as a new dataset like uploads do
                            base_label = f"{catalog}.{schema}.{table}"
                            ds_name = ensure_unique_name(set(ss.datasets.keys()), base_label)
                            ss.datasets[ds_name] = df
                            ss.raw_datasets[ds_name] = df.copy()
                            ss.active_ds = ds_name
                            # Track in file list with a pseudo key
                            meta_key = f"trino:{ds_name}"
                            ss.file_meta[meta_key] = {"name": ds_name, "filename": f"trino › {base_label}"}
                            log(f"Loaded '{base_label}' from DataOS (Trino) into '{ds_name}'.")
                            st.success(f"Loaded {len(df):,} rows.")
                            st.dataframe(df.head(50), use_container_width=True)
                            st.download_button(
                                "Download as CSV",
                                data=df.to_csv(index=False).encode("utf-8"),
                                file_name=f"{catalog}_{schema}_{table}.csv",
                                mime="text/csv",
                            )
                        except Exception as e:
                            st.error(f"Query failed: {e}")


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
            render_table(pd.DataFrame(stats_rows), height=220)

            # Rename only; propagate to mappings so only the new name is referenced
            rn_src = st.selectbox("Rename dataset", ["—"] + names, index=0, key="rn_src")
            new_name = st.text_input("New name", value="", key="rn_new")
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

            schema_tbl = pd.DataFrame({"column": df.columns, "dtype": [str(t) for t in df.dtypes]})
            render_table(schema_tbl, height=260)

            st.markdown("**Preview (first 20 rows)**")
            st.dataframe(df.head(20), use_container_width=True)

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

        # --- Export: Summary as PDF (end of Summary page) ---
        st.markdown("---")
        try:
            pdf_bytes = build_summary_pdf(
                active_name=ss.active_ds,
                df=df,
                datasets=ss.datasets,   # include all datasets so they appear in the PDF
            )
            st.download_button(
                label="⬇️ Download Summary (PDF)",
                data=pdf_bytes,
                file_name=f"{ss.active_ds}_summary.pdf",
                mime="application/pdf",
                key=f"download_summary_pdf_{ss.active_ds}"
            )
        except ImportError:
            st.info("Install reportlab to enable PDF export: `pip install reportlab`")

    # =========================================================
    # EDA STEP
    # =========================================================
    elif ss.step == "EDA":
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
    elif ss.step == "Preprocess":
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
    elif ss.step == "Final Summary":
        if not ss.datasets:
            st.info("Upload one or more files in **Summary** to begin.")
            st.stop()

        # keep active_ds valid
        names_all = sorted(ss.datasets.keys())
        if ss.active_ds not in names_all:
            ss.active_ds = names_all[0]

        df = ss.datasets[ss.active_ds]
        # Outstanding issues + KPIs + compact preview (from core/final_summary.py)
        render_final_summary(df, st.session_state)
