# core/preprocess_text.py
from __future__ import annotations
import io, re
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

from ui.components import section, render_table

# -------- small, dependency-light helpers --------
_EN_STOP = {
    # compact default English stop list (can be expanded)
    "a","an","and","are","as","at","be","by","for","from","has","he","in","is","it","its","of","on","that","the","to","was","were","will","with","you","your","yours","me","my","we","our","ours","they","their","this","those","these","i"
}

def _strip_html(text: str) -> str:
    # BS4 is nice but optional; regex fallback is fine for short user notes
    text = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", " ", text)
    text = re.sub(r"(?s)<.*?>", " ", text)
    return text

def _remove_punct(text: str) -> str:
    return re.sub(r"[^\w\s']", " ", text)  # keep word chars & apostrophes

def _remove_numbers(text: str) -> str:
    return re.sub(r"\d+", " ", text)

def _remove_non_ascii(text: str) -> str:
    return text.encode("ascii", "ignore").decode("ascii")

def _stemmer():
    try:
        from nltk.stem import PorterStemmer
        ps = PorterStemmer()
        return lambda w: ps.stem(w)
    except Exception:
        return lambda w: w  # no-op if nltk missing

def _tokenize(text: str) -> list[str]:
    return [w for w in re.split(r"\s+", text.strip()) if w]

def _process_text_series(
    s: pd.Series,
    *,
    lowercase: bool,
    strip_html: bool,
    remove_punct: bool,
    remove_numbers: bool,
    remove_stop: bool,
    keep_unicode: bool,
    light_stem: bool,
    custom_stoplist: set[str],
) -> pd.Series:
    stem = _stemmer() if light_stem else (lambda w: w)
    stop = set(_EN_STOP)
    stop |= {w.lower() for w in custom_stoplist or set()}
    def clean_one(t: str | float) -> str:
        if pd.isna(t):
            return ""
        x = str(t)
        if strip_html:    x = _strip_html(x)
        if not keep_unicode: x = _remove_non_ascii(x)
        if lowercase:     x = x.lower()
        if remove_punct:  x = _remove_punct(x)
        if remove_numbers:x = _remove_numbers(x)
        # normalize whitespace
        x = re.sub(r"\s+", " ", x).strip()
        if not x:
            return x
        toks = _tokenize(x)
        if remove_stop:
            toks = [w for w in toks if w not in stop]
        if light_stem:
            toks = [stem(w) for w in toks]
        return " ".join(toks)
    return s.astype("string").map(clean_one)

def _token_counts(texts: Iterable[str], top_n: int) -> tuple[list[tuple[str,int]], int]:
    ctr = Counter()
    total = 0
    for t in texts:
        if not isinstance(t, str):
            continue
        toks = _tokenize(t)
        ctr.update(toks)
        total += len(toks)
    top = ctr.most_common(top_n)
    return top, total

def _wordcloud_or_bar(top_pairs: list[tuple[str,int]]):
    labels = [w for w,_ in top_pairs]
    counts = [c for _,c in top_pairs]
    try:
        from wordcloud import WordCloud
        wc = WordCloud(width=1000, height=400, background_color="white")
        img = wc.generate_from_frequencies(dict(top_pairs)).to_image()
        st.image(img, use_container_width=True)
    except Exception:
        # fallback: simple bar chart
        st.bar_chart(pd.DataFrame({"word": labels, "count": counts}).set_index("word"))

# --------- the tab renderer ---------
def render_preprocess_text(ss) -> None:
    if not ss.active_ds or ss.active_ds not in ss.datasets:
        st.info("Pick a dataset to begin.")
        st.stop()

    df = ss.datasets[ss.active_ds]
    # text-like columns
    text_cols = [c for c in df.columns if str(df[c].dtype) in ("string", "object", "category")]
    if not text_cols:
        st.info("No text-like columns found (string/object/category).")
        return

    # keep session state
    ss.setdefault("pp_text_stoplist", set())   # custom stop words
    ss.setdefault("pp_text_options", {})       # last used options per column

    # --- Main controls (top area) ---
    c_col = st.selectbox("Text column", text_cols, index=0, key="ppt_col")

    opts = ss.pp_text_options.get(c_col, {
        "lowercase": True,
        "strip_html": True,
        "remove_punct": True,
        "remove_numbers": False,
        "remove_stop": True,
        "light_stem": False,
        "keep_unicode": False,
        "replace_inplace": False,
    })

    with section("Options", expandable=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            opts["lowercase"]      = st.checkbox("Lowercase", value=opts["lowercase"], key="ppt_opt_lower")
            opts["strip_html"]     = st.checkbox("Strip HTML", value=opts["strip_html"], key="ppt_opt_html")
            opts["remove_punct"]   = st.checkbox("Remove punctuation", value=opts["remove_punct"], key="ppt_opt_punct")
        with c2:
            opts["remove_numbers"] = st.checkbox("Remove numbers", value=opts["remove_numbers"], key="ppt_opt_num")
            opts["remove_stop"]    = st.checkbox("Remove stopwords", value=opts["remove_stop"], key="ppt_opt_stop")
            opts["light_stem"]     = st.checkbox("Light stemming", value=opts["light_stem"], key="ppt_opt_stem")
        with c3:
            opts["keep_unicode"]   = st.checkbox("Keep raw unicode", value=opts["keep_unicode"], key="ppt_opt_uni")
            opts["replace_inplace"]= st.checkbox("Replace original column", value=opts["replace_inplace"], key="ppt_opt_inplace")

        # output name (ignored when replacing)
        default_out = f"{c_col}_clean"
        out_name = st.text_input("Output column name (ignored if replacing)",
                                 value=default_out, key="ppt_outname", disabled=opts["replace_inplace"])

    ss.pp_text_options[c_col] = opts

    # --- Live preview (original vs processed) ---
    with section("Live preview (original vs processed)", expandable=False):
        c_side, c_rows = st.columns([0.20, 0.80])
        with c_side:
            sample_kind = st.radio("Preview sample", ["Head", "Random"], horizontal=True, key="ppt_sample_kind")
        with c_rows:
            n_rows = st.slider("Rows", min_value=5, max_value=50, value=15, key="ppt_nrows")

        series = df[c_col]
        sample = series.head(n_rows) if sample_kind == "Head" else series.sample(n_rows, random_state=1)
        prev = _process_text_series(
            sample,
            lowercase=opts["lowercase"],
            strip_html=opts["strip_html"],
            remove_punct=opts["remove_punct"],
            remove_numbers=opts["remove_numbers"],
            remove_stop=opts["remove_stop"],
            keep_unicode=opts["keep_unicode"],
            light_stem=opts["light_stem"],
            custom_stoplist=ss.pp_text_stoplist,
        )
        render_table(pd.DataFrame({"original": sample.to_list(), "processed": prev.to_list()}), height=340)

        with st.expander("Preview stats", expanded=False):
            # a few quick metrics post-processing
            proc_all = _process_text_series(
                df[c_col].astype("string").fillna(""),
                lowercase=opts["lowercase"],
                strip_html=opts["strip_html"],
                remove_punct=opts["remove_punct"],
                remove_numbers=opts["remove_numbers"],
                remove_stop=opts["remove_stop"],
                keep_unicode=opts["keep_unicode"],
                light_stem=opts["light_stem"],
                custom_stoplist=ss.pp_text_stoplist,
            )
            n_empty = int((proc_all.str.len() == 0).sum())
            avg_len = float(proc_all.str.len().mean())
            st.write(f"- Rows: **{len(proc_all):,}**  •  Avg length: **{avg_len:.1f}** chars  •  Empty after cleaning: **{n_empty:,}**")

    # --- Word cloud + Custom stoplist ---
    with section("Word cloud & custom stoplist", expandable=False):
        # sliders
        total_rows = len(df)
        # Robust slider for small datasets
        max_rows = int(max(1, min(7074, total_rows)))  # upper cap and at least 1
        min_rows = 1
        default_rows = int(min(5000, max_rows))

        if max_rows <= min_rows:
            # Not enough data to show a range — just use what's available
            sample_n = max_rows
            st.caption(f"Using {sample_n} row(s) for word cloud (dataset is small).")
        else:
            sample_n = st.slider(
                "Rows for word cloud (sampled for speed)",
                min_value=min_rows,
                max_value=max_rows,
                value=default_rows,
                key="ppt_wc_rows",
            )

        top_n     = st.slider("Show top N words", min_value=10, max_value=200, value=100, key="ppt_wc_topn")

        sample_wc = df[c_col].sample(sample_n, random_state=7) if sample_n < total_rows else df[c_col]
        proc_wc = _process_text_series(
            sample_wc,
            lowercase=opts["lowercase"],
            strip_html=opts["strip_html"],
            remove_punct=opts["remove_punct"],
            remove_numbers=opts["remove_numbers"],
            remove_stop=opts["remove_stop"],
            keep_unicode=opts["keep_unicode"],
            light_stem=opts["light_stem"],
            custom_stoplist=ss.pp_text_stoplist,
        )
        top_pairs, total_tokens = _token_counts(proc_wc, top_n)
        _wordcloud_or_bar(top_pairs)

        st.caption("Top words (after current preprocessing)")
        render_table(pd.DataFrame(top_pairs, columns=["user_note", "count"]), height=280)

        # Build custom stoplist UI
        st.markdown("Build a custom stoplist (applies only when **Remove stopwords** is checked).")
        col_sel, col_add = st.columns([0.5, 0.5])

        with col_sel:
            choices = ["— choose —"] + [w for w,_ in top_pairs]
            pick = st.selectbox("Select words to add to custom stoplist", choices, index=0, key="ppt_pick_stop")
            if pick != "— choose —" and st.button("➕ Add selected", key="ppt_add_sel"):
                ss.pp_text_stoplist.add(pick)

        with col_add:
            to_add = st.text_input("Or type words (comma-separated) to add", value="", key="ppt_add_text", placeholder="e.g., ok, inc, products")
            if st.button("➕ Add to custom stoplist", key="ppt_add_many") and to_add.strip():
                for w in [w.strip() for w in to_add.split(",") if w.strip()]:
                    ss.pp_text_stoplist.add(w.lower())

        if ss.pp_text_stoplist:
            st.caption("Current custom stoplist")
            st.code(", ".join(sorted(ss.pp_text_stoplist)), language="text")

    # --- Apply ---
    with section("Apply", expandable=False):
        if st.button("Apply processing", type="primary", key="ppt_apply"):
            out_series = _process_text_series(
                df[c_col],
                lowercase=opts["lowercase"],
                strip_html=opts["strip_html"],
                remove_punct=opts["remove_punct"],
                remove_numbers=opts["remove_numbers"],
                remove_stop=opts["remove_stop"],
                keep_unicode=opts["keep_unicode"],
                light_stem=opts["light_stem"],
                custom_stoplist=ss.pp_text_stoplist,
            )
            ss.df_history.append(df.copy())
            if opts["replace_inplace"]:
                df[c_col] = out_series
            else:
                name = out_name or f"{c_col}_clean"
                df[name] = out_series
            ss.datasets[ss.active_ds] = df
            st.success("Text preprocessing applied.")
