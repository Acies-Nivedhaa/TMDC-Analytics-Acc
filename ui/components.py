import streamlit as st
import pandas as pd
from typing import Iterable, Tuple, List, Dict

__all__ = ["header_bar", "kpi_row", "section", "render_table", "control_bar", "steps_nav"]


def header_bar(title: str):
    st.markdown(
        f"""
        <div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:0.75rem;'>
            <h2 style='margin:0'>{title}</h2>
        </div>
        <hr style='margin-top:0.25rem;margin-bottom:1rem;'>
        """,
        unsafe_allow_html=True,
    )


def control_bar() -> Dict[str, bool]:
    # Right-aligned trio with wider columns to avoid wrapping
    spacer, c1, c2, c3 = st.columns([6,2,2,2])
    with c1: clear = st.button("🗑️ Clear")
    with c2: undo = st.button("↩️ Undo")
    # NBSP keeps label on one line
    with c3: restore = st.button("⟳ Restore\u00A0RAW")
    return {"clear": clear, "undo": undo, "restore": restore}


def steps_nav(steps: List[str], selected: str = "Summary"):
    idx = steps.index(selected) if selected in steps else 0
    st.radio("Steps", steps, index=idx)


def kpi_row(items: Iterable[Tuple[str, str]]):
    items = list(items)
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        with col:
            st.metric(label, value)


class section:
    def __init__(self, title: str, expandable: bool = True, expanded: bool = True):
        self.title = title
        self.expandable = expandable
        self.expanded = expanded
        self.ctx = None
    def __enter__(self):
        if self.expandable:
            self.ctx = st.expander(self.title, expanded=self.expanded)
            self.ctx.__enter__()
        else:
            self.ctx = st.container()
            self.ctx.__enter__()
            st.subheader(self.title)
        return self.ctx
    def __exit__(self, exc_type, exc, tb):
        if self.ctx is not None:
            self.ctx.__exit__(exc_type, exc, tb)


def render_table(df: pd.DataFrame, height: int = 360):
    st.dataframe(df, use_container_width=True, height=height)


