# ui/components.py
# ----------------------------------------------------------------------
# Common UI helpers shared across the Streamlit app.
# Keep these tiny & dependency-free—only Streamlit + pandas.
# ----------------------------------------------------------------------

from typing import Iterable, Tuple, List, Dict
import pandas as pd
import streamlit as st

__all__ = ["header_bar", "kpi_row", "section", "render_table", "control_bar", "steps_nav"]


def header_bar(title: str) -> None:
    """
    Page/section header used on the right pane (beneath the big app title).

    Parameters
    ----------
    title : str
        Text to render as the section heading.
    """
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
    """
    Top-right control strip with three action buttons.
    Returns a dict of booleans keyed by action name.

    Returns
    -------
    Dict[str, bool]
        {"clear": bool, "undo": bool, "restore": bool}
    """
    # Right-aligned trio with wider columns to avoid wrapping on smaller screens
    spacer, c1, c2, c3 = st.columns([6, 2, 2, 2])
    with c1:
        clear = st.button("🗑️ Clear")
    with c2:
        undo = st.button("↩️ Undo")
    # NBSP keeps "Restore RAW" on a single line
    with c3:
        restore = st.button("⟳ Restore\u00A0RAW")
    return {"clear": clear, "undo": undo, "restore": restore}


def steps_nav(steps: List[str], selected: str = "Summary") -> None:
    """
    Minimal radio-based step navigator (unused in the main layout but kept for reuse).

    Parameters
    ----------
    steps : List[str]
        Ordered list of labels to show.
    selected : str, optional
        Which label should be pre-selected, by default "Summary".
    """
    idx = steps.index(selected) if selected in steps else 0
    st.radio("Steps", steps, index=idx)


def kpi_row(items: Iterable[Tuple[str, str]]) -> None:
    """
    Display a single row of compact KPI cards without truncation.

    Parameters
    ----------
    items : Iterable[Tuple[str, str]]
        Sequence of (label, value) pairs. Values can be strings or numbers
        (they will be rendered as strings).
    """
    items = list(items)  # allow callers to pass generators/iterables

    # Styles scoped to a simple wrapper class to avoid side-effects
    st.markdown(
        """
        <style>
          .kpi-wrap {display:flex; gap:12px;}
          .kpi {flex:1; border:1px solid rgba(0,0,0,.08); border-radius:10px;
                padding:10px 12px; background: var(--background-color);}
          .kpi .label {font-size:.78rem; color:rgba(0,0,0,.6);}
          .kpi .value {font-size:1.05rem; font-weight:600; line-height:1.2;
                       white-space:normal; overflow:visible; text-overflow:unset;
                       font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
                                    "Liberation Mono", "Courier New", monospace;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Use Streamlit columns so content wraps nicely on narrow widths
    cols = st.columns(len(items))
    for (label, value), col in zip(items, cols):
        with col:
            st.markdown(
                f'<div class="kpi"><div class="label">{label}</div>'
                f'<div class="value">{value}</div></div>',
                unsafe_allow_html=True,
            )


class section:
    """
    Context manager that renders a titled area.

    - If `expandable=True`, uses an expander with the given title.
    - Otherwise, renders a container with a Streamlit subheader.

    Example
    -------
    with section("Select Files", expandable=True):
        ...
    """

    def __init__(self, title: str, expandable: bool = True, expanded: bool = True):
        self.title = title
        self.expandable = expandable
        self.expanded = expanded
        self.ctx = None  # type: ignore[var-annotated]

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


def render_table(df: pd.DataFrame, height: int = 360) -> None:
    """
    Thin wrapper to keep a consistent look for dataframes.

    Parameters
    ----------
    df : pd.DataFrame
        Table to render.
    height : int, optional
        Fixed height in pixels, by default 360.
    """
    st.dataframe(df, use_container_width=True, height=height)
