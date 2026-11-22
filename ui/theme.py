"""
AutoData-Hub UI Theme Engine.

Injects the Dark Obsidian design system CSS tokens into Streamlit,
covering global layout, metric cards, badge pills, tab styles,
code blocks, scrollbars, and Plotly chart containers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import streamlit as st


# ---------------------------------------------------------------------------
# Design Token Constants (importable by tab modules)
# ---------------------------------------------------------------------------
COLORS = {
    "bg":           "#0E1117",
    "surface":      "#161B22",
    "surface_2":    "#1C2333",
    "border":       "#30363D",
    "accent":       "#6366F1",
    "accent_glow":  "rgba(99,102,241,0.18)",
    "success":      "#10B981",
    "warning":      "#F59E0B",
    "danger":       "#EF4444",
    "text":         "#F0F6FC",
    "text_muted":   "#8B949E",
    "chart_grid":   "#21262D",
}

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", color=COLORS["text"]),
    xaxis=dict(gridcolor=COLORS["chart_grid"], zerolinecolor=COLORS["chart_grid"]),
    yaxis=dict(gridcolor=COLORS["chart_grid"], zerolinecolor=COLORS["chart_grid"]),
    margin=dict(l=40, r=20, t=50, b=40),
    hoverlabel=dict(
        bgcolor=COLORS["surface_2"],
        bordercolor=COLORS["border"],
        font_size=12,
    ),
)


# ---------------------------------------------------------------------------
# CSS Injection
# ---------------------------------------------------------------------------

_CSS = f"""
<style>
/* ── Google Font ─────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global Reset ───────────────────────────────── */
html, body, [class*="css"] {{
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    background-color: {COLORS["bg"]} !important;
    color: {COLORS["text"]} !important;
}}

/* ── Main App Container ─────────────────────────── */
.main .block-container {{
    background-color: {COLORS["bg"]};
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1320px;
}}

/* ── Streamlit Header Bar ───────────────────────── */
header[data-testid="stHeader"] {{
    background-color: {COLORS["bg"]};
    border-bottom: 1px solid {COLORS["border"]};
}}

/* ── Sidebar ────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background-color: {COLORS["surface"]} !important;
    border-right: 1px solid {COLORS["border"]} !important;
}}
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {{
    color: {COLORS["accent"]} !important;
    font-weight: 600;
    letter-spacing: 0.03em;
}}

/* ── Metric Cards ───────────────────────────────── */
.adh-card {{
    background: {COLORS["surface"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}}
.adh-card:hover {{
    border-color: {COLORS["accent"]};
    box-shadow: 0 0 0 2px {COLORS["accent_glow"]};
}}
.adh-card-label {{
    font-size: 0.72rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {COLORS["text_muted"]};
    margin-bottom: 0.3rem;
}}
.adh-card-value {{
    font-size: 1.75rem;
    font-weight: 700;
    color: {COLORS["text"]};
    line-height: 1.1;
}}
.adh-card-sub {{
    font-size: 0.75rem;
    color: {COLORS["text_muted"]};
    margin-top: 0.25rem;
}}

/* ── KPI Row ────────────────────────────────────── */
.adh-kpi-row {{
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin-bottom: 1.25rem;
}}
.adh-kpi {{
    flex: 1;
    min-width: 130px;
    background: {COLORS["surface"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
}}
.adh-kpi-label {{
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {COLORS["text_muted"]};
}}
.adh-kpi-value {{
    font-size: 1.6rem;
    font-weight: 700;
    margin-top: 0.15rem;
}}

/* ── Badge Pills ─────────────────────────────────── */
.adh-badge {{
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}}
.adh-badge-success {{ background: rgba(16,185,129,0.15); color: {COLORS["success"]}; border: 1px solid {COLORS["success"]}; }}
.adh-badge-warning {{ background: rgba(245,158,11,0.15); color: {COLORS["warning"]}; border: 1px solid {COLORS["warning"]}; }}
.adh-badge-danger  {{ background: rgba(239,68,68,0.15);  color: {COLORS["danger"]};  border: 1px solid {COLORS["danger"]};  }}
.adh-badge-accent  {{ background: rgba(99,102,241,0.15); color: {COLORS["accent"]};  border: 1px solid {COLORS["accent"]};  }}

/* ── Section Headers ─────────────────────────────── */
.adh-section-header {{
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: {COLORS["text_muted"]};
    border-bottom: 1px solid {COLORS["border"]};
    padding-bottom: 0.4rem;
    margin: 1.2rem 0 0.8rem 0;
}}

/* ── Streamlit Tabs ──────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    background: {COLORS["surface"]};
    border-radius: 8px;
    padding: 3px;
    gap: 2px;
    border: 1px solid {COLORS["border"]};
}}
[data-testid="stTabs"] [data-baseweb="tab"] {{
    background: transparent;
    color: {COLORS["text_muted"]};
    border-radius: 6px;
    font-weight: 500;
    font-size: 0.85rem;
    padding: 0.45rem 1rem;
    border: none;
    transition: all 0.18s ease;
}}
[data-testid="stTabs"] [aria-selected="true"] {{
    background: {COLORS["accent"]} !important;
    color: white !important;
    font-weight: 600;
    box-shadow: 0 1px 6px rgba(99,102,241,0.4);
}}

/* ── Buttons ─────────────────────────────────────── */
.stButton > button {{
    background: {COLORS["accent"]};
    color: white;
    border: none;
    border-radius: 7px;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 0.48rem 1.2rem;
    transition: all 0.18s ease;
    letter-spacing: 0.02em;
}}
.stButton > button:hover {{
    background: #4F52D4;
    box-shadow: 0 0 0 3px {COLORS["accent_glow"]};
    transform: translateY(-1px);
}}
.stButton > button:active {{
    transform: translateY(0px);
}}

/* ── Secondary / Ghost Buttons ───────────────────── */
[data-testid="baseButton-secondary"] > button {{
    background: transparent !important;
    border: 1px solid {COLORS["border"]} !important;
    color: {COLORS["text_muted"]} !important;
}}
[data-testid="baseButton-secondary"] > button:hover {{
    border-color: {COLORS["accent"]} !important;
    color: {COLORS["accent"]} !important;
}}

/* ── Selectbox / Inputs ──────────────────────────── */
[data-baseweb="select"] > div,
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {{
    background-color: {COLORS["surface_2"]} !important;
    border: 1px solid {COLORS["border"]} !important;
    border-radius: 7px !important;
    color: {COLORS["text"]} !important;
}}

/* ── Dataframe / Table ───────────────────────────── */
[data-testid="stDataFrame"] iframe {{
    border-radius: 8px;
    border: 1px solid {COLORS["border"]};
}}

/* ── Code blocks ─────────────────────────────────── */
.stCode, code, pre {{
    background-color: {COLORS["surface_2"]} !important;
    border: 1px solid {COLORS["border"]} !important;
    border-radius: 7px !important;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 0.8rem !important;
}}

/* ── Alerts / Info boxes ─────────────────────────── */
[data-testid="stAlert"] {{
    background: {COLORS["surface"]} !important;
    border: 1px solid {COLORS["border"]} !important;
    border-radius: 8px !important;
}}

/* ── Progress bar ────────────────────────────────── */
[data-testid="stProgress"] > div > div {{
    background: linear-gradient(90deg, {COLORS["accent"]}, #8B5CF6) !important;
    border-radius: 4px;
}}

/* ── Scrollbar ───────────────────────────────────── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {COLORS["bg"]}; }}
::-webkit-scrollbar-thumb {{
    background: {COLORS["border"]};
    border-radius: 4px;
}}
::-webkit-scrollbar-thumb:hover {{ background: {COLORS["accent"]}; }}

/* ── Expander ────────────────────────────────────── */
[data-testid="stExpander"] {{
    background: {COLORS["surface"]} !important;
    border: 1px solid {COLORS["border"]} !important;
    border-radius: 8px !important;
}}

/* ── Divider ─────────────────────────────────────── */
hr {{
    border-color: {COLORS["border"]} !important;
    margin: 1rem 0;
}}

/* ── File Uploader ───────────────────────────────── */
[data-testid="stFileUploader"] {{
    background: {COLORS["surface"]};
    border: 2px dashed {COLORS["border"]};
    border-radius: 10px;
    transition: border-color 0.2s ease;
}}
[data-testid="stFileUploader"]:hover {{
    border-color: {COLORS["accent"]};
}}

/* ── Slider ──────────────────────────────────────── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {{
    background: {COLORS["accent"]} !important;
}}

/* ── Spinner ─────────────────────────────────────── */
[data-testid="stSpinner"] {{
    color: {COLORS["accent"]} !important;
}}

/* ── Hide Streamlit branding ─────────────────────── */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
</style>
"""


def inject_theme() -> None:
    """
    Injects the Dark Obsidian CSS design system into the active Streamlit page.
    Must be called once at the top of app.py before any other st.* calls.
    """
    st.markdown(_CSS, unsafe_allow_html=True)


def _is_plotly_figure(artifact: Any) -> bool:
    """Return True only when Plotly can interpret the value as one figure."""
    try:
        from plotly.basedatatypes import BaseFigure
        from plotly.graph_objects import Figure

        if isinstance(artifact, BaseFigure):
            return True
        if not isinstance(artifact, Mapping):
            return False
        Figure(dict(artifact))
        return True
    except (TypeError, ValueError):
        return False


def render_plotly_artifact(
    artifact: Any,
    *,
    key_prefix: str,
    columns: int = 1,
) -> int:
    """Safely render a figure or recursively flatten a mapping of named figures."""
    figures: list[tuple[str, Any]] = []

    def collect(value: Any, label: str) -> None:
        if _is_plotly_figure(value):
            figures.append((label, value))
        elif isinstance(value, Mapping):
            for name, nested_value in value.items():
                collect(nested_value, str(name))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, nested_value in enumerate(value):
                collect(nested_value, f"{label}_{index}")

    collect(artifact, key_prefix)
    if not figures:
        st.warning("No valid Plotly figures were available for this visualization.")
        return 0

    column_count = max(1, min(columns, len(figures)))

    def render_one(index: int, label: str, figure: Any) -> None:
        try:
            st.plotly_chart(
                figure,
                width="stretch",
                config={"displayModeBar": False},
                key=f"{key_prefix}_{index}",
            )
        except Exception as exc:
            st.warning(f"Could not render {label}: {exc}")

    for row_start in range(0, len(figures), column_count):
        row_figures = figures[row_start:row_start + column_count]
        if column_count == 1:
            label, figure = row_figures[0]
            render_one(row_start, label, figure)
            continue

        chart_columns = st.columns(column_count)
        for offset, (label, figure) in enumerate(row_figures):
            with chart_columns[offset]:
                render_one(row_start + offset, label, figure)

    return len(figures)


def metric_card(label: str, value: str, sub: str = "", color: str = "") -> str:
    """Returns HTML for a styled metric card."""
    val_style = f"color:{color};" if color else ""
    return f"""
    <div class="adh-card">
        <div class="adh-card-label">{label}</div>
        <div class="adh-card-value" style="{val_style}">{value}</div>
        {"" if not sub else f'<div class="adh-card-sub">{sub}</div>'}
    </div>
    """


def badge(text: str, variant: str = "accent") -> str:
    """Returns HTML for a colored badge pill. variant: success|warning|danger|accent."""
    return f'<span class="adh-badge adh-badge-{variant}">{text}</span>'


def section_header(title: str) -> str:
    """Returns HTML for a styled section header."""
    return f'<div class="adh-section-header">{title}</div>'


def kpi_row(items: list[dict]) -> str:
    """
    Renders a horizontal KPI row.
    Each item: {"label": str, "value": str, "color": str (optional)}
    """
    cards = ""
    for item in items:
        color = f"color:{item['color']};" if item.get("color") else ""
        cards += f"""
        <div class="adh-kpi">
            <div class="adh-kpi-label">{item["label"]}</div>
            <div class="adh-kpi-value" style="{color}">{item["value"]}</div>
        </div>"""
    return f'<div class="adh-kpi-row">{cards}</div>'
