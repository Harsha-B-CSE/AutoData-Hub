"""Standardized In-Tab UX guidance components for AutoData-Hub workspaces."""

from __future__ import annotations

import streamlit as st


def render_tab_guidance_header(
    title: str,
    objective: str,
    methodology: str,
    deliverable: str,
    *,
    expanded: bool = False,
) -> None:
    """Render a standardized collapsible guidance header at the top of a tab."""
    label = f"ℹ️ **Tab Capability Guide: {title}** · Workflow & Methodology Guide"
    with st.expander(label, expanded=expanded):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**🎯 Objective & Scope**\n\n{objective}")
        with col2:
            st.markdown(f"**🔬 Core Methodology**\n\n{methodology}")
        with col3:
            st.markdown(f"**📄 Output Deliverable**\n\n{deliverable}")
