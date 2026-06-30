"""
components/cards.py
====================
Reusable HTML/CSS-driven UI components: KPI cards, risk badges, and
recommendation cards. Centralizing these keeps every page visually
consistent (the core requirement of a "Power BI-grade" look and feel).
"""

import streamlit as st

from config import RISK_COLORS, COLOR_PRIMARY
def kpi_card(label: str, value: str, icon: str = "📊", delta: str | None = None, delta_positive: bool = True):
    """Render a single large KPI card with optional delta indicator."""
    delta_html = ""
    if delta is not None:
        color = "#22C55E" if delta_positive else "#EF4444"
        arrow = "⬆" if delta_positive else "⬇"
        delta_html = f'<div class="kpi-delta" style="color:{color};">{arrow} {delta}</div>'

    # DHYAN DEIN: {delta_html} ko ab alag line pe nahi rakha hai
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-content">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>{delta_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def kpi_row(cards: list[dict]):
    """Render a responsive row of KPI cards. Each dict: label, value, icon, delta(optional)."""
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        with col:
            kpi_card(**card)


def risk_badge(risk_label: str) -> str:
    """Return HTML for a colored risk pill badge."""
    color = RISK_COLORS.get(risk_label, COLOR_PRIMARY)
    return f'<span class="risk-badge" style="background-color:{color}1A; color:{color}; border:1px solid {color};">{risk_label}</span>'


def render_risk_badge(risk_label: str):
    st.markdown(risk_badge(risk_label), unsafe_allow_html=True)


def recommendation_card(title: str, description: str, icon: str = "💡", severity: str = "info"):
    """Render a single recommendation card. severity: 'info' | 'warning' | 'danger'."""
    border_colors = {"info": COLOR_PRIMARY, "warning": "#F59E0B", "danger": "#EF4444"}
    border = border_colors.get(severity, COLOR_PRIMARY)
    st.markdown(
        f"""
        <div class="recommendation-card" style="border-left: 4px solid {border};">
            <div class="recommendation-icon">{icon}</div>
            <div>
                <div class="recommendation-title">{title}</div>
                <div class="recommendation-desc">{description}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str | None = None, icon: str = ""):
    """Consistent section header used at the top of every chart block / page section."""
    subtitle_html = f'<div class="section-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="section-header">
            <div class="section-title">{icon} {title}</div>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = "", icon: str = ""):
    """Large page-level header banner used at the top of every page."""
    st.markdown(
        f"""
        <div class="page-header">
            <div class="page-header-title">{icon} {title}</div>
            <div class="page-header-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(message: str, icon: str = "📭"):
    """Friendly empty-state message when filters return zero rows."""
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="empty-state-icon">{icon}</div>
            <div class="empty-state-text">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
