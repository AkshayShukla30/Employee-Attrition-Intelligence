"""
streamlit_app.py
=================
Entry point for the Employee Attrition Intelligence Platform (EAIP).
Sets global page config, injects the custom theme (style.css), trains/loads
the ML model on first boot, and renders the landing / executive summary view.
Individual analytical pages live under pages/ as a native Streamlit
multi-page app.
"""

import sys
from pathlib import Path

import streamlit as st

# Ensure local package imports (config, utils, components) resolve from any page
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from config import APP_NAME, APP_TAGLINE, COLOR_PRIMARY, CSS_PATH
from components.cards import kpi_row, page_header, section_header
from components.sidebar import render_sidebar
from utils.preprocessing import load_clean_data
from utils.prediction import load_model, load_model_metadata


def inject_css():
    """Load and inject the global stylesheet."""
    if CSS_PATH.exists():
        with open(CSS_PATH) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def configure_page():
    st.set_page_config(
        page_title=f"{APP_NAME}",
        page_icon="🏢",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()


def main():
    configure_page()

    # Train (first run only) / load the model so metadata is available app-wide
    with st.spinner("Initializing AI engine..."):
        load_model()
        metadata = load_model_metadata()

    df = load_clean_data()
    filtered_df = render_sidebar(df, metadata)
    st.session_state["filtered_df"] = filtered_df
    st.session_state["full_df"] = df

    page_header(
        APP_NAME,
        subtitle=f"{APP_TAGLINE} &nbsp;|&nbsp; AI-Powered HR Decision Support",
        icon="🏢",
    )

    # ---- Executive Summary KPIs ----
    section_header("Executive Summary", "Live snapshot based on current filters", icon="📌")
    total = len(filtered_df)
    left = int((filtered_df["Attrition"] == "Yes").sum())
    rate = (left / total * 100) if total else 0
    avg_income = filtered_df["MonthlyIncome"].mean() if total else 0
    avg_tenure = filtered_df["YearsAtCompany"].mean() if total else 0
    avg_age = filtered_df["Age"].mean() if total else 0

    kpi_row([
        {"label": "Total Employees", "value": f"{total:,}", "icon": "👥"},
        {"label": "Employees Left", "value": f"{left:,}", "icon": "🚪"},
        {"label": "Attrition Rate", "value": f"{rate:.1f}%", "icon": "📉"},
        {"label": "Avg. Monthly Income", "value": f"${avg_income:,.0f}", "icon": "💵"},
        {"label": "Avg. Years at Company", "value": f"{avg_tenure:.1f}", "icon": "📅"},
        {"label": "Avg. Age", "value": f"{avg_age:.1f}", "icon": "🎂"},
    ])

    st.write("")
    col1, col2 = st.columns([2, 1])
    with col1:
        section_header("Welcome", icon="👋")
        st.markdown(
            """
            **EAIP** is an enterprise HR analytics platform that combines descriptive analytics,
            predictive machine learning, and explainable AI to help HR teams understand, predict,
            and reduce employee attrition.

            Use the navigation panel on the left to explore:

            - **📊 Dashboard** — Executive KPIs and attrition breakdowns
            - **🔍 Analytics** — Deep-dive, fully filterable analytics workbench
            - **🎯 Prediction** — Score an individual employee's attrition risk in real time
            - **🧠 AI Insights** — Global model explainability (SHAP)
            - **🏆 Model Performance** — Compare candidate models and metrics
            - **ℹ️ About** — Project background, methodology, and roadmap
            """
        )
    with col2:
        section_header("Active Model", icon="🤖")
        st.markdown(
            f"""
            <div class="recommendation-card" style="border-left:4px solid {COLOR_PRIMARY};">
                <div class="recommendation-icon">🏆</div>
                <div>
                    <div class="recommendation-title">{metadata.get('best_model', 'N/A')}</div>
                    <div class="recommendation-desc">
                        Selected automatically by highest ROC AUC across 5 candidate algorithms.
                        Trained on {metadata.get('n_train', '—')} records, validated on
                        {metadata.get('n_test', '—')} held-out records.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
