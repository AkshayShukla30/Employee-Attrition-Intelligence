"""
pages/6_About.py
=================
Project background page: description, workflow, dataset, models used,
business objective, future improvements, and developer information.
"""

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from config import APP_NAME, APP_TAGLINE, CSS_PATH
from components.cards import page_header, section_header
from components.sidebar import render_sidebar
from utils.preprocessing import load_clean_data
from utils.prediction import load_model_metadata


def inject_css():
    if CSS_PATH.exists():
        with open(CSS_PATH) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


st.set_page_config(page_title=f"About | {APP_NAME}", page_icon="ℹ️", layout="wide")
inject_css()

df = load_clean_data()
metadata = load_model_metadata()
render_sidebar(df, metadata)

page_header("About This Platform", APP_TAGLINE, icon="ℹ️")

section_header("Project Description", icon="📘")
st.markdown(
    f"""
    **{APP_NAME} ({"EAIP"})** is an end-to-end HR analytics and machine learning platform that
    helps organizations understand, predict, and proactively manage employee attrition. It
    combines descriptive analytics, predictive modeling, and explainable AI into a single
    decision-support tool for HR leaders, business partners, and people managers.
    """
)

section_header("Workflow", icon="🔄")
st.markdown(
    """
    1. **Data Ingestion & Cleaning** — load and validate HR records, drop non-informative columns.
    2. **Feature Engineering** — derive tenure, promotion-cadence, and income-based features.
    3. **Model Training** — train and compare five candidate algorithms via a leak-free
       `ColumnTransformer` + `Pipeline` architecture.
    4. **Model Selection** — automatically select the best model by ROC AUC on a held-out test set.
    5. **Explainability** — generate SHAP-based global and per-employee explanations.
    6. **Deployment** — serve real-time predictions, what-if simulations, and PDF reporting
       through this Streamlit application.
    """
)

section_header("Dataset", icon="🗂️")
st.markdown(
    f"""
    The platform is built on the **IBM HR Analytics Employee Attrition dataset**, containing
    **{len(df):,} employee records** across **{df.shape[1]} attributes** spanning demographics,
    compensation, job role, satisfaction scores, and tenure history.
    """
)

section_header("Models Used", icon="🤖")
st.markdown(
    """
    - Logistic Regression (balanced class weights)
    - Random Forest Classifier
    - Gradient Boosting Classifier
    - Extra Trees Classifier
    - XGBoost Classifier *(when available in the runtime environment)*

    The best-performing model (by ROC AUC) is automatically selected and deployed for live
    predictions — see the **Model Performance** page for the full comparison.
    """
)

section_header("Business Objective", icon="🎯")
st.markdown(
    """
    Voluntary employee attrition carries significant, often underestimated costs: recruitment,
    onboarding, lost institutional knowledge, and reduced team productivity. EAIP's objective is
    to convert raw HR data into **early, individualized, and explainable risk signals** so HR
    teams can intervene before attrition happens — not after an employee has already resigned.
    """
)

section_header("Future Improvements", icon="🚀")
st.markdown(
    """
    - Integrate live HRIS data feeds for continuous, automated re-scoring
    - Add survival analysis to estimate *time-to-attrition*, not just probability
    - Incorporate employee sentiment / engagement survey text via NLP
    - Role-based access control and audit logging for enterprise deployment
    - A/B testing framework to measure the impact of HR interventions over time
    """
)


section_header("Developer Information", icon="👨‍💻")

st.info("""
### 👨‍💻 Akshay Shukla



🔗 LinkedIn  
https://www.linkedin.com/in/akshayshukla-/

💻 GitHub  
https://github.com/AkshayShukla30
""")