"""
pages/3_Prediction.py
======================
Single-employee Attrition Prediction page. Two-column premium layout:
left = employee input form, right = prediction result card with risk gauge.
Below: SHAP-based AI explanation, HR recommendation cards, an interactive
What-If analysis, and a downloadable PDF report.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from config import APP_NAME, CSS_PATH, FORM_CATEGORICAL_FIELDS, FORM_NUMERIC_FIELDS
from components.cards import page_header, recommendation_card, render_risk_badge, section_header
from components.charts import risk_gauge, what_if_comparison_bar
from components.recommendation_engine import generate_recommendations
from components.sidebar import render_sidebar
from utils.preprocessing import load_clean_data, single_employee_to_dataframe
from utils.prediction import load_model, load_model_metadata, predict_employee
from utils.report_generator import generate_prediction_report
from utils.shap_utils import (
    SHAP_AVAILABLE, compute_shap_values, natural_language_explanation,
    top_contributing_factors,
)


def inject_css():
    if CSS_PATH.exists():
        with open(CSS_PATH) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


st.set_page_config(page_title=f"Prediction | {APP_NAME}", page_icon="🎯", layout="wide")
inject_css()

df = load_clean_data()
metadata = load_model_metadata()
render_sidebar(df, metadata)
model = load_model()

page_header("Employee Attrition Prediction", "Score an individual employee's attrition risk in real time", icon="🎯")

left_col, right_col = st.columns([1.15, 1])

# ---------------------------------------------------------------------------
# LEFT: Employee Input Form
# ---------------------------------------------------------------------------
with left_col:
    section_header("Employee Profile", "Enter employee attributes", icon="📝")

    with st.form("employee_form"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Age", 18, 60, 32)
            gender = st.selectbox("Gender", sorted(df["Gender"].unique()))
            department = st.selectbox("Department", sorted(df["Department"].unique()))
            job_role = st.selectbox("Job Role", sorted(df["JobRole"].unique()))
            marital_status = st.selectbox("Marital Status", sorted(df["MaritalStatus"].unique()))
            education_field = st.selectbox("Education Field", sorted(df["EducationField"].unique()))
            education = st.slider("Education Level (1=Below College, 5=Doctor)", 1, 5, 3)
            business_travel = st.selectbox("Business Travel", sorted(df["BusinessTravel"].unique()))
            distance_from_home = st.slider("Distance From Home (km)", 1, 30, 8)
            overtime = st.selectbox("OverTime", ["No", "Yes"])
            stock_option = st.slider("Stock Option Level", 0, 3, 1)
            job_level = st.slider("Job Level", 1, 5, 2)

        with c2:
            monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=25000, value=5000, step=100)
            env_satisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
            job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
            relationship_satisfaction = st.slider("Relationship Satisfaction (1-4)", 1, 4, 3)
            work_life_balance = st.slider("Work Life Balance (1-4)", 1, 4, 3)
            job_involvement = st.slider("Job Involvement (1-4)", 1, 4, 3)
            performance_rating = st.slider("Performance Rating (1-4)", 1, 4, 3)
            years_at_company = st.slider("Years At Company", 0, 40, 5)
            years_in_role = st.slider("Years In Current Role", 0, 20, 3)
            years_since_promotion = st.slider("Years Since Last Promotion", 0, 15, 1)
            years_with_manager = st.slider("Years With Current Manager", 0, 20, 3)
            num_companies = st.slider("Num Companies Worked", 0, 10, 2)
            total_working_years = st.slider("Total Working Years", 0, 40, 8)
            training_times = st.slider("Training Times Last Year", 0, 6, 2)
            salary_hike = st.slider("Percent Salary Hike", 10, 25, 14)

        submitted = st.form_submit_button("🔮 Predict Attrition Risk", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Build employee record (used for both initial prediction and what-if)
# ---------------------------------------------------------------------------
def build_employee_dict(overrides: dict | None = None) -> dict:
    base = {
        "Age": age, "Gender": gender, "Department": department, "JobRole": job_role,
        "MaritalStatus": marital_status, "EducationField": education_field, "Education": education,
        "BusinessTravel": business_travel, "DistanceFromHome": distance_from_home,
        "OverTime": overtime, "StockOptionLevel": stock_option, "JobLevel": job_level,
        "MonthlyIncome": monthly_income, "EnvironmentSatisfaction": env_satisfaction,
        "JobSatisfaction": job_satisfaction, "RelationshipSatisfaction": relationship_satisfaction,
        "WorkLifeBalance": work_life_balance, "JobInvolvement": job_involvement,
        "PerformanceRating": performance_rating, "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_role, "YearsSinceLastPromotion": years_since_promotion,
        "YearsWithCurrManager": years_with_manager, "NumCompaniesWorked": num_companies,
        "TotalWorkingYears": total_working_years, "TrainingTimesLastYear": training_times,
        "PercentSalaryHike": salary_hike,
        # Reasonable defaults for fields not exposed on the form
        "HourlyRate": int(df["HourlyRate"].median()),
        "DailyRate": int(df["DailyRate"].median()),
        "MonthlyRate": int(df["MonthlyRate"].median()),
    }
    if overrides:
        base.update(overrides)
    return base


if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False

if submitted:
    st.session_state.prediction_made = True
    st.session_state.employee_dict = build_employee_dict()

# ---------------------------------------------------------------------------
# RIGHT: Prediction Result Card
# ---------------------------------------------------------------------------
with right_col:
    section_header("Prediction Result", icon="📋")

    if not st.session_state.prediction_made:
        st.info("Fill out the employee profile on the left and click **Predict Attrition Risk** to see results here.")
    else:
        employee_dict = st.session_state.employee_dict
        employee_row = single_employee_to_dataframe(employee_dict, df)
        result = predict_employee(employee_row)

        st.plotly_chart(risk_gauge(result["probability"]), use_container_width=True)

        rc1, rc2 = st.columns(2)
        with rc1:
            st.metric("Predicted Outcome", result["prediction"])
        with rc2:
            st.metric("Confidence Score", f"{result['confidence']:.1%}")

        render_risk_badge(result["risk_label"])
        st.write("")
        st.metric("Attrition Probability", f"{result['probability']:.1%}")

# ---------------------------------------------------------------------------
# AI EXPLANATION (SHAP)
# ---------------------------------------------------------------------------
if st.session_state.prediction_made:
    st.divider()
    section_header("Why Did the Model Predict This?", "AI-generated explanation (SHAP)", icon="🧠")

    employee_dict = st.session_state.employee_dict
    employee_row = single_employee_to_dataframe(employee_dict, df)
    result = predict_employee(employee_row)

    positive_factors, negative_factors = [], []
    if SHAP_AVAILABLE:
        shap_values, X_transformed, base_value = compute_shap_values(model, employee_row)
        if shap_values is not None:
            positive_factors, negative_factors = top_contributing_factors(
                shap_values[0], X_transformed.columns.tolist(), employee_row.iloc[0], top_n=5
            )

            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                st.markdown("**🔺 Top Factors Increasing Risk**")
                for f in positive_factors:
                    st.markdown(f"- `{f['feature']}` — impact +{f['value']:.3f}")
            with exp_col2:
                st.markdown("**🔻 Top Factors Decreasing Risk**")
                for f in negative_factors:
                    st.markdown(f"- `{f['feature']}` — impact {f['value']:.3f}")

            st.write("")
            st.markdown("**🗣️ Natural Language Explanation**")
            st.success(natural_language_explanation(positive_factors, employee_row.iloc[0]))
        else:
            st.warning("SHAP explainer is not available for the currently active model type.")
    else:
        st.warning("Install the `shap` package (`pip install shap`) to enable AI explainability for this prediction.")

    # -----------------------------------------------------------------------
    # HR RECOMMENDATION ENGINE
    # -----------------------------------------------------------------------
    st.divider()
    section_header("HR Recommendations", "Actions generated from this employee's risk profile", icon="💡")

    recommendations = generate_recommendations(employee_dict, result["risk_label"])
    for rec in recommendations:
        recommendation_card(rec["title"], rec["description"], rec["icon"], rec["severity"])

    # -----------------------------------------------------------------------
    # WHAT-IF ANALYSIS
    # -----------------------------------------------------------------------
    st.divider()
    section_header("What-If Analysis", "Adjust key levers and see the risk update instantly", icon="🧮")

    wi1, wi2, wi3 = st.columns(3)
    with wi1:
        new_salary = st.slider("Adjusted Monthly Income ($)", 1000, 25000, int(employee_dict["MonthlyIncome"]), step=100, key="whatif_salary")
        new_overtime = st.selectbox("Adjusted OverTime", ["No", "Yes"], index=["No", "Yes"].index(employee_dict["OverTime"]), key="whatif_overtime")
    with wi2:
        new_promotion = st.slider("Adjusted Years Since Promotion", 0, 15, int(employee_dict["YearsSinceLastPromotion"]), key="whatif_promo")
        new_job_sat = st.slider("Adjusted Job Satisfaction", 1, 4, int(employee_dict["JobSatisfaction"]), key="whatif_jobsat")
    with wi3:
        new_wlb = st.slider("Adjusted Work Life Balance", 1, 4, int(employee_dict["WorkLifeBalance"]), key="whatif_wlb")

    whatif_overrides = {
        "MonthlyIncome": new_salary, "OverTime": new_overtime,
        "YearsSinceLastPromotion": new_promotion, "JobSatisfaction": new_job_sat,
        "WorkLifeBalance": new_wlb,
    }
    whatif_employee = build_employee_dict(whatif_overrides)
    whatif_row = single_employee_to_dataframe(whatif_employee, df)
    whatif_result = predict_employee(whatif_row)

    st.plotly_chart(
        what_if_comparison_bar(result["probability"], whatif_result["probability"]),
        use_container_width=True,
    )

    delta = result["probability"] - whatif_result["probability"]
    if delta > 0:
        st.success(f"These adjustments would reduce attrition risk by **{delta:.1%}** "
                    f"({result['probability']:.1%} → {whatif_result['probability']:.1%}).")
    elif delta < 0:
        st.error(f"These adjustments would INCREASE attrition risk by **{abs(delta):.1%}** "
                  f"({result['probability']:.1%} → {whatif_result['probability']:.1%}).")
    else:
        st.info("These adjustments produce no meaningful change in predicted risk.")

    # -----------------------------------------------------------------------
    # PDF REPORT DOWNLOAD
    # -----------------------------------------------------------------------
    st.divider()
    section_header("Download Report", "Export this prediction as a PDF for HR records", icon="📄")

    risk_factor_strings = [
        f"{f['feature']} (impact: +{f['value']:.3f})" for f in positive_factors
    ] if positive_factors else []

    profile_for_report = {k: v for k, v in employee_dict.items()
                           if k in FORM_NUMERIC_FIELDS + FORM_CATEGORICAL_FIELDS}

    pdf_bytes = generate_prediction_report(
        employee_profile=profile_for_report,
        prediction_result=result,
        risk_factors=risk_factor_strings,
        recommendations=recommendations,
        employee_name=f"{job_role} — {department}",
    )

    st.download_button(
        "⬇️ Download PDF Report",
        data=pdf_bytes,
        file_name="employee_attrition_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
