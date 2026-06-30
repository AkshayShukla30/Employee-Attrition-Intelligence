"""
pages/1_Dashboard.py
=====================
Executive Dashboard — top KPI cards, demographic distributions, and
attrition breakdowns across key dimensions. All charts respond to the
global sidebar filters.
"""

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from config import APP_NAME, CSS_PATH
from components.cards import empty_state, kpi_row, page_header, section_header
from components.charts import (
    attrition_by_age_histogram, attrition_by_category, attrition_by_salary,
    attrition_rate_by_category, bar_distribution, correlation_heatmap,
    pie_distribution,
)
from components.sidebar import render_sidebar
from utils.preprocessing import load_clean_data
from utils.prediction import load_model_metadata


def inject_css():
    if CSS_PATH.exists():
        with open(CSS_PATH) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


st.set_page_config(page_title=f"Dashboard | {APP_NAME}", page_icon="📊", layout="wide")
inject_css()

df = load_clean_data()
metadata = load_model_metadata()
filtered_df = render_sidebar(df, metadata)

page_header("Executive Dashboard", "Company-wide attrition KPIs and breakdowns", icon="📊")

if filtered_df.empty:
    empty_state("No employees match the current filter selection. Try resetting filters.")
    st.stop()

# ---------------------------------------------------------------------------
# KPI Row
# ---------------------------------------------------------------------------
total = len(filtered_df)
left = int((filtered_df["Attrition"] == "Yes").sum())
rate = (left / total * 100) if total else 0

kpi_row([
    {"label": "Total Employees", "value": f"{total:,}", "icon": "👥"},
    {"label": "Employees Left", "value": f"{left:,}", "icon": "🚪"},
    {"label": "Attrition Rate", "value": f"{rate:.1f}%", "icon": "📉"},
    {"label": "Avg. Monthly Income", "value": f"${filtered_df['MonthlyIncome'].mean():,.0f}", "icon": "💵"},
    {"label": "Avg. Years at Company", "value": f"{filtered_df['YearsAtCompany'].mean():.1f}", "icon": "📅"},
    {"label": "Avg. Age", "value": f"{filtered_df['Age'].mean():.1f}", "icon": "🎂"},
])

st.write("")

# ---------------------------------------------------------------------------
# Demographic distributions
# ---------------------------------------------------------------------------
section_header("Workforce Composition", "Demographic and organizational distribution", icon="🧩")

c1, c2, c3 = st.columns(3)
with c1:
    st.plotly_chart(pie_distribution(filtered_df, "Department"), use_container_width=True)
with c2:
    st.plotly_chart(pie_distribution(filtered_df, "Gender"), use_container_width=True)
with c3:
    st.plotly_chart(pie_distribution(filtered_df, "EducationField", "Education Field"), use_container_width=True)

c4, c5 = st.columns(2)
with c4:
    st.plotly_chart(bar_distribution(filtered_df, "BusinessTravel"), use_container_width=True)
with c5:
    st.plotly_chart(bar_distribution(filtered_df, "JobRole"), use_container_width=True)

st.write("")

# ---------------------------------------------------------------------------
# Attrition breakdowns
# ---------------------------------------------------------------------------
section_header("Attrition Breakdown", "Where attrition concentrates across the organization", icon="🔥")

c6, c7 = st.columns(2)
with c6:
    st.plotly_chart(attrition_by_category(filtered_df, "Department"), use_container_width=True)
with c7:
    st.plotly_chart(attrition_rate_by_category(filtered_df, "JobRole"), use_container_width=True)

c8, c9 = st.columns(2)
with c8:
    st.plotly_chart(attrition_by_category(filtered_df, "Gender"), use_container_width=True)
with c9:
    st.plotly_chart(attrition_by_age_histogram(filtered_df), use_container_width=True)

c10, c11 = st.columns(2)
with c10:
    st.plotly_chart(attrition_by_salary(filtered_df), use_container_width=True)
with c11:
    st.plotly_chart(attrition_by_category(filtered_df, "OverTime"), use_container_width=True)

st.plotly_chart(
    attrition_rate_by_category(filtered_df, "TenureGroup", title="Attrition Rate by Tenure Group"),
    use_container_width=True,
)

section_header("Correlation Analysis", "Relationships between numeric workforce metrics", icon="🧮")
st.plotly_chart(correlation_heatmap(filtered_df), use_container_width=True)

# ---------------------------------------------------------------------------
# Employee Risk Intelligence: scored employees, search, leaderboard
# ---------------------------------------------------------------------------
import pandas as pd
from utils.prediction import load_model

section_header("Employee Risk Intelligence", "Model-scored employee risk across the workforce", icon="🛰️")

model = load_model()
score_df = filtered_df.copy().reset_index(drop=True)
X_score = score_df.drop(columns=["Attrition", "AttritionFlag"])
score_df["RiskProbability"] = model.predict_proba(X_score)[:, 1]
score_df["EmployeeID"] = score_df.index + 1

tab_top_risk, tab_top_safe, tab_leaderboard, tab_search, tab_compare = st.tabs(
    ["🔴 Top Risk Employees", "🟢 Top Safe Employees", "🏢 Department Leaderboard", "🔎 Search", "⚖️ Compare"]
)

with tab_top_risk:
    top_risk = score_df.sort_values("RiskProbability", ascending=False).head(10)
    st.dataframe(
        top_risk[["EmployeeID", "Department", "JobRole", "Age", "MonthlyIncome", "OverTime", "RiskProbability"]]
        .style.format({"RiskProbability": "{:.1%}"}),
        use_container_width=True, hide_index=True,
    )

with tab_top_safe:
    top_safe = score_df.sort_values("RiskProbability", ascending=True).head(10)
    st.dataframe(
        top_safe[["EmployeeID", "Department", "JobRole", "Age", "MonthlyIncome", "OverTime", "RiskProbability"]]
        .style.format({"RiskProbability": "{:.1%}"}),
        use_container_width=True, hide_index=True,
    )

with tab_leaderboard:
    leaderboard = score_df.groupby("Department")["RiskProbability"].mean().sort_values(ascending=False).reset_index()
    leaderboard.columns = ["Department", "Avg. Risk Score"]
    st.dataframe(leaderboard.style.format({"Avg. Risk Score": "{:.1%}"}), use_container_width=True, hide_index=True)

with tab_search:
    search_role = st.selectbox("Filter by Job Role", ["All"] + sorted(score_df["JobRole"].unique().tolist()), key="search_role")
    search_dept = st.selectbox("Filter by Department", ["All"] + sorted(score_df["Department"].unique().tolist()), key="search_dept")
    search_results = score_df.copy()
    if search_role != "All":
        search_results = search_results[search_results["JobRole"] == search_role]
    if search_dept != "All":
        search_results = search_results[search_results["Department"] == search_dept]
    st.dataframe(
        search_results[["EmployeeID", "Department", "JobRole", "Gender", "Age", "MonthlyIncome", "RiskProbability"]]
        .style.format({"RiskProbability": "{:.1%}"}),
        use_container_width=True, hide_index=True,
    )

with tab_compare:
    cmp1, cmp2 = st.columns(2)
    with cmp1:
        emp_a = st.selectbox("Employee A (ID)", score_df["EmployeeID"].tolist(), key="cmp_a")
    with cmp2:
        emp_b = st.selectbox("Employee B (ID)", score_df["EmployeeID"].tolist(), index=min(1, len(score_df) - 1), key="cmp_b")

    row_a = score_df[score_df["EmployeeID"] == emp_a].iloc[0]
    row_b = score_df[score_df["EmployeeID"] == emp_b].iloc[0]
    compare_cols = ["Department", "JobRole", "Age", "MonthlyIncome", "JobSatisfaction",
                    "WorkLifeBalance", "OverTime", "YearsAtCompany", "RiskProbability"]
    compare_table = pd.DataFrame({"Employee A": row_a[compare_cols], "Employee B": row_b[compare_cols]})


    st.dataframe(compare_table, use_container_width=True)

    
st.divider()
st.markdown("#### 📥 Export Data")

# Convert the scored dataframe to CSV
csv_data = score_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download Full Prediction Dataset (CSV)",
    data=csv_data,
    file_name="employee_risk_predictions.csv",
    mime="text/csv",
    use_container_width=True
)