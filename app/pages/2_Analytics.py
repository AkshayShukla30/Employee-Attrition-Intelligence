"""
pages/2_Analytics.py
=====================
Deep-dive Analytics workbench. In addition to the global sidebar filters,
this page exposes extra page-local filters and a wide variety of chart
types (bar, pie, treemap, heatmap, scatter, box, violin, histogram) driven
by user-selected dimensions.
"""

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from config import APP_NAME, CSS_PATH
from components.cards import empty_state, page_header, section_header
from components.charts import (
    box_plot, correlation_heatmap, histogram_chart, pie_distribution,
    scatter_plot, treemap_chart, violin_plot,
)
from components.sidebar import render_sidebar
from utils.preprocessing import load_clean_data
from utils.prediction import load_model_metadata


def inject_css():
    if CSS_PATH.exists():
        with open(CSS_PATH) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


st.set_page_config(page_title=f"Analytics | {APP_NAME}", page_icon="🔍", layout="wide")
inject_css()

df = load_clean_data()
metadata = load_model_metadata()
filtered_df = render_sidebar(df, metadata)

page_header("Analytics Workbench", "Multi-dimensional, fully interactive exploration", icon="🔍")

# ---------------------------------------------------------------------------
# Page-local filters (in addition to global sidebar filters)
# ---------------------------------------------------------------------------
section_header("Refine Analysis", "Additional page-level filters", icon="🎛️")

f1, f2, f3 = st.columns(3)
with f1:
    wlb_options = ["All"] + sorted(filtered_df["WorkLifeBalance"].unique().tolist())
    wlb_filter = st.selectbox("Work Life Balance", wlb_options)
with f2:
    marital_options = ["All"] + sorted(filtered_df["MaritalStatus"].unique().tolist())
    marital_filter = st.selectbox("Marital Status", marital_options)
with f3:
    edu_options = ["All"] + sorted(filtered_df["Education"].unique().tolist())
    edu_filter = st.selectbox("Education Level", edu_options)

view_df = filtered_df.copy()
if wlb_filter != "All":
    view_df = view_df[view_df["WorkLifeBalance"] == wlb_filter]
if marital_filter != "All":
    view_df = view_df[view_df["MaritalStatus"] == marital_filter]
if edu_filter != "All":
    view_df = view_df[view_df["Education"] == edu_filter]

if view_df.empty:
    empty_state("No employees match the current filter combination.")
    st.stop()

st.caption(f"Showing **{len(view_df):,}** of {len(df):,} total employees based on active filters.")

# ---------------------------------------------------------------------------
# Categorical breakdowns
# ---------------------------------------------------------------------------
section_header("Categorical Breakdown", icon="📐")
t1, t2, t3 = st.tabs(["Pie Charts", "Treemap", "Heatmap"])

with t1:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(pie_distribution(view_df, "MaritalStatus"), use_container_width=True)
    with c2:
        st.plotly_chart(pie_distribution(view_df, "WorkLifeBalance", "Work Life Balance"), use_container_width=True)

with t2:
    st.plotly_chart(
        treemap_chart(view_df, path=["Department", "JobRole"], title="Department → Job Role Headcount"),
        use_container_width=True,
    )

with t3:
    st.plotly_chart(correlation_heatmap(view_df), use_container_width=True)

st.write("")

# ---------------------------------------------------------------------------
# Distribution & relationship explorer (user-selectable axes)
# ---------------------------------------------------------------------------
section_header("Distribution & Relationship Explorer", "Choose your own variables", icon="🧪")

numeric_cols = view_df.select_dtypes(include="number").columns.tolist()
categorical_cols = view_df.select_dtypes(include="object").columns.tolist()

tab_scatter, tab_box, tab_violin, tab_hist = st.tabs(
    ["Scatter Plot", "Box Plot", "Violin Plot", "Histogram"]
)

with tab_scatter:
    sc1, sc2 = st.columns(2)
    with sc1:
        x_axis = st.selectbox("X Axis", numeric_cols, index=numeric_cols.index("Age") if "Age" in numeric_cols else 0, key="scatter_x")
    with sc2:
        y_axis = st.selectbox("Y Axis", numeric_cols, index=numeric_cols.index("MonthlyIncome") if "MonthlyIncome" in numeric_cols else 1, key="scatter_y")
    st.plotly_chart(scatter_plot(view_df, x_axis, y_axis), use_container_width=True)

with tab_box:
    bx1, bx2 = st.columns(2)
    with bx1:
        box_x = st.selectbox("Category", categorical_cols, index=categorical_cols.index("Department") if "Department" in categorical_cols else 0, key="box_x")
    with bx2:
        box_y = st.selectbox("Numeric Value", numeric_cols, index=numeric_cols.index("MonthlyIncome") if "MonthlyIncome" in numeric_cols else 0, key="box_y")
    st.plotly_chart(box_plot(view_df, box_x, box_y), use_container_width=True)

with tab_violin:
    vx1, vx2 = st.columns(2)
    with vx1:
        violin_x = st.selectbox("Category", categorical_cols, index=categorical_cols.index("JobRole") if "JobRole" in categorical_cols else 0, key="violin_x")
    with vx2:
        violin_y = st.selectbox("Numeric Value", numeric_cols, index=numeric_cols.index("Age") if "Age" in numeric_cols else 0, key="violin_y")
    st.plotly_chart(violin_plot(view_df, violin_x, violin_y), use_container_width=True)

with tab_hist:
    hist_col = st.selectbox("Variable", numeric_cols, index=numeric_cols.index("TotalWorkingYears") if "TotalWorkingYears" in numeric_cols else 0, key="hist_col")
    st.plotly_chart(histogram_chart(view_df, hist_col, color="Attrition"), use_container_width=True)
