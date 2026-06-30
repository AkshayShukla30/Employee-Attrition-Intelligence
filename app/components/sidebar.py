"""
components/sidebar.py
======================
Renders the global sidebar: branding, dataset/model info panel, theme toggle,
and the global filter set used by Dashboard/Analytics pages. Filters are
stored in st.session_state so every page reads the same filtered dataset.
"""

import streamlit as st

from config import APP_NAME, APP_TAGLINE, COLOR_PRIMARY


def _init_filter_state(df):
    """Initialize session_state filter defaults exactly once."""
    defaults = {
        "filter_department": "All",
        "filter_gender": "All",
        "filter_job_role": "All",
        "filter_education": "All",
        "filter_business_travel": "All",
        "filter_overtime": "All",
        "filter_work_life_balance": "All",
        "filter_age_range": (int(df["Age"].min()), int(df["Age"].max())),
        "filter_years_at_company": (int(df["YearsAtCompany"].min()), int(df["YearsAtCompany"].max())),
        "theme_mode": "Light",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar(df, model_metadata: dict | None = None):
    """Render the full sidebar and return the currently filtered DataFrame."""
    _init_filter_state(df)

    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-brand">
                <div class="sidebar-logo">🏢</div>
                <div>
                    <div class="sidebar-title">{APP_NAME.split()[0]} Analytics</div>
                    <div class="sidebar-subtitle">{APP_TAGLINE}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown("#### 🔎 Global Filters")
        st.selectbox(
            "Department", ["All"] + sorted(df["Department"].unique().tolist()),
            key="filter_department",
        )
        st.selectbox(
            "Job Role", ["All"] + sorted(df["JobRole"].unique().tolist()),
            key="filter_job_role",
        )
        st.selectbox(
            "Gender", ["All"] + sorted(df["Gender"].unique().tolist()),
            key="filter_gender",
        )
        st.selectbox(
            "Education Field", ["All"] + sorted(df["EducationField"].unique().tolist()),
            key="filter_education",
        )
        st.selectbox(
            "Business Travel", ["All"] + sorted(df["BusinessTravel"].unique().tolist()),
            key="filter_business_travel",
        )
        st.selectbox(
            "OverTime", ["All"] + sorted(df["OverTime"].unique().tolist()),
            key="filter_overtime",
        )
        st.slider(
            "Age Range",
            int(df["Age"].min()), int(df["Age"].max()),
            key="filter_age_range",
        )
        st.slider(
            "Years At Company",
            int(df["YearsAtCompany"].min()), int(df["YearsAtCompany"].max()),
            key="filter_years_at_company",
        )

        if st.button("↺ Reset Filters", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key.startswith("filter_"):
                    del st.session_state[key]
            st.rerun()

        st.divider()
        st.markdown("#### 🌓 Display")
        st.radio("Theme", ["Light", "Compact"], key="theme_mode", horizontal=True)

        st.divider()
        st.markdown("#### 🤖 Model Information")
        if model_metadata:
            st.markdown(
                f"""
                <div class="sidebar-info-box">
                    <b>Active Model:</b> {model_metadata.get('best_model', 'N/A')}<br>
                    <b>Trained:</b> {model_metadata.get('trained_at', 'N/A')}<br>
                    <b>Train / Test:</b> {model_metadata.get('n_train', '—')} / {model_metadata.get('n_test', '—')}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.caption("Model not yet trained.")

        st.markdown("#### 📦 Dataset Information")
        st.markdown(
            f"""
            <div class="sidebar-info-box">
                <b>Total Records:</b> {len(df):,}<br>
                <b>Features:</b> {df.shape[1]}<br>
                <b>Source:</b> IBM HR Analytics Dataset
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        st.caption(f"© {APP_NAME} — v1.0.0")

    return apply_filters(df)


def apply_filters(df):
    """Apply the current session_state filter selections to the dataframe."""
    filtered = df.copy()

    if st.session_state.filter_department != "All":
        filtered = filtered[filtered["Department"] == st.session_state.filter_department]
    if st.session_state.filter_job_role != "All":
        filtered = filtered[filtered["JobRole"] == st.session_state.filter_job_role]
    if st.session_state.filter_gender != "All":
        filtered = filtered[filtered["Gender"] == st.session_state.filter_gender]
    if st.session_state.filter_education != "All":
        filtered = filtered[filtered["EducationField"] == st.session_state.filter_education]
    if st.session_state.filter_business_travel != "All":
        filtered = filtered[filtered["BusinessTravel"] == st.session_state.filter_business_travel]
    if st.session_state.filter_overtime != "All":
        filtered = filtered[filtered["OverTime"] == st.session_state.filter_overtime]

    age_lo, age_hi = st.session_state.filter_age_range
    filtered = filtered[(filtered["Age"] >= age_lo) & (filtered["Age"] <= age_hi)]

    yrs_lo, yrs_hi = st.session_state.filter_years_at_company
    filtered = filtered[(filtered["YearsAtCompany"] >= yrs_lo) & (filtered["YearsAtCompany"] <= yrs_hi)]

    return filtered
