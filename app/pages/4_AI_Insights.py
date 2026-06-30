"""
pages/4_AI_Insights.py
=======================
Global explainability page: SHAP summary plot, global feature importance,
a sample waterfall plot, a dependence plot, and a narrative summary of the
top organization-wide attrition risk drivers.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from config import APP_NAME, CSS_PATH, TARGET_COLUMN
from components.cards import page_header, section_header
from components.charts import feature_importance_bar
from components.sidebar import render_sidebar
from utils.preprocessing import load_clean_data
from utils.prediction import load_model, load_model_metadata
from utils.shap_utils import SHAP_AVAILABLE, compute_shap_values, transform_for_shap

if SHAP_AVAILABLE:
    import shap


def inject_css():
    if CSS_PATH.exists():
        with open(CSS_PATH) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


st.set_page_config(page_title=f"AI Insights | {APP_NAME}", page_icon="🧠", layout="wide")
inject_css()

df = load_clean_data()
metadata = load_model_metadata()
render_sidebar(df, metadata)
model = load_model()

page_header("AI Insights", "Global model explainability and organization-wide risk drivers", icon="🧠")

if not SHAP_AVAILABLE:
    st.warning(
        "The `shap` package is not installed in this environment. Run `pip install shap` "
        "to unlock SHAP-based global explainability on this page. Feature importance from "
        "the model itself is shown below as a fallback."
    )

X = df.drop(columns=[TARGET_COLUMN, "AttritionFlag"])
y = df["AttritionFlag"]

# Use a manageable sample for SHAP computation speed
sample_size = min(200, len(X))
X_sample = X.sample(sample_size, random_state=42)

# ---------------------------------------------------------------------------
# Feature importance (model-native fallback, always available)
# ---------------------------------------------------------------------------
section_header("Global Feature Importance", "Model-native importance scores", icon="📊")

classifier = model.named_steps["classifier"]
preprocessor = model.named_steps["preprocessor"]

from utils.preprocessing import get_feature_names
feature_names = get_feature_names(preprocessor)

if hasattr(classifier, "feature_importances_"):
    importances = pd.Series(classifier.feature_importances_, index=feature_names)
elif hasattr(classifier, "coef_"):
    importances = pd.Series(np.abs(classifier.coef_[0]), index=feature_names)
else:
    importances = pd.Series(dtype=float)

if not importances.empty:
    st.plotly_chart(feature_importance_bar(importances), use_container_width=True)

# ---------------------------------------------------------------------------
# SHAP Summary Plot
# ---------------------------------------------------------------------------
if SHAP_AVAILABLE:
    section_header("SHAP Summary Plot", "Feature impact direction and magnitude across employees", icon="🌐")

    with st.spinner("Computing SHAP values..."):
        shap_values, X_transformed, base_value = compute_shap_values(model, X_sample)

    if shap_values is not None:
        fig, ax = plt.subplots(figsize=(9, 6))
        shap.summary_plot(shap_values, X_transformed, show=False)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # ---------------------------------------------------------------
        # Dependence plot
        # ---------------------------------------------------------------
        section_header("Dependence Plot", "How a single feature drives predictions", icon="🔗")
        dep_feature = st.selectbox(
            "Select a feature to inspect",
            options=X_transformed.columns.tolist(),
            index=X_transformed.columns.tolist().index("OverTime_Yes") if "OverTime_Yes" in X_transformed.columns else 0,
        )
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        shap.dependence_plot(dep_feature, shap_values, X_transformed, show=False, ax=ax2)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

        # ---------------------------------------------------------------
        # Sample waterfall plot
        # ---------------------------------------------------------------
        section_header("Sample Waterfall Plot", "Explanation for one representative employee", icon="💧")
        sample_idx = st.slider("Select a sample employee index", 0, len(X_sample) - 1, 0)

        explanation = shap.Explanation(
            values=shap_values[sample_idx],
            base_values=base_value,
            data=X_transformed.iloc[sample_idx],
            feature_names=feature_names,
        )
        fig3, ax3 = plt.subplots(figsize=(9, 6))
        shap.plots.waterfall(explanation, show=False)
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

        # ---------------------------------------------------------------
        # Top risk drivers narrative
        # ---------------------------------------------------------------
        section_header("Top Organization-Wide Risk Drivers", icon="🚨")
        mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names).sort_values(ascending=False)
        top5 = mean_abs_shap.head(5)
        bullets = "\n".join([f"- **{feat}** — average impact magnitude: {val:.4f}" for feat, val in top5.items()])
        st.markdown(
            f"""
            Across the sampled workforce, the model's predictions are most sensitive to:

            {bullets}

            These features should anchor HR's top-level attrition-reduction strategy, as they
            consistently move the model's risk estimate up or down across the broadest set
            of employees.
            """
        )
    else:
        st.info("SHAP TreeExplainer is not compatible with the currently active model type "
                 "(e.g. this can happen for some linear models). Feature importance above "
                 "remains a reliable explainability signal.")
else:
    st.caption("SHAP-based summary, dependence, and waterfall plots will appear here once `shap` is installed.")
