"""
pages/5_Model_Performance.py
=============================
Model comparison and evaluation page: metrics table, confusion matrices,
ROC and Precision-Recall curves, and feature importance for the deployed
best model, with a short explanation of why it was selected.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from config import APP_NAME, CSS_PATH, TARGET_COLUMN
from components.cards import page_header, section_header
from components.charts import confusion_matrix_chart, feature_importance_bar, pr_curve_chart, roc_curve_chart
from components.sidebar import render_sidebar
from utils.preprocessing import get_feature_names, load_clean_data
from utils.prediction import load_model, load_model_metadata, load_model_metrics


def inject_css():
    if CSS_PATH.exists():
        with open(CSS_PATH) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


st.set_page_config(page_title=f"Model Performance | {APP_NAME}", page_icon="🏆", layout="wide")
inject_css()

df = load_clean_data()
metadata = load_model_metadata()
render_sidebar(df, metadata)
model = load_model()
metrics_data = load_model_metrics()

page_header("Model Performance", "Candidate model comparison and evaluation", icon="🏆")

metrics_df = pd.DataFrame(metrics_data["metrics"]).sort_values("ROC AUC", ascending=False).reset_index(drop=True)
curves = metrics_data["curves"]
best_model_name = metadata.get("best_model")

# ---------------------------------------------------------------------------
# Comparison Table
# ---------------------------------------------------------------------------
section_header("Model Comparison Table", "Ranked by ROC AUC on the held-out test set", icon="📋")

styled = metrics_df.style.format({
    "Accuracy": "{:.1%}", "Precision": "{:.1%}", "Recall": "{:.1%}",
    "F1": "{:.1%}", "ROC AUC": "{:.3f}",
}).highlight_max(subset=["ROC AUC"], color="#DCFCE7")
st.dataframe(styled, use_container_width=True, hide_index=True)

st.success(
    f"🏆 **Best Model: {best_model_name}** — selected automatically as the highest ROC AUC "
    f"performer and deployed for live predictions across the platform."
)

st.write("")

# ---------------------------------------------------------------------------
# ROC & PR Curves
# ---------------------------------------------------------------------------
section_header("ROC & Precision-Recall Curves", icon="📈")
cc1, cc2 = st.columns(2)
with cc1:
    st.plotly_chart(roc_curve_chart(curves), use_container_width=True)
with cc2:
    st.plotly_chart(pr_curve_chart(curves), use_container_width=True)

# ---------------------------------------------------------------------------
# Confusion Matrices
# ---------------------------------------------------------------------------
section_header("Confusion Matrices", "Test-set prediction breakdown per model", icon="🧮")
model_names = list(curves.keys())
n_cols = min(3, len(model_names))
cols = st.columns(n_cols)
for i, name in enumerate(model_names):
    with cols[i % n_cols]:
        st.plotly_chart(
            confusion_matrix_chart(curves[name]["confusion_matrix"], title=name),
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# Feature Importance for the best model
# ---------------------------------------------------------------------------
section_header("Feature Importance — Best Model", icon="🌟")
classifier = model.named_steps["classifier"]
preprocessor = model.named_steps["preprocessor"]
feature_names = get_feature_names(preprocessor)

import numpy as np
if hasattr(classifier, "feature_importances_"):
    importances = pd.Series(classifier.feature_importances_, index=feature_names)
    st.plotly_chart(feature_importance_bar(importances, f"Top 10 Feature Importance — {best_model_name}"), use_container_width=True)
elif hasattr(classifier, "coef_"):
    importances = pd.Series(np.abs(classifier.coef_[0]), index=feature_names)
    st.plotly_chart(feature_importance_bar(importances, f"Top 10 |Coefficient| — {best_model_name}"), use_container_width=True)
else:
    st.info("Feature importance is not directly available for this model type.")
