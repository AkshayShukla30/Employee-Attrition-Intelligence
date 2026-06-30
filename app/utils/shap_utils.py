"""
utils/shap_utils.py
====================
SHAP-based explainability utilities. All functions degrade gracefully (return
None / informative flags) if the `shap` package is not installed, so the rest
of the app never crashes — pages simply show an "install shap" notice instead
of the explanation widgets.
"""

import numpy as np
import pandas as pd
import streamlit as st

from utils.preprocessing import get_feature_names

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


@st.cache_resource(show_spinner="Building explainability engine...")
def get_explainer(_model_pipeline):
    """
    Build a SHAP TreeExplainer for the classifier step of a fitted pipeline.
    Leading underscore on the parameter tells Streamlit's cache not to hash
    the (unhashable) pipeline object.
    """
    if not SHAP_AVAILABLE:
        return None
    classifier = _model_pipeline.named_steps["classifier"]
    try:
        return shap.TreeExplainer(classifier)
    except Exception:
        # Fallback for non-tree models (e.g. Logistic Regression won best model)
        return None


def transform_for_shap(model_pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Run raw feature rows through the fitted preprocessor for SHAP consumption."""
    preprocessor = model_pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = get_feature_names(preprocessor)
    return pd.DataFrame(transformed, columns=feature_names)


def compute_shap_values(model_pipeline, X: pd.DataFrame):
    """
    Compute SHAP values for the rows in X. Returns (shap_values, X_transformed,
    base_value) or (None, None, None) if SHAP / tree explainer unavailable.
    """
    if not SHAP_AVAILABLE:
        return None, None, None

    explainer = get_explainer(model_pipeline)
    if explainer is None:
        return None, None, None

    X_transformed = transform_for_shap(model_pipeline, X)
    raw_values = explainer.shap_values(X_transformed)

    # Handle both binary-list and single-array SHAP outputs across model types
    if isinstance(raw_values, list):
        shap_values = raw_values[1]
        base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    else:
        shap_values = raw_values
        base_value = explainer.expected_value if not isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value[-1]

    return shap_values, X_transformed, base_value


def top_contributing_factors(shap_values_row, feature_names, raw_row: pd.Series, top_n: int = 5):
    """
    Return (positive_factors, negative_factors) lists of dicts:
    {"feature": str, "value": shap_value, "raw_value": original input value}
    sorted by absolute impact, split by sign of contribution.
    """
    contributions = pd.Series(shap_values_row, index=feature_names)
    contributions = contributions.sort_values(key=np.abs, ascending=False)

    positive = contributions[contributions > 0].head(top_n)
    negative = contributions[contributions < 0].head(top_n)

    def _format(series):
        out = []
        for feat, val in series.items():
            base_feat = feat.split("_")[0] if "_" in feat and feat not in raw_row.index else feat
            raw_val = raw_row.get(base_feat, raw_row.get(feat, "—"))
            out.append({"feature": feat, "value": float(val), "raw_value": raw_val})
        return out

    return _format(positive), _format(negative)


def natural_language_explanation(positive_factors, raw_row: pd.Series) -> str:
    """
    Build a plain-English explanation string from the top positive (risk-increasing)
    SHAP factors, in the style requested: 'Employee has high attrition risk because...'
    """
    if not positive_factors:
        return "No dominant risk-increasing factors were identified for this employee."

    readable_map = {
        "OverTime_Yes": "works overtime",
        "JobSatisfaction": "has low job satisfaction",
        "EnvironmentSatisfaction": "has low environment satisfaction",
        "MonthlyIncome": "has relatively low monthly income",
        "YearsSinceLastPromotion": "has gone a long time without a promotion",
        "BusinessTravel_Travel_Frequently": "travels frequently for business",
        "DistanceFromHome": "lives far from the office",
        "WorkLifeBalance": "has poor work-life balance",
        "Age": "is in a higher-attrition-risk age band",
        "StockOptionLevel": "holds a low stock option level",
    }

    bullets = []
    for factor in positive_factors[:5]:
        feat = factor["feature"]
        bullets.append(readable_map.get(feat, f"has an elevated value of '{feat}'"))

    bullet_text = "\n".join([f"- Employee {b}." for b in bullets])
    return f"This employee shows elevated attrition risk primarily because:\n\n{bullet_text}"
