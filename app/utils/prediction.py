"""
utils/prediction.py
====================
Model training (one-time, cached to disk), loading, and prediction utilities.

On first run, if no saved model is found in app/models/, a full training run
is executed (multiple algorithms compared by ROC AUC, best one persisted).
On subsequent runs the saved artifacts are loaded instantly via joblib.
"""

import json
import time

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    roc_curve, precision_recall_curve, confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config import (
    MODEL_PATH, PREPROCESSOR_PATH, METRICS_PATH, METADATA_PATH,
    RANDOM_STATE, TARGET_COLUMN, RISK_THRESHOLDS, MODELS_DIR,
)
from utils.preprocessing import load_clean_data, get_feature_lists, build_preprocessor

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def _build_models(y_train):
    """Instantiate the candidate model dictionary."""
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced", n_estimators=300, random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "Extra Trees": ExtraTreesClassifier(
            class_weight="balanced", n_estimators=300, random_state=RANDOM_STATE
        ),
    }
    if XGBOOST_AVAILABLE:
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        models["XGBoost"] = XGBClassifier(
            random_state=RANDOM_STATE, eval_metric="logloss",
            scale_pos_weight=scale_pos_weight, n_estimators=300,
        )
    return models


def train_and_save_models(force: bool = False):
    """
    Train all candidate models, evaluate them, persist the best pipeline plus
    full metrics/metadata to disk. Skips training if artifacts already exist
    unless force=True.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists() and METRICS_PATH.exists() and not force:
        return  # already trained

    df = load_clean_data()
    X = df.drop(columns=[TARGET_COLUMN, "AttritionFlag"])
    y = df["AttritionFlag"]

    numeric_features, categorical_features = get_feature_lists(df)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    models = _build_models(y_train)
    all_metrics = []
    curves = {}
    trained_pipelines = {}

    start = time.time()
    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        all_metrics.append({
            "Model": name,
            "Accuracy": float(accuracy_score(y_test, y_pred)),
            "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "F1": float(f1_score(y_test, y_pred, zero_division=0)),
            "ROC AUC": float(roc_auc_score(y_test, y_proba)),
        })
        curves[name] = {
            "fpr": fpr.tolist(), "tpr": tpr.tolist(),
            "precision": prec.tolist(), "recall": rec.tolist(),
            "confusion_matrix": cm.tolist(),
        }

    metrics_df = pd.DataFrame(all_metrics).sort_values("ROC AUC", ascending=False).reset_index(drop=True)
    best_name = metrics_df.iloc[0]["Model"]
    best_pipeline = trained_pipelines[best_name]

    joblib.dump(best_pipeline, MODEL_PATH)
    joblib.dump(best_pipeline.named_steps["preprocessor"], PREPROCESSOR_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump({"metrics": all_metrics, "curves": curves}, f)

    metadata = {
        "best_model": best_name,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_seconds": round(time.time() - start, 2),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "xgboost_available": XGBOOST_AVAILABLE,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f)


@st.cache_resource(show_spinner="Loading prediction model...")
def load_model() -> Pipeline:
    """Load the persisted best model pipeline (preprocessor + classifier)."""
    train_and_save_models()
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_model_metrics() -> dict:
    """Load persisted evaluation metrics / curves for the Model Performance page."""
    train_and_save_models()
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_model_metadata() -> dict:
    """Load persisted training metadata (best model name, training time, etc.)."""
    train_and_save_models()
    with open(METADATA_PATH) as f:
        return json.load(f)


def classify_risk(probability: float) -> str:
    """Map a predicted probability to a risk band label."""
    for label, (low, high) in RISK_THRESHOLDS.items():
        if low <= probability < high:
            return label
    return "High Risk"


def predict_employee(employee_df: pd.DataFrame) -> dict:
    """
    Run the persisted model on a single-employee DataFrame and return a
    structured prediction result.
    """
    model = load_model()
    proba = float(model.predict_proba(employee_df)[0, 1])
    risk_label = classify_risk(proba)
    return {
        "probability": proba,
        "prediction": "Attrition" if proba >= 0.5 else "No Attrition",
        "risk_label": risk_label,
        "confidence": float(max(proba, 1 - proba)),
    }
