"""
utils/preprocessing.py
=======================
Data loading, cleaning, feature engineering, and preprocessing-pipeline
construction for the EAIP application. Mirrors the logic used in the
01_Employee_Attrition_Analysis.ipynb notebook so the deployed app and the
research notebook stay consistent.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import DATA_PATH, DROP_COLUMNS, TARGET_COLUMN


@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    """Load the raw employee attrition CSV from disk."""
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-derived features used throughout the app and the model."""
    df = df.copy()
    df["IncomePerYear"] = df["MonthlyIncome"] * 12
    df["YearsPerPromotion"] = df["YearsAtCompany"] / (df["YearsSinceLastPromotion"] + 1)
    df["ExperienceRatio"] = df["YearsAtCompany"] / (df["TotalWorkingYears"] + 1)
    df["AgeGroup"] = pd.cut(
        df["Age"], bins=[17, 25, 35, 45, 60], labels=["18-25", "26-35", "36-45", "46-60"]
    ).astype(str)
    df["TenureGroup"] = pd.cut(
        df["YearsAtCompany"], bins=[-1, 2, 5, 10, 40],
        labels=["0-2 yrs", "3-5 yrs", "6-10 yrs", "10+ yrs"]
    ).astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_clean_data() -> pd.DataFrame:
    """
    Full cleaning pipeline used across the app:
    drop unneeded columns, engineer features, keep a human-readable
    'Attrition' (Yes/No) column AND a numeric 'AttritionFlag' (1/0) column.
    """
    df = load_raw_data()
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    df = engineer_features(df)
    df["AttritionFlag"] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})
    return df


def get_feature_lists(df: pd.DataFrame):
    """Return (numeric_features, categorical_features) excluding target columns."""
    feature_df = df.drop(columns=[TARGET_COLUMN, "AttritionFlag"], errors="ignore")
    numeric_features = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = feature_df.select_dtypes(include=["object"]).columns.tolist()
    return numeric_features, categorical_features


def build_preprocessor(numeric_features, categorical_features) -> ColumnTransformer:
    """Build the standard ColumnTransformer: StandardScaler + OneHotEncoder."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_features),
        ],
        remainder="drop",
    )


def get_feature_names(preprocessor: ColumnTransformer):
    """Recover human-readable feature names after a fitted ColumnTransformer."""
    num_names = preprocessor.transformers_[0][2]
    cat_encoder = preprocessor.transformers_[1][1]
    cat_names = cat_encoder.get_feature_names_out(preprocessor.transformers_[1][2])
    return list(num_names) + list(cat_names)


def single_employee_to_dataframe(form_values: dict, reference_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a dict of form inputs (Prediction page) into a single-row DataFrame
    with the exact schema the model pipeline expects (including engineered
    features), using reference_df only to source the correct column order.
    """
    base = {col: form_values.get(col, reference_df[col].mode()[0])
            for col in reference_df.columns if col not in (TARGET_COLUMN, "AttritionFlag")}
    row = pd.DataFrame([base])
    row = engineer_features(row)
    return row
