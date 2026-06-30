"""
config.py
=========
Central configuration for the Employee Attrition Intelligence Platform (EAIP).

Holds theme constants, file paths, and shared metadata so that no other module
hard-codes colors, paths, or column lists. Import from here everywhere else.
"""

from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
ASSETS_DIR = APP_DIR / "assets"
MODELS_DIR = APP_DIR / "models"

DATA_PATH = ASSETS_DIR / "employee_attrition.csv"
MODEL_PATH = MODELS_DIR / "best_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
METRICS_PATH = MODELS_DIR / "model_metrics.json"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

LOGO_PATH = ASSETS_DIR / "logo.png"
BANNER_PATH = ASSETS_DIR / "banner.png"
CSS_PATH = APP_DIR / "style.css"

# ----------------------------------------------------------------------------
# Brand / Theme  (Microsoft Power BI inspired)
# ----------------------------------------------------------------------------
APP_NAME = "Employee Attrition Intelligence Platform"
APP_SHORT_NAME = "EAIP"
APP_TAGLINE = "Predict • Analyze • Explain • Recommend"

COLOR_PRIMARY = "#2563EB"
COLOR_PRIMARY_DARK = "#1D4ED8"
COLOR_PRIMARY_LIGHT = "#DBEAFE"
COLOR_SUCCESS = "#22C55E"
COLOR_WARNING = "#F59E0B"
COLOR_DANGER = "#EF4444"
COLOR_BACKGROUND = "#F5F7FA"
COLOR_CARD = "#FFFFFF"
COLOR_TEXT = "#1E293B"
COLOR_TEXT_MUTED = "#64748B"
COLOR_BORDER = "#E2E8F0"

RISK_COLORS = {
    "Low Risk": COLOR_SUCCESS,
    "Medium Risk": COLOR_WARNING,
    "High Risk": COLOR_DANGER,
}

CHART_COLORWAY = [
    COLOR_PRIMARY, COLOR_DANGER, COLOR_SUCCESS, COLOR_WARNING,
    "#6E8898", "#9DB4C0", "#7C3AED", "#0EA5E9",
]

PLOTLY_TEMPLATE_CONFIG = dict(
    font_family="Segoe UI, -apple-system, Helvetica, Arial, sans-serif",
    paper_bgcolor="white",
    plot_bgcolor="white",
)

# ----------------------------------------------------------------------------
# Modeling constants
# ----------------------------------------------------------------------------
RANDOM_STATE = 42
TARGET_COLUMN = "Attrition"

DROP_COLUMNS = ["EmployeeNumber", "EmployeeCount", "StandardHours", "Over18"]

# Columns presented on the Prediction page input form, with display order
FORM_NUMERIC_FIELDS = [
    "Age", "MonthlyIncome", "DistanceFromHome", "Education",
    "EnvironmentSatisfaction", "JobSatisfaction", "WorkLifeBalance",
    "YearsAtCompany", "YearsSinceLastPromotion", "YearsWithCurrManager",
    "StockOptionLevel", "RelationshipSatisfaction", "JobInvolvement",
    "PerformanceRating", "NumCompaniesWorked", "TotalWorkingYears",
    "TrainingTimesLastYear", "PercentSalaryHike", "JobLevel",
    "YearsInCurrentRole", "HourlyRate", "DailyRate", "MonthlyRate",
]

FORM_CATEGORICAL_FIELDS = [
    "Gender", "Department", "JobRole", "BusinessTravel",
    "MaritalStatus", "EducationField", "OverTime",
]

# Risk thresholds applied to the model's predicted probability of attrition
RISK_THRESHOLDS = {
    "Low Risk": (0.0, 0.35),
    "Medium Risk": (0.35, 0.65),
    "High Risk": (0.65, 1.01),
}
