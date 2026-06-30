"""
components/charts.py
=====================
All Plotly chart builders live here so every page reuses the same styling
function instead of duplicating layout/theme code (single source of truth
for the visual language of the app).
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import CHART_COLORWAY, COLOR_DANGER, COLOR_PRIMARY, COLOR_SUCCESS, PLOTLY_TEMPLATE_CONFIG

ATTRITION_COLOR_MAP = {"Yes": COLOR_DANGER, "No": COLOR_PRIMARY}


def _style(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    """Apply consistent EAIP styling to any Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#1E293B", family=PLOTLY_TEMPLATE_CONFIG["font_family"]), x=0.02, y=0.96),
        font=dict(family=PLOTLY_TEMPLATE_CONFIG["font_family"], color="#334155", size=12),
        paper_bgcolor=PLOTLY_TEMPLATE_CONFIG["paper_bgcolor"],
        plot_bgcolor=PLOTLY_TEMPLATE_CONFIG["plot_bgcolor"],
        height=height,
        # Top aur Bottom margins ko bada diya gaya hai taaki text ko saans lene ki jagah mile
        margin=dict(l=20, r=20, t=70, b=80), 
        # Legend ko chart ke neeche (bottom-center) shift kar diya gaya hai
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5, font=dict(size=11)),
        colorway=CHART_COLORWAY,
        hoverlabel=dict(bgcolor="white", font_size=12, bordercolor="#E2E8F0"),
    )
    fig.update_xaxes(showgrid=False, linecolor="#E2E8F0")
    fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9", linecolor="#E2E8F0")
    return fig

def bar_distribution(df: pd.DataFrame, column: str, title: str = "") -> go.Figure:
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, "Count"]
    fig = px.bar(counts, x=column, y="Count", text="Count", color=column)
    fig.update_traces(textposition="outside")
    
    # Ye nayi line legend ko hide kar degi jisse overlap khatam ho jayega
    fig.update_layout(showlegend=False)
    
    return _style(fig, title or f"{column} Distribution")

def pie_distribution(df: pd.DataFrame, column: str, title: str = "") -> go.Figure:
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, "Count"]
    fig = px.pie(counts, names=column, values="Count", hole=0.45)
    fig.update_traces(textinfo="percent+label")
    return _style(fig, title or f"{column} Distribution", height=380)


def attrition_by_category(df: pd.DataFrame, column: str, title: str = "") -> go.Figure:
    """Grouped bar of attrition Yes/No counts across a categorical column."""
    grouped = df.groupby([column, "Attrition"]).size().reset_index(name="Count")
    fig = px.bar(
        grouped, x=column, y="Count", color="Attrition", barmode="group",
        color_discrete_map=ATTRITION_COLOR_MAP, text="Count",
    )
    fig.update_traces(textposition="outside")
    return _style(fig, title or f"Attrition by {column}")


def attrition_rate_by_category(df: pd.DataFrame, column: str, title: str = "") -> go.Figure:
    """Horizontal bar of attrition RATE (%) by category, sorted descending."""
    rate = (
        df.groupby(column)["AttritionFlag"].mean().mul(100).round(1)
        .sort_values(ascending=True).reset_index()
    )
    rate.columns = [column, "Attrition Rate (%)"]
    fig = px.bar(rate, x="Attrition Rate (%)", y=column, orientation="h", text="Attrition Rate (%)",
                 color="Attrition Rate (%)", color_continuous_scale=["#DBEAFE", COLOR_DANGER])
    fig.update_traces(textposition="outside")
    fig.update_layout(coloraxis_showscale=False)
    return _style(fig, title or f"Attrition Rate by {column}", height=420)


def attrition_by_age_histogram(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df, x="Age", color="Attrition", barmode="overlay", nbins=20,
        color_discrete_map=ATTRITION_COLOR_MAP, opacity=0.75,
    )
    return _style(fig, "Attrition by Age Distribution")


def attrition_by_salary(df: pd.DataFrame) -> go.Figure:
    fig = px.box(df, x="Attrition", y="MonthlyIncome", color="Attrition",
                 color_discrete_map=ATTRITION_COLOR_MAP, points="outliers")
    return _style(fig, "Monthly Income by Attrition")


def correlation_heatmap(df: pd.DataFrame, title: str = "Correlation Heatmap") -> go.Figure:
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="RdBu", zmid=0, colorbar=dict(title="r"),
    ))
    return _style(fig, title, height=650)


def treemap_chart(df: pd.DataFrame, path: list, value_col: str = None, title: str = "") -> go.Figure:
    if value_col is None:
        plot_df = df.groupby(path).size().reset_index(name="Count")
        value_col = "Count"
    else:
        plot_df = df
    fig = px.treemap(plot_df, path=path, values=value_col, color=value_col,
                      color_continuous_scale=["#DBEAFE", COLOR_PRIMARY])
    return _style(fig, title, height=450)


def scatter_plot(df: pd.DataFrame, x: str, y: str, color: str = "Attrition", title: str = "") -> go.Figure:
    fig = px.scatter(df, x=x, y=y, color=color, opacity=0.7,
                      color_discrete_map=ATTRITION_COLOR_MAP if color == "Attrition" else None)
    return _style(fig, title or f"{y} vs {x}")


def box_plot(df: pd.DataFrame, x: str, y: str, color: str = "Attrition", title: str = "") -> go.Figure:
    fig = px.box(df, x=x, y=y, color=color, color_discrete_map=ATTRITION_COLOR_MAP if color == "Attrition" else None)
    return _style(fig, title or f"{y} by {x}")


def violin_plot(df: pd.DataFrame, x: str, y: str, color: str = "Attrition", title: str = "") -> go.Figure:
    fig = px.violin(df, x=x, y=y, color=color, box=True, points=False,
                     color_discrete_map=ATTRITION_COLOR_MAP if color == "Attrition" else None)
    return _style(fig, title or f"{y} Distribution by {x}")


def histogram_chart(df: pd.DataFrame, column: str, color: str = None, title: str = "") -> go.Figure:
    fig = px.histogram(df, x=column, color=color, barmode="overlay" if color else None,
                        color_discrete_map=ATTRITION_COLOR_MAP if color == "Attrition" else None)
    return _style(fig, title or f"{column} Histogram")


def risk_gauge(probability: float) -> go.Figure:
    """Speedometer-style gauge for the attrition risk probability."""
    pct = probability * 100
    if pct < 35:
        bar_color = COLOR_SUCCESS
    elif pct < 65:
        bar_color = "#F59E0B"
    else:
        bar_color = COLOR_DANGER

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 38, "color": "#1E293B"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94A3B8"},
            "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 35], "color": "#DCFCE7"},
                {"range": [35, 65], "color": "#FEF3C7"},
                {"range": [65, 100], "color": "#FEE2E2"},
            ],
        },
    ))
    fig.update_layout(
        height=260, margin=dict(l=20, r=20, t=20, b=0),
        paper_bgcolor="white", font=dict(family=PLOTLY_TEMPLATE_CONFIG["font_family"]),
    )
    return fig


def feature_importance_bar(importances: pd.Series, title: str = "Top 10 Feature Importance", top_n: int = 10) -> go.Figure:
    top = importances.sort_values(ascending=True).tail(top_n)
    fig = px.bar(x=top.values, y=top.index, orientation="h", color=top.values,
                 color_continuous_scale=["#DBEAFE", COLOR_PRIMARY])
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(title="Importance")
    fig.update_yaxes(title="")
    return _style(fig, title, height=420)


def roc_curve_chart(curves: dict) -> go.Figure:
    fig = go.Figure()
    for name, data in curves.items():
        fig.add_trace(go.Scatter(x=data["fpr"], y=data["tpr"], mode="lines", name=name))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#CBD5E1"), name="Random Baseline"))
    fig.update_xaxes(title="False Positive Rate")
    fig.update_yaxes(title="True Positive Rate")
    return _style(fig, "ROC Curve Comparison", height=450)


def pr_curve_chart(curves: dict) -> go.Figure:
    fig = go.Figure()
    for name, data in curves.items():
        fig.add_trace(go.Scatter(x=data["recall"], y=data["precision"], mode="lines", name=name))
    fig.update_xaxes(title="Recall")
    fig.update_yaxes(title="Precision")
    return _style(fig, "Precision-Recall Curve Comparison", height=450)


def confusion_matrix_chart(cm: list, title: str = "Confusion Matrix") -> go.Figure:
    cm = np.array(cm)
    labels = ["No Attrition", "Attrition"]
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels, colorscale="Blues",
        text=cm, texttemplate="%{text}", showscale=False,
    ))
    fig.update_yaxes(title="Actual", autorange="reversed")
    fig.update_xaxes(title="Predicted")
    return _style(fig, title, height=380)


def what_if_comparison_bar(before: float, after: float) -> go.Figure:
    fig = go.Figure(data=[
        go.Bar(name="Risk", x=["Current Risk", "Adjusted Risk"], y=[before * 100, after * 100],
               marker_color=[COLOR_DANGER, COLOR_SUCCESS], text=[f"{before:.0%}", f"{after:.0%}"], textposition="outside")
    ])
    fig.update_yaxes(title="Attrition Probability (%)", range=[0, 100])
    return _style(fig, "Before vs. After — What-If Analysis", height=350)
