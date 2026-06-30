"""
utils/report_generator.py
==========================
Generates a downloadable PDF report for a single employee's attrition
prediction, including the input profile, prediction result, top SHAP risk
factors, and HR recommendations. Built with reportlab (no external binary
dependencies, safe for cloud deployment).
"""

import io

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from config import COLOR_PRIMARY, COLOR_DANGER, COLOR_SUCCESS, COLOR_WARNING, RISK_COLORS


def _risk_color(risk_label: str):
    hex_color = RISK_COLORS.get(risk_label, COLOR_WARNING)
    return colors.HexColor(hex_color)


def generate_prediction_report(
    employee_profile: dict,
    prediction_result: dict,
    risk_factors: list,
    recommendations: list,
    employee_name: str = "Employee",
) -> bytes:
    """
    Build a single-employee PDF report and return it as raw bytes, ready for
    st.download_button.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Title"], textColor=colors.HexColor(COLOR_PRIMARY),
        fontSize=20, spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "SubtitleStyle", parent=styles["Normal"], textColor=colors.HexColor("#64748B"),
        fontSize=10, spaceAfter=16,
    )
    section_style = ParagraphStyle(
        "SectionStyle", parent=styles["Heading2"], textColor=colors.HexColor("#1E293B"),
        fontSize=13, spaceBefore=14, spaceAfter=8,
    )
    body_style = styles["BodyText"]

    elements = []
    elements.append(Paragraph("Employee Attrition Intelligence Platform", title_style))
    elements.append(Paragraph("AI-Generated Attrition Risk Report — Confidential HR Document", subtitle_style))

    # ---- Employee Profile ----
    elements.append(Paragraph(f"Employee Profile: {employee_name}", section_style))
    profile_rows = [["Attribute", "Value"]]
    for key, value in employee_profile.items():
        profile_rows.append([str(key), str(value)])
    profile_table = Table(profile_rows, colWidths=[7 * cm, 8 * cm])
    profile_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(COLOR_PRIMARY)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7FA")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(profile_table)

    # ---- Prediction Result ----
    elements.append(Paragraph("Prediction Result", section_style))
    risk_label = prediction_result.get("risk_label", "N/A")
    probability = prediction_result.get("probability", 0.0)
    confidence = prediction_result.get("confidence", 0.0)

    result_rows = [
        ["Predicted Outcome", prediction_result.get("prediction", "N/A")],
        ["Risk Level", risk_label],
        ["Attrition Probability", f"{probability:.1%}"],
        ["Model Confidence", f"{confidence:.1%}"],
    ]
    result_table = Table(result_rows, colWidths=[7 * cm, 8 * cm])
    result_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 1), (1, 1), _risk_color(risk_label)),
        ("TEXTCOLOR", (0, 1), (1, 1), colors.white),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(result_table)

    # ---- Top Risk Factors ----
    elements.append(Paragraph("Top Risk Factors (SHAP Explainability)", section_style))
    if risk_factors:
        for factor in risk_factors:
            elements.append(Paragraph(f"• {factor}", body_style))
    else:
        elements.append(Paragraph("No explainability data available for this prediction.", body_style))

    # ---- HR Recommendations ----
    elements.append(Paragraph("HR Recommendations", section_style))
    if recommendations:
        for rec in recommendations:
            title = rec.get("title", "")
            desc = rec.get("description", "")
            elements.append(Paragraph(f"<b>{title}:</b> {desc}", body_style))
            elements.append(Spacer(1, 4))
    else:
        elements.append(Paragraph("This employee shows no major risk indicators at this time.", body_style))

    elements.append(Spacer(1, 16))
    footer_style = ParagraphStyle(
        "Footer", parent=styles["Normal"], textColor=colors.HexColor("#94A3B8"), fontSize=8,
    )
    elements.append(Paragraph(
        "Generated by EAIP — Employee Attrition Intelligence Platform. "
        "This report is AI-generated and intended to support, not replace, HR judgment.",
        footer_style,
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer.read()

