"""
components/recommendation_engine.py
====================================
Rule-based HR recommendation engine. Translates an employee's input profile
(plus, optionally, top SHAP risk factors) into concrete, actionable HR
recommendations rendered as cards on the Prediction page and included in the
PDF report.
"""

import pandas as pd


def generate_recommendations(employee: dict, risk_label: str) -> list[dict]:
    """
    Inspect an employee's profile and return a list of recommendation dicts:
    {"title": str, "description": str, "icon": str, "severity": "info"|"warning"|"danger"}
    Ordered roughly by severity / impact.
    """
    recs = []

    if str(employee.get("OverTime", "No")) == "Yes":
        recs.append({
            "title": "Reduce Workload / Manage Overtime",
            "description": "This employee regularly works overtime, a top driver of burnout-related "
                            "attrition. Consider workload redistribution or temporary support staffing.",
            "icon": "⏱️", "severity": "danger",
        })

    job_sat = employee.get("JobSatisfaction", 3)
    if job_sat is not None and job_sat <= 2:
        recs.append({
            "title": "Schedule a Manager Feedback Session",
            "description": "Job satisfaction is low. A structured 1:1 with their manager to surface "
                            "concerns and set development goals is recommended within the next 2 weeks.",
            "icon": "🗣️", "severity": "warning",
        })

    env_sat = employee.get("EnvironmentSatisfaction", 3)
    if env_sat is not None and env_sat <= 2:
        recs.append({
            "title": "Review Team / Workplace Environment",
            "description": "Low environment satisfaction may reflect team dynamics or workspace "
                            "issues — consider a confidential pulse-check with the immediate team.",
            "icon": "🏢", "severity": "warning",
        })

    income = employee.get("MonthlyIncome", None)
    if income is not None and income < 4000:
        recs.append({
            "title": "Initiate a Salary Review",
            "description": "Monthly income is below typical retention thresholds for comparable roles. "
                            "Recommend a compensation benchmarking review.",
            "icon": "💰", "severity": "danger",
        })

    promo_years = employee.get("YearsSinceLastPromotion", 0)
    if promo_years is not None and promo_years >= 4:
        recs.append({
            "title": "Conduct a Promotion Evaluation",
            "description": f"It has been {promo_years} years since the last promotion. Evaluate "
                            "readiness for advancement to prevent stagnation-driven attrition.",
            "icon": "📈", "severity": "warning",
        })

    wlb = employee.get("WorkLifeBalance", 3)
    if wlb is not None and wlb <= 2:
        recs.append({
            "title": "Offer Flexible Working Arrangements",
            "description": "Work-life balance score is low. Consider flexible hours, hybrid work, "
                            "or workload review to improve sustainability.",
            "icon": "🧘", "severity": "warning",
        })

    travel = employee.get("BusinessTravel", "Non-Travel")
    if travel == "Travel_Frequently":
        recs.append({
            "title": "Optimize Business Travel Schedule",
            "description": "Frequent travel is associated with elevated attrition risk. Consider "
                            "rotating travel duties or compensating with travel-recovery time off.",
            "icon": "✈️", "severity": "info",
        })

    distance = employee.get("DistanceFromHome", 0)
    if distance is not None and distance >= 20:
        recs.append({
            "title": "Consider Hybrid / Remote Flexibility",
            "description": f"Commute distance is {distance} km. Hybrid or remote options could "
                            "meaningfully reduce commute-related attrition risk.",
            "icon": "🏠", "severity": "info",
        })

    stock = employee.get("StockOptionLevel", 0)
    if stock is not None and stock == 0:
        recs.append({
            "title": "Review Long-Term Incentive Eligibility",
            "description": "Employee currently holds no stock options. Long-term incentives can "
                            "meaningfully improve retention for tenured, high-value staff.",
            "icon": "📊", "severity": "info",
        })

    if not recs:
        recs.append({
            "title": "Maintain Current Engagement",
            "description": "No major risk indicators detected. Continue regular check-ins and "
                            "recognition to sustain engagement.",
            "icon": "✅", "severity": "info",
        })

    # Surface the most severe recommendations first
    severity_rank = {"danger": 0, "warning": 1, "info": 2}
    recs.sort(key=lambda r: severity_rank.get(r["severity"], 3))
    return recs
