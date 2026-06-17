def get_risk_level(risk_score):

    if risk_score < 25:
        return "LOW"

    elif risk_score < 50:
        return "MODERATE"

    elif risk_score < 75:
        return "HIGH"

    else:
        return "CRITICAL"


def calculate_forecast_risk_score(
    predicted_incidents,
    incident_p95
):

    if incident_p95 <= 0:
        incident_p95 = 1.0

    risk_score = (
        predicted_incidents
        /
        incident_p95
    ) * 100

    risk_score = max(
        0,
        min(risk_score, 100)
    )

    risk_level = get_risk_level(
        risk_score
    )

    return risk_score, risk_level


def calculate_final_operational_risk(
    forecast_risk_score,
    event_impact_score
):

    weighted_score = (
        0.45 * forecast_risk_score
        +
        0.55 * event_impact_score
    )

    event_floor_score = (
        0.85 * event_impact_score
    )

    final_score = max(
        weighted_score,
        event_floor_score
    )

    if event_impact_score >= 85:
        final_score = max(
            final_score,
            75
        )

    elif event_impact_score >= 70:
        final_score = max(
            final_score,
            55
        )

    final_score = max(
        0,
        min(final_score, 100)
    )

    final_level = get_risk_level(
        final_score
    )

    return final_score, final_level