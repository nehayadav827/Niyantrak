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
    incident_p95=1.0,
    incident_p99=None,
    alert_probability=None,
    context_multiplier=1.0
):
    """
    Converts predicted incident count into forecast risk.

    Why this version:
    - Sparse traffic data has many zeros.
    - incident_p95 can be too small.
    - Using p99 gives more stable risk scaling.
    - alert_probability helps when using the hurdle model.
    - context_multiplier lets us dampen broad fallback categories like Non-corridor.
    """

    predicted_incidents = max(
        float(predicted_incidents),
        0.0
    )

    if predicted_incidents <= 0:
        return 0.0, "LOW"

    if incident_p99 is None:
        incident_p99 = incident_p95

    incident_p95 = max(
        float(incident_p95),
        1.0
    )

    incident_p99 = max(
        float(incident_p99),
        incident_p95,
        1.0
    )

    # More stable than only p95.
    risk_scale = max(
        incident_p99 * 1.5,
        incident_p95 * 2.0,
        1.0
    )

    count_score = (
        predicted_incidents
        /
        risk_scale
    ) * 100

    count_score = max(
        0,
        min(count_score, 100)
    )

    if alert_probability is not None:

        alert_score = max(
            0,
            min(float(alert_probability) * 100, 100)
        )

        risk_score = (
            0.65 * count_score
            +
            0.35 * alert_score
        )

    else:

        risk_score = count_score

    risk_score = (
        risk_score
        *
        context_multiplier
    )

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