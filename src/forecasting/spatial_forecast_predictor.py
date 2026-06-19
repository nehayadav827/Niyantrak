import numpy as np


def safe_float(value, fallback=0.0):
    try:
        if value is None:
            return fallback

        return float(value)

    except Exception:
        return fallback


def get_positive_probability(model, X):
    classifier = model.get(
        "classifier"
    )

    if classifier is None:
        return None

    probabilities = classifier.predict_proba(
        X
    )

    if probabilities.ndim == 2 and probabilities.shape[1] > 1:
        return probabilities[:, 1]

    return probabilities.reshape(-1)


def predict_single_spatial_forecast(model, X):
    classifier = model.get(
        "classifier"
    )

    regressor = model.get(
        "regressor"
    )

    if classifier is None or regressor is None:
        raise ValueError(
            "Invalid spatial forecast model bundle."
        )

    alert_probability = get_positive_probability(
        model,
        X
    )

    positive_prediction = regressor.predict(
        X
    )

    threshold = safe_float(
        model.get("alert_threshold"),
        0.50
    )

    predictions = []

    for prob, positive_count in zip(
        alert_probability,
        positive_prediction
    ):
        prob = safe_float(prob)
        positive_count = max(
            safe_float(positive_count),
            0.0
        )

        if prob <= threshold:
            strength = 0.0

        else:
            strength = (
                prob
                -
                threshold
            ) / max(
                1.0 - threshold,
                1e-6
            )

        strength = max(
            0.0,
            min(strength, 1.0)
        )

        expected_count = (
            strength
            *
            positive_count
        )

        predictions.append(
            max(expected_count, 0.0)
        )

    details = {
        "alert_probability": alert_probability,
        "positive_prediction": positive_prediction,
        "alert_threshold": threshold,
    }

    return predictions[0], details