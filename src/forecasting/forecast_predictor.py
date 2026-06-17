import numpy as np


def predict_forecast_count(
    model_bundle,
    X
):
    """
    Supports both:
    1. New zero-inflated / hurdle model bundle
    2. Old single CatBoostRegressor model

    For sparse traffic incident data:
    - alert_probability is used for risk ranking
    - expected_count is threshold-gated to avoid overpredicting zeros
    """

    # =====================================================
    # NEW ZERO-INFLATED / HURDLE MODEL
    # =====================================================

    if (
        isinstance(model_bundle, dict)
        and model_bundle.get("model_type") == "zero_inflated_hurdle_v1"
    ):

        classifier = model_bundle["classifier"]
        regressor = model_bundle["regressor"]

        alert_threshold = model_bundle.get(
            "alert_threshold",
            0.35
        )

        positive_count_mean = model_bundle.get(
            "positive_count_mean",
            1.0
        )

        alert_proba = classifier.predict_proba(X)[:, 1]

        alert_pred = (
            alert_proba
            >=
            alert_threshold
        ).astype(int)

        # =================================================
        # Positive-count prediction
        # =================================================

        if regressor is not None:

            positive_log_pred = regressor.predict(X)

            positive_count_pred = np.expm1(
                positive_log_pred
            )

            positive_count_pred = np.maximum(
                positive_count_pred,
                0.0
            )

        else:

            positive_count_pred = np.full(
                len(X),
                positive_count_mean
            )

        # =================================================
        # CONSERVATIVE COUNT CALIBRATION
        #
        # Old formula:
        # expected_count = alert_probability * positive_count
        #
        # Problem:
        # It overpredicts many zero rows.
        #
        # New formula:
        # Only probability above threshold contributes strongly.
        # =================================================

        probability_strength = (
            alert_proba
            -
            alert_threshold
        ) / (
            1.0
            -
            alert_threshold
            +
            1e-9
        )

        probability_strength = np.clip(
            probability_strength,
            0.0,
            1.0
        )

        expected_count = (
            probability_strength
            *
            positive_count_pred
        )

        expected_count = np.maximum(
            expected_count,
            0.0
        )

        # Optional raw expected count for diagnostics
        raw_expected_count = (
            alert_proba
            *
            positive_count_pred
        )

        raw_expected_count = np.maximum(
            raw_expected_count,
            0.0
        )

        return {
            "expected_count": expected_count,
            "raw_expected_count": raw_expected_count,
            "alert_probability": alert_proba,
            "alert_prediction": alert_pred,
            "positive_count_prediction": positive_count_pred,
            "model_type": "zero_inflated_hurdle_v1"
        }

    # =====================================================
    # OLD SINGLE REGRESSOR FALLBACK
    # =====================================================

    preds = model_bundle.predict(X)

    preds = np.maximum(
        preds,
        0.0
    )

    return {
        "expected_count": preds,
        "raw_expected_count": preds,
        "alert_probability": None,
        "alert_prediction": None,
        "positive_count_prediction": None,
        "model_type": "single_regressor"
    }


def predict_single_forecast(
    model_bundle,
    X
):

    result = predict_forecast_count(
        model_bundle,
        X
    )

    return (
        float(result["expected_count"][0]),
        result
    )