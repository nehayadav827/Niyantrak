import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from src.forecasting.train_timeseries_model import (
    FEATURES,
    TARGET,
    fit_hurdle_bundle,
    tune_alert_threshold
)

from src.forecasting.forecast_predictor import (
    predict_forecast_count
)


def evaluate_fold(
    bundle,
    X_test,
    y_test
):

    pred = predict_forecast_count(
        bundle,
        X_test
    )

    expected_count = pred["expected_count"]
    alert_proba = pred["alert_probability"]
    alert_pred = pred["alert_prediction"]

    y_test = y_test.astype(float)

    y_alert = (
        y_test > 0
    ).astype(int)

    mae = mean_absolute_error(
        y_test,
        expected_count
    )

    rmse = mean_squared_error(
        y_test,
        expected_count
    ) ** 0.5

    r2 = r2_score(
        y_test,
        expected_count
    )

    precision = precision_score(
        y_alert,
        alert_pred,
        zero_division=0
    )

    recall = recall_score(
        y_alert,
        alert_pred,
        zero_division=0
    )

    alert_f1 = f1_score(
        y_alert,
        alert_pred,
        zero_division=0
    )

    try:

        roc_auc = roc_auc_score(
            y_alert,
            alert_proba
        )

    except Exception:

        roc_auc = 0.0

    try:

        pr_auc = average_precision_score(
            y_alert,
            alert_proba
        )

    except Exception:

        pr_auc = 0.0

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(alert_f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
    }


def cross_validate_timeseries(
    ts_df,
    n_splits=5
):

    print("\n" + "=" * 60)
    print("ZERO-INFLATED TIME SERIES CROSS VALIDATION")
    print("=" * 60)

    missing = [
        col
        for col in FEATURES + [TARGET, "time_bucket"]
        if col not in ts_df.columns
    ]

    if missing:

        raise ValueError(
            "Missing columns in time-series dataset: "
            + str(missing)
        )

    df = (
        ts_df
        .sort_values(
            [
                "time_bucket",
                "corridor"
            ]
        )
        .reset_index(drop=True)
        .copy()
    )

    unique_times = np.array(
        sorted(df["time_bucket"].unique())
    )

    if len(unique_times) < n_splits + 2:

        raise ValueError(
            "Not enough unique time buckets for time-series cross validation."
        )

    fold_size = len(unique_times) // (n_splits + 1)

    results = []

    for fold in range(1, n_splits + 1):

        train_end = fold * fold_size
        test_start = train_end

        if fold == n_splits:

            test_end = len(unique_times)

        else:

            test_end = (fold + 1) * fold_size

        train_times = unique_times[:train_end]
        test_times = unique_times[test_start:test_end]

        train_df = df[
            df["time_bucket"].isin(train_times)
        ].copy()

        test_df = df[
            df["time_bucket"].isin(test_times)
        ].copy()

        X_train = train_df[FEATURES]
        y_train = train_df[TARGET]

        X_test = test_df[FEATURES]
        y_test = test_df[TARGET]

        # simple fold-local threshold calibration
        y_train_alert = (
            y_train > 0
        ).astype(int)

        temp_bundle = fit_hurdle_bundle(
            X_train=X_train,
            y_train=y_train,
            alert_threshold=0.35
        )

        train_pred = predict_forecast_count(
            temp_bundle,
            X_train
        )

        threshold, _ = tune_alert_threshold(
            y_true_alert=y_train_alert,
            alert_proba=train_pred["alert_probability"]
        )

        bundle = fit_hurdle_bundle(
            X_train=X_train,
            y_train=y_train,
            alert_threshold=threshold
        )

        metrics = evaluate_fold(
            bundle=bundle,
            X_test=X_test,
            y_test=y_test
        )

        results.append(metrics)

        print(
            f"Fold {fold}: "
            f"MAE={metrics['mae']:.4f}, "
            f"RMSE={metrics['rmse']:.4f}, "
            f"R²={metrics['r2']:.4f}, "
            f"AlertF1={metrics['f1']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"PR-AUC={metrics['pr_auc']:.4f}, "
            f"Train={len(train_df)}, "
            f"Test={len(test_df)}"
        )

    mean_metrics = {}

    for key in results[0].keys():

        mean_metrics[key] = float(
            np.mean(
                [
                    row[key]
                    for row in results
                ]
            )
        )

    print("\n" + "=" * 60)
    print("MEAN CROSS VALIDATION RESULTS")
    print("=" * 60)

    print(f"Mean MAE       : {mean_metrics['mae']:.4f}")
    print(f"Mean RMSE      : {mean_metrics['rmse']:.4f}")
    print(f"Mean R²        : {mean_metrics['r2']:.4f}")
    print(f"Mean Precision : {mean_metrics['precision']:.4f}")
    print(f"Mean Recall    : {mean_metrics['recall']:.4f}")
    print(f"Mean Alert F1  : {mean_metrics['f1']:.4f}")
    print(f"Mean ROC-AUC   : {mean_metrics['roc_auc']:.4f}")
    print(f"Mean PR-AUC    : {mean_metrics['pr_auc']:.4f}")

    return {
        "folds": results,
        "mean": mean_metrics
    }