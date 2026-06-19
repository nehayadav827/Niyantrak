import os

import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from catboost import CatBoostRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
)

from src.forecasting.forecast_predictor import (
    predict_forecast_count,
    HurdleModelBundle,
)


FEATURES = [
    "corridor",

    "hour",
    "weekday",
    "month",

    "is_event_day",
    "calendar_event_type",
    "calendar_event_intensity",

    "hour_sin",
    "hour_cos",

    "lag_1",
    "lag_2",
    "lag_3",
    "lag_24",
    "lag_48",
    "lag_72",
    "lag_168",

    "any_incident_last_3h",
    "incidents_last_24h",
    "above_corridor_avg",

    "rolling_6",
    "rolling_12",
    "rolling_24",
    "rolling_168",

    "corridor_avg",
    "corridor_volatility",

    "zone_risk",
    "junction_risk",
    "cause_risk",
    "closure_risk",
    "cluster_risk",
]


CAT_FEATURES = [
    "corridor",
    "calendar_event_type",
]


TARGET = "incident_count"


def validate_dataset(ts_df):
    missing = [
        col
        for col in FEATURES + [TARGET, "time_bucket"]
        if col not in ts_df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns for forecasting:\n"
            + str(missing)
        )


def prepare_feature_frame(df):
    X = df[FEATURES].copy()

    categorical_cols = [
        col
        for col in CAT_FEATURES
        if col in X.columns
    ]

    for col in categorical_cols:
        X[col] = (
            X[col]
            .fillna("none")
            .astype(str)
        )

    for col in X.columns:
        if col in categorical_cols:
            continue

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

        median_value = X[col].median()

        if pd.isna(median_value):
            median_value = 0.0

        X[col] = X[col].fillna(
            median_value
        )

    return X


def split_by_time(
    df,
    train_ratio=0.8
):
    df = df.copy()

    df["time_bucket"] = pd.to_datetime(
        df["time_bucket"],
        errors="coerce",
        utc=True
    ).dt.tz_convert(None)

    df = (
        df
        .dropna(subset=["time_bucket"])
        .sort_values(
            [
                "time_bucket",
                "corridor"
            ]
        )
        .reset_index(drop=True)
    )

    unique_times = sorted(
        df["time_bucket"].unique()
    )

    split_idx = int(
        len(unique_times)
        *
        train_ratio
    )

    train_times = unique_times[:split_idx]
    test_times = unique_times[split_idx:]

    train_df = df[
        df["time_bucket"].isin(train_times)
    ].copy()

    test_df = df[
        df["time_bucket"].isin(test_times)
    ].copy()

    return train_df, test_df


def tune_alert_threshold(
    y_true_alert,
    alert_proba
):
    best_threshold = 0.50
    best_score = -1

    thresholds = np.arange(
        0.20,
        0.91,
        0.05
    )

    beta = 2.0

    for threshold in thresholds:
        preds = (
            alert_proba >= threshold
        ).astype(int)

        precision = precision_score(
            y_true_alert,
            preds,
            zero_division=0
        )

        recall = recall_score(
            y_true_alert,
            preds,
            zero_division=0
        )

        if precision == 0 and recall == 0:
            score = 0.0

        else:
            score = (
                (1 + beta ** 2)
                *
                precision
                *
                recall
            ) / (
                (beta ** 2 * precision)
                +
                recall
                +
                1e-9
            )

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return float(best_threshold), float(best_score)


def fit_hurdle_bundle(
    X_train,
    y_train,
    alert_threshold=0.35
):
    X_train = X_train.copy()

    categorical_cols = [
        col
        for col in CAT_FEATURES
        if col in X_train.columns
    ]

    for col in categorical_cols:
        X_train[col] = (
            X_train[col]
            .fillna("none")
            .astype(str)
        )

    for col in X_train.columns:
        if col in categorical_cols:
            continue

        X_train[col] = pd.to_numeric(
            X_train[col],
            errors="coerce"
        )

        median_value = X_train[col].median()

        if pd.isna(median_value):
            median_value = 0.0

        X_train[col] = X_train[col].fillna(
            median_value
        )

    y_train = y_train.astype(float)

    y_alert = (
        y_train > 0
    ).astype(int)

    classifier = CatBoostClassifier(
        iterations=700,
        depth=6,
        learning_rate=0.03,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=False,
    )

    classifier.fit(
        X_train,
        y_alert,
        cat_features=categorical_cols
    )

    positive_mask = (
        y_train > 0
    )

    positive_count_mean = 1.0
    regressor = None

    if positive_mask.sum() >= 20:
        X_pos = X_train.loc[
            positive_mask
        ].copy()

        y_pos = y_train.loc[
            positive_mask
        ]

        positive_count_mean = float(
            y_pos.mean()
        )

        y_pos_log = np.log1p(
            y_pos
        )

        regressor = CatBoostRegressor(
            iterations=700,
            depth=6,
            learning_rate=0.03,
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
        )

        regressor.fit(
            X_pos,
            y_pos_log,
            cat_features=categorical_cols
        )

    else:
        if positive_mask.sum() > 0:
            positive_count_mean = float(
                y_train.loc[positive_mask].mean()
            )

    bundle = HurdleModelBundle({
        "model_type": "zero_inflated_hurdle_v1",

        "classifier": classifier,
        "regressor": regressor,

        "features": FEATURES,
        "cat_features": categorical_cols,
        "categorical_cols": categorical_cols,

        "alert_threshold": float(alert_threshold),
        "positive_count_mean": float(positive_count_mean),
    })

    return bundle


def build_calibrated_threshold(
    train_df
):
    if len(train_df) < 100:
        return 0.35

    calibration_train_df, calibration_df = split_by_time(
        train_df,
        train_ratio=0.80
    )

    if len(calibration_train_df) == 0 or len(calibration_df) == 0:
        return 0.35

    X_cal_train = prepare_feature_frame(
        calibration_train_df
    )

    y_cal_train = calibration_train_df[
        TARGET
    ].astype(float)

    X_cal = prepare_feature_frame(
        calibration_df
    )

    y_cal = calibration_df[
        TARGET
    ].astype(float)

    temp_bundle = fit_hurdle_bundle(
        X_train=X_cal_train,
        y_train=y_cal_train,
        alert_threshold=0.35
    )

    pred = predict_forecast_count(
        temp_bundle,
        X_cal
    )

    y_cal_alert = (
        y_cal > 0
    ).astype(int)

    threshold, best_f2 = tune_alert_threshold(
        y_true_alert=y_cal_alert,
        alert_proba=pred["alert_probability"]
    )

    print("\nCalibrated Alert Threshold:")
    print(f"Threshold : {threshold:.2f}")
    print(f"Calib F2  : {best_f2:.4f}")

    return threshold


def evaluate_hurdle_model(
    bundle,
    X_test,
    y_test
):
    X_test = X_test.copy()

    cat_cols = bundle.get(
        "cat_features",
        []
    )

    for col in cat_cols:
        if col in X_test.columns:
            X_test[col] = (
                X_test[col]
                .fillna("none")
                .astype(str)
            )

    for col in X_test.columns:
        if col in cat_cols:
            continue

        X_test[col] = pd.to_numeric(
            X_test[col],
            errors="coerce"
        ).fillna(0.0)

    pred = predict_forecast_count(
        bundle,
        X_test
    )

    expected_count = pred[
        "expected_count"
    ]

    alert_proba = pred[
        "alert_probability"
    ]

    alert_pred = pred[
        "alert_prediction"
    ]

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

    try:
        r2 = r2_score(
            y_test,
            expected_count
        )

    except Exception:
        r2 = 0.0

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

    alert_accuracy = accuracy_score(
        y_alert,
        alert_pred
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

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),

        "alert_accuracy": float(alert_accuracy),
        "alert_precision": float(precision),
        "alert_recall": float(recall),
        "alert_f1": float(alert_f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
    }

    return metrics


def save_bundle(
    bundle
):
    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        bundle,
        "models/timeseries_forecast_model.pkl"
    )

    joblib.dump(
        bundle,
        "models/timeseries_forecast.pkl"
    )

    print("\nModel Saved:")
    print("models/timeseries_forecast_model.pkl")
    print("models/timeseries_forecast.pkl")


def train_timeseries_model(
    ts_df
):
    print("\n" + "=" * 60)
    print("TRAINING ZERO-INFLATED TRAFFIC FORECAST MODEL")
    print("=" * 60)

    validate_dataset(
        ts_df
    )

    df = ts_df.copy()

    df["time_bucket"] = pd.to_datetime(
        df["time_bucket"],
        errors="coerce",
        utc=True
    ).dt.tz_convert(None)

    df = (
        df
        .dropna(subset=["time_bucket"])
        .sort_values(
            [
                "time_bucket",
                "corridor"
            ]
        )
        .reset_index(drop=True)
        .copy()
    )

    train_df, test_df = split_by_time(
        df,
        train_ratio=0.8
    )

    print("\nTrain rows:", len(train_df))
    print("Test rows :", len(test_df))

    print(
        "Train time range:",
        train_df["time_bucket"].min(),
        "to",
        train_df["time_bucket"].max()
    )

    print(
        "Test time range :",
        test_df["time_bucket"].min(),
        "to",
        test_df["time_bucket"].max()
    )

    print("\nTarget Distribution:")
    print(
        df[TARGET]
        .value_counts()
        .head(15)
    )

    print("\nZero Ratio:")
    print(
        round(
            float((df[TARGET] == 0).mean()),
            4
        )
    )

    alert_threshold = build_calibrated_threshold(
        train_df
    )

    X_train = prepare_feature_frame(
        train_df
    )

    y_train = train_df[
        TARGET
    ].astype(float)

    X_test = prepare_feature_frame(
        test_df
    )

    y_test = test_df[
        TARGET
    ].astype(float)

    print("\nTraining holdout evaluation model...")

    eval_bundle = fit_hurdle_bundle(
        X_train=X_train,
        y_train=y_train,
        alert_threshold=alert_threshold
    )

    metrics = evaluate_hurdle_model(
        bundle=eval_bundle,
        X_test=X_test,
        y_test=y_test
    )

    print("\n" + "=" * 60)
    print("FORECAST HOLDOUT RESULTS")
    print("=" * 60)

    print(f"MAE              : {metrics['mae']:.4f}")
    print(f"RMSE             : {metrics['rmse']:.4f}")
    print(f"R²               : {metrics['r2']:.4f}")

    print("\n" + "-" * 60)
    print("ALERT CLASSIFICATION RESULTS")
    print("-" * 60)

    print(f"Accuracy         : {metrics['alert_accuracy']:.4f}")
    print(f"Precision        : {metrics['alert_precision']:.4f}")
    print(f"Recall           : {metrics['alert_recall']:.4f}")
    print(f"F1               : {metrics['alert_f1']:.4f}")
    print(f"ROC-AUC          : {metrics['roc_auc']:.4f}")
    print(f"PR-AUC           : {metrics['pr_auc']:.4f}")

    print("\nTraining production hurdle model on full dataset...")

    X_full = prepare_feature_frame(
        df
    )

    y_full = df[
        TARGET
    ].astype(float)

    final_bundle = fit_hurdle_bundle(
        X_train=X_full,
        y_train=y_full,
        alert_threshold=alert_threshold
    )

    final_bundle["holdout_metrics"] = metrics
    final_bundle["train_rows"] = int(len(df))
    final_bundle["positive_rows"] = int((y_full > 0).sum())
    final_bundle["zero_ratio"] = float((y_full == 0).mean())

    save_bundle(
        final_bundle
    )

    return final_bundle