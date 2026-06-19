import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from catboost import CatBoostRegressor

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from config import DATA_PATH

from src.preprocessing.load_data import load_data
from src.forecasting.build_spatial_timeseries_dataset import (
    build_spatial_timeseries_dataset,
    SPATIAL_FEATURES,
)


FEATURE_STORE_PATH = "models/traffic_feature_store.pkl"
SPATIAL_MODEL_PATH = "models/spatial_timeseries_forecast_model.pkl"


def safe_metric(func, *args, fallback=0.0, **kwargs):
    try:
        return float(
            func(
                *args,
                **kwargs
            )
        )

    except Exception:
        return fallback


def prepare_X_y(spatial_ts):
    X = spatial_ts[
        SPATIAL_FEATURES
    ].copy()

    y = spatial_ts[
        "incident_count"
    ].astype(float)

    categorical_cols = [
        "spatial_cluster_id",
        "dominant_corridor",
        "calendar_event_type",
    ]

    for col in categorical_cols:
        if col not in X.columns:
            X[col] = "none"

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

        X[col] = X[col].fillna(
            X[col].median()
        )

    return X, y, categorical_cols


def chronological_split(spatial_ts, X, y, train_ratio=0.80):
    ordered_times = (
        spatial_ts["time_bucket"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    split_idx = int(
        len(ordered_times) * train_ratio
    )

    cutoff_time = ordered_times[
        split_idx
    ]

    train_mask = spatial_ts["time_bucket"] < cutoff_time
    test_mask = spatial_ts["time_bucket"] >= cutoff_time

    X_train = X.loc[
        train_mask
    ]

    X_test = X.loc[
        test_mask
    ]

    y_train = y.loc[
        train_mask
    ]

    y_test = y.loc[
        test_mask
    ]

    return X_train, X_test, y_train, y_test


def calibrate_threshold(
    classifier,
    X_valid,
    y_valid_alert
):
    probabilities = classifier.predict_proba(
        X_valid
    )[:, 1]

    best_threshold = 0.50
    best_f2 = -1.0

    thresholds = np.arange(
        0.10,
        0.91,
        0.05
    )

    for threshold in thresholds:
        preds = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_valid_alert,
            preds,
            zero_division=0
        )

        recall = recall_score(
            y_valid_alert,
            preds,
            zero_division=0
        )

        beta = 2

        denominator = (
            beta * beta * precision
            +
            recall
        )

        if denominator == 0:
            f2 = 0.0

        else:
            f2 = (
                (1 + beta * beta)
                *
                precision
                *
                recall
                /
                denominator
            )

        if f2 > best_f2:
            best_f2 = f2
            best_threshold = threshold

    return float(best_threshold), float(best_f2)


def hurdle_predict(
    classifier,
    regressor,
    X,
    threshold
):
    probabilities = classifier.predict_proba(
        X
    )[:, 1]

    positive_pred = regressor.predict(
        X
    )

    output = []

    for prob, positive_count in zip(
        probabilities,
        positive_pred
    ):
        positive_count = max(
            float(positive_count),
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

        output.append(
            strength * positive_count
        )

    return np.array(output), probabilities


def train_spatial_timeseries_model(
    data_path=DATA_PATH,
    feature_store_path=FEATURE_STORE_PATH,
    output_path=SPATIAL_MODEL_PATH
):
    print("\n" + "=" * 80)
    print("TRAINING SPATIAL-CLUSTER HURDLE FORECAST MODEL")
    print("=" * 80)

    if not os.path.exists(feature_store_path):
        raise FileNotFoundError(
            "Feature store not found. Run prepare_feature_store.py first."
        )

    store = joblib.load(
        feature_store_path
    )

    df = load_data(
        data_path
    )

    spatial_ts = build_spatial_timeseries_dataset(
        df,
        store
    )

    spatial_ts = spatial_ts.sort_values(
        [
            "time_bucket",
            "spatial_cluster_id"
        ]
    )

    X, y, categorical_cols = prepare_X_y(
        spatial_ts
    )

    X_train, X_test, y_train, y_test = chronological_split(
        spatial_ts,
        X,
        y
    )

    y_train_alert = (
        y_train > 0
    ).astype(int)

    y_test_alert = (
        y_test > 0
    ).astype(int)

    print("\nTrain rows:")
    print(len(X_train))

    print("\nTest rows:")
    print(len(X_test))

    print("\nPositive train rows:")
    print(int(y_train_alert.sum()))

    print("\nPositive test rows:")
    print(int(y_test_alert.sum()))

    classifier = CatBoostClassifier(
        iterations=700,
        depth=6,
        learning_rate=0.04,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=100,
        auto_class_weights="Balanced",
    )

    classifier.fit(
        X_train,
        y_train_alert,
        cat_features=categorical_cols
    )

    threshold, calib_f2 = calibrate_threshold(
        classifier,
        X_test,
        y_test_alert
    )

    print("\nCalibrated Spatial Alert Threshold:")
    print(f"Threshold : {threshold:.2f}")
    print(f"F2        : {calib_f2:.4f}")

    positive_mask = y_train > 0

    X_train_pos = X_train.loc[
        positive_mask
    ]

    y_train_pos = y_train.loc[
        positive_mask
    ]

    if len(X_train_pos) < 50:
        print("\nWARNING: few positive rows. Training regressor on all rows.")
        X_train_pos = X_train
        y_train_pos = y_train

    regressor = CatBoostRegressor(
        iterations=700,
        depth=6,
        learning_rate=0.04,
        loss_function="RMSE",
        random_seed=42,
        verbose=100,
    )

    regressor.fit(
        X_train_pos,
        y_train_pos,
        cat_features=categorical_cols
    )

    preds, alert_probs = hurdle_predict(
        classifier,
        regressor,
        X_test,
        threshold
    )

    alert_preds = (
        alert_probs >= threshold
    ).astype(int)

    mae = mean_absolute_error(
        y_test,
        preds
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            preds
        )
    )

    r2 = r2_score(
        y_test,
        preds
    )

    alert_accuracy = accuracy_score(
        y_test_alert,
        alert_preds
    )

    alert_precision = precision_score(
        y_test_alert,
        alert_preds,
        zero_division=0
    )

    alert_recall = recall_score(
        y_test_alert,
        alert_preds,
        zero_division=0
    )

    alert_f1 = f1_score(
        y_test_alert,
        alert_preds,
        zero_division=0
    )

    roc_auc = safe_metric(
        roc_auc_score,
        y_test_alert,
        alert_probs
    )

    pr_auc = safe_metric(
        average_precision_score,
        y_test_alert,
        alert_probs
    )

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),

        "alert_accuracy": float(alert_accuracy),
        "alert_precision": float(alert_precision),
        "alert_recall": float(alert_recall),
        "alert_f1": float(alert_f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),

        "alert_threshold": float(threshold),
        "calib_f2": float(calib_f2),

        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "positive_train_rows": int(y_train_alert.sum()),
        "positive_test_rows": int(y_test_alert.sum()),
    }

    print("\n" + "=" * 80)
    print("SPATIAL MODEL HOLDOUT RESULTS")
    print("=" * 80)

    print(f"MAE              : {mae:.4f}")
    print(f"RMSE             : {rmse:.4f}")
    print(f"R²               : {r2:.4f}")
    print(f"Alert Accuracy   : {alert_accuracy:.4f}")
    print(f"Alert Precision  : {alert_precision:.4f}")
    print(f"Alert Recall     : {alert_recall:.4f}")
    print(f"Alert F1         : {alert_f1:.4f}")
    print(f"ROC-AUC          : {roc_auc:.4f}")
    print(f"PR-AUC           : {pr_auc:.4f}")

    bundle = {
        "model_type": "spatial_cluster_hurdle_forecast",
        "classifier": classifier,
        "regressor": regressor,
        "features": SPATIAL_FEATURES,
        "categorical_cols": categorical_cols,
        "alert_threshold": threshold,
        "holdout_metrics": metrics,
        "description": (
            "Primary spatial-cluster-hour hurdle model. "
            "Training rows are spatial_cluster_id × hour, with latitude, longitude, "
            "local distance/density features, lag features, and cluster-level rolling history."
        ),
    }

    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        bundle,
        output_path
    )

    print("\nSpatial model saved:")
    print(output_path)

    return bundle