import os
import joblib
import pandas as pd

from catboost import CatBoostClassifier

from sklearn.metrics import (
    f1_score,
    accuracy_score
)

from sklearn.model_selection import TimeSeriesSplit


FEATURE_COLS = [

    "event_cause",

    "requires_road_closure",

    "corridor",

    "veh_type",

    "police_station",

    "hotspot_id",

    "corridor_risk",

    "rush_hour",

    "corridor_event_count",

    "police_station_risk",

    "event_cause_freq",

    "junction_risk",

    "spatial_density",

    "dbscan_hotspot",

    "hour_sin",

    "hour_cos",

    "closure_rush",

    "hotspot_rush",

    "corridor_hotspot"

]


CATEGORICAL_COLS = [

    "event_cause",

    "corridor",

    "veh_type",

    "police_station"

]


TARGET_COL = "severity_class"


def prepare_severity_dataset(df):

    missing_features = [

        col
        for col in FEATURE_COLS
        if col not in df.columns

    ]

    if missing_features:

        raise ValueError(
            "Missing required severity model features: "
            + str(missing_features)
        )

    if TARGET_COL not in df.columns:

        raise ValueError(
            f"Missing target column: {TARGET_COL}"
        )

    X = df[FEATURE_COLS].copy()

    y = df[TARGET_COL].copy().astype(str)

    # =====================================================
    # CATEGORICAL CLEANING
    # =====================================================

    categorical_cols = [

        col
        for col in CATEGORICAL_COLS
        if col in X.columns

    ]

    for col in categorical_cols:

        X[col] = (
            X[col]
            .fillna("UNKNOWN")
            .astype(str)
        )

    # =====================================================
    # BOOLEAN CLEANING
    # =====================================================

    if "requires_road_closure" in X.columns:

        X["requires_road_closure"] = (
            X["requires_road_closure"]
            .fillna(False)
            .astype(int)
        )

    # =====================================================
    # NUMERIC CLEANING
    # =====================================================

    numeric_cols = [

        col
        for col in X.columns
        if col not in categorical_cols
    ]

    for col in numeric_cols:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

        median_value = X[col].median()

        if pd.isna(median_value):
            median_value = 0

        X[col] = X[col].fillna(
            median_value
        )

    return X, y, categorical_cols


def sort_by_time_if_available(
    df,
    X,
    y
):

    if "start_datetime" not in df.columns:
        return X, y

    dt = pd.to_datetime(
        df["start_datetime"],
        errors="coerce"
    )

    sort_idx = (
        dt
        .fillna(pd.Timestamp.max)
        .argsort()
    )

    X = X.iloc[sort_idx].reset_index(drop=True)

    y = y.iloc[sort_idx].reset_index(drop=True)

    return X, y


def train_catboost(df):

    print("\n" + "=" * 70)
    print("TRAINING SEVERITY CATBOOST MODEL")
    print("=" * 70)

    # =====================================================
    # PREPARE DATASET
    # =====================================================

    X, y, categorical_cols = prepare_severity_dataset(
        df
    )

    X, y = sort_by_time_if_available(
        df,
        X,
        y
    )

    # =====================================================
    # DEBUG OUTPUT
    # =====================================================

    print("\nFeatures Used")
    print("-" * 50)
    print(X.columns.tolist())

    print("\nCategorical Features")
    print("-" * 50)
    print(categorical_cols)

    print("\nMissing Values")
    print("-" * 50)
    print(X.isnull().sum())

    print("\nData Types")
    print("-" * 50)
    print(X.dtypes)

    print("\nTarget Distribution")
    print("-" * 50)
    print(y.value_counts())

    # =====================================================
    # CROSS VALIDATION
    # =====================================================

    tscv = TimeSeriesSplit(
        n_splits=5
    )

    fold_scores = []

    print("\n" + "=" * 70)
    print("CROSS VALIDATION")
    print("=" * 70)

    for fold_no, (train_idx, test_idx) in enumerate(
        tscv.split(X),
        start=1
    ):

        print("\n" + "=" * 50)
        print(f"FOLD {fold_no}")
        print("=" * 50)

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model = CatBoostClassifier(

            iterations=500,

            depth=8,

            learning_rate=0.05,

            loss_function="MultiClass",

            eval_metric="TotalF1",

            random_seed=42,

            verbose=False

        )

        model.fit(

            X_train,
            y_train,

            cat_features=categorical_cols

        )

        preds = model.predict(
            X_test
        ).flatten()

        f1 = f1_score(

            y_test,

            preds,

            average="weighted"

        )

        acc = accuracy_score(
            y_test,
            preds
        )

        print(f"Accuracy : {acc:.4f}")
        print(f"F1 Score : {f1:.4f}")

        fold_scores.append(f1)

    # =====================================================
    # RESULTS
    # =====================================================

    print("\n" + "=" * 70)
    print("CROSS VALIDATION RESULTS")
    print("=" * 70)

    for i, score in enumerate(
        fold_scores,
        start=1
    ):

        print(f"Fold {i}: {score:.4f}")

    mean_f1 = sum(fold_scores) / len(fold_scores)

    print(f"\nMean F1: {mean_f1:.4f}")

    # =====================================================
    # FINAL TRAINING
    # =====================================================

    print("\nTraining Final Severity Model...")

    final_model = CatBoostClassifier(

        iterations=500,

        depth=8,

        learning_rate=0.05,

        loss_function="MultiClass",

        eval_metric="TotalF1",

        random_seed=42,

        verbose=False

    )

    final_model.fit(

        X,
        y,

        cat_features=categorical_cols

    )

    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(

        final_model,

        "models/catboost_severity.pkl"

    )

    print("\nModel Saved:")
    print("models/catboost_severity.pkl")

    return final_model, X