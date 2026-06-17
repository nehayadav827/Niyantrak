import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, accuracy_score

from catboost import CatBoostClassifier


def prepare_cv_data(
    df,
    feature_cols,
    target_col
):

    missing_features = [
        col
        for col in feature_cols
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing feature columns: "
            + str(missing_features)
        )

    if target_col not in df.columns:
        raise ValueError(
            f"Missing target column: {target_col}"
        )

    X = df[feature_cols].copy()
    y = df[target_col].copy().astype(str)

    categorical_cols = []

    for col in X.columns:

        if (
            X[col].dtype == "object"
            or str(X[col].dtype) == "category"
        ):
            X[col] = (
                X[col]
                .fillna("UNKNOWN")
                .astype(str)
            )

            categorical_cols.append(col)

        elif X[col].dtype == "bool":
            X[col] = (
                X[col]
                .fillna(False)
                .astype(int)
            )

        else:
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


def run_cross_validation(
    df,
    feature_cols,
    target_col,
    n_splits=5
):

    print("\n" + "=" * 70)
    print("CROSS VALIDATION RESULTS")
    print("=" * 70)

    X, y, categorical_cols = prepare_cv_data(
        df,
        feature_cols,
        target_col
    )

    X, y = sort_by_time_if_available(
        df,
        X,
        y
    )

    print("\nFeatures Used:")
    print(X.columns.tolist())

    print("\nCategorical Features:")
    print(categorical_cols)

    print("\nTarget Distribution:")
    print(y.value_counts())

    tscv = TimeSeriesSplit(
        n_splits=n_splits
    )

    f1_scores = []
    accuracy_scores = []

    for fold, (train_idx, test_idx) in enumerate(
        tscv.split(X),
        start=1
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model = CatBoostClassifier(

            iterations=500,

            depth=6,

            learning_rate=0.05,

            loss_function="MultiClass",

            eval_metric="TotalF1",

            verbose=False,

            random_seed=42

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

        f1_scores.append(f1)
        accuracy_scores.append(acc)

        print(
            f"Fold {fold}: "
            f"F1={f1:.4f}, "
            f"Accuracy={acc:.4f}"
        )

    mean_f1 = float(
        np.mean(f1_scores)
    )

    mean_accuracy = float(
        np.mean(accuracy_scores)
    )

    print("\nMean F1:", round(mean_f1, 4))
    print("Mean Accuracy:", round(mean_accuracy, 4))

    return {
        "mean_f1": mean_f1,
        "mean_accuracy": mean_accuracy,
        "fold_f1": f1_scores,
        "fold_accuracy": accuracy_scores
    }