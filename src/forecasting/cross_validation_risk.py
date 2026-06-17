import numpy as np
import pandas as pd

from catboost import CatBoostRegressor

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score


def cross_validate_risk(risk_df):

    features = [

        "corridor",
        "event_cause",
        "requires_road_closure",

        "hour",
        "weekday",
        "month",

        "corridor_risk",
        "cause_risk"

    ]

    X = risk_df[features].copy()

    y = risk_df["incident_count"]

    X["corridor"] = (
        X["corridor"]
        .fillna("UNKNOWN")
        .astype(str)
    )

    X["event_cause"] = (
        X["event_cause"]
        .fillna("UNKNOWN")
        .astype(str)
    )

    X["requires_road_closure"] = (
        X["requires_road_closure"]
        .astype(int)
    )

    kf = KFold(

        n_splits=5,
        shuffle=True,
        random_state=42

    )

    scores = []

    fold = 1

    print("\n")
    print("=" * 60)
    print("RISK FORECAST CROSS VALIDATION")
    print("=" * 60)

    for train_idx, test_idx in kf.split(X):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model = CatBoostRegressor(

            iterations=1000,
            depth=8,
            learning_rate=0.03,

            verbose=False,
            random_seed=42

        )
        print("\nFeature Columns:")
        print(X.columns.tolist())

        print("\nDtypes:")
        print(X.dtypes)

        model.fit(

            X_train,
            y_train,

            cat_features=[
                "corridor",
                "event_cause"
            ]

        )

        preds = model.predict(X_test)

        score = r2_score(
            y_test,
            preds
        )

        scores.append(score)

        print(
            f"Fold {fold}: {score:.4f}"
        )

        fold += 1

    print(
        "\nMean R²:",
        round(
            np.mean(scores),
            4
        )
    )

    return np.mean(scores)