import numpy as np

from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error


FEATURES = [
    "corridor",

    "hour",
    "weekday",
    "month",

    "hour_sin",
    "hour_cos",

    "lag_1",
    "lag_2",
    "lag_3",

    "lag_24",
    "lag_48",
    "lag_72",
    "lag_168",

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
    "cluster_risk"
]


def cross_validate_timeseries(ts_df, n_splits=5):

    print("\n" + "=" * 60)
    print("TIME SERIES CROSS VALIDATION")
    print("=" * 60)

    missing = [
        col for col in FEATURES + ["incident_count"]
        if col not in ts_df.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns in time-series dataset: "
            + str(missing)
        )

    if "time_bucket" not in ts_df.columns:
        raise ValueError(
            "time_bucket column is required for true time-based validation."
        )

    df = (
        ts_df
        .sort_values(["time_bucket", "corridor"])
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

    r2_scores = []
    mae_scores = []

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
        ]

        test_df = df[
            df["time_bucket"].isin(test_times)
        ]

        X_train = train_df[FEATURES]
        y_train = train_df["incident_count"]

        X_test = test_df[FEATURES]
        y_test = test_df["incident_count"]

        model = CatBoostRegressor(
            iterations=700,
            depth=6,
            learning_rate=0.03,
            loss_function="RMSE",
            random_seed=42,
            verbose=False
        )

        model.fit(
            X_train,
            y_train,
            cat_features=["corridor"]
        )

        preds = model.predict(X_test)

        r2 = r2_score(
            y_test,
            preds
        )

        mae = mean_absolute_error(
            y_test,
            preds
        )

        r2_scores.append(r2)
        mae_scores.append(mae)

        print(
            f"Fold {fold}: "
            f"R²={r2:.4f}, "
            f"MAE={mae:.4f}, "
            f"Train rows={len(train_df)}, "
            f"Test rows={len(test_df)}"
        )

    print("\nMean R²:", round(float(np.mean(r2_scores)), 4))
    print("Mean MAE:", round(float(np.mean(mae_scores)), 4))

    return {
        "mean_r2": float(np.mean(r2_scores)),
        "mean_mae": float(np.mean(mae_scores)),
        "fold_r2": r2_scores,
        "fold_mae": mae_scores
    }