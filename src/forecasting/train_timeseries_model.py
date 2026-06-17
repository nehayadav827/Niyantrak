import os
import joblib

from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score


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


def train_timeseries_model(ts_df):

    print("\n" + "=" * 60)
    print("TRAINING FINAL TIME-SERIES MODEL")
    print("=" * 60)

    target = "incident_count"

    # =====================================================
    # VALIDATION
    # =====================================================

    missing = [
        col for col in FEATURES + [target]
        if col not in ts_df.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns in time-series dataset: "
            + str(missing)
        )

    if "time_bucket" not in ts_df.columns:
        raise ValueError(
            "time_bucket column is required for chronological train/test split."
        )

    df = (
        ts_df
        .sort_values(["time_bucket", "corridor"])
        .reset_index(drop=True)
        .copy()
    )

    # =====================================================
    # TRUE TIME-BASED SPLIT
    # Past 80% time buckets -> Future 20% time buckets
    # =====================================================

    unique_times = sorted(
        df["time_bucket"].unique()
    )

    split_time_idx = int(
        len(unique_times) * 0.8
    )

    train_times = unique_times[:split_time_idx]
    test_times = unique_times[split_time_idx:]

    train_df = df[
        df["time_bucket"].isin(train_times)
    ].copy()

    test_df = df[
        df["time_bucket"].isin(test_times)
    ].copy()

    X_train = train_df[FEATURES]
    y_train = train_df[target]

    X_test = test_df[FEATURES]
    y_test = test_df[target]

    print("\nTrain rows:", len(train_df))
    print("Test rows :", len(test_df))
    print("Train time range:", train_df["time_bucket"].min(), "to", train_df["time_bucket"].max())
    print("Test time range :", test_df["time_bucket"].min(), "to", test_df["time_bucket"].max())

    # =====================================================
    # EVALUATION MODEL
    # =====================================================

    eval_model = CatBoostRegressor(
        iterations=700,
        depth=6,
        learning_rate=0.03,
        loss_function="RMSE",
        random_seed=42,
        verbose=100
    )

    eval_model.fit(
        X_train,
        y_train,
        cat_features=["corridor"]
    )

    preds = eval_model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        preds
    )

    r2 = r2_score(
        y_test,
        preds
    )

    print("\n" + "=" * 60)
    print("FORECAST HOLDOUT RESULTS")
    print("=" * 60)
    print(f"MAE: {mae:.4f}")
    print(f"R² : {r2:.4f}")

    # =====================================================
    # FINAL MODEL TRAINED ON FULL DATA
    # This is the model used for inference.
    # =====================================================

    print("\nTraining production model on full dataset...")

    X_full = df[FEATURES]
    y_full = df[target]

    final_model = CatBoostRegressor(
        iterations=700,
        depth=6,
        learning_rate=0.03,
        loss_function="RMSE",
        random_seed=42,
        verbose=100
    )

    final_model.fit(
        X_full,
        y_full,
        cat_features=["corridor"]
    )

    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        final_model,
        "models/timeseries_forecast_model.pkl"
    )

    # Compatibility with older inference paths
    joblib.dump(
        final_model,
        "models/timeseries_forecast.pkl"
    )

    print("\nModel Saved:")
    print("models/timeseries_forecast_model.pkl")
    print("models/timeseries_forecast.pkl")

    return final_model