import os

from config import DATA_PATH

from src.preprocessing.load_data import load_data

from src.forecasting.build_timeseries_dataset import (
    build_timeseries_dataset
)

from src.forecasting.cross_validate_timeseries import (
    cross_validate_timeseries
)

from src.forecasting.train_timeseries_model import (
    train_timeseries_model
)

from src.forecasting.forecast_feature_importance import (
    forecast_feature_importance
)


FORECAST_FEATURES = [
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


def validate_data_path():

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}\n"
            "Update DATA_PATH in config.py or place your dataset at that path."
        )


def validate_forecast_features(ts_df):

    missing_features = [
        col
        for col in FORECAST_FEATURES
        if col not in ts_df.columns
    ]

    if missing_features:

        raise ValueError(
            "Time-series dataset is missing required forecast features:\n"
            + str(missing_features)
        )

    if "incident_count" not in ts_df.columns:

        raise ValueError(
            "Time-series dataset is missing target column: incident_count"
        )


def main():

    print("\n" + "=" * 70)
    print("TRAINING TRAFFIC INTELLIGENCE BACKEND")
    print("=" * 70)

    os.makedirs(
        "models",
        exist_ok=True
    )

    validate_data_path()

    # =====================================================
    # STEP 1: LOAD DATA
    # =====================================================

    print("\nStep 1: Loading Data")

    df = load_data(
        DATA_PATH
    )

    # =====================================================
    # STEP 2: BUILD FORECAST DATASET
    # =====================================================

    print("\nStep 2: Building Time-Series Forecast Dataset")

    ts_df = build_timeseries_dataset(
        df
    )

    print("\nFinal Forecast Dataset Shape:")
    print(
        ts_df.shape
    )

    validate_forecast_features(
        ts_df
    )

    # =====================================================
    # STEP 3: CROSS VALIDATION
    # =====================================================

    print("\nStep 3: Time-Series Cross Validation")

    cross_validate_timeseries(
        ts_df
    )

    # =====================================================
    # STEP 4: TRAIN MODEL
    # =====================================================

    print("\nStep 4: Training Final Forecast Model")

    model = train_timeseries_model(
        ts_df
    )

    # =====================================================
    # STEP 5: FEATURE IMPORTANCE
    # =====================================================

    print("\nStep 5: Forecast Feature Importance")

    forecast_feature_importance(
        model,
        FORECAST_FEATURES
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print("\nGenerated files:")
    print("models/timeseries_forecast_model.pkl")
    print("models/timeseries_forecast.pkl")
    print("forecast_feature_importance.png")

    print("\nNext command:")
    print("python prepare_feature_store.py")


if __name__ == "__main__":
    main()