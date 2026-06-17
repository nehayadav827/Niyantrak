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


def main():

    print("\n" + "=" * 60)
    print("TRAFFIC RISK FORECASTING PIPELINE V2")
    print("=" * 60)

    # ==================================================
    # LOAD DATA
    # ==================================================

    df = load_data(
        "data.csv"
    )

    # ==================================================
    # BUILD TIME SERIES DATASET
    # ==================================================

    ts_df = build_timeseries_dataset(
        df
    )

    print("\nFinal Dataset Shape:")
    print(ts_df.shape)

    print("\nColumns:")
    print(ts_df.columns.tolist())

    # ==================================================
    # CROSS VALIDATION
    # ==================================================

    cross_validate_timeseries(
        ts_df
    )

    # ==================================================
    # TRAIN MODEL
    # ==================================================

    model = train_timeseries_model(
        ts_df
    )

    # ==================================================
    # FEATURE IMPORTANCE
    # ==================================================

    features = [

        # categorical

        "corridor",

        # time

        "hour",
        "weekday",
        "month",

        # cyclic time

        "hour_sin",
        "hour_cos",

        # lag features

        "lag_1",
        "lag_2",
        "lag_3",

        "lag_24",
        "lag_48",
        "lag_72",
        "lag_168",

        # rolling windows

        "rolling_6",
        "rolling_12",
        "rolling_24",
        "rolling_168",

        # corridor stats

        "corridor_avg",
        "corridor_volatility",

        # spatial intelligence

        "zone_risk",
        "junction_risk",
        "cause_risk",
        "closure_risk",
        "cluster_risk"
    ]

    forecast_feature_importance(
        model,
        features
    )

    print("\n" + "=" * 60)
    print("PIPELINE FINISHED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()