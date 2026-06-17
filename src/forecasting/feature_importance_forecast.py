import pandas as pd
import joblib


def forecast_feature_importance():

    model = joblib.load(
        "models/timeseries_forecast_model.pkl"
    )

    features = [

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
        "rolling_24",
        "rolling_168",

        "corridor_avg",
        "corridor_volatility"

    ]

    imp = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    })

    imp = imp.sort_values(
        "importance",
        ascending=False
    )

    print("\n")
    print("=" * 60)
    print("FORECAST FEATURE IMPORTANCE")
    print("=" * 60)

    print(imp)