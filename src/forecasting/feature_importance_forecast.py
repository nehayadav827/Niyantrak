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