import os
import joblib
import numpy as np
import pandas as pd

from src.routing.diversion_engine import recommend_diversions

from src.scoring.event_impact import calculate_event_impact

from src.scoring.risk_score import (
    calculate_forecast_risk_score,
    calculate_final_operational_risk
)


MODEL_PATHS = [
    "models/timeseries_forecast_model.pkl",
    "models/timeseries_forecast.pkl",
    "models/timeseries_forecast_model",
]

FEATURE_STORE_PATH = "models/traffic_feature_store.pkl"


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


def make_key(corridor, hour):
    return f"{str(corridor)}__{int(hour)}"


def parse_bool(value):

    value = str(value).strip().lower()

    return value in [
        "yes",
        "y",
        "true",
        "1"
    ]


def safe_int_input(
    prompt,
    min_value=None,
    max_value=None
):

    while True:

        try:
            value = int(
                input(prompt)
            )

            if min_value is not None and value < min_value:
                print(
                    f"Value must be >= {min_value}"
                )
                continue

            if max_value is not None and value > max_value:
                print(
                    f"Value must be <= {max_value}"
                )
                continue

            return value

        except ValueError:
            print(
                "Please enter a valid integer."
            )


def load_forecast_model():

    for path in MODEL_PATHS:

        if os.path.exists(path):

            print(
                f"\nLoaded model: {path}"
            )

            return joblib.load(path)

    raise FileNotFoundError(
        "No forecasting model found. Expected one of:\n"
        +
        "\n".join(MODEL_PATHS)
    )


def load_feature_store():

    if not os.path.exists(FEATURE_STORE_PATH):

        raise FileNotFoundError(
            "Feature store not found.\n"
            "Run this first:\n"
            "python prepare_feature_store.py"
        )

    return joblib.load(
        FEATURE_STORE_PATH
    )


def resolve_corridor_name(
    user_corridor,
    store
):

    user_clean = (
        str(user_corridor)
        .strip()
        .lower()
    )

    for corridor in store.get("corridors", []):

        if (
            str(corridor)
            .strip()
            .lower()
            ==
            user_clean
        ):
            return corridor

    return str(user_corridor).strip()


def find_nearest_hour_profile(
    store,
    corridor,
    requested_hour
):

    corridor_hour_profiles = store[
        "corridor_hour_profiles"
    ]

    available_hours = []

    for key in corridor_hour_profiles.keys():

        try:
            c, h = key.rsplit(
                "__",
                1
            )

            if c == corridor:
                available_hours.append(
                    int(h)
                )

        except ValueError:
            continue

    if not available_hours:
        return None, None

    nearest_hour = min(
        available_hours,
        key=lambda h: min(
            abs(h - requested_hour),
            24 - abs(h - requested_hour)
        )
    )

    nearest_key = make_key(
        corridor,
        nearest_hour
    )

    return (
        corridor_hour_profiles[nearest_key],
        nearest_hour
    )


def get_profile(
    store,
    corridor,
    hour
):

    key = make_key(
        corridor,
        hour
    )

    corridor_hour_profiles = store[
        "corridor_hour_profiles"
    ]

    corridor_profiles = store[
        "corridor_profiles"
    ]

    global_profile = store[
        "global_profile"
    ]

    if key in corridor_hour_profiles:

        return (
            corridor_hour_profiles[key],
            "exact corridor-hour history",
            hour
        )

    nearest_profile, nearest_hour = find_nearest_hour_profile(
        store,
        corridor,
        hour
    )

    if nearest_profile is not None:

        return (
            nearest_profile,
            f"nearest corridor-hour history, hour {nearest_hour}",
            nearest_hour
        )

    if corridor in corridor_profiles:

        return (
            corridor_profiles[corridor],
            "corridor-level fallback history",
            None
        )

    return (
        global_profile,
        "global fallback history",
        None
    )


def recommend_resources(
    final_risk_level,
    predicted_incidents,
    road_closure
):

    if final_risk_level == "LOW":
        officers = 2
        barricades = 0

    elif final_risk_level == "MODERATE":
        officers = 4
        barricades = 1

    elif final_risk_level == "HIGH":
        officers = 6
        barricades = 2

    else:
        officers = 8
        barricades = 4

    if predicted_incidents >= 3:
        officers += 1

    if predicted_incidents >= 5:
        barricades += 1

    if predicted_incidents >= 8:
        officers += 2
        barricades += 1

    if road_closure:
        officers += 1
        barricades += 2

    return officers, barricades


def build_input_row(
    corridor,
    hour,
    weekday,
    month,
    profile
):

    hour_sin = np.sin(
        2 * np.pi * hour / 24
    )

    hour_cos = np.cos(
        2 * np.pi * hour / 24
    )

    row = {}

    for feature in FEATURES:

        if feature == "corridor":
            row[feature] = corridor

        elif feature == "hour":
            row[feature] = hour

        elif feature == "weekday":
            row[feature] = weekday

        elif feature == "month":
            row[feature] = month

        elif feature == "hour_sin":
            row[feature] = hour_sin

        elif feature == "hour_cos":
            row[feature] = hour_cos

        else:
            row[feature] = profile.get(
                feature,
                0.0
            )

    return pd.DataFrame(
        [row],
        columns=FEATURES
    )


def get_action_message(
    final_level,
    road_closure
):

    if final_level == "LOW":
        return "Normal monitoring is sufficient."

    if final_level == "MODERATE":
        return "Increase patrol visibility and monitor CCTV feeds."

    if final_level == "HIGH":

        if road_closure:
            return (
                "Deploy officers, prepare barricades, and keep diversion support ready."
            )

        return (
            "Deploy traffic officers and prepare diversion support."
        )

    if road_closure:
        return (
            "Immediate deployment required. Activate barricading and diversion plan."
        )

    return (
        "Immediate deployment required. Deploy officers and monitor corridor continuously."
    )


def predict_traffic_risk():

    print("\n" + "=" * 60)
    print("TRAFFIC INTELLIGENCE ENGINE")
    print("=" * 60)

    model = load_forecast_model()

    store = load_feature_store()

    print("\nAvailable sample corridors:")

    for corridor in store["corridors"][:15]:
        print("-", corridor)

    raw_corridor = input(
        "\nCorridor: "
    ).strip()

    corridor = resolve_corridor_name(
        raw_corridor,
        store
    )

    if corridor != raw_corridor:
        print(
            f"Using matched corridor: {corridor}"
        )

    hour = safe_int_input(
        "Hour (0-23): ",
        min_value=0,
        max_value=23
    )

    weekday = safe_int_input(
        "Weekday (0=Mon, 6=Sun): ",
        min_value=0,
        max_value=6
    )

    month = safe_int_input(
        "Month (1-12): ",
        min_value=1,
        max_value=12
    )

    print("\nEvent Details")
    print("-" * 60)

    event_cause = input(
        "Event Cause: "
    ).strip()

    veh_type = input(
        "Vehicle Type: "
    ).strip()

    road_closure = parse_bool(
        input("Road Closure Required (yes/no): ")
    )

    profile, profile_source, matched_hour = get_profile(
        store,
        corridor,
        hour
    )

    X = build_input_row(
        corridor=corridor,
        hour=hour,
        weekday=weekday,
        month=month,
        profile=profile
    )

    from src.forecasting.forecast_predictor import (
        predict_single_forecast
    )

    predicted_incidents, forecast_details = predict_single_forecast(
        model,
        X
    )

    predicted_incidents = max(
        predicted_incidents,
        0.0
    )
    alert_probability = None

    if (
            forecast_details.get("alert_probability") is not None
    ):
        alert_probability = float(
            forecast_details["alert_probability"][0]
        )

    context_multiplier = 1.0

    if corridor.strip().lower() == "non-corridor":
        context_multiplier = 0.65

    forecast_risk_score, forecast_risk_level = calculate_forecast_risk_score(
        predicted_incidents=predicted_incidents,
        incident_p95=store.get(
            "incident_p95",
            1.0
        ),
        incident_p99=store.get(
            "incident_p99",
            None
        ),
        alert_probability=alert_probability,
        context_multiplier=context_multiplier
    )



    rush_hour = (
        (7 <= hour <= 10)
        or
        (17 <= hour <= 21)
    )

    event_impact_score, event_impact_level = calculate_event_impact(
        event_cause=event_cause,
        veh_type=veh_type,
        road_closure=road_closure,
        rush_hour=rush_hour
    )

    final_risk_score, final_risk_level = calculate_final_operational_risk(
        forecast_risk_score=forecast_risk_score,
        event_impact_score=event_impact_score
    )

    officers, barricades = recommend_resources(
        final_risk_level=final_risk_level,
        predicted_incidents=predicted_incidents,
        road_closure=road_closure
    )

    diversion = recommend_diversions(
        affected_corridor=corridor,
        final_risk_level=final_risk_level,
        road_closure=road_closure
    )

    action = get_action_message(
        final_risk_level,
        road_closure
    )

    print("\n" + "=" * 60)
    print("TRAFFIC INTELLIGENCE REPORT")
    print("=" * 60)

    print("\nInput Summary")
    print("-" * 60)
    print(f"Corridor              : {corridor}")
    print(f"Hour                  : {hour}")
    print(f"Weekday               : {weekday}")
    print(f"Month                 : {month}")
    print(f"Event Cause           : {event_cause}")
    print(f"Vehicle Type          : {veh_type}")
    print(f"Road Closure Required : {road_closure}")

    print("\nForecast Layer")
    print("-" * 60)
    print(f"Predicted Incidents   : {predicted_incidents:.2f}")
    print(f"Forecast Risk Score   : {forecast_risk_score:.2f}%")
    print(f"Forecast Risk Level   : {forecast_risk_level}")

    print("\nEvent Impact Layer")
    print("-" * 60)
    print(f"Event Impact Score    : {event_impact_score:.2f}%")
    print(f"Event Impact Level    : {event_impact_level}")

    print("\nFinal Operational Decision")
    print("-" * 60)
    print(f"Final Risk Score      : {final_risk_score:.2f}%")
    print(f"Final Risk Level      : {final_risk_level}")

    print("\nHistorical Context Used")
    print("-" * 60)
    print(f"Feature Source        : {profile_source}")

    if matched_hour is not None and matched_hour != hour:
        print(f"Matched History Hour  : {matched_hour}")

    print(f"Lag 1 Hour            : {X.loc[0, 'lag_1']:.2f}")
    print(f"Lag 2 Hours           : {X.loc[0, 'lag_2']:.2f}")
    print(f"Lag 3 Hours           : {X.loc[0, 'lag_3']:.2f}")
    print(f"Lag 24 Hours          : {X.loc[0, 'lag_24']:.2f}")
    print(f"Rolling 6 Hours       : {X.loc[0, 'rolling_6']:.2f}")
    print(f"Rolling 24 Hours      : {X.loc[0, 'rolling_24']:.2f}")
    print(f"Rolling 168 Hours     : {X.loc[0, 'rolling_168']:.2f}")
    print(f"Corridor Avg          : {X.loc[0, 'corridor_avg']:.2f}")
    print(f"Corridor Volatility   : {X.loc[0, 'corridor_volatility']:.2f}")

    print("\nRecommended Deployment")
    print("-" * 60)
    print(f"Officers Needed       : {officers}")
    print(f"Barricades Needed     : {barricades}")

    print("\nRecommended Diversion")
    print("-" * 60)
    print(f"Primary Detour        : {diversion['primary_detour']}")
    print(f"Secondary Detour      : {diversion['secondary_detour']}")

    if diversion["support_corridors"]:
        print(
            "Support Corridors     : "
            + ", ".join(diversion["support_corridors"])
        )
    else:
        print("Support Corridors     : None")

    print(f"Diversion Action      : {diversion['message']}")

    print("\nRecommended Action")
    print("-" * 60)
    print(action)

    print("=" * 60)