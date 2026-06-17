import os
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from django.conf import settings

from src.scoring.event_impact import calculate_event_impact

from src.scoring.risk_score import (
    calculate_forecast_risk_score,
    calculate_final_operational_risk
)

from src.routing.diversion_engine import recommend_diversions


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(settings.BASE_DIR)

MODEL_PATHS = [
    BASE_DIR / "models" / "timeseries_forecast_model.pkl",
    BASE_DIR / "models" / "timeseries_forecast.pkl",
]

FEATURE_STORE_PATH = (
    BASE_DIR
    / "models"
    / "traffic_feature_store.pkl"
)


# ============================================================
# MODEL FEATURES
# Must match the exact training feature order.
# ============================================================

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


# ============================================================
# SIMPLE CACHE
# Avoid loading model/store again and again for every request.
# ============================================================

_MODEL_CACHE = None
_STORE_CACHE = None


# ============================================================
# BASIC HELPERS
# ============================================================

def parse_bool(value):

    value = str(value).strip().lower()

    return value in [
        "yes",
        "true",
        "1",
        "y",
        "on"
    ]


def safe_float(value, fallback=0.0):

    try:

        if value is None:
            return fallback

        value = str(value).strip()

        if value == "":
            return fallback

        return float(value)

    except Exception:

        return fallback


def make_key(
    corridor,
    hour
):

    return f"{str(corridor)}__{int(hour)}"


def load_model():

    global _MODEL_CACHE

    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    for path in MODEL_PATHS:

        if os.path.exists(path):

            _MODEL_CACHE = joblib.load(path)

            return _MODEL_CACHE

    raise FileNotFoundError(
        "Forecast model not found. Run:\n"
        "python train_all.py"
    )


def load_feature_store():

    global _STORE_CACHE

    if _STORE_CACHE is not None:
        return _STORE_CACHE

    if not os.path.exists(FEATURE_STORE_PATH):

        raise FileNotFoundError(
            "Feature store not found. Run:\n"
            "python prepare_feature_store.py"
        )

    _STORE_CACHE = joblib.load(FEATURE_STORE_PATH)

    return _STORE_CACHE


def resolve_corridor_name(
    corridor,
    store
):

    user_clean = (
        str(corridor)
        .strip()
        .lower()
    )

    for c in store.get("corridors", []):

        if (
            str(c)
            .strip()
            .lower()
            ==
            user_clean
        ):
            return c

    return str(corridor).strip()


# ============================================================
# FEATURE STORE PROFILE LOOKUP
# ============================================================

def find_nearest_hour_profile(
    store,
    corridor,
    requested_hour
):

    profiles = store.get(
        "corridor_hour_profiles",
        {}
    )

    available_hours = []

    for key in profiles.keys():

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
        profiles[nearest_key],
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

    corridor_hour_profiles = store.get(
        "corridor_hour_profiles",
        {}
    )

    corridor_profiles = store.get(
        "corridor_profiles",
        {}
    )

    global_profile = store.get(
        "global_profile",
        {}
    )

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


# ============================================================
# MODEL INPUT ROW
# ============================================================

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


# ============================================================
# RECOMMENDATION HELPERS
# ============================================================

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


def get_action_message(
    final_risk_level,
    road_closure
):

    if final_risk_level == "LOW":

        return "Normal monitoring is sufficient."

    if final_risk_level == "MODERATE":

        return "Increase patrol visibility and monitor CCTV feeds."

    if final_risk_level == "HIGH":

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


def normalize_timestamp(
    timestamp_string
):

    if not timestamp_string:

        return datetime.now()

    try:

        return datetime.fromisoformat(
            timestamp_string
        )

    except Exception:

        return datetime.now()


def estimate_duration_minutes(
    predicted_incidents,
    final_risk_level,
    road_closure
):

    base = 30 + predicted_incidents * 20

    if final_risk_level == "HIGH":

        base += 20

    elif final_risk_level == "CRITICAL":

        base += 40

    if road_closure:

        base += 30

    return int(
        max(
            15,
            min(base, 240)
        )
    )


def estimate_affected_radius_meters(
    final_risk_level,
    predicted_incidents,
    road_closure,
    event_cause
):

    event_cause = (
        str(event_cause)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    if final_risk_level == "LOW":

        base_radius = 250

    elif final_risk_level == "MODERATE":

        base_radius = 500

    elif final_risk_level == "HIGH":

        base_radius = 900

    else:

        base_radius = 1400

    if road_closure:

        base_radius += 500

    if predicted_incidents >= 3:

        base_radius += 300

    if predicted_incidents >= 6:

        base_radius += 500

    high_spread_causes = [
        "accident",
        "congestion",
        "vip_movement",
        "protest",
        "procession",
        "public_event",
        "water_logging"
    ]

    if event_cause in high_spread_causes:

        base_radius += 300

    return int(
        min(
            max(base_radius, 200),
            3000
        )
    )


# ============================================================
# MAIN ENGINE
# ============================================================

def predict_event_impact(payload):

    model = load_model()

    store = load_feature_store()

    timestamp = normalize_timestamp(
        payload.get("timestamp")
    )

    hour = timestamp.hour
    weekday = timestamp.weekday()
    month = timestamp.month

    raw_corridor = payload.get(
        "corridor",
        "UNKNOWN"
    )

    corridor = resolve_corridor_name(
        raw_corridor,
        store
    )

    event_type = payload.get(
        "event_type",
        "unplanned"
    )

    event_cause = payload.get(
        "event_cause",
        "others"
    )

    priority = payload.get(
        "priority",
        "Medium"
    )

    veh_type = payload.get(
        "veh_type",
        "unknown"
    )

    police_station = payload.get(
        "police_station",
        "Unknown"
    )

    road_closure = parse_bool(
        payload.get(
            "requires_road_closure",
            False
        )
    )

    latitude = safe_float(
        payload.get("latitude"),
        12.9716
    )

    longitude = safe_float(
        payload.get("longitude"),
        77.5946
    )

    end_latitude = safe_float(
        payload.get("end_latitude"),
        None
    )

    end_longitude = safe_float(
        payload.get("end_longitude"),
        None
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

    predicted_incidents = float(
        model.predict(X)[0]
    )

    predicted_incidents = max(
        predicted_incidents,
        0.0
    )

    forecast_score, forecast_level = calculate_forecast_risk_score(
        predicted_incidents=predicted_incidents,
        incident_p95=store.get(
            "incident_p95",
            1.0
        )
    )

    rush_hour = (
        (7 <= hour <= 10)
        or
        (17 <= hour <= 21)
    )

    event_score, event_level = calculate_event_impact(
        event_cause=event_cause,
        veh_type=veh_type,
        road_closure=road_closure,
        rush_hour=rush_hour
    )

    # ========================================================
    # FORM-LEVEL OPERATIONAL BOOSTS
    # Priority and event type are available in the UI.
    # ========================================================

    priority_boost = {
        "low": 0,
        "medium": 4,
        "high": 8,
        "critical": 12
    }.get(
        str(priority).strip().lower(),
        4
    )

    event_type_boost = {
        "planned": 2,
        "unplanned": 6
    }.get(
        str(event_type).strip().lower(),
        4
    )

    event_score = min(
        event_score
        +
        priority_boost
        +
        event_type_boost,
        100
    )

    # Recompute event level after boosts
    if event_score < 25:

        event_level = "LOW"

    elif event_score < 50:

        event_level = "MODERATE"

    elif event_score < 75:

        event_level = "HIGH"

    else:

        event_level = "CRITICAL"

    final_score, final_level = calculate_final_operational_risk(
        forecast_risk_score=forecast_score,
        event_impact_score=event_score
    )

    officers, barricades = recommend_resources(
        final_risk_level=final_level,
        predicted_incidents=predicted_incidents,
        road_closure=road_closure
    )

    diversion = recommend_diversions(
        affected_corridor=corridor,
        final_risk_level=final_level,
        road_closure=road_closure
    )

    duration = estimate_duration_minutes(
        predicted_incidents=predicted_incidents,
        final_risk_level=final_level,
        road_closure=road_closure
    )

    affected_radius = estimate_affected_radius_meters(
        final_risk_level=final_level,
        predicted_incidents=predicted_incidents,
        road_closure=road_closure,
        event_cause=event_cause
    )

    secondary_radius = int(
        affected_radius * 1.75
    )

    action = get_action_message(
        final_risk_level=final_level,
        road_closure=road_closure
    )

    closure_probability = min(
        max(
            (
                forecast_score * 0.35
                +
                event_score * 0.65
            ) / 100,
            0
        ),
        1
    )

    return {
        "input": {
            "event_type": event_type,
            "event_cause": event_cause,
            "priority": priority,
            "corridor": corridor,
            "veh_type": veh_type,
            "police_station": police_station,
            "timestamp": timestamp,
            "hour": hour,
            "weekday": weekday,
            "month": month,
            "road_closure": road_closure,
            "latitude": latitude,
            "longitude": longitude,
            "end_latitude": end_latitude,
            "end_longitude": end_longitude,
        },

        "forecast": {
            "predicted_incidents": predicted_incidents,
            "forecast_score": forecast_score,
            "forecast_level": forecast_level,
        },

        "event": {
            "event_score": event_score,
            "event_level": event_level,
            "rush_hour": rush_hour,
        },

        "final": {
            "final_score": final_score,
            "final_level": final_level,
            "duration_minutes": duration,
            "closure_probability": closure_probability,
            "affected_radius_m": affected_radius,
            "secondary_radius_m": secondary_radius,
        },

        "history": {
            "profile_source": profile_source,
            "matched_hour": matched_hour,
            "lag_1": X.loc[0, "lag_1"],
            "lag_2": X.loc[0, "lag_2"],
            "lag_3": X.loc[0, "lag_3"],
            "lag_24": X.loc[0, "lag_24"],
            "rolling_6": X.loc[0, "rolling_6"],
            "rolling_24": X.loc[0, "rolling_24"],
            "rolling_168": X.loc[0, "rolling_168"],
            "corridor_avg": X.loc[0, "corridor_avg"],
            "corridor_volatility": X.loc[0, "corridor_volatility"],
        },

        "resources": {
            "officers": officers,
            "barricades": barricades,
            "station": police_station or "Unknown",
            "shift_hours": 2 if final_level in ["LOW", "MODERATE"] else 4,
        },

        "diversion": diversion,

        "map": {
            "latitude": latitude,
            "longitude": longitude,
            "end_latitude": end_latitude,
            "end_longitude": end_longitude,
            "affected_radius_m": affected_radius,
            "secondary_radius_m": secondary_radius,
        },

        "action": action,
    }