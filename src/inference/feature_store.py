import os
import joblib
import pandas as pd

from src.preprocessing.load_data import load_data
from src.forecasting.build_timeseries_dataset import build_timeseries_dataset


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


PROFILE_FEATURES = [
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


def safe_float(value):
    if pd.isna(value):
        return 0.0

    return float(value)


def build_feature_store(
    data_path="data/traffic_events.csv",
    output_path="models/traffic_feature_store.pkl"
):

    print("\n" + "=" * 60)
    print("BUILDING CORRIDOR-HOUR FEATURE STORE")
    print("=" * 60)

    df = load_data(data_path)

    ts_df = build_timeseries_dataset(df)

    os.makedirs(
        "models",
        exist_ok=True
    )

    # ==================================================
    # CORRIDOR + HOUR PROFILE
    # ==================================================

    corridor_hour_profiles = {}

    grouped_ch = (
        ts_df
        .groupby(["corridor", "hour"])[PROFILE_FEATURES]
        .median()
        .reset_index()
    )

    for _, row in grouped_ch.iterrows():

        key = make_key(
            row["corridor"],
            row["hour"]
        )

        profile = {}

        for feature in PROFILE_FEATURES:
            profile[feature] = safe_float(
                row[feature]
            )

        corridor_hour_profiles[key] = profile

    # ==================================================
    # CORRIDOR FALLBACK PROFILE
    # ==================================================

    corridor_profiles = {}

    grouped_c = (
        ts_df
        .groupby("corridor")[PROFILE_FEATURES]
        .median()
        .reset_index()
    )

    for _, row in grouped_c.iterrows():

        corridor = str(
            row["corridor"]
        )

        profile = {}

        for feature in PROFILE_FEATURES:
            profile[feature] = safe_float(
                row[feature]
            )

        corridor_profiles[corridor] = profile

    # ==================================================
    # GLOBAL FALLBACK PROFILE
    # ==================================================

    global_profile = {}

    for feature in PROFILE_FEATURES:

        global_profile[feature] = safe_float(
            ts_df[feature].median()
        )

    # ==================================================
    # RISK THRESHOLDS
    # ==================================================

    incident_p50 = safe_float(
        ts_df["incident_count"].quantile(0.50)
    )

    incident_p75 = safe_float(
        ts_df["incident_count"].quantile(0.75)
    )

    incident_p90 = safe_float(
        ts_df["incident_count"].quantile(0.90)
    )

    incident_p95 = safe_float(
        ts_df["incident_count"].quantile(0.95)
    )

    incident_p99 = safe_float(
        ts_df["incident_count"].quantile(0.99)
    )

    if incident_p95 <= 0:
        incident_p95 = max(
            safe_float(ts_df["incident_count"].max()),
            1.0
        )

    store = {
        "features": FEATURES,

        "profile_features": PROFILE_FEATURES,

        "corridor_hour_profiles": corridor_hour_profiles,

        "corridor_profiles": corridor_profiles,

        "global_profile": global_profile,

        "incident_p50": incident_p50,
        "incident_p75": incident_p75,
        "incident_p90": incident_p90,
        "incident_p95": incident_p95,
        "incident_p99": incident_p99,

        "corridors": sorted(
            list(corridor_profiles.keys())
        )
    }

    joblib.dump(
        store,
        output_path
    )

    print("\nFeature store saved:")
    print(output_path)

    print("\nTotal corridor-hour profiles:")
    print(len(corridor_hour_profiles))

    print("\nTotal corridor profiles:")
    print(len(corridor_profiles))

    print("\nRisk thresholds:")
    print(f"P50: {incident_p50:.2f}")
    print(f"P75: {incident_p75:.2f}")
    print(f"P90: {incident_p90:.2f}")
    print(f"P95: {incident_p95:.2f}")
    print(f"P99: {incident_p99:.2f}")

    print("\nSample corridors:")
    for corridor in store["corridors"][:20]:
        print("-", corridor)

    return store