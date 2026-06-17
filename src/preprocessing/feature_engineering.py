import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


def normalize_text(value):

    value = str(value).strip().lower()

    value = value.replace("-", "_")
    value = value.replace(" ", "_")

    while "__" in value:
        value = value.replace("__", "_")

    return value


def engineer_features(df):

    print("\nBuilding Base Features...")

    df = df.copy()

    # ==================================================
    # REQUIRED COLUMN SAFETY
    # ==================================================

    required_cols = [
        "start_datetime",
        "event_cause",
        "corridor",
        "veh_type",
        "police_station",
        "junction",
        "requires_road_closure",
        "latitude",
        "longitude"
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            "Missing required columns for feature engineering: "
            + str(missing_cols)
        )

    # ==================================================
    # DATETIME
    # ==================================================

    df["start_datetime"] = pd.to_datetime(
        df["start_datetime"],
        errors="coerce"
    )

    df["hour"] = (
        df["start_datetime"]
        .dt.hour
        .fillna(0)
        .astype(int)
    )

    df["weekday"] = (
        df["start_datetime"]
        .dt.weekday
        .fillna(0)
        .astype(int)
    )

    df["month"] = (
        df["start_datetime"]
        .dt.month
        .fillna(0)
        .astype(int)
    )

    # ==================================================
    # CYCLIC TIME FEATURES
    # ==================================================

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    # ==================================================
    # RUSH HOUR
    # ==================================================

    df["rush_hour"] = np.where(
        (
            ((df["hour"] >= 7) & (df["hour"] <= 10))
            |
            ((df["hour"] >= 17) & (df["hour"] <= 21))
        ),
        1,
        0
    )

    # ==================================================
    # CATEGORICAL CLEANING
    # ==================================================

    df["event_cause"] = (
        df["event_cause"]
        .fillna("others")
        .apply(normalize_text)
    )

    df["corridor"] = (
        df["corridor"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.strip()
    )

    df["veh_type"] = (
        df["veh_type"]
        .fillna("unknown")
        .apply(normalize_text)
    )

    df["police_station"] = (
        df["police_station"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.strip()
    )

    df["junction"] = (
        df["junction"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.strip()
    )

    # ==================================================
    # ROAD CLOSURE
    # ==================================================

    df["requires_road_closure"] = (
        df["requires_road_closure"]
        .fillna(False)
        .astype(int)
    )

    # ==================================================
    # LATITUDE / LONGITUDE CLEANING
    # ==================================================

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce"
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce"
    )

    df["latitude"] = df["latitude"].fillna(
        df["latitude"].median()
    )

    df["longitude"] = df["longitude"].fillna(
        df["longitude"].median()
    )

    # ==================================================
    # CORRIDOR RISK
    # ==================================================

    corridor_counts = (
        df["corridor"]
        .value_counts()
    )

    df["corridor_risk"] = (
        df["corridor"]
        .map(corridor_counts)
        .fillna(0)
    )

    # ==================================================
    # HOTSPOT CLUSTERS
    # ==================================================

    coords = df[
        ["latitude", "longitude"]
    ].copy()

    n_clusters = min(
        20,
        len(df)
    )

    if n_clusters <= 1:

        df["hotspot_id"] = 0

    else:

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )

        df["hotspot_id"] = kmeans.fit_predict(
            coords
        )

    # ==================================================
    # LOCATION GRID
    # ==================================================

    df["lat_round"] = (
        df["latitude"]
        .round(3)
    )

    df["lon_round"] = (
        df["longitude"]
        .round(3)
    )

    # ==================================================
    # OPTIONAL DURATION
    # For analytics only. Do not use as operational feature.
    # ==================================================

    if "end_datetime" in df.columns:

        end_dt = pd.to_datetime(
            df["end_datetime"],
            errors="coerce"
        )

        duration = (
            end_dt
            -
            df["start_datetime"]
        ).dt.total_seconds() / 60

        df["duration_minutes"] = (
            duration
            .fillna(0)
            .clip(lower=0)
        )

    print("Base Features Complete")

    important_cols = [
        "hour",
        "weekday",
        "month",
        "rush_hour",
        "hour_sin",
        "hour_cos",
        "corridor_risk",
        "hotspot_id"
    ]

    print("\nCurrent Features:")
    print(
        df[important_cols]
        .head()
    )

    return df