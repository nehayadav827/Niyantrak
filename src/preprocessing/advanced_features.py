import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN


def add_advanced_features(df):

    print("Building Advanced Features...")

    df = df.copy()

    # =====================================================
    # REQUIRED COLUMN SAFETY
    # =====================================================

    required_cols = [
        "corridor",
        "police_station",
        "event_cause",
        "junction",
        "latitude",
        "longitude",
        "hour",
        "rush_hour",
        "hotspot_id",
        "corridor_risk",
        "requires_road_closure"
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            "Missing required columns for advanced features: "
            + str(missing_cols)
        )

    # =====================================================
    # BASIC CLEANING
    # =====================================================

    df["corridor"] = (
        df["corridor"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.strip()
    )

    df["police_station"] = (
        df["police_station"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.strip()
    )

    df["event_cause"] = (
        df["event_cause"]
        .fillna("others")
        .astype(str)
        .str.strip()
    )

    df["junction"] = (
        df["junction"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.strip()
    )

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

    df["requires_road_closure"] = (
        df["requires_road_closure"]
        .fillna(False)
        .astype(int)
    )

    df["rush_hour"] = (
        df["rush_hour"]
        .fillna(0)
        .astype(int)
    )

    df["hotspot_id"] = pd.to_numeric(
        df["hotspot_id"],
        errors="coerce"
    ).fillna(0)

    df["corridor_risk"] = pd.to_numeric(
        df["corridor_risk"],
        errors="coerce"
    ).fillna(0)

    # =====================================================
    # CORRIDOR EVENT COUNT
    # =====================================================

    corridor_counts = (
        df.groupby("corridor")
        .size()
        .to_dict()
    )

    df["corridor_event_count"] = (
        df["corridor"]
        .map(corridor_counts)
        .fillna(0)
    )

    # =====================================================
    # POLICE STATION RISK
    # =====================================================

    police_counts = (
        df.groupby("police_station")
        .size()
        .to_dict()
    )

    df["police_station_risk"] = (
        df["police_station"]
        .map(police_counts)
        .fillna(0)
    )

    # =====================================================
    # EVENT CAUSE FREQUENCY
    # =====================================================

    cause_counts = (
        df.groupby("event_cause")
        .size()
        .to_dict()
    )

    df["event_cause_freq"] = (
        df["event_cause"]
        .map(cause_counts)
        .fillna(0)
    )

    # =====================================================
    # JUNCTION RISK
    # =====================================================

    junction_counts = (
        df.groupby("junction")
        .size()
        .to_dict()
    )

    df["junction_risk"] = (
        df["junction"]
        .map(junction_counts)
        .fillna(0)
    )

    # =====================================================
    # SPATIAL DENSITY
    # =====================================================

    coords = df[
        ["latitude", "longitude"]
    ].values

    n_neighbors = min(
        10,
        len(df)
    )

    if n_neighbors <= 1:

        df["spatial_density"] = 0.0

    else:

        nn = NearestNeighbors(
            n_neighbors=n_neighbors
        )

        nn.fit(coords)

        distances, _ = nn.kneighbors(coords)

        df["spatial_density"] = (
            1
            /
            (
                distances.mean(axis=1)
                +
                1e-6
            )
        )

    # =====================================================
    # DBSCAN HOTSPOT
    # =====================================================

    if len(df) < 10:

        df["dbscan_hotspot"] = 0

    else:

        cluster = DBSCAN(
            eps=0.005,
            min_samples=10
        )

        df["dbscan_hotspot"] = cluster.fit_predict(
            coords
        )

    # =====================================================
    # CYCLICAL TIME FEATURES
    # Keep here too for safety.
    # =====================================================

    df["hour"] = (
        df["hour"]
        .fillna(0)
        .astype(int)
    )

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    # =====================================================
    # INTERACTION FEATURES
    # =====================================================

    df["closure_rush"] = (
        df["requires_road_closure"]
        *
        df["rush_hour"]
    )

    df["hotspot_rush"] = (
        df["hotspot_id"]
        *
        df["rush_hour"]
    )

    df["corridor_hotspot"] = (
        df["corridor_risk"]
        *
        df["hotspot_id"]
    )

    # =====================================================
    # FINAL NUMERIC SAFETY
    # =====================================================

    numeric_cols = [
        "corridor_event_count",
        "police_station_risk",
        "event_cause_freq",
        "junction_risk",
        "spatial_density",
        "dbscan_hotspot",
        "hour_sin",
        "hour_cos",
        "closure_rush",
        "hotspot_rush",
        "corridor_hotspot"
    ]

    for col in numeric_cols:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        ).fillna(0)

    print("Advanced Features Complete")

    print("\nAdvanced Feature Preview:")
    print(
        df[
            [
                "corridor_event_count",
                "police_station_risk",
                "event_cause_freq",
                "junction_risk",
                "spatial_density",
                "dbscan_hotspot",
                "closure_rush",
                "hotspot_rush",
                "corridor_hotspot"
            ]
        ]
        .head()
    )

    return df