from pathlib import Path

import numpy as np
import pandas as pd

from config import EVENT_CALENDAR_PATH

from src.inference.location_resolver import haversine_distance_meters


CALENDAR_EVENT_FEATURES = [
    "is_event_day",
    "calendar_event_type",
    "calendar_event_intensity",
]


EVENT_TYPE_WEIGHTS = {
    "none": 0,
    "sports": 75,
    "festival": 70,
    "protest": 85,
    "political": 85,
    "vip": 90,
    "election": 80,
    "public_event": 70,
    "procession": 75,
    "construction": 55,
    "other": 45,
}


CROWD_MULTIPLIERS = {
    "none": 0.00,
    "small": 0.60,
    "medium": 0.80,
    "large": 1.00,
    "mega": 1.20,
}


DEFAULT_RADIUS_BY_CROWD = {
    "none": 0,
    "small": 800,
    "medium": 1500,
    "large": 2500,
    "mega": 4000,
}


def safe_float(value, fallback=0.0):
    try:
        if value is None:
            return fallback

        if pd.isna(value):
            return fallback

        value = str(value).strip()

        if value == "":
            return fallback

        return float(value)

    except Exception:
        return fallback


def normalize_text(value, fallback="none"):
    text = (
        str(value or fallback)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    if text in ["", "nan", "none", "null", "unknown"]:
        return fallback

    return text


def parse_datetime_robust(series):
    raw_series = series.copy()

    parsed = pd.to_datetime(
        raw_series,
        errors="coerce",
        utc=True
    )

    failed_mask = (
        parsed.isna()
        &
        raw_series.notna()
    )

    if failed_mask.any():
        try:
            second_pass = pd.to_datetime(
                raw_series.loc[failed_mask],
                format="mixed",
                errors="coerce",
                utc=True
            )

        except Exception:
            second_pass = pd.to_datetime(
                raw_series.loc[failed_mask],
                errors="coerce",
                utc=True
            )

        parsed.loc[failed_mask] = second_pass

    return parsed.dt.tz_convert(None)


def default_event_features(df):
    df = df.copy()

    df["is_event_day"] = 0
    df["calendar_event_type"] = "none"
    df["calendar_event_intensity"] = 0.0

    return df


def load_event_calendar(event_calendar_path=EVENT_CALENDAR_PATH):
    path = Path(event_calendar_path)

    if not path.exists():
        return pd.DataFrame()

    calendar = pd.read_csv(
        path
    )

    required_cols = {
        "event_name": "",
        "event_type": "other",
        "start_datetime": "",
        "end_datetime": "",
        "latitude": np.nan,
        "longitude": np.nan,
        "corridor": "",
        "impact_radius_m": np.nan,
        "crowd_size": "medium",
    }

    for col, default in required_cols.items():
        if col not in calendar.columns:
            calendar[col] = default

    calendar["start_datetime"] = parse_datetime_robust(
        calendar["start_datetime"]
    )

    calendar["end_datetime"] = parse_datetime_robust(
        calendar["end_datetime"]
    )

    calendar["end_datetime"] = calendar["end_datetime"].fillna(
        calendar["start_datetime"]
    )

    calendar["latitude"] = pd.to_numeric(
        calendar["latitude"],
        errors="coerce"
    )

    calendar["longitude"] = pd.to_numeric(
        calendar["longitude"],
        errors="coerce"
    )

    calendar["event_type"] = calendar["event_type"].apply(
        lambda x: normalize_text(x, "other")
    )

    calendar["crowd_size"] = calendar["crowd_size"].apply(
        lambda x: normalize_text(x, "medium")
    )

    calendar["corridor"] = (
        calendar["corridor"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    calendar["impact_radius_m"] = calendar.apply(
        lambda row: safe_float(
            row.get("impact_radius_m"),
            DEFAULT_RADIUS_BY_CROWD.get(
                row.get("crowd_size", "medium"),
                1500
            )
        ),
        axis=1
    )

    calendar = calendar.dropna(
        subset=[
            "start_datetime"
        ]
    ).copy()

    return calendar


def calculate_event_intensity(event_type, crowd_size):
    event_type = normalize_text(
        event_type,
        "other"
    )

    crowd_size = normalize_text(
        crowd_size,
        "medium"
    )

    base = EVENT_TYPE_WEIGHTS.get(
        event_type,
        45
    )

    crowd_multiplier = CROWD_MULTIPLIERS.get(
        crowd_size,
        0.80
    )

    return max(
        0.0,
        min(base * crowd_multiplier, 100.0)
    )


def build_corridor_centroids(raw_df):
    df = raw_df.copy()

    if "latitude" not in df.columns or "longitude" not in df.columns:
        return {}

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce"
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce"
    )

    if "corridor" not in df.columns:
        return {}

    df["corridor"] = (
        df["corridor"]
        .fillna("Non-corridor")
        .astype(str)
        .str.strip()
    )

    df = df.dropna(
        subset=[
            "latitude",
            "longitude"
        ]
    )

    centroids = (
        df
        .groupby("corridor")
        .agg(
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean")
        )
        .reset_index()
    )

    output = {}

    for _, row in centroids.iterrows():
        output[str(row["corridor"])] = {
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
        }

    return output


def get_affected_corridors(event_row, corridor_centroids):
    affected = set()

    event_corridor = str(
        event_row.get("corridor", "")
    ).strip()

    if event_corridor:
        affected.add(
            event_corridor
        )

    event_lat = safe_float(
        event_row.get("latitude"),
        None
    )

    event_lon = safe_float(
        event_row.get("longitude"),
        None
    )

    if event_lat is None or event_lon is None:
        return affected

    radius_m = safe_float(
        event_row.get("impact_radius_m"),
        1500
    )

    for corridor, center in corridor_centroids.items():
        try:
            distance = haversine_distance_meters(
                event_lat,
                event_lon,
                center["latitude"],
                center["longitude"]
            )

            if distance <= radius_m:
                affected.add(
                    corridor
                )

        except Exception:
            continue

    return affected


def add_event_calendar_features_to_corridor_timeseries(
    ts_df,
    raw_df,
    event_calendar_path=EVENT_CALENDAR_PATH
):
    ts_df = default_event_features(
        ts_df
    )

    calendar = load_event_calendar(
        event_calendar_path
    )

    if calendar.empty:
        print("\nEvent calendar not found or empty. Using no-event defaults.")
        return ts_df

    corridor_centroids = build_corridor_centroids(
        raw_df
    )

    ts_df["_calendar_score"] = 0.0

    for _, event in calendar.iterrows():
        affected_corridors = get_affected_corridors(
            event,
            corridor_centroids
        )

        if not affected_corridors:
            continue

        start_hour = event["start_datetime"].floor("h")
        end_hour = event["end_datetime"].ceil("h")

        event_type = normalize_text(
            event.get("event_type"),
            "other"
        )

        intensity = calculate_event_intensity(
            event_type=event_type,
            crowd_size=event.get("crowd_size", "medium")
        )

        mask = (
            (ts_df["time_bucket"] >= start_hour)
            &
            (ts_df["time_bucket"] <= end_hour)
            &
            (ts_df["corridor"].isin(affected_corridors))
        )

        update_mask = (
            mask
            &
            (intensity >= ts_df["_calendar_score"])
        )

        ts_df.loc[update_mask, "is_event_day"] = 1
        ts_df.loc[update_mask, "calendar_event_type"] = event_type
        ts_df.loc[update_mask, "calendar_event_intensity"] = intensity
        ts_df.loc[update_mask, "_calendar_score"] = intensity

    ts_df = ts_df.drop(
        columns=[
            "_calendar_score"
        ],
        errors="ignore"
    )

    print("\nCalendar Event Feature Coverage")
    print("-" * 50)
    print("Event rows marked:", int(ts_df["is_event_day"].sum()))
    print("Event types:")
    print(ts_df["calendar_event_type"].value_counts().head(10))

    return ts_df


def add_event_calendar_features_to_spatial_timeseries(
    spatial_ts,
    cluster_centers,
    event_calendar_path=EVENT_CALENDAR_PATH
):
    spatial_ts = default_event_features(
        spatial_ts
    )

    calendar = load_event_calendar(
        event_calendar_path
    )

    if calendar.empty:
        print("\nEvent calendar not found or empty for spatial model. Using no-event defaults.")
        return spatial_ts

    spatial_ts["_calendar_score"] = 0.0

    for _, event in calendar.iterrows():
        event_lat = safe_float(
            event.get("latitude"),
            None
        )

        event_lon = safe_float(
            event.get("longitude"),
            None
        )

        if event_lat is None or event_lon is None:
            continue

        radius_m = safe_float(
            event.get("impact_radius_m"),
            1500
        )

        affected_clusters = []

        for cluster_id, center in cluster_centers.items():
            try:
                distance = haversine_distance_meters(
                    event_lat,
                    event_lon,
                    center["latitude"],
                    center["longitude"]
                )

                if distance <= radius_m:
                    affected_clusters.append(
                        int(cluster_id)
                    )

            except Exception:
                continue

        if not affected_clusters:
            continue

        start_hour = event["start_datetime"].floor("h")
        end_hour = event["end_datetime"].ceil("h")

        event_type = normalize_text(
            event.get("event_type"),
            "other"
        )

        intensity = calculate_event_intensity(
            event_type=event_type,
            crowd_size=event.get("crowd_size", "medium")
        )

        mask = (
            (spatial_ts["time_bucket"] >= start_hour)
            &
            (spatial_ts["time_bucket"] <= end_hour)
            &
            (spatial_ts["spatial_cluster_id"].isin(affected_clusters))
        )

        update_mask = (
            mask
            &
            (intensity >= spatial_ts["_calendar_score"])
        )

        spatial_ts.loc[update_mask, "is_event_day"] = 1
        spatial_ts.loc[update_mask, "calendar_event_type"] = event_type
        spatial_ts.loc[update_mask, "calendar_event_intensity"] = intensity
        spatial_ts.loc[update_mask, "_calendar_score"] = intensity

    spatial_ts = spatial_ts.drop(
        columns=[
            "_calendar_score"
        ],
        errors="ignore"
    )

    print("\nSpatial Calendar Event Feature Coverage")
    print("-" * 50)
    print("Event rows marked:", int(spatial_ts["is_event_day"].sum()))
    print("Event types:")
    print(spatial_ts["calendar_event_type"].value_counts().head(10))

    return spatial_ts