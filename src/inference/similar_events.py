from functools import lru_cache

import pandas as pd

from config import DATA_PATH
from src.inference.location_resolver import haversine_distance_meters


def normalize_text(value):
    return (
        str(value or "")
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


@lru_cache(maxsize=1)
def load_historical_events():
    try:
        df = pd.read_csv(
            DATA_PATH
        )

    except Exception:
        return pd.DataFrame()

    if "start_datetime" in df.columns:
        df["start_datetime"] = pd.to_datetime(
            df["start_datetime"],
            errors="coerce",
            utc=True
        ).dt.tz_convert(None)

    if "end_datetime" in df.columns:
        df["end_datetime"] = pd.to_datetime(
            df["end_datetime"],
            errors="coerce",
            utc=True
        ).dt.tz_convert(None)

    for col in [
        "corridor",
        "event_cause",
        "veh_type",
        "requires_road_closure"
    ]:
        if col not in df.columns:
            df[col] = "UNKNOWN"

        df[col] = (
            df[col]
            .fillna("UNKNOWN")
            .astype(str)
        )

    for col in [
        "latitude",
        "longitude"
    ]:
        if col not in df.columns:
            df[col] = None

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    if (
        "start_datetime" in df.columns
        and
        "end_datetime" in df.columns
    ):
        duration = (
            df["end_datetime"]
            -
            df["start_datetime"]
        ).dt.total_seconds() / 60

        df["duration_minutes"] = duration.fillna(0)

    else:
        df["duration_minutes"] = 0

    df["hour"] = (
        df["start_datetime"]
        .dt.hour
        .fillna(0)
        .astype(int)
        if "start_datetime" in df.columns
        else 0
    )

    return df


def hour_distance(a, b):
    diff = abs(int(a) - int(b))

    return min(
        diff,
        24 - diff
    )


def find_similar_events(
    event_cause,
    corridor,
    latitude,
    longitude,
    hour,
    limit=3
):
    df = load_historical_events()

    if df.empty:
        return []

    cause_key = normalize_text(
        event_cause
    )

    corridor_key = normalize_text(
        corridor
    )

    candidates = df.copy()

    candidates["cause_match"] = (
        candidates["event_cause"]
        .map(normalize_text)
        ==
        cause_key
    )

    candidates["corridor_match"] = (
        candidates["corridor"]
        .map(normalize_text)
        ==
        corridor_key
    )

    candidates["hour_gap"] = candidates["hour"].apply(
        lambda h: hour_distance(h, hour)
    )

    distance_values = []

    for _, row in candidates.iterrows():
        try:
            if pd.isna(row["latitude"]) or pd.isna(row["longitude"]):
                distance_values.append(999999.0)
            else:
                distance_values.append(
                    haversine_distance_meters(
                        latitude,
                        longitude,
                        row["latitude"],
                        row["longitude"]
                    )
                )

        except Exception:
            distance_values.append(999999.0)

    candidates["distance_m"] = distance_values

    candidates["similarity_score"] = (
        candidates["cause_match"].astype(int) * 45
        +
        candidates["corridor_match"].astype(int) * 30
        +
        candidates["hour_gap"].apply(
            lambda x: max(0, 15 - (x * 4))
        )
        +
        candidates["distance_m"].apply(
            lambda d: max(0, 10 - (d / 500))
        )
    )

    candidates = candidates.sort_values(
        by=[
            "similarity_score",
            "distance_m"
        ],
        ascending=[
            False,
            True
        ]
    )

    output = []

    for _, row in candidates.head(limit).iterrows():
        start_value = row.get(
            "start_datetime",
            ""
        )

        if pd.isna(start_value):
            start_text = "Unknown"
        else:
            start_text = str(start_value)

        output.append({
            "event_cause": str(row.get("event_cause", "UNKNOWN")),
            "corridor": str(row.get("corridor", "UNKNOWN")),
            "hour": int(row.get("hour", 0)),
            "start_datetime": start_text,
            "distance_m": round(float(row.get("distance_m", 0)), 1),
            "duration_minutes": round(float(row.get("duration_minutes", 0)), 1),
            "road_closure": str(row.get("requires_road_closure", "UNKNOWN")),
            "similarity_score": round(float(row.get("similarity_score", 0)), 1),
        })

    return output