import numpy as np
import pandas as pd


def normalize_text(value):

    value = str(value).strip().lower()

    value = value.replace("-", "_")
    value = value.replace(" ", "_")

    while "__" in value:
        value = value.replace("__", "_")

    return value


def create_target(df):

    print("\nCreating Severity Target...")

    df = df.copy()

    # =====================================================
    # REQUIRED COLUMN SAFETY
    # =====================================================

    required_cols = [
        "event_cause",
        "requires_road_closure",
        "corridor",
        "latitude",
        "longitude",
        "start_datetime"
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            "Missing required columns for target creation: "
            + str(missing_cols)
        )

    # =====================================================
    # CLEAN BASIC COLUMNS
    # =====================================================

    df["event_cause"] = (
        df["event_cause"]
        .fillna("others")
        .apply(normalize_text)
    )

    df["corridor"] = (
        df["corridor"]
        .fillna("UNKNOWN")
        .astype(str)
    )

    df["requires_road_closure"] = (
        df["requires_road_closure"]
        .fillna(False)
        .astype(bool)
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

    # =====================================================
    # BASE SCORE
    # =====================================================

    score = np.zeros(
        len(df),
        dtype=float
    )

    # =====================================================
    # EVENT CAUSE WEIGHT
    # =====================================================

    cause_weights = {
        "accident": 90,
        "vip_movement": 95,
        "protest": 90,
        "public_event": 80,
        "procession": 80,
        "construction": 75,
        "tree_fall": 75,
        "water_logging": 70,
        "congestion": 65,
        "road_conditions": 60,
        "pot_holes": 50,
        "vehicle_breakdown": 45,
        "others": 40,
        "debris": 50,
        "fog_low_visibility": 60,
        "test_demo": 20
    }

    score += (
        df["event_cause"]
        .map(cause_weights)
        .fillna(40)
        .to_numpy()
    )

    # =====================================================
    # ROAD CLOSURE IMPACT
    # =====================================================

    score += np.where(
        df["requires_road_closure"],
        20,
        0
    )

    # =====================================================
    # CORRIDOR FREQUENCY IMPACT
    # =====================================================

    corridor_freq = (
        df["corridor"]
        .value_counts(normalize=True)
    )

    corridor_score = (
        df["corridor"]
        .map(corridor_freq)
        .fillna(0)
        .to_numpy()
        * 30
    )

    score += corridor_score

    # =====================================================
    # HOTSPOT IMPACT
    # =====================================================

    df["lat_round"] = (
        df["latitude"]
        .round(3)
    )

    df["lon_round"] = (
        df["longitude"]
        .round(3)
    )

    hotspot_counts = (
        df.groupby(
            [
                "lat_round",
                "lon_round"
            ]
        )
        .size()
    )

    hotspot_map = hotspot_counts.to_dict()

    hotspot_score = []

    for lat, lon in zip(
        df["lat_round"],
        df["lon_round"]
    ):

        hotspot_score.append(
            hotspot_map.get(
                (lat, lon),
                1
            )
        )

    score += np.array(
        hotspot_score,
        dtype=float
    )

    # =====================================================
    # RUSH HOUR IMPACT
    # =====================================================

    dt = pd.to_datetime(
        df["start_datetime"],
        errors="coerce"
    )

    hours = (
        dt.dt.hour
        .fillna(0)
        .astype(int)
    )

    rush = (
        ((hours >= 7) & (hours <= 10))
        |
        ((hours >= 17) & (hours <= 21))
    )

    score += np.where(
        rush,
        15,
        0
    )

    # =====================================================
    # NORMALIZE SCORE SAFELY
    # =====================================================

    score_min = np.min(score)
    score_max = np.max(score)

    if score_max == score_min:

        normalized_score = np.zeros_like(
            score,
            dtype=float
        )

    else:

        normalized_score = (
            (score - score_min)
            /
            (score_max - score_min)
        ) * 100

    df["severity_score"] = normalized_score

    # =====================================================
    # BALANCED CLASSES
    # =====================================================

    try:

        df["severity_class"] = pd.qcut(
            df["severity_score"],
            q=4,
            labels=[
                "LOW",
                "MODERATE",
                "HIGH",
                "CRITICAL"
            ],
            duplicates="drop"
        )

        # If qcut dropped bins, fallback to fixed bins
        if df["severity_class"].isna().any():

            raise ValueError(
                "qcut produced NaN labels"
            )

    except Exception:

        df["severity_class"] = pd.cut(
            df["severity_score"],
            bins=[
                -1,
                25,
                50,
                75,
                100
            ],
            labels=[
                "LOW",
                "MODERATE",
                "HIGH",
                "CRITICAL"
            ]
        )

    df["severity_class"] = (
        df["severity_class"]
        .astype(str)
    )

    print("\nSeverity Distribution")
    print(
        df["severity_class"]
        .value_counts()
    )

    print("\nSample Severity")
    print(
        df[
            [
                "event_cause",
                "severity_score",
                "severity_class"
            ]
        ]
        .head()
    )

    return df