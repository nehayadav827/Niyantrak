import pandas as pd


CAUSE_WEIGHTS = {

    "vehicle_breakdown": 15,
    "others": 10,
    "pot_holes": 10,
    "construction": 20,
    "water_logging": 20,
    "accident": 25,
    "tree_fall": 25,
    "road_conditions": 15,
    "congestion": 30,
    "public_event": 30,
    "procession": 35,
    "vip_movement": 35,
    "protest": 40

}

def build_severity(df):

    priority_score = (
        df["priority"]
        .map({
            "Low": 10,
            "High": 35
        })
        .fillna(10)
    )

    closure_score = (
        df["requires_road_closure"]
        .astype(int)
        * 35
    )

    cause_score = (
        df["event_cause"]
        .map(CAUSE_WEIGHTS)
        .fillna(10)
    )

    duration_score = (
        pd.qcut(
            df["duration_minutes"],
            q=5,
            labels=False,
            duplicates="drop"
        )
        * 8
    )

    hotspot_score = (
        df.groupby("hotspot_id")["hotspot_id"]
        .transform("count")
    )

    hotspot_score = (
        hotspot_score
        /
        hotspot_score.max()
    ) * 20

    rush_score = (
        df["rush_hour"]
        * 10
    )

    severity = (
        priority_score
        + closure_score
        + cause_score
        + duration_score
        + hotspot_score
        + rush_score
    )

    severity = (
        100 *
        (severity - severity.min())
        /
        (severity.max() - severity.min())
    )

    df["severity_score"] = severity

    return df



def build_class(df):

    df["severity_class"] = pd.qcut(
        df["severity_score"],
        q=4,
        labels=[
            "LOW",
            "MODERATE",
            "HIGH",
            "CRITICAL"
        ]
    )

    return df