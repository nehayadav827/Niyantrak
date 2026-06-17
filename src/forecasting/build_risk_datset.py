import pandas as pd


def build_risk_dataset(df):

    # =====================================================
    # DATETIME
    # =====================================================

    df["start_datetime"] = pd.to_datetime(
        df["start_datetime"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["start_datetime"]
    )

    df["hour"] = df["start_datetime"].dt.hour
    df["weekday"] = df["start_datetime"].dt.weekday
    df["month"] = df["start_datetime"].dt.month

    # =====================================================
    # CLEAN COLUMNS
    # =====================================================

    df["corridor"] = (
        df["corridor"]
        .fillna("UNKNOWN")
        .astype(str)
    )

    df["event_cause"] = (
        df["event_cause"]
        .fillna("UNKNOWN")
        .astype(str)
    )

    df["requires_road_closure"] = (
        df["requires_road_closure"]
        .fillna(False)
    )

    # =====================================================
    # BUILD RISK DATASET
    # =====================================================

    risk_df = (
        df.groupby(
            [
                "corridor",
                "hour",
                "weekday",
                "month",
                "event_cause",
                "requires_road_closure"
            ]
        )
        .size()
        .reset_index(name="incident_count")
    )

    # =====================================================
    # CORRIDOR RISK
    # =====================================================

    corridor_risk = (
        risk_df.groupby("corridor")
        ["incident_count"]
        .mean()
    )

    risk_df["corridor_risk"] = (
        risk_df["corridor"]
        .map(corridor_risk)
    )

    # =====================================================
    # EVENT CAUSE RISK
    # =====================================================

    cause_risk = (
        risk_df.groupby("event_cause")
        ["incident_count"]
        .mean()
    )

    risk_df["cause_risk"] = (
        risk_df["event_cause"]
        .map(cause_risk)
    )

    # =====================================================
    # FEATURE ORDER
    # =====================================================

    risk_df = risk_df[

        [
            "corridor",
            "event_cause",
            "requires_road_closure",

            "hour",
            "weekday",
            "month",

            "corridor_risk",
            "cause_risk",

            "incident_count"
        ]

    ]

    print("\n" + "=" * 60)
    print("RISK DATASET")
    print("=" * 60)

    print(risk_df.shape)

    print("\nColumns:")
    print(risk_df.columns.tolist())

    print("\nDtypes:")
    print(risk_df.dtypes)

    return risk_df