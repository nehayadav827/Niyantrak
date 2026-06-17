import pandas as pd
import numpy as np


def load_data(path):
    return pd.read_csv(path)


def clean_dates(df):

    cols = [
        "start_datetime",
        "end_datetime",
        "modified_datetime",
        "created_date"
    ]

    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col],
                errors="coerce"
            )

    return df


def clean_coordinates(df):

    df["endlatitude"] = (
        df["endlatitude"]
        .fillna(df["latitude"])
    )

    df["endlongitude"] = (
        df["endlongitude"]
        .fillna(df["longitude"])
    )

    return df


def clean_duration(df):

    effective_end = (
        df["end_datetime"]
        .fillna(df["modified_datetime"])
    )

    df["duration_minutes"] = (
        effective_end
        - df["start_datetime"]
    ).dt.total_seconds() / 60

    df.loc[
        (df["duration_minutes"] < 0)
        |
        (df["duration_minutes"] > 4320),
        "duration_minutes"
    ] = np.nan

    median_duration = (
        df["duration_minutes"]
        .median()
    )

    df["duration_minutes"] = (
        df["duration_minutes"]
        .fillna(median_duration)
    )

    return df


def remove_leakage(df):

    leakage_cols = [

        "status",

        "closed_datetime",
        "closed_by_id",

        "resolved_datetime",
        "resolved_by_id",

        "resolved_at_address",
        "resolved_at_latitude",
        "resolved_at_longitude"

    ]

    existing = [
        c
        for c in leakage_cols
        if c in df.columns
    ]

    return df.drop(
        columns=existing
    )