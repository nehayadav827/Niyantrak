import os
import pandas as pd


REQUIRED_COLUMNS = [
    "start_datetime",
    "corridor",
    "latitude",
    "longitude",
    "zone",
    "junction",
    "event_cause",
    "requires_road_closure",
    "veh_type"
]


def load_data(file_path):

    print("\nLoading Dataset...")

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"Dataset not found at: {file_path}"
        )

    file_path_lower = file_path.lower()

    if file_path_lower.endswith(".csv"):

        df = pd.read_csv(
            file_path
        )

    elif (
        file_path_lower.endswith(".xlsx")
        or
        file_path_lower.endswith(".xls")
    ):

        df = pd.read_excel(
            file_path
        )

    elif file_path_lower.endswith(".parquet"):

        df = pd.read_parquet(
            file_path
        )

    else:

        raise ValueError(
            "Unsupported file format. Use CSV, Excel, or Parquet."
        )

    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    missing_required = [
        col
        for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_required:

        print("\nWARNING: Missing expected columns:")
        print(missing_required)

    else:

        print("\nRequired columns check: OK")

    return df