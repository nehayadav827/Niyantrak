import pandas as pd

# CHANGE PATH IF NEEDED
DATA_PATH = "data.csv"

df = pd.read_csv(DATA_PATH)

df["start_datetime"] = pd.to_datetime(
    df["start_datetime"],
    errors="coerce"
)

df = df.dropna(subset=["start_datetime"])

risk_df = (
    df.groupby(
        [
            "corridor",
            df["start_datetime"].dt.hour
        ]
    )
    .size()
    .reset_index(name="count")
)

print("\n")
print("=" * 60)
print("RISK DATASET DISTRIBUTION")
print("=" * 60)

print(risk_df["count"].describe())

print("\n")
print("=" * 60)
print("TOP 20")
print("=" * 60)

print(
    risk_df.sort_values(
        "count",
        ascending=False
    ).head(20)
)