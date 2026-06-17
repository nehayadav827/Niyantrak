import joblib
import pandas as pd


def predict_future_risk():

    model = joblib.load(
        "models/timeseries_forecast.pkl"
    )

    corridor = input(
        "\nCorridor: "
    )

    hour = int(
        input("Hour: ")
    )

    weekday = int(
        input("Weekday (0-6): ")
    )

    month = int(
        input("Month: ")
    )

    lag_1 = float(
        input("Last Hour Count: ")
    )

    lag_2 = float(
        input("2 Hours Ago Count: ")
    )

    lag_3 = float(
        input("3 Hours Ago Count: ")
    )

    rolling_3 = (
        lag_1 +
        lag_2 +
        lag_3
    ) / 3

    rolling_6 = rolling_3

    sample = pd.DataFrame(

        [{
            "corridor": corridor,

            "hour": hour,
            "weekday": weekday,
            "month": month,

            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_3": lag_3,

            "rolling_3": rolling_3,
            "rolling_6": rolling_6
        }]

    )

    pred = model.predict(sample)[0]

    print("\n")
    print("=" * 60)
    print("NEXT HOUR FORECAST")
    print("=" * 60)

    print(
        f"Expected Incidents: {pred:.2f}"
    )