import pandas as pd
import joblib


def predict_risk():

    model = joblib.load(
        "models/risk_forecast.pkl"
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

    sample = pd.DataFrame([{

        "corridor": corridor,
        "hour": hour,
        "weekday": weekday,
        "month": month

    }])

    predicted_count = model.predict(
        sample
    )[0]

    risk_score = min(
        predicted_count / 70,
        1
    )

    risk_pct = risk_score * 100

    if risk_pct < 25:
        level = "LOW"

    elif risk_pct < 50:
        level = "MODERATE"

    elif risk_pct < 75:
        level = "HIGH"

    else:
        level = "CRITICAL"

    print("\n")
    print("=" * 50)

    print(
        f"Expected Incidents : {predicted_count:.1f}"
    )

    print(
        f"Risk Score : {risk_pct:.1f}%"
    )

    print(
        f"Risk Level : {level}"
    )

    print("=" * 50)