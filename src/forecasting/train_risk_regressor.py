import joblib
import pandas as pd

from catboost import CatBoostRegressor

from sklearn.metrics import (
    mean_absolute_error,
    r2_score
)

from sklearn.model_selection import (
    train_test_split
)


def train_risk_regressor(risk_df):

    print("\nPreparing Risk Forecast Features...")

    features = [
        "corridor",

        "hour",
        "weekday",
        "month",
        "is_event_day",
        "calendar_event_type",
        "calendar_event_intensity",

        "hour_sin",
        "hour_cos",

        "lag_1",
        "lag_2",
        "lag_3",
        "lag_24",
        "lag_48",
        "lag_72",
        "lag_168",

        "any_incident_last_3h",
        "incidents_last_24h",
        "above_corridor_avg",

        "rolling_6",
        "rolling_12",
        "rolling_24",
        "rolling_168",

        "corridor_avg",
        "corridor_volatility",

        "zone_risk",
        "junction_risk",
        "cause_risk",
        "closure_risk",
        "cluster_risk",
    ]

    X = risk_df[features].copy()

    y = risk_df["incident_count"]

    # =====================================================
    # CLEAN CATEGORICAL FEATURES
    # =====================================================

    categorical_cols = [

        "corridor",
        "event_cause"

    ]

    for col in categorical_cols:

        X[col] = (
            X[col]
            .fillna("UNKNOWN")
            .astype(str)
        )

    # =====================================================
    # BOOLEAN -> INT
    # =====================================================

    X["requires_road_closure"] = (
        X["requires_road_closure"]
        .fillna(False)
        .astype(int)
    )

    # =====================================================
    # NUMERIC FEATURES
    # =====================================================

    numeric_cols = [

        "hour",
        "weekday",
        "month",

        "corridor_risk",
        "cause_risk"

    ]

    for col in numeric_cols:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        ).fillna(0)

    # =====================================================
    # SPLIT
    # =====================================================

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.2,
        random_state=42

    )

    # =====================================================
    # CATBOOST
    # =====================================================

    model = CatBoostRegressor(

        iterations=2000,
        depth=10,
        learning_rate=0.03,

        loss_function="RMSE",

        random_seed=42,

        verbose=200

    )
    print("\nFeature Columns:")
    print(X.columns.tolist())

    print("\nDtypes:")
    print(X.dtypes)

    model.fit(
        X_train,
        y_train,
        cat_features=[
            "corridor",
            "event_cause"
        ]
    )

    preds = model.predict(X_test)

    print("\n")
    print("=" * 60)
    print("RISK FORECAST MODEL")
    print("=" * 60)

    print(
        "MAE:",
        round(
            mean_absolute_error(
                y_test,
                preds
            ),
            4
        )
    )

    print(
        "R²:",
        round(
            r2_score(
                y_test,
                preds
            ),
            4
        )
    )

    joblib.dump(
        model,
        "models/risk_forecast.pkl"
    )

    print("\nRisk Model Saved")

    return model