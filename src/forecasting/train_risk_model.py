import joblib

from catboost import CatBoostClassifier

from sklearn.metrics import (
    classification_report,
    f1_score
)

from sklearn.model_selection import (
    train_test_split
)


def train_risk_model(risk_df):

    features = [

        "corridor",
        "hour",
        "weekday",
        "month"

    ]

    X = risk_df[
        features
    ].copy()

    y = risk_df[
        "high_risk"
    ]

    X["corridor"] = (

        X["corridor"]
        .fillna("UNKNOWN")
        .astype(str)

    )

    X_train, X_test, y_train, y_test = (

        train_test_split(

            X,
            y,

            test_size=0.2,

            random_state=42,

            stratify=y

        )

    )

    model = CatBoostClassifier(

        iterations=500,

        depth=6,

        learning_rate=0.05,

        verbose=False

    )

    model.fit(

        X_train,
        y_train,

        cat_features=[0]

    )

    preds = model.predict(
        X_test
    )

    print("\n")
    print("=" * 60)
    print("RISK FORECAST MODEL")
    print("=" * 60)

    print(

        classification_report(
            y_test,
            preds
        )

    )

    print(
        "F1:",
        f1_score(
            y_test,
            preds
        )
    )

    joblib.dump(

        model,

        "models/risk_model.pkl"

    )

    return model