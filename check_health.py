import os
import sys
import traceback
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent

PASSED = []
WARNINGS = []
FAILED = []


def ok(message):
    PASSED.append(message)
    print(f"[OK] {message}")


def warn(message):
    WARNINGS.append(message)
    print(f"[WARN] {message}")


def fail(message):
    FAILED.append(message)
    print(f"[FAIL] {message}")


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


REQUIRED_FILES = [
    "config.py",
    "train_all.py",
    "prepare_feature_store.py",
    "predict.py",

    "src/preprocessing/load_data.py",

    "src/forecasting/build_timeseries_dataset.py",
    "src/forecasting/cross_validate_timeseries.py",
    "src/forecasting/train_timeseries_model.py",
    "src/forecasting/forecast_predictor.py",
    "src/forecasting/forecast_feature_importance.py",

    "src/inference/feature_store.py",
    "src/inference/predict_traffic_risk.py",

    "src/scoring/event_impact.py",
    "src/scoring/risk_score.py",

    "src/routing/diversion_engine.py",

    "dashboard/services/ml_engine.py",
    "dashboard/templates/dashboard/index.html",
    "dashboard/static/dashboard/style.css",

    "traffic_web/settings.py",
    "traffic_web/urls.py",
    "manage.py",
]

OPTIONAL_FILES = [
    "src/training/train_catboost.py",
    "src/training/cross_validation.py",
    "src/training/feature_importance.py",
    "src/training/shap_analysis.py",

    "src/preprocessing/create_target.py",
    "src/preprocessing/feature_engineering.py",
    "src/preprocessing/advanced_features.py",
]

MODEL_FILES = [
    "models/timeseries_forecast_model.pkl",
    "models/timeseries_forecast.pkl",
    "models/traffic_feature_store.pkl",
]

FORECAST_FEATURES = [
    "corridor",

    "hour",
    "weekday",
    "month",

    "hour_sin",
    "hour_cos",

    "lag_1",
    "lag_2",
    "lag_3",
    "lag_24",
    "lag_48",
    "lag_72",
    "lag_168",

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


def check_python_version():
    section("PYTHON VERSION CHECK")

    version = sys.version_info

    print(
        f"Python version: "
        f"{version.major}.{version.minor}.{version.micro}"
    )

    if version.major == 3 and version.minor in [10, 11, 12]:
        ok("Python version is acceptable.")
    else:
        warn(
            "Recommended Python version is 3.10 or 3.11. "
            "Some ML libraries may fail on very new Python versions."
        )


def check_required_files():
    section("PROJECT FILE CHECK")

    for file_path in REQUIRED_FILES:
        path = ROOT / file_path

        if path.exists():
            ok(f"Found {file_path}")
        else:
            fail(f"Missing required file: {file_path}")

    for file_path in OPTIONAL_FILES:
        path = ROOT / file_path

        if path.exists():
            ok(f"Found optional file: {file_path}")
        else:
            warn(f"Optional old pipeline file missing: {file_path}")


def check_imports():
    section("PACKAGE IMPORT CHECK")

    packages = [
        "pandas",
        "numpy",
        "sklearn",
        "catboost",
        "joblib",
        "networkx",
        "django",
        "matplotlib",
    ]

    for package in packages:
        try:
            __import__(package)
            ok(f"Package import works: {package}")

        except Exception as e:
            fail(f"Package import failed: {package} -> {e}")


def load_config():
    section("CONFIG CHECK")

    try:
        import config

        ok("config.py imported successfully")

        data_path = getattr(
            config,
            "DATA_PATH",
            None
        )

        if data_path is None:
            fail("DATA_PATH missing in config.py")
        else:
            print(f"DATA_PATH = {data_path}")

            full_data_path = ROOT / data_path

            if full_data_path.exists():
                ok(f"Dataset path exists: {data_path}")
            else:
                fail(f"Dataset path does not exist: {data_path}")

        return config

    except Exception as e:
        fail(f"Failed to import config.py: {e}")
        return None


def check_dataset(config):
    section("DATASET CHECK")

    if config is None:
        fail("Skipping dataset check because config failed.")
        return None

    data_path = getattr(
        config,
        "DATA_PATH",
        None
    )

    if data_path is None:
        fail("DATA_PATH missing.")
        return None

    full_path = ROOT / data_path

    if not full_path.exists():
        fail(f"Dataset not found: {full_path}")
        return None

    try:
        file_path = str(full_path).lower()

        if file_path.endswith(".csv"):
            df = pd.read_csv(full_path)

        elif file_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(full_path)

        elif file_path.endswith(".parquet"):
            df = pd.read_parquet(full_path)

        else:
            fail("Unsupported dataset format.")
            return None

        ok("Dataset loaded successfully")

        print("Dataset shape:", df.shape)
        print("Columns:", list(df.columns))

        required_cols = [
            "start_datetime",
            "corridor",
            "latitude",
            "longitude",
            "zone",
            "junction",
            "event_cause",
            "requires_road_closure",
            "veh_type",
        ]

        missing_cols = [
            col
            for col in required_cols
            if col not in df.columns
        ]

        if missing_cols:
            fail(f"Dataset missing columns: {missing_cols}")
        else:
            ok("Dataset has required columns.")

        if "start_datetime" in df.columns:
            try:
                dt = pd.to_datetime(
                    df["start_datetime"],
                    errors="coerce",
                    utc=True
                ).dt.tz_convert(None)

                bad_count = dt.isna().sum()

                print("Invalid datetime rows:", bad_count)

                if bad_count == len(df):
                    fail("All start_datetime values failed parsing.")
                elif bad_count > 0:
                    warn(f"{bad_count} rows have invalid start_datetime.")
                else:
                    ok("Datetime parsing is safe with utc=True.")

            except Exception as e:
                fail(f"Datetime parsing failed: {e}")

        return df

    except Exception as e:
        fail(f"Dataset loading failed: {e}")
        traceback.print_exc()
        return None


def check_model_files():
    section("MODEL FILE CHECK")

    for file_path in MODEL_FILES:
        path = ROOT / file_path

        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)

            ok(
                f"Found {file_path} "
                f"({size_mb:.2f} MB)"
            )
        else:
            fail(
                f"Missing model artifact: {file_path}. "
                "Run python train_all.py or python prepare_feature_store.py"
            )


def check_feature_store():
    section("FEATURE STORE CHECK")

    store_path = ROOT / "models" / "traffic_feature_store.pkl"

    if not store_path.exists():
        fail("traffic_feature_store.pkl missing.")
        return None

    try:
        store = joblib.load(store_path)

        ok("Feature store loaded successfully.")

        expected_keys = [
            "features",
            "profile_features",
            "corridor_hour_profiles",
            "corridor_profiles",
            "global_profile",
            "incident_p50",
            "incident_p75",
            "incident_p90",
            "incident_p95",
            "incident_p99",
            "corridors",
        ]

        for key in expected_keys:
            if key in store:
                ok(f"Feature store key exists: {key}")
            else:
                fail(f"Feature store missing key: {key}")

        corridors = store.get("corridors", [])

        print("Total corridors:", len(corridors))
        print("Sample corridors:", corridors[:10])

        if len(corridors) == 0:
            fail("Feature store has no corridors.")
        else:
            ok("Feature store has corridor profiles.")

        print("Incident thresholds:")
        print("P50:", store.get("incident_p50"))
        print("P75:", store.get("incident_p75"))
        print("P90:", store.get("incident_p90"))
        print("P95:", store.get("incident_p95"))
        print("P99:", store.get("incident_p99"))

        return store

    except Exception as e:
        fail(f"Feature store load failed: {e}")
        traceback.print_exc()
        return None


def check_forecast_model():
    section("FORECAST MODEL CHECK")

    model_path = ROOT / "models" / "timeseries_forecast_model.pkl"

    if not model_path.exists():
        fail("timeseries_forecast_model.pkl missing.")
        return None

    try:
        model = joblib.load(model_path)

        ok("Forecast model loaded successfully.")

        if hasattr(model, "predict"):
            ok("Model supports .predict(X)")
        else:
            fail(
                "Model does NOT support .predict(X). "
                "Bug 2 still exists. Add HurdleModelBundle wrapper."
            )

        if isinstance(model, dict):
            print("Model type:", model.get("model_type"))

            if model.get("model_type") == "zero_inflated_hurdle_v1":
                ok("Detected zero-inflated hurdle model bundle.")

                for key in [
                    "classifier",
                    "regressor",
                    "features",
                    "cat_features",
                    "alert_threshold",
                    "positive_count_mean",
                ]:
                    if key in model:
                        ok(f"Model bundle key exists: {key}")
                    else:
                        fail(f"Model bundle missing key: {key}")

                print("Alert threshold:", model.get("alert_threshold"))
                print("Positive count mean:", model.get("positive_count_mean"))

        else:
            warn(
                "Model is not a dict-like hurdle bundle. "
                "It may be an older single regressor."
            )

        return model

    except Exception as e:
        fail(f"Model load failed: {e}")
        traceback.print_exc()
        return None


def build_sample_input(store):
    if store is None:
        return None

    corridors = store.get("corridors", [])

    if "ORR East 1" in corridors:
        corridor = "ORR East 1"
    elif "CBD 1" in corridors:
        corridor = "CBD 1"
    elif len(corridors) > 0:
        corridor = corridors[0]
    else:
        corridor = "UNKNOWN"

    hour = 9
    weekday = 1
    month = 2

    profile = store.get("global_profile", {})

    row = {}

    import numpy as np

    for feature in FORECAST_FEATURES:
        if feature == "corridor":
            row[feature] = corridor

        elif feature == "hour":
            row[feature] = hour

        elif feature == "weekday":
            row[feature] = weekday

        elif feature == "month":
            row[feature] = month

        elif feature == "hour_sin":
            row[feature] = np.sin(
                2 * np.pi * hour / 24
            )

        elif feature == "hour_cos":
            row[feature] = np.cos(
                2 * np.pi * hour / 24
            )

        else:
            row[feature] = profile.get(
                feature,
                0.0
            )

    return pd.DataFrame(
        [row],
        columns=FORECAST_FEATURES
    )


def check_sample_prediction(model, store):
    section("SAMPLE MODEL PREDICTION CHECK")

    if model is None:
        fail("Skipping prediction check because model failed.")
        return

    if store is None:
        fail("Skipping prediction check because feature store failed.")
        return

    try:
        X = build_sample_input(store)

        print("Sample input:")
        print(X)

        preds = model.predict(X)

        print("Prediction:", preds)

        if len(preds) == 0:
            fail("Prediction returned empty output.")
        else:
            ok("Sample model.predict(X) works.")

        try:
            from src.forecasting.forecast_predictor import (
                predict_single_forecast
            )

            predicted_incidents, details = predict_single_forecast(
                model,
                X
            )

            print("predict_single_forecast output:")
            print("Predicted incidents:", predicted_incidents)
            print("Details keys:", list(details.keys()))

            ok("predict_single_forecast works.")

        except Exception as e:
            fail(f"predict_single_forecast failed: {e}")
            traceback.print_exc()

    except Exception as e:
        fail(f"Sample prediction failed: {e}")
        traceback.print_exc()


def check_main_imports():
    section("PROJECT MODULE IMPORT CHECK")

    imports = [
        "src.preprocessing.load_data",
        "src.forecasting.build_timeseries_dataset",
        "src.forecasting.cross_validate_timeseries",
        "src.forecasting.train_timeseries_model",
        "src.forecasting.forecast_predictor",
        "src.forecasting.forecast_feature_importance",
        "src.inference.feature_store",
        "src.inference.predict_traffic_risk",
        "src.scoring.event_impact",
        "src.scoring.risk_score",
        "src.routing.diversion_engine",
    ]

    for module_name in imports:
        try:
            __import__(module_name)
            ok(f"Module import works: {module_name}")
        except Exception as e:
            fail(f"Module import failed: {module_name} -> {e}")


def check_django_project():
    section("DJANGO PROJECT CHECK")

    try:
        os.environ.setdefault(
            "DJANGO_SETTINGS_MODULE",
            "traffic_web.settings"
        )

        import django

        django.setup()

        ok("Django setup works.")

        from django.conf import settings

        print("DEBUG:", settings.DEBUG)

        if "dashboard" in settings.INSTALLED_APPS:
            ok("Dashboard app registered.")
        else:
            fail("Dashboard app missing from INSTALLED_APPS.")

    except Exception as e:
        fail(f"Django setup failed: {e}")
        traceback.print_exc()


def check_common_bug_patterns():
    section("COMMON BUG PATTERN CHECK")

    build_ts_path = ROOT / "src" / "forecasting" / "build_timeseries_dataset.py"

    if build_ts_path.exists():
        text = build_ts_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        if "utc=True" in text and (
            "tz_convert(None)" in text
            or
            "tz_localize(None)" in text
        ):
            ok("Bug 1 protection found in build_timeseries_dataset.py")
        else:
            warn(
                "Bug 1 protection may be missing in build_timeseries_dataset.py. "
                "Use pd.to_datetime(..., utc=True).dt.tz_convert(None)."
            )

    forecast_predictor_path = ROOT / "src" / "forecasting" / "forecast_predictor.py"

    if forecast_predictor_path.exists():
        text = forecast_predictor_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        if "class HurdleModelBundle" in text and "def predict" in text:
            ok("Bug 2 wrapper exists: HurdleModelBundle with .predict()")
        else:
            warn(
                "Bug 2 wrapper may be missing. "
                "forecast_predictor.py should define HurdleModelBundle with .predict()."
            )

    train_catboost_path = ROOT / "src" / "training" / "train_catboost.py"

    if train_catboost_path.exists():
        warn(
            "train_catboost.py exists but is optional in current main flow. "
            "Ignore unless you run old severity classifier pipeline."
        )


def print_summary():
    section("HEALTH CHECK SUMMARY")

    print(f"Passed  : {len(PASSED)}")
    print(f"Warnings: {len(WARNINGS)}")
    print(f"Failed  : {len(FAILED)}")

    if FAILED:
        print("\nFAILED CHECKS:")
        for item in FAILED:
            print("-", item)

    if WARNINGS:
        print("\nWARNINGS:")
        for item in WARNINGS:
            print("-", item)

    if not FAILED:
        print("\nSTATUS: PROJECT HEALTH CHECK PASSED")
    else:
        print("\nSTATUS: PROJECT HAS ISSUES TO FIX")


def main():
    print("\nTRAFFIC INTELLIGENCE PROJECT HEALTH CHECK")
    print("Root:", ROOT)

    check_python_version()
    check_required_files()
    check_imports()

    config = load_config()

    check_dataset(config)
    check_model_files()

    store = check_feature_store()
    model = check_forecast_model()

    check_main_imports()
    check_sample_prediction(model, store)
    check_django_project()
    check_common_bug_patterns()

    print_summary()


if __name__ == "__main__":
    main()