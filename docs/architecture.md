# Architecture

## System Pipeline

```text
Raw Traffic Event Data
        ↓
Data Cleaning and Timestamp Normalization
        ↓
Corridor-Hour Time-Series Dataset
        ↓
Lag / Rolling / Spatial Risk Feature Engineering
        ↓
Zero-Inflated CatBoost Hurdle Model
        ↓
Coordinate-Aware Feature Store
        ↓
Location Resolver
        ↓
KMeans Spatial Cluster Fallback
        ↓
Forecast Risk Score
        ↓
Event Impact Score
        ↓
Crowd + Weather Adjustment
        ↓
Calibrated EIS Score
        ↓
Final Operational Risk
        ↓
Map Impact Circles + Deployment Recommendation
        ↓
Feedback Collection + Validation Evidence
```

## Why Coordinate-First

Earlier versions of this system depended on the user manually selecting a corridor name. That approach was weak because real incidents can happen:

- between corridors
- near service roads
- near junctions
- near temporary event venues
- at unknown or new locations

The system was upgraded to work directly from coordinates instead:

- latitude
- longitude
- nearest corridor
- nearest hotspot
- nearest spatial cluster
- spatial density
- cluster-hour fallback profile

The user now clicks on the map or enters coordinates directly. The backend infers the corridor internally — there is no need to know corridor names in advance.

## Main User Flow

1. User opens the dashboard.
2. User clicks an event location on the map.
3. Latitude and longitude are auto-filled.
4. User enters event details (cause, vehicle type, road closure, crowd size, weather).
5. Backend validates the coordinates against Bengaluru bounds.
6. Backend infers the nearest corridor and spatial cluster.
7. The feature store supplies historical lag/rolling features for that location and hour.
8. The hurdle model predicts incident risk.
9. The event scoring layer calculates live impact.
10. The EIS layer combines forecast risk, live event impact, and historical cause risk.
11. The dashboard shows the final decision: risk, affected area, deployment, and diversion.

## What Is ML vs. Statistical vs. Rule-Based

Niyantrak is intentionally a hybrid system. Traffic operations need explainable recommendations, not only black-box predictions.

**Machine Learning**
- `CatBoostClassifier` (alert/incident-likelihood classifier)
- `CatBoostRegressor` (positive incident-count regressor)
- CatBoost quantile regressors (prediction intervals)
- KMeans spatial clustering
- Time-series cross-validation
- Feature importance analysis

**Statistical Feature Store**
- Lag features
- Rolling features
- Corridor-hour profiles
- Cluster-hour profiles
- Risk percentiles
- Spatial density
- Hotspot points

**Rule-Based Operational Intelligence**
- Event cause weights
- Vehicle type weights
- Road closure boost
- Crowd multiplier
- Weather multiplier
- EIS formula
- Resource allocation rules
- Affected radius rules
- Diversion graph
- Deployment order generation
- Feedback storage

## Project Structure

```text
config.py
train_all.py
predict.py
prepare_feature_store.py

src/
    preprocessing/
        load_data.py

    forecasting/
        build_timeseries_dataset.py
        cross_validate_timeseries.py
        train_timeseries_model.py
        forecast_predictor.py
        forecast_feature_importance.py
        train_quantile_intervals.py

    inference/
        feature_store.py
        location_resolver.py
        predict_traffic_risk.py
        similar_events.py

    scoring/
        event_impact.py
        risk_score.py

    routing/
        diversion_engine.py

    evaluation/
        cluster_fallback_ablation.py
        eis_weight_calibration.py

dashboard/
    services/
        ml_engine.py
        feedback_store.py

    templates/
        dashboard/
            index.html

    static/
        dashboard/
            style.css

models/
    timeseries_forecast_model.pkl
    traffic_feature_store.pkl
    cluster_fallback_ablation.json
    eis_weight_calibration.json
```

> The live repository's top level also contains `traffic_web/`, `main.py`, `risk_main.py`, `risk_main_v2.py`, and `check.py`, which are not yet reflected in the structure above. These appear to be either an alternate/legacy entry point for the Django app or earlier iterations of the risk pipeline. Until confirmed, treat `train_all.py`, `predict.py`, and `manage.py` (with the Django app, whichever of `dashboard/`/`traffic_web/` is current) as the canonical entry points — see [Setup & Usage](setup-and-usage.md).

## Related Docs

- [Data & Features](data-and-features.md) — what feeds into the model
- [Forecasting Model](forecasting-model.md) — how the hurdle model works
- [Location Intelligence](location-intelligence.md) — how coordinates become corridors
- [Event Impact Scoring](event-impact-scoring.md) — how EIS is calculated
- [Operational Outputs](operational-outputs.md) — what the system recommends