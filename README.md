# Intelligent Event-Driven Traffic Management System

An AI-powered traffic intelligence prototype that predicts event-driven traffic impact, estimates operational risk, recommends manpower and barricading plans, and suggests diversion corridors using machine learning, historical corridor-hour patterns, and live event details.

---

## 1. Project Overview

Traffic congestion caused by accidents, vehicle breakdowns, VIP movement, public events, protests, road closures, and construction is often handled reactively. This project builds a decision-support system that helps traffic authorities act earlier.

The system takes event details such as:

* Event type
* Event cause
* Priority
* Latitude and longitude
* Corridor
* Vehicle type
* Timestamp
* Road closure requirement

and produces:

* Predicted incident volume
* Forecast risk score
* Event impact score
* Final operational risk level
* Affected area radius
* Officer deployment plan
* Barricading recommendation
* Diversion corridor suggestion
* Recommended action

---

## 2. Core Idea

The system combines two intelligence layers:

```text
Historical Forecast Layer
+
Live Event Impact Layer
=
Final Operational Traffic Risk
```

The historical forecast layer learns from past corridor-hour incident patterns.

The live event impact layer evaluates the seriousness of the current event, such as accident, congestion, heavy vehicle involvement, road closure, and rush-hour timing.

This hybrid approach makes the system more practical than using only a regression model.

Example:

```text
Historical forecast says: LOW
Live event says: accident + heavy vehicle + road closure
Final operational risk: HIGH / CRITICAL
```

---

## 3. Features

### Machine Learning

* Zero-inflated hurdle forecasting model
* CatBoost classifier for incident likelihood
* CatBoost regressor for positive incident count
* Time-series cross-validation
* Alert threshold calibration using F2 score
* Feature importance generation
* Sparse incident-count handling

### Feature Engineering

* Corridor-hour aggregation
* Full hourly grid with zero-incident rows
* Lag features
* Rolling-window features
* Corridor average and volatility
* Zone, junction, cause, closure, and cluster risk features
* Cyclic time encoding using sine and cosine

### Decision Intelligence

* Event impact scoring
* Final operational risk scoring
* Officer recommendation
* Barricade recommendation
* Diversion corridor recommendation
* Affected area estimation

### Web Dashboard

* Django-based dashboard
* Dark liquid-glass UI
* Leaflet + OpenStreetMap map
* Real event location marker
* Primary affected zone
* Secondary monitoring zone
* Suggested officer checkpoint markers
* Risk cards and operational recommendation cards

---

## 4. Final System Architecture

```text
Raw Traffic Event Dataset
        ↓
Data Loading
        ↓
Time-Series Dataset Builder
        ↓
Zero-Inflated Hurdle Forecast Model
        ↓
Feature Store Builder
        ↓
Live Prediction Engine
        ↓
Event Impact Scoring
        ↓
Final Risk Scoring
        ↓
Resource Recommendation
        ↓
Diversion Recommendation
        ↓
Terminal Output / Django Dashboard
```

---

## 5. Project Structure

```text
gridv1/
│
├── manage.py
├── config.py
├── train_all.py
├── prepare_feature_store.py
├── predict.py
├── requirements.txt
├── .gitignore
├── README.md
│
├── data/
│   └── traffic_events.csv
│
├── models/
│   ├── timeseries_forecast_model.pkl
│   ├── timeseries_forecast.pkl
│   └── traffic_feature_store.pkl
│
├── src/
│   ├── preprocessing/
│   │   ├── load_data.py
│   │   ├── create_target.py
│   │   ├── feature_engineering.py
│   │   └── advanced_features.py
│   │
│   ├── forecasting/
│   │   ├── build_timeseries_dataset.py
│   │   ├── cross_validate_timeseries.py
│   │   ├── train_timeseries_model.py
│   │   ├── forecast_predictor.py
│   │   └── forecast_feature_importance.py
│   │
│   ├── inference/
│   │   ├── feature_store.py
│   │   └── predict_traffic_risk.py
│   │
│   ├── scoring/
│   │   ├── event_impact.py
│   │   └── risk_score.py
│   │
│   ├── routing/
│   │   └── diversion_engine.py
│   │
│   └── training/
│       ├── train_catboost.py
│       ├── cross_validation.py
│       ├── feature_importance.py
│       └── shap_analysis.py
│
├── traffic_web/
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py
│   └── wsgi.py
│
└── dashboard/
    ├── apps.py
    ├── urls.py
    ├── views.py
    │
    ├── services/
    │   └── ml_engine.py
    │
    ├── templates/
    │   └── dashboard/
    │       └── index.html
    │
    └── static/
        └── dashboard/
            └── style.css
```

---

## 6. Active Main Pipeline

The main working flow uses these commands:

```bash
python train_all.py
python prepare_feature_store.py
python predict.py
python manage.py runserver
```

The old severity classifier files are kept for optional experimentation, but the current main project depends on the forecasting model, feature store, scoring engine, and routing engine.

---

## 7. Installation

### Step 1: Clone or open the project

```bash
cd gridv1
```

### Step 2: Create virtual environment

```bash
python -m venv .venv
```

### Step 3: Activate virtual environment

On Windows:

```bash
.venv\Scripts\activate
```

On Linux / macOS:

```bash
source .venv/bin/activate
```

### Step 4: Install requirements

```bash
pip install -r requirements.txt
```

Recommended Python version:

```text
Python 3.10 or Python 3.11
```

---

## 8. Dataset Setup

Place your dataset here:

```text
data/traffic_events.csv
```

The expected important columns are:

```text
start_datetime
corridor
latitude
longitude
zone
junction
event_cause
requires_road_closure
veh_type
```

Optional useful columns:

```text
event_type
priority
police_station
end_datetime
endlatitude
endlongitude
```

---

## 9. Configuration

Edit `config.py` if your dataset path is different.

Example:

```python
DATA_PATH = "data/traffic_events.csv"

FORECAST_MODEL_PATH = "models/timeseries_forecast.pkl"

FORECAST_MODEL_COMPAT_PATH = "models/timeseries_forecast_model.pkl"

FEATURE_STORE_PATH = "models/traffic_feature_store.pkl"

FORECAST_FEATURE_IMPORTANCE_PATH = "forecast_feature_importance.png"

SEVERITY_MODEL_PATH = "models/catboost_severity.pkl"
```

---

## 10. Training the Forecast Model

Run:

```bash
python train_all.py
```

This command performs:

```text
1. Load traffic event dataset
2. Build corridor-hour time-series dataset
3. Run time-series cross-validation
4. Train zero-inflated hurdle model
5. Save trained model
6. Generate feature importance plot
```

Generated files:

```text
models/timeseries_forecast_model.pkl
models/timeseries_forecast.pkl
forecast_feature_importance.png
```

---

## 11. Building the Feature Store

Run:

```bash
python prepare_feature_store.py
```

This creates:

```text
models/traffic_feature_store.pkl
```

The feature store contains:

* Corridor-hour historical profiles
* Corridor-level fallback profiles
* Global fallback profile
* Incident percentile thresholds
* Valid corridor list

This is required for prediction because the live user input does not contain lag and rolling-window features.

---

## 12. Terminal Prediction

Run:

```bash
python predict.py
```

Example input:

```text
Corridor: ORR East 1
Hour: 9
Weekday: 4
Month: 4

Event Cause: accident
Vehicle Type: heavy_vehicle
Road Closure Required: yes
```

Example output:

```text
Forecast Layer
Predicted Incidents   : 0.00
Forecast Risk Level   : LOW

Event Impact Layer
Event Impact Score    : 73.90%
Event Impact Level    : HIGH

Final Operational Decision
Final Risk Score      : 62.82%
Final Risk Level      : HIGH

Recommended Deployment
Officers Needed       : 7
Barricades Needed     : 4

Recommended Diversion
Primary Detour        : ORR East 2
Secondary Detour      : Old Airport Road
```

---

## 13. Running the Web Dashboard

After training and feature store preparation, run:

```bash
python manage.py runserver
```

Open:

```text
http://127.0.0.1:8000/
```

The dashboard allows the user to enter event details and see:

* Real map location
* Primary affected radius
* Secondary monitoring radius
* Risk score
* Officer requirement
* Barricade requirement
* Diversion plan
* Historical context

---

## 14. Machine Learning Model

The final forecasting model is a zero-inflated hurdle model.

This was chosen because the dataset is highly sparse.

In our training data:

```text
Zero Ratio: around 93%
```

That means most corridor-hour rows have zero incidents.

A normal regression model struggles with this kind of data. So we split the task into two stages:

```text
Stage 1:
Predict whether any incident is likely.

Stage 2:
If incident is likely, predict expected incident count.
```

Final prediction:

```text
Expected incident count = threshold-gated probability strength × positive count prediction
```

This avoids overpredicting incidents in zero-heavy data.

---

## 15. Model Metrics

Example latest performance:

```text
MAE              : 0.1548
RMSE             : 0.4890
R²               : 0.2340

Precision        : 0.2242
Recall           : 0.7124
F1               : 0.3410
ROC-AUC          : 0.8541
PR-AUC           : 0.4217
```

Interpretation:

* MAE is low for sparse corridor-hour forecasting.
* R² is not the most important metric because incidents are rare and spiky.
* ROC-AUC shows the model ranks risky hours well.
* PR-AUC is strong compared to the low positive baseline.
* Recall is high, which is important for traffic operations because missing real incidents is costly.

---

## 16. Why F2 Threshold Tuning Was Used

The alert threshold is calibrated using F2 score.

F2 gives more importance to recall than precision.

This is suitable for traffic operations because:

```text
Missing a real incident is worse than raising an early warning.
```

The calibrated threshold was around:

```text
0.55
```

This gave stronger recall while keeping the model operationally useful.

---

## 17. Feature Engineering Details

The forecasting dataset uses:

```text
corridor
hour
weekday
month
hour_sin
hour_cos
lag_1
lag_2
lag_3
lag_24
lag_48
lag_72
lag_168
rolling_6
rolling_12
rolling_24
rolling_168
corridor_avg
corridor_volatility
zone_risk
junction_risk
cause_risk
closure_risk
cluster_risk
```

Target:

```text
incident_count
```

Important design choices:

* Lags are corridor-specific.
* Rolling windows are shifted to avoid leakage.
* Missing corridor-hours are filled with zero incidents.
* Time is encoded cyclically using sine and cosine.

---

## 18. Event Impact Scoring

The live event impact score uses:

```text
event_cause
vehicle_type
road_closure
rush_hour
priority
event_type
```

Examples:

```text
accident          → high impact
vip_movement      → very high impact
congestion        → high impact
vehicle_breakdown → moderate impact
heavy_vehicle     → extra impact
road_closure      → extra impact
rush_hour         → extra impact
```

This allows live operational conditions to influence the final risk even if historical forecast is low.

---

## 19. Final Operational Risk

The final risk combines:

```text
forecast risk score
event impact score
```

The system uses an event-floor rule so high-impact live events are not suppressed by historically low forecast risk.

Example:

```text
Forecast Risk: LOW
Event Impact: HIGH
Final Risk: HIGH
```

This is useful in real operations because live events can suddenly disrupt traffic even on historically quiet corridors.

---

## 20. Resource Recommendation

The system recommends officers and barricades based on final risk.

Base logic:

```text
LOW      → 2 officers, 0 barricades
MODERATE → 4 officers, 1 barricade
HIGH     → 6 officers, 2 barricades
CRITICAL → 8 officers, 4 barricades
```

Additional resources are added for:

```text
high predicted incident count
road closure requirement
```

---

## 21. Diversion Recommendation

The diversion engine uses a corridor graph built with NetworkX.

Example connections:

```text
ORR East 1 → ORR East 2
ORR East 1 → Old Airport Road
ORR East 1 → Varthur Road
```

The system recommends:

* Primary detour
* Secondary detour
* Support corridors
* Diversion action

Example:

```text
Primary Detour    : ORR East 2
Secondary Detour  : Old Airport Road
Support Corridors : Varthur Road, CBD 2, Mysore Road, CBD 1, Hosur Road
```

---

## 22. Test Scenarios

### Low Scenario

```text
Event type: planned
Event cause: others
Priority: Low
Latitude: 12.9810
Longitude: 77.6090
Corridor: CBD 1
Vehicle type: private_car
Police station: High Grounds
Timestamp: 02:00 PM
Requires road closure: No
```

Expected:

```text
Final Risk: LOW / MODERATE
Officers: 2 to 4
Barricades: 0 or 1
Small affected area
```

### Moderate Scenario

```text
Event type: unplanned
Event cause: vehicle_breakdown
Priority: High
Latitude: 12.9716
Longitude: 77.5946
Corridor: CBD 1
Vehicle type: lcv
Police station: Cubbon Park
Timestamp: 04:30 PM
Requires road closure: No
```

Expected:

```text
Final Risk: MODERATE
Officers: around 4
Barricades: 0 or 1
Standby diversion
```

### High Scenario

```text
Event type: unplanned
Event cause: accident
Priority: Critical
Latitude: 12.9200
Longitude: 77.6200
Corridor: ORR East 1
Vehicle type: heavy_vehicle
Police station: HSR Layout
Timestamp: 09:00 AM
Requires road closure: Yes
```

Expected:

```text
Final Risk: HIGH / CRITICAL
Officers: 7+
Barricades: 4+
Large affected area
Active diversion
```

---

## 23. Important Notes

`Non-corridor` is a broad fallback bucket, not a real single corridor. It may produce high forecast risk because it aggregates many unrelated locations.

For clean demo scenarios, prefer real corridors such as:

```text
ORR East 1
Mysore Road
CBD 1
CBD 2
Hosur Road
Old Airport Road
Old Madras Road
Bellary Road 1
```

---

## 24. Active vs Optional Files

### Active final system files

```text
src/preprocessing/load_data.py

src/forecasting/build_timeseries_dataset.py
src/forecasting/cross_validate_timeseries.py
src/forecasting/train_timeseries_model.py
src/forecasting/forecast_predictor.py
src/forecasting/forecast_feature_importance.py

src/inference/feature_store.py
src/inference/predict_traffic_risk.py

src/scoring/event_impact.py
src/scoring/risk_score.py

src/routing/diversion_engine.py

train_all.py
prepare_feature_store.py
predict.py
dashboard/services/ml_engine.py
```

### Optional old severity classifier files

```text
src/preprocessing/create_target.py
src/preprocessing/feature_engineering.py
src/preprocessing/advanced_features.py

src/training/train_catboost.py
src/training/cross_validation.py
src/training/feature_importance.py
src/training/shap_analysis.py
```

These files are not required for the current final dashboard flow.

---

## 25. Troubleshooting

### Model not found

Error:

```text
Forecast model not found
```

Fix:

```bash
python train_all.py
```

### Feature store not found

Error:

```text
Feature store not found
```

Fix:

```bash
python prepare_feature_store.py
```

### Dataset not found

Check `config.py`:

```python
DATA_PATH = "data/traffic_events.csv"
```

Make sure the dataset exists at that path.

### Map not loading

Check internet connection because Leaflet uses OpenStreetMap tiles.

### Predictions too high for Non-corridor

Use a real corridor. `Non-corridor` is a fallback category and may contain many unrelated events.

---

## 26. Future Improvements

Possible upgrades:

```text
real-time traffic speed data
weather data
holiday/event calendar features
road capacity features
OpenStreetMap road graph routing
live CCTV/GPS integration
MLflow model registry
automatic retraining
post-event learning loop
alert classification dashboard
```

---

## 27. Final Summary

This project is a hybrid ML and decision-intelligence system for event-driven traffic management.

It does not only predict incident count. It converts historical patterns and live event details into an actionable operational plan:

```text
Forecast risk
+
Event impact
+
Final operational risk
+
Officer deployment
+
Barricading plan
+
Diversion recommendation
```

This makes the system suitable as a practical prototype for traffic police command-center decision support.
