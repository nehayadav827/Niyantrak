# GridLock IQ — Coordinate-First Event-Driven Traffic Intelligence Engine

GridLock IQ is a machine learning powered traffic intelligence system for forecasting event-driven congestion risk, estimating operational impact, recommending traffic deployment, identifying diversion routes, and visualizing affected zones on a map.

The system is built for traffic-control scenarios such as:

* Vehicle breakdowns
* Accidents
* Congestion
* Public events
* Protests
* Processions
* VIP movement
* Festivals
* Sports events
* Construction activity
* Water logging
* Road closures

The final system is **coordinate-first**. The user can select an event location on the map or enter latitude and longitude. The backend automatically resolves the nearest corridor, spatial cluster, hotspot distance, and nearest police station.

---

## 1. Project Objective

Traffic planning for large events and sudden incidents is often reactive. Event impact is not always quantified in advance, resource deployment is experience-driven, and post-event feedback is usually not stored in a structured way.

GridLock IQ solves this by providing:

```text
1. Coordinate-first event impact prediction
2. Spatial-cluster-hour traffic forecasting
3. Corridor-hour fallback forecasting
4. Event Impact Score from 0 to 100
5. Forecasted incident volume
6. Final operational risk level
7. Officer and barricade recommendation
8. State-aware diversion recommendation
9. Affected area visualization on map
10. Nearest police station identification
11. Model validation metrics on dashboard
12. Historical similar event lookup
13. Post-event feedback collection for future retraining
```

---

## 2. Core Idea

```text
User selects event location on map
        ↓
System extracts latitude and longitude
        ↓
Backend validates Bengaluru coverage
        ↓
Nearest corridor is inferred
        ↓
Nearest spatial cluster is identified
        ↓
Nearest police station is resolved
        ↓
Historical feature store is queried
        ↓
Spatial ML model predicts incident risk
        ↓
Corridor model is used as fallback if needed
        ↓
Event impact layer adjusts for live event severity
        ↓
Final operational risk is calculated
        ↓
Dashboard shows map, risk, deployment, diversion, and evidence
```

---

## 3. Final System Architecture

```text
Raw Traffic Event Data
        ↓
Robust Datetime Parsing
        ↓
Corridor-Hour Time-Series Dataset
        ↓
Spatial-Cluster-Hour Time-Series Dataset
        ↓
Lag / Rolling / Spatial / Calendar Feature Engineering
        ↓
Zero-Inflated CatBoost Hurdle Models
        ↓
Coordinate-Aware Feature Store
        ↓
Location Resolver
        ↓
Police Station Resolver
        ↓
Spatial Forecast Model
        ↓
Corridor Forecast Fallback Model
        ↓
Forecast Risk Score
        ↓
Event Impact Score
        ↓
Calibrated EIS Score
        ↓
State-Aware Diversion Engine
        ↓
Operational Dashboard
```

---

## 4. Final Project Structure

```text
gridv1/
│
├── config.py
├── manage.py
├── train_all.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── events_calendar.csv
│   └── police_stations.csv
│
├── scripts/
│   ├── predict.py
│   ├── prepare_feature_store.py
│   ├── train_spatial_model.py
│   ├── run_cluster_ablation.py
│   ├── run_eis_calibration.py
│   └── train_quantile_intervals.py
│
├── src/
│   ├── features/
│   │   ├── __init__.py
│   │   └── event_calendar.py
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── load_data.py
│   │   └── clean.py
│   │
│   ├── forecasting/
│   │   ├── __init__.py
│   │   ├── build_timeseries_dataset.py
│   │   ├── build_spatial_timeseries_dataset.py
│   │   ├── train_timeseries_model.py
│   │   ├── train_spatial_timeseries_model.py
│   │   ├── train_quantile_intervals.py
│   │   ├── forecast_predictor.py
│   │   ├── spatial_forecast_predictor.py
│   │   ├── cross_validate_timeseries.py
│   │   └── forecast_feature_importance.py
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── feature_store.py
│   │   ├── location_resolver.py
│   │   ├── police_station_resolver.py
│   │   ├── predict_traffic_risk.py
│   │   └── similar_events.py
│   │
│   ├── scoring/
│   │   ├── __init__.py
│   │   ├── event_impact.py
│   │   └── risk_score.py
│   │
│   ├── routing/
│   │   ├── __init__.py
│   │   └── diversion_engine.py
│   │
│   ├── recommendation/
│   │   ├── __init__.py
│   │   └── resource_recommender.py
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── cluster_fallback_ablation.py
│       └── eis_weight_calibration.py
│
├── dashboard/
│   ├── views.py
│   ├── urls.py
│   ├── services/
│   │   ├── ml_engine.py
│   │   └── feedback_store.py
│   ├── templates/
│   │   └── dashboard/
│   │       └── index.html
│   └── static/
│       └── dashboard/
│           └── style.css
│
└── traffic_web/
    ├── __init__.py
    ├── settings.py
    ├── urls.py
    ├── asgi.py
    └── wsgi.py
```

---

## 5. Data Inputs

### Main traffic dataset

The main dataset contains traffic event records.

Important columns:

```text
id
event_type
latitude
longitude
endlatitude
endlongitude
address
end_address
event_cause
requires_road_closure
start_datetime
end_datetime
status
corridor
priority
veh_type
veh_no
junction
zone
police_station
```

Each row represents one traffic-related event.

Example:

```text
accident on ORR East 1 at 09:12
vehicle breakdown near CBD at 18:30
public event near Mysore Road at 17:00
```

---

## 6. Event Calendar Data

File:

```text
data/events_calendar.csv
```

Purpose:

The system no longer treats event-driven prediction as purely hardcoded rules. Historical known events can be labelled in an event calendar and joined to training rows.

Expected schema:

```csv
event_name,event_type,start_datetime,end_datetime,latitude,longitude,corridor,impact_radius_m,crowd_size
IPL Match,sports,2024-03-25 16:00:00,2024-03-25 23:00:00,12.9788,77.5996,CBD 1,2500,large
Public Rally,protest,2024-02-10 09:00:00,2024-02-10 15:00:00,12.9716,77.5946,CBD 2,3000,large
Festival Crowd,festival,2024-01-15 17:00:00,2024-01-15 23:00:00,12.9600,77.5800,Mysore Road,2500,medium
```

Generated ML features:

```text
is_event_day
calendar_event_type
calendar_event_intensity
```

Why this matters:

```text
The model can learn that sports events, protests, festivals, and public gatherings increase incident probability near affected corridors or clusters.
```

The event calendar feature is learned by CatBoost, while live crowd and weather multipliers remain part of the operational adjustment layer.

---

## 7. Police Station Data

File:

```text
data/police_stations.csv
```

Expected schema:

```csv
police_station,latitude,longitude
Cubbon Park Traffic Police Station,12.9766,77.5993
Ashok Nagar Traffic Police Station,12.9662,77.6068
Indiranagar Traffic Police Station,12.9784,77.6408
Madiwala Traffic Police Station,12.9212,77.6175
Whitefield Traffic Police Station,12.9698,77.7500
```

The system uses this file to resolve:

```text
event latitude / longitude → nearest responsible police station
```

If official police station coordinates are not provided, the system can fall back to historical police-station incident centroids from the dataset.

Honest limitation:

```text
If using historical centroids, the result estimates the nearest operational police station area, not necessarily the exact physical station building.
```

---

## 8. Robust Datetime Parsing

The system includes robust datetime parsing.

Why:

Some raw rows may contain mixed datetime formats or timezone strings. A single `pd.to_datetime(..., utc=True)` pass can fail and silently drop useful rows.

Fix:

```text
First parse pass:
    pd.to_datetime(..., utc=True)

Second parse pass:
    pd.to_datetime(..., format="mixed", utc=True)
```

The training log prints:

```text
Datetime Parse Recovery
--------------------------------------------------
Initially failed : X
Recovered        : Y
Still failed     : Z
```

This improves data coverage and avoids unnecessary row loss.

---

## 9. Corridor-Hour Dataset

File:

```text
src/forecasting/build_timeseries_dataset.py
```

Raw event-level rows are converted into a corridor-hour time-series dataset.

Raw format:

```text
one row = one traffic event
```

Training format:

```text
one row = one corridor-hour
```

Example:

```text
Raw events:
ORR East 1 | 2024-04-01 09:12 | accident
ORR East 1 | 2024-04-01 09:47 | congestion

After aggregation:
ORR East 1 | 2024-04-01 09:00 | incident_count = 2
```

Target column:

```text
incident_count
```

Meaning:

```text
number of traffic incidents in a corridor during a specific hour
```

This model is retained as a fallback model.

---

## 10. Spatial-Cluster-Hour Dataset

File:

```text
src/forecasting/build_spatial_timeseries_dataset.py
```

The spatial model fixes the earlier corridor-only limitation.

Old approach:

```text
corridor × hour
```

New primary approach:

```text
spatial_cluster_id × hour
```

This means two different points on the same corridor can produce different predictions if they belong to different spatial clusters or local spatial contexts.

Spatial features include:

```text
spatial_cluster_id
latitude
longitude
dominant_corridor
nearest_corridor_distance_m
nearest_hotspot_distance_m
spatial_density_at_point
```

The spatial-cluster model is the primary model. The corridor-hour model remains as fallback.

---

## 11. Target Distribution and Rare Events

Traffic incident prediction is a rare-event forecasting problem.

Most corridor-hour and spatial-cluster-hour rows contain zero incidents.

Typical behavior:

```text
incident_count = 0 for more than 90% of rows
```

Because of this, exact incident count regression is difficult. The main operational goal is early risk detection, not only perfect count prediction.

---

## 12. Lag and Rolling Features

The model uses time-history features.

Lag features:

```text
lag_1
lag_2
lag_3
lag_24
lag_48
lag_72
lag_168
```

Meaning:

```text
lag_1    = incident count 1 hour ago
lag_24   = same hour yesterday
lag_168  = same hour last week
```

Rolling features:

```text
rolling_6
rolling_12
rolling_24
rolling_168
```

Meaning:

```text
rolling_6    = average incident count over last 6 hours
rolling_24   = average incident count over last 24 hours
rolling_168  = average incident count over last week
```

---

## 13. Derived Lag Features

Raw lag values are sparse because most hours have zero incidents. To make recent-history signal stronger, the system adds compressed lag-derived features:

```text
any_incident_last_3h
incidents_last_24h
above_corridor_avg
```

Meaning:

| Feature                | Meaning                                                                  |
| ---------------------- | ------------------------------------------------------------------------ |
| `any_incident_last_3h` | Whether any incident occurred in the last 3 hours                        |
| `incidents_last_24h`   | Estimated incident volume in the last 24 hours                           |
| `above_corridor_avg`   | Whether recent rolling activity is above normal corridor/cluster average |

Why useful:

```text
These features compress sparse lag signals into binary/aggregate signals that CatBoost can split on more effectively.
```

---

## 14. Time Features

Time features:

```text
hour
weekday
month
hour_sin
hour_cos
```

Why use `hour_sin` and `hour_cos`:

```text
23:00 and 00:00 are close in real life, but numerically 23 and 0 look far apart.
Cyclical encoding fixes this.
```

---

## 15. Spatial and Historical Risk Features

The model uses:

```text
corridor_avg
corridor_volatility
zone_risk
junction_risk
cause_risk
closure_risk
cluster_risk
```

Meaning:

| Feature               | Meaning                                             |
| --------------------- | --------------------------------------------------- |
| `corridor_avg`        | Average historical incident count                   |
| `corridor_volatility` | How unstable or spike-prone the corridor/cluster is |
| `zone_risk`           | Historical event density in the zone                |
| `junction_risk`       | Historical risk around the junction                 |
| `cause_risk`          | Historical frequency/risk of event cause            |
| `closure_risk`        | Historical risk linked to road closure              |
| `cluster_risk`        | Spatial cluster event density                       |

---

## 16. Forecast Model Architecture

The forecasting model uses a:

```text
Zero-Inflated CatBoost Hurdle Model
```

This is required because incident counts are highly zero-heavy.

It has two stages:

```text
Stage 1: CatBoostClassifier
Stage 2: CatBoostRegressor
```

### Stage 1 — Alert Classifier

Predicts whether a corridor-hour or spatial-cluster-hour will have at least one incident.

Target:

```text
alert_target = 1 if incident_count > 0 else 0
```

Output:

```text
alert_probability
```

Example:

```text
alert_probability = 0.72
```

Meaning:

```text
The model estimates a 72% chance of at least one incident.
```

### Stage 2 — Positive Count Regressor

Predicts expected incident count when incident risk exists.

Conceptually:

```text
if alert probability is below threshold:
    expected incidents ≈ 0

if alert probability is above threshold:
    positive-count regressor output is used
```

This avoids predicting small fake incident counts everywhere.

---

## 17. Primary and Fallback Models

The system uses two forecast models:

### Primary model

```text
models/spatial_timeseries_forecast_model.pkl
```

Role:

```text
Point-aware spatial-cluster-hour prediction
```

Uses:

```text
spatial_cluster_id
latitude
longitude
nearest corridor distance
nearest hotspot distance
spatial density
spatial lag/rolling history
calendar event features
```

### Fallback model

```text
models/timeseries_forecast_model.pkl
models/timeseries_forecast.pkl
```

Role:

```text
Corridor-hour fallback prediction
```

Used when:

```text
spatial model is unavailable
spatial feature construction fails
location does not have usable cluster context
```

---

## 18. Coordinate-Aware Feature Store

File:

```text
src/inference/feature_store.py
```

Generated file:

```text
models/traffic_feature_store.pkl
```

The feature store provides historical context at inference time.

It stores:

```text
corridor-hour profiles
corridor-level profiles
spatial cluster-hour profiles
spatial cluster-level profiles
global fallback profile
corridor location profiles
hotspot points
spatial cluster model
spatial cluster centers
police station points
risk thresholds
```

Why needed:

At prediction time, the user only provides event details and coordinates. The model still needs historical lag, rolling, and spatial features. The feature store supplies those features.

---

## 19. Location Resolver

File:

```text
src/inference/location_resolver.py
```

Input:

```text
latitude
longitude
feature store
```

Output:

```python
{
    "corridor": "ORR East 1",
    "matched_by": "nearest real corridor historical point",
    "distance_m": 420.5,
    "confidence": "HIGH",
    "spatial_cluster_id": 7,
    "nearest_hotspot_distance_m": 180.4,
    "spatial_density_at_point": 0.72,
    "outside_bengaluru": False
}
```

It performs:

```text
Bengaluru boundary validation
nearest real corridor matching
nearest corridor centroid matching
spatial cluster lookup
nearest hotspot distance calculation
spatial density calculation
```

---

## 20. Bengaluru Boundary Validation

The system validates coordinates against Bengaluru coverage.

Invalid example:

```text
Latitude: 28.6139
Longitude: 77.2090
```

This is Delhi, not Bengaluru.

The system returns:

```text
Selected location is outside Bengaluru coverage area.
Please choose a location within Bengaluru.
```

This prevents false corridor matching outside the project area.

---

## 21. Police Station Resolver

File:

```text
src/inference/police_station_resolver.py
```

Input:

```text
latitude
longitude
feature store
```

Output:

```python
{
    "police_station": "Cubbon Park Traffic Police Station",
    "matched_by": "nearest police station by coordinates",
    "distance_m": 850.0,
    "confidence": "HIGH",
    "source": "official_station_coordinates"
}
```

If official station coordinates are available, they are used. Otherwise, the system falls back to historical police-station incident centroids.

---

## 22. Event Impact Layer

File:

```text
src/scoring/event_impact.py
```

The event impact layer uses live event information:

```text
event_type
event_cause
vehicle_type
road_closure
rush_hour
priority
crowd_size
weather
```

Examples:

```text
accident + heavy vehicle + closure       → high impact
vehicle breakdown + LCV + no closure     → moderate impact
public event + large crowd + rush hour   → high impact
```

This layer captures live operational severity that may not exist in historical forecasts.

---

## 23. Crowd Size Handling

Crowd size is used only for planned or crowd-bearing events.

Crowd-relevant examples:

```text
public_event
protest
procession
vip_movement
festival
sports
political
election
```

For unplanned events:

```text
accident
vehicle_breakdown
tree_fall
water_logging
road_conditions
pot_holes
```

the system treats crowd as:

```text
not_applicable
```

This prevents unrealistic risk inflation when crowd size is unknown for unplanned incidents.

Behavior:

```text
Unplanned accident + unknown crowd      → crowd multiplier = 1.00
Vehicle breakdown + unknown crowd       → crowd multiplier = 1.00
Planned public event + unknown crowd     → conservative medium default
Planned protest + large crowd            → large crowd multiplier
```

---

## 24. Weather Adjustment

Supported weather inputs:

```text
clear
light_rain
heavy_rain
fog
```

Weather affects:

```text
event impact
duration estimate
operational pressure
```

Heavy rain and fog increase traffic instability.

---

## 25. Event Impact Score

The system calculates a live event impact score based on event details.

Output:

```text
Event Impact Score: 72.5%
Event Impact Level: HIGH
```

Risk levels:

```text
0–25     LOW
25–50    MODERATE
50–75    HIGH
75–100   CRITICAL
```

---

## 26. Composite EIS Score

EIS means:

```text
Event Impact Score
```

It combines:

```text
forecast risk
live event impact
historical cause risk
```

Formula:

```text
EIS =
    forecast_weight × forecast_score
  + event_weight × adjusted_event_score
  + cause_weight × cause_risk_score
```

The EIS gives a single 0–100 score to compare different events.

---

## 27. EIS Weight Micro-Calibration

File:

```text
src/evaluation/eis_weight_calibration.py
```

Output:

```text
models/eis_weight_calibration.json
EIS_WEIGHT_CALIBRATION.md
```

Purpose:

Avoid choosing EIS weights blindly.

The calibration tests several candidate weight combinations against a historical severity proxy.

Severity proxy:

```text
45% duration score
+ 25% same corridor-hour incident volume score
+ 20% road closure score
+ 10% event cause severity prior
```

Honest note:

```text
This is a micro-calibration using proxy severity, not manually labelled ground truth.
It provides evidence for the EIS weights and can later be replaced by officer-labelled feedback data.
```

---

## 28. Forecast Risk Score

The ML model outputs expected incident count and alert probability.

This is converted into an operational forecast risk score using:

```text
predicted_incidents
alert_probability
incident_p95
incident_p99
context multiplier
```

Forecast risk is purely model/historical based.

If forecast risk is zero, it means:

```text
The model does not expect incidents from historical corridor/cluster patterns.
```

The final risk can still be high if the live event is severe.

---

## 29. Why Forecast Risk or Expected Incidents Can Be Zero

Because the dataset is zero-heavy, the hurdle classifier often predicts that the current hour is below the incident threshold.

This can produce:

```text
Expected Incidents : 0.00
Lower Estimate     : 0.00
Upper Estimate     : 0.61
```

Meaning:

```text
Most likely incident count is zero, but historical uncertainty still allows possible spillover risk up to the upper range.
```

This is expected behavior in rare-event forecasting.

---

## 30. Prediction Interval

File:

```text
src/forecasting/train_quantile_intervals.py
```

The system supports prediction uncertainty.

Preferred method:

```text
CatBoost quantile interval
```

Two additional quantile models can be trained:

```text
lower quantile model: alpha = 0.10
upper quantile model: alpha = 0.90
```

Together they produce:

```text
80% prediction interval
```

Fallback method:

```text
Estimated uncertainty using holdout RMSE
```

Dashboard wording is honest:

```text
Estimated uncertainty using holdout RMSE. This is a validation-error based estimate, not a guaranteed statistical interval.
```

---

## 31. Cluster Fallback Ablation Study

File:

```text
src/evaluation/cluster_fallback_ablation.py
```

Output:

```text
models/cluster_fallback_ablation.json
```

Purpose:

Test whether spatial-cluster fallback should replace corridor-hour history.

Measured result example:

```text
Rows tested       : 5000
Normal MAE        : 0.1537
Cluster MAE       : 0.2384
Improvement       : -55.18%
```

Conclusion:

```text
Cluster fallback is weaker than corridor-hour history when corridor matching is reliable.
Therefore, cluster fallback is used only when corridor matching is weak or unavailable.
```

This makes the fallback design evidence-based.

---

## 32. State-Aware Diversion Engine

File:

```text
src/routing/diversion_engine.py
```

Earlier, diversion was a static lookup.

Now, the system:

```text
1. Generates alternate corridor candidates
2. Runs lightweight risk checks for each alternate corridor
3. Computes route pressure score
4. Ranks alternates by current predicted route risk
5. Avoids HIGH/CRITICAL routes when safer routes exist
```

Route pressure score uses:

```text
historical load
alert probability
volatility
ML incident forecast
```

This prevents recommending a diversion route that is already risky.

---

## 33. Resource Recommendation

File:

```text
src/recommendation/resource_recommender.py
```

The system recommends:

```text
officers
barricades
shift duration
```

Based on:

```text
final risk level
road closure
predicted incidents
event severity
```

Example:

```text
Final Risk Level      : HIGH
Officers Needed       : 6
Barricades Needed     : 2
```

---

## 34. Historical Similar Events

File:

```text
src/inference/similar_events.py
```

The system finds similar past events using:

```text
event cause
corridor
hour
distance
event type
```

Dashboard output can show:

```text
similar event
corridor
time
duration
closure
similarity score
```

This grounds the prediction in historical evidence.

---

## 35. Map Visualization

The dashboard uses Leaflet and OpenStreetMap.

Map output includes:

```text
event marker
primary impact circle
secondary spillover circle
risk color
popup details
```

Meaning:

```text
inner circle = primary affected zone
outer circle = secondary spillover zone
marker = event location
```

This visually answers:

```text
What area will be affected?
```

---

## 36. Pre-Event vs Post-Event Comparison

The dashboard compares:

```text
normal condition
with event
delta
```

Example:

```text
Incident volume
0.15 → 0.60
+0.46

Risk score
38.3% → 42.5%
+4.2 percentage points
```

The incident delta is shown as an absolute difference to avoid misleading percentage spikes from small baselines.

---

## 37. Model Metrics

The dashboard shows raw metrics and operational translations.

Important metrics:

```text
MAE
RMSE
R²
Alert Accuracy
Alert Precision
Alert Recall
Alert F1
ROC-AUC
PR-AUC
```

Operational interpretation:

```text
Recall      → how many real incident-hours were caught
Precision   → how many alerts were correct
ROC-AUC     → how well risky hours are ranked above quiet hours
PR-AUC      → alert quality when positive incidents are rare
MAE         → average incident-count error
R²          → exact regression fit, secondary for sparse rare-event data
```

---

## 38. Why R² Can Be Modest

This is not normal continuous regression.

Traffic incidents are rare and spike-driven.

Because most rows have zero incidents, exact count prediction is difficult.

Judge-safe explanation:

```text
R² is reported honestly, but it is not the main decision metric.
This is a sparse rare-event forecasting problem.
The alert classifier is the main operational layer.
ROC-AUC, recall, PR-AUC, and MAE are more important for traffic operations.
```

---

## 39. Post-Event Feedback Collection

File:

```text
dashboard/services/feedback_store.py
```

Output:

```text
data/post_event_feedback.csv
```

The feedback form stores:

```text
actual duration
actual officers deployed
actual barricades used
actual road closure
actual incident count
officer notes
```

Important honesty note:

```text
The current prototype stores feedback for audit, analysis, and future retraining.
It does not automatically retrain the ML model after each feedback entry.
```

Correct wording:

```text
feedback collection system with retraining capability planned
```

---

## 40. Final Training Pipeline

Main command:

```bash
python train_all.py
```

The training pipeline runs:

```text
1. Load raw traffic dataset
2. Build corridor-hour time-series dataset
3. Run time-series cross-validation
4. Train corridor-hour hurdle fallback model
5. Save forecast feature importance
6. Build coordinate-aware feature store
7. Train primary spatial-cluster-hour model
8. Run cluster fallback ablation study
9. Train CatBoost quantile interval models
10. Run EIS weight micro-calibration
```

Generated files:

```text
models/timeseries_forecast_model.pkl
models/timeseries_forecast.pkl
models/spatial_timeseries_forecast_model.pkl
models/traffic_feature_store.pkl
models/cluster_fallback_ablation.json
models/eis_weight_calibration.json
EIS_WEIGHT_CALIBRATION.md
forecast_feature_importance.png
```

Generated model files should not be pushed to GitHub.

---

## 41. Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run full ML pipeline:

```bash
python train_all.py
```

Run Django dashboard:

```bash
python manage.py runserver
```

Optional helper scripts:

```bash
python scripts/prepare_feature_store.py
python scripts/predict.py
python scripts/train_spatial_model.py
python scripts/run_cluster_ablation.py
python scripts/run_eis_calibration.py
python scripts/train_quantile_intervals.py
```

---

## 42. Dashboard Output

The dashboard shows:

```text
Final operational risk
EIS score
Forecast risk
Event impact score
Expected incident count
Prediction interval
Map impact circles
Nearest corridor
Nearest police station
Historical context variables
Officer recommendation
Barricade recommendation
State-aware diversion ranking
Similar historical events
Model metrics
R² explanation
Cluster fallback ablation
EIS calibration evidence
Post-event feedback form
```

---

## 43. What Is ML and What Is Rule-Based?

### Machine Learning

```text
CatBoostClassifier
CatBoostRegressor
CatBoost quantile regressors
KMeans spatial clustering
Time-series cross-validation
Feature importance
Spatial-cluster-hour forecasting
Corridor-hour fallback forecasting
Calendar-event learning features
```

### Statistical Feature Store

```text
lag features
rolling features
derived lag features
corridor-hour profiles
cluster-hour profiles
risk percentiles
spatial density
hotspot points
police station points
```

### Rule-Based Operational Intelligence

```text
event severity scoring
weather adjustment
crowd adjustment for planned events
EIS formula
resource allocation rules
affected radius rules
state-aware diversion ranking
deployment order generation
feedback storage
```

This hybrid approach is intentional because traffic operations need explainable recommendations, not only black-box predictions.

---

## 44. Known Limitations

```text
real-time traffic API is not yet integrated
SMS/WhatsApp/email alert sending is not yet implemented
feedback is stored but not yet used for automatic retraining
weather is manually selected, not API-driven
crowd size is manually entered for planned events
similar-event lookup is heuristic
diversion is corridor-level, not full live road-network routing
police station matching depends on station coordinate quality
```

---

## 45. Future Improvements

```text
integrate live traffic speed feeds
integrate live weather API
send automatic alerts to officers
use OpenStreetMap road graph for dynamic diversions
use officer feedback for scheduled retraining
add MLflow model registry
add model drift monitoring
add real-time congestion heatmap
add CCTV/GPS/sensor integration
expand police station coordinate database
expand event calendar with official event feeds
```

---

## 46. GitHub Notes

Do not push generated or local files:

```text
models/
*.pkl
*.joblib
db.sqlite3
.venv/
.env
__pycache__/
data/post_event_feedback.csv
forecast_feature_importance.png
EIS_WEIGHT_CALIBRATION.md
```

If the raw dataset is large or private, do not push:

```text
data.csv
```

Instead, push a small sample:

```text
data/sample_data.csv
```

Recommended root files:

```text
config.py
manage.py
train_all.py
requirements.txt
README.md
.gitignore
```

---

## 47. One-Line Summary

```text
Map coordinates → spatial location intelligence → CatBoost hurdle forecast → calibrated event impact score → final operational risk → map impact zone → police station, deployment, and diversion recommendation
```

---

## 48. Judge Explanation

```text
GridLock IQ is a coordinate-first event-driven traffic intelligence system. The user selects an exact event location on a map, and the backend infers the nearest corridor, spatial cluster, hotspot distance, and police station.

The forecasting layer uses a primary spatial-cluster-hour CatBoost hurdle model. This fixes the corridor-only limitation because coordinates are converted into spatial cluster ID, latitude/longitude, local density, hotspot distance, and distance-to-corridor features. A corridor-hour model remains as fallback.

The model is zero-inflated because traffic incidents are rare and most corridor-hour records have zero incidents. First, a classifier estimates whether an incident is likely, and then a regressor estimates the count when risk exists.

The system also includes event-calendar learning features, so historical sports events, protests, festivals, and public events can influence training instead of relying only on hardcoded multipliers.

The final operational decision combines ML forecast risk, live event severity, road closure, weather, crowd relevance, historical cause risk, and calibrated EIS weights. The dashboard shows final risk, affected map radius, nearest police station, officers required, barricades required, state-aware diversion routes, prediction interval, similar historical events, model metrics, and feedback collection.

The prototype stores post-event feedback for audit and future retraining, but it does not claim automatic online learning yet.
```
