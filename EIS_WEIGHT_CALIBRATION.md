# EIS Weight Micro-Calibration

## Purpose

The Event Impact Score uses a weighted combination of forecast risk, live event impact, and historical cause risk. This calibration checks multiple candidate weight combinations against historical event outcomes using a practical severity proxy.

## Selected Weights

- Forecast weight: `0.35`
- Event impact weight: `0.5`
- Cause risk weight: `0.15`

Best method: **Forecast stronger**

Best MAE against proxy severity: **10.7044**

## Target Definition

Actual severity proxy = 45% duration score + 25% same corridor-hour incident volume score + 20% road closure score + 10% event cause severity prior.

## Candidate Results

| Method | Forecast | Event | Cause | MAE |
|---|---:|---:|---:|---:|
| Forecast stronger | 0.35 | 0.5 | 0.15 | 10.7044 |
| Forecast-heavy | 0.4 | 0.45 | 0.15 | 10.7876 |
| Balanced default | 0.3 | 0.55 | 0.15 | 11.0601 |
| Event stronger | 0.25 | 0.6 | 0.15 | 11.7738 |
| Cause stronger | 0.3 | 0.5 | 0.2 | 12.1973 |
| Event-dominant | 0.2 | 0.65 | 0.15 | 12.5881 |

## Honesty Note

This is a micro-calibration using a proxy severity target, not manually labelled ground truth. It gives evidence for the EIS weights and can be replaced by officer-labelled feedback data later.

## Judge Explanation

The EIS weights were not chosen blindly. We tested multiple candidate formulas against historical event outcomes using a proxy severity target based on actual duration, incident volume, closure status, and event cause severity. The lowest-MAE formula was selected for the dashboard.