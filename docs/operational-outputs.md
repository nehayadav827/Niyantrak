# Operational Outputs

This covers everything the system produces *after* risk has been scored: the final risk decision, resourcing, diversion, and the formatted deployment order.

## Final Operational Risk

The final risk combines:

```text
forecast risk
calibrated EIS score
```

This matters because the historical forecast can be low while the live event is still severe. Example:

```text
Forecast risk: LOW
Event: accident + heavy vehicle + road closure
EIS: HIGH
Final operational risk: HIGH
```

This logic prevents under-response to serious live events just because a corridor doesn't have a historically risky profile.

## Resource Recommendation

The system recommends officers and barricades based on the final risk level.

Base logic:

| Risk Level | Officers | Barricades |
|---|---|---|
| LOW | 2 | 0 |
| MODERATE | 4 | 1 |
| HIGH | 6 | 2 |
| CRITICAL | 8 | 4 |

Additional resources are added on top of this base for:

- road closure
- high predicted incidents
- critical final risk

## Diversion Recommendation

File: `src/routing/diversion_engine.py`

The diversion engine uses a graph of corridors, where each corridor is a node and edges represent possible route alternatives.

Output fields:

```text
primary_detour
secondary_detour
support_corridors
diversion_action
```

Example:

```text
Affected corridor : ORR East 1
Primary detour    : ORR East 2
Secondary detour  : Old Airport Road
Support corridors : Varthur Road, CBD 2, Mysore Road
```

> This is corridor-level routing based on a predefined graph, not live road-network routing — see [Limitations & Roadmap](limitations-and-roadmap.md).

## Affected Radius Calculation

The system estimates:

```text
affected_radius_m
secondary_radius_m
```

Based on:

- final risk level
- predicted incidents
- road closure
- event cause

High-spread causes increase the radius, including: accident, congestion, VIP movement, protest, procession, public event, and water logging.

## Deployment Order

The dashboard generates a formatted, operator-ready deployment order:

```text
TRAFFIC DEPLOYMENT ORDER
----------------------------------------
Risk Level       : HIGH
Risk Score       : 67.20%
Event Cause      : accident
Location         : 12.920000, 77.620000
Inferred Corridor: ORR East 1
Duration Estimate: 95 minutes

RESOURCE PLAN
----------------------------------------
Officers Required: 7
Barricades Needed: 4

DIVERSION PLAN
----------------------------------------
Primary Detour   : ORR East 2
Secondary Detour : Old Airport Road
Support Corridors: Varthur Road, CBD 2

ACTION
----------------------------------------
Deploy officers, prepare barricades, and keep diversion support ready.
```

This is what makes the system's output **operational** rather than purely analytical — it's meant to be actioned directly, not just read as a risk score.

## Related Docs

- [Event Impact Scoring](event-impact-scoring.md) — how EIS feeds into final risk
- [Dashboard](dashboard.md) — where these outputs are displayed