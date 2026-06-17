def normalize_text(value):
    value = str(value).strip().lower()

    value = value.replace("-", "_")
    value = value.replace(" ", "_")

    while "__" in value:
        value = value.replace("__", "_")

    return value


def calculate_event_impact(
    event_cause,
    veh_type,
    road_closure,
    rush_hour=False
):
    event_cause = normalize_text(event_cause)
    veh_type = normalize_text(veh_type)

    cause_weights = {
        "vehicle_breakdown": 45,
        "accident": 85,
        "construction": 75,
        "water_logging": 78,
        "tree_fall": 72,
        "road_conditions": 60,
        "pot_holes": 50,
        "congestion": 75,
        "public_event": 82,
        "procession": 88,
        "vip_movement": 92,
        "protest": 92,
        "others": 40,
        "debris": 55,
        "fog_low_visibility": 65,
        "test_demo": 20
    }

    vehicle_weights = {
        "heavy_vehicle": 30,
        "truck": 30,
        "private_bus": 28,
        "bmtc_bus": 30,
        "ksrtc_bus": 28,
        "lcv": 18,
        "private_car": 10,
        "taxi": 8,
        "auto": 6,
        "others": 10,
        "unknown": 8,
        "": 8
    }

    cause_score = cause_weights.get(
        event_cause,
        40
    )

    vehicle_score = vehicle_weights.get(
        veh_type,
        8
    )

    closure_score = 30 if road_closure else 0

    rush_score = 15 if rush_hour else 0

    public_transport_boost = 0

    if veh_type in [
        "bmtc_bus",
        "ksrtc_bus",
        "private_bus"
    ]:
        public_transport_boost = 8

    impact_score = (
        0.58 * cause_score
        +
        0.20 * vehicle_score
        +
        0.12 * closure_score
        +
        rush_score
        +
        public_transport_boost
    )

    impact_score = max(
        0,
        min(impact_score, 100)
    )

    if impact_score < 25:
        impact_level = "LOW"

    elif impact_score < 50:
        impact_level = "MODERATE"

    elif impact_score < 75:
        impact_level = "HIGH"

    else:
        impact_level = "CRITICAL"

    return impact_score, impact_level