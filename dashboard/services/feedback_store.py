import csv
from datetime import datetime
from pathlib import Path

from django.conf import settings


FEEDBACK_PATH = (
    Path(settings.BASE_DIR)
    / "data"
    / "post_event_feedback.csv"
)


FIELDNAMES = [
    "submitted_at",

    "event_type",
    "event_cause",
    "priority",

    "latitude",
    "longitude",
    "corridor",

    "predicted_incidents",
    "predicted_duration_minutes",
    "predicted_officers",
    "predicted_barricades",
    "predicted_final_risk",

    "actual_duration_minutes",
    "actual_officers_deployed",
    "actual_barricades_used",
    "actual_road_closure",
    "actual_incident_count",

    "officer_notes",
]


def save_post_event_feedback(payload):
    FEEDBACK_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    file_exists = FEEDBACK_PATH.exists()

    row = {
        "submitted_at": datetime.now().isoformat(timespec="seconds"),

        "event_type": payload.get("event_type", ""),
        "event_cause": payload.get("event_cause", ""),
        "priority": payload.get("priority", ""),

        "latitude": payload.get("latitude", ""),
        "longitude": payload.get("longitude", ""),
        "corridor": payload.get("corridor", ""),

        "predicted_incidents": payload.get("predicted_incidents", ""),
        "predicted_duration_minutes": payload.get("predicted_duration_minutes", ""),
        "predicted_officers": payload.get("predicted_officers", ""),
        "predicted_barricades": payload.get("predicted_barricades", ""),
        "predicted_final_risk": payload.get("predicted_final_risk", ""),

        "actual_duration_minutes": payload.get("actual_duration_minutes", ""),
        "actual_officers_deployed": payload.get("actual_officers_deployed", ""),
        "actual_barricades_used": payload.get("actual_barricades_used", ""),
        "actual_road_closure": payload.get("actual_road_closure", ""),
        "actual_incident_count": payload.get("actual_incident_count", ""),

        "officer_notes": payload.get("officer_notes", ""),
    }

    with open(
        FEEDBACK_PATH,
        mode="a",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    return str(FEEDBACK_PATH)