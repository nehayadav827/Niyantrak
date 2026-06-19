from django.shortcuts import redirect
from django.shortcuts import render

from .services.ml_engine import predict_event_impact
from .services.feedback_store import save_post_event_feedback


DEFAULT_FORM = {
    "event_type": "unplanned",
    "event_cause": "vehicle_breakdown",
    "priority": "High",

    "latitude": "12.9716",
    "longitude": "77.5946",
    "end_latitude": "",
    "end_longitude": "",

    "corridor": "Auto inferred after prediction",

    "veh_type": "",
    "police_station": "",
    "timestamp": "",
    "requires_road_closure": "no",

    "crowd_size": "unknown",
    "weather": "clear",
}


def dashboard_view(request):
    result = None
    error = None

    feedback_status = request.GET.get(
        "feedback"
    )

    form_data = DEFAULT_FORM.copy()

    if request.method == "POST":
        form_data.update(
            request.POST.dict()
        )

        try:
            result = predict_event_impact(
                form_data
            )

            form_data["corridor"] = result["input"].get(
                "corridor",
                "Auto inferred after prediction"
            )

            form_data["latitude"] = str(
                result["input"].get(
                    "latitude",
                    form_data.get("latitude", "")
                )
            )

            form_data["longitude"] = str(
                result["input"].get(
                    "longitude",
                    form_data.get("longitude", "")
                )
            )

            form_data["crowd_size"] = result["input"].get(
                "crowd_size",
                form_data.get("crowd_size", "small")
            )

            form_data["weather"] = result["input"].get(
                "weather",
                form_data.get("weather", "clear")
            )

        except Exception as e:
            error = str(e)

    return render(
        request,
        "dashboard/index.html",
        {
            "form": form_data,
            "result": result,
            "error": error,
            "feedback_status": feedback_status,
        }
    )


def feedback_view(request):
    if request.method == "POST":
        save_post_event_feedback(
            request.POST.dict()
        )

    return redirect(
        "/?feedback=saved"
    )