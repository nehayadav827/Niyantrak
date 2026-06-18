from django.urls import path

from .views import dashboard_view
from .views import feedback_view


urlpatterns = [
    path(
        "",
        dashboard_view,
        name="dashboard"
    ),

    path(
        "feedback/",
        feedback_view,
        name="post_event_feedback"
    ),
]