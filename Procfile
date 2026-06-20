web: python manage.py migrate && gunicorn traffic_web.wsgi:application --bind 0.0.0.0:${PORT:-8000}
