from prometheus_client import make_asgi_app

# Create prometheus metric exporter for FastAPI
metrics_app = make_asgi_app()
