from fastapi import FastAPI
from prometheus_client import Counter, Histogram, Gauge
import importlib.util
spec = importlib.util.spec_from_file_location("prometheus_exporter", "3.prometheus_exporter.py")
prometheus_exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prometheus_exporter)
metrics_app = prometheus_exporter.metrics_app
import time
import random

app = FastAPI()

app.mount("/metrics", metrics_app)

REQUEST_COUNT = Counter("api_requests_total", "Total API requests")
ERROR_COUNT = Counter("api_errors_total", "Total API errors")
LATENCY = Histogram("api_latency_seconds", "API latency in seconds")
PREDICTION_COUNT = Counter("model_predictions_total", "Total predictions made")
PREDICTION_VALUE = Histogram("model_prediction_values", "Model prediction values")
LOAD_TIME = Gauge("model_load_time_seconds", "Time taken to load the model")
MEMORY_USAGE = Gauge("process_memory_usage_bytes", "Memory usage of the process")
CPU_USAGE = Gauge("process_cpu_usage_percent", "CPU usage of the process")
ACTIVE_REQUESTS = Gauge("api_active_requests", "Number of active requests")
UPTIME = Gauge("process_uptime_seconds", "Process uptime in seconds")

start_time = time.time()

@app.on_event("startup")
def startup_event():
    LOAD_TIME.set(random.uniform(0.1, 0.5))

@app.get("/")
def read_root():
    UPTIME.set(time.time() - start_time)
    return {"status": "ok"}

@app.post("/predict")
def predict():
    ACTIVE_REQUESTS.inc()
    REQUEST_COUNT.inc()
    start = time.time()
    
    try:
        time.sleep(random.uniform(0.01, 0.1))
        pred = random.choice([0, 1])
        
        PREDICTION_COUNT.inc()
        PREDICTION_VALUE.observe(pred)
        
        return {"prediction": pred}
    except Exception as e:
        ERROR_COUNT.inc()
        return {"error": str(e)}
    finally:
        LATENCY.observe(time.time() - start)
        ACTIVE_REQUESTS.dec()
        MEMORY_USAGE.set(random.uniform(50000000, 100000000))
        CPU_USAGE.set(random.uniform(0, 100))
        UPTIME.set(time.time() - start_time)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
