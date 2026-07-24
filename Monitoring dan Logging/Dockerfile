FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY 3.prometheus_exporter.py .
COPY 7.Inference.py .

CMD ["python", "7.Inference.py"]
