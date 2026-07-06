FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY core/ ./core/
COPY flows/ ./flows/
COPY models/ ./models/
COPY services/ ./services/
COPY main.py .

CMD ["sh", "-c", "test -n \"$PORT\" || (echo 'PORT env var is required' >&2; exit 1); exec uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
