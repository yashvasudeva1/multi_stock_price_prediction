# syntax: docker/dockerfile
FROM python:3.12-slim

# Install OS-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY app/ app/

# Copy model files
COPY us_stock_lstm.pth .
COPY multi_stock_lstm_v2.pth .

# Copy static frontend (served from root)
COPY index.html .

# Copy results directory (used by monitoring scripts at runtime)
COPY results/ results/

# Create monitoring directory for prediction persistence
RUN mkdir -p monitoring

# Non-root user for security
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# Use $PORT if set by Render, default to 8000
ENV PORT=8000

# Production server — no --reload
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
