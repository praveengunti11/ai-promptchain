# Multi-stage Dockerfile to produce a smaller runtime image
FROM python:3.11-slim as builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install build dependencies needed to compile some wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       gcc \
       libffi-dev \
       libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and build wheels to avoid recompiling on every change
COPY requirements.txt ./

RUN pip install --upgrade pip setuptools wheel \
    && pip wheel --wheel-dir /wheels -r requirements.txt

## Final image: keep it small and runtime-only
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal runtime deps (add more if your requirements need them)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy built wheels from the builder stage and install them
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* \
    && rm -rf /wheels

# Copy application source
COPY . /app

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit so it's reachable by the host
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
