# Kepler Exoplanet Detection - Multi-stage Docker Build
# Supports Keras CNN, XGBoost, and Random Forest models

# Stage 1: Base image with common dependencies
FROM python:3.11-slim as base

LABEL maintainer="System Architecture Team"
LABEL description="Kepler Exoplanet Detection Inference Server"
LABEL version="1.0.0"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Build dependencies
FROM base as builder

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 3: Production image
FROM base as production

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY src/deployment/model_server_template.py /app/src/deployment/
COPY src/deployment/model_registry_manager.py /app/src/deployment/

# Create directories
RUN mkdir -p /app/models /app/logs

# Set environment variables
ENV MODEL_REGISTRY_PATH=/app/models
ENV CACHE_SIZE=3
ENV ENABLE_AB_TESTING=false
ENV DEFAULT_MODEL=auto
ENV PYTHONUNBUFFERED=1

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=40s \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run application
CMD ["uvicorn", "src.deployment.model_server_template:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]

# ============================================================================
# Build Instructions:
# ============================================================================
#
# Basic build:
#   docker build -t kepler-model-server:latest -f src/deployment/Dockerfile .
#
# Build with specific tag:
#   docker build -t kepler-model-server:1.0.0 -f src/deployment/Dockerfile .
#
# Run container:
#   docker run -d \
#     -p 8000:8000 \
#     -v $(pwd)/models:/app/models:ro \
#     --name kepler-server \
#     kepler-model-server:latest
#
# Run with GPU support (NVIDIA):
#   docker run -d \
#     --gpus all \
#     -p 8000:8000 \
#     -v $(pwd)/models:/app/models:ro \
#     --name kepler-server-gpu \
#     kepler-model-server:latest
#
# Run with environment variables:
#   docker run -d \
#     -p 8000:8000 \
#     -v $(pwd)/models:/app/models:ro \
#     -e DEFAULT_MODEL=keras_cnn \
#     -e CACHE_SIZE=5 \
#     --name kepler-server \
#     kepler-model-server:latest
#
# Check logs:
#   docker logs -f kepler-server
#
# Health check:
#   curl http://localhost:8000/api/v1/health
#
# ============================================================================
