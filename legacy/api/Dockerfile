# Battery SOH Prediction API
# Multi-stage build for smaller image
# Endpoints include SOH-band disposal and lab precautions (/disposal/ranges, /disposal/guidance/{soh})

# ============== Build Stage ==============
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# ============== Runtime Stage ==============
FROM python:3.11-slim AS runtime

WORKDIR /app

# Create non-root runtime user
RUN useradd --create-home --shell /bin/bash appuser

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY api/ ./api/
COPY src/ ./src/
COPY models/ ./models/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV PORT=8000

# Ensure runtime user can access app files
RUN chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

LABEL description="Battery SOH API (no bundled frontend in this image)"
LABEL org.opencontainers.image.description="Li-ion SOH API with disposal and analysis precautions by SOH band"

# Health check using Python stdlib
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)"

# Run server with production worker manager
CMD ["gunicorn", "api.main:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120", "--graceful-timeout", "30", "--keep-alive", "5", "--access-logfile", "-", "--error-logfile", "-"]
