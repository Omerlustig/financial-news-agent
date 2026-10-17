# ==============================================================================
# Market News Agent - Production Container
# Multi-stage build for minimal footprint, hardened non-root security,
# and high-performance continuous financial intelligence monitoring.
# ==============================================================================

# Stage 1: Build & dependency resolution
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Minimal runtime image
FROM python:3.11-slim AS runner

WORKDIR /app

# Install runtime dependencies (ca-certificates for HTTPS feed feeds)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Create non-root application user and persistent data directory
RUN groupadd -r appuser && useradd -r -g appuser -u 1000 -d /app appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app /app/data \
    && chmod 755 /app/data

# Environment configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    DATABASE_PATH=/app/data/agent.db

# Copy configuration templates and application source
COPY --chown=appuser:appuser config/ /app/config/
COPY --chown=appuser:appuser src/ /app/src/

# Declare persistent volume mount for SQLite database
VOLUME ["/app/data"]

# Switch to non-root user
USER appuser

# Health check to ensure agent connectivity and schema readiness
HEALTHCHECK --interval=5m --timeout=15s --start-period=30s --retries=3 \
    CMD python -m src.main test-connections || exit 1

# Default entrypoint: continuous background monitoring daemon (15 min interval)
ENTRYPOINT ["python", "-m", "src.main", "run", "--daemon", "--interval", "15"]
