# Multi-stage Dockerfile for Causal Eval Bench
# Optimized for security, performance, and minimal image size

# =============================================================================
# Build Stage
# =============================================================================
FROM python:3.11-slim-bullseye AS builder

# Set build-time environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create application user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Set working directory
WORKDIR /app

# Install Poetry
RUN pip install poetry==1.7.1

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Configure Poetry and install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --only=main --no-dev --no-interaction --no-ansi

# =============================================================================
# Runtime Stage
# =============================================================================
FROM python:3.11-slim-bullseye AS runtime

# Set runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH="/home/appuser/.local/bin:$PATH" \
    ENVIRONMENT=production

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create application user (matching builder stage)
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create application directories
RUN mkdir -p /app/logs /app/data /app/cache \
    && chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Remove development files
RUN rm -rf tests/ docs/ scripts/ .git/ .github/ \
    && find . -name "*.pyc" -delete \
    && find . -name "__pycache__" -delete

# Switch to non-root user
USER appuser

# Create necessary directories for the application user
RUN mkdir -p /home/appuser/.cache/pip

# Expose application port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "causal_eval.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# =============================================================================
# Development Stage (for development builds)
# =============================================================================
FROM runtime AS development

# Switch back to root to install dev dependencies
USER root

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry for development
RUN pip install poetry==1.7.1

# Copy all files including dev dependencies
COPY pyproject.toml poetry.lock* ./

# Install all dependencies including dev
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Switch back to app user
USER appuser

# Override command for development
CMD ["uvicorn", "causal_eval.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--workers", "1"]

# =============================================================================
# Production Optimized Stage
# =============================================================================
FROM runtime AS production

# Additional production optimizations
USER root

# Install additional production tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Copy production configuration files
COPY --chown=appuser:appuser docker/nginx.conf /etc/nginx/nginx.conf
COPY --chown=appuser:appuser docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Create log directories
RUN mkdir -p /var/log/nginx /var/log/supervisor \
    && chown -R appuser:appuser /var/log/nginx /var/log/supervisor

# Switch to app user
USER appuser

# Expose both application and nginx ports
EXPOSE 8000 80

# Use supervisor to manage multiple processes
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]

# =============================================================================
# Testing Stage
# =============================================================================
FROM development AS testing

# Switch to root for test setup
USER root

# Install additional testing tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    redis-server \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Switch back to app user
USER appuser

# Copy test files
COPY --chown=appuser:appuser tests/ tests/

# Override command for testing
CMD ["python", "-m", "pytest", "tests/", "-v", "--cov=causal_eval", "--cov-report=html", "--cov-report=xml"]

# =============================================================================
# Labels and Metadata
# =============================================================================
LABEL maintainer="Daniel Schmidt <daniel@terragon-labs.com>" \
      version="0.1.0" \
      description="Causal Evaluation Benchmark - Comprehensive evaluation framework for causal reasoning in language models" \
      org.opencontainers.image.title="causal-eval-bench" \
      org.opencontainers.image.description="Comprehensive evaluation framework for testing genuine causal reasoning in language models" \
      org.opencontainers.image.url="https://github.com/your-org/causal-eval-bench" \
      org.opencontainers.image.source="https://github.com/your-org/causal-eval-bench" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.created="2025-01-27" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.licenses="MIT"