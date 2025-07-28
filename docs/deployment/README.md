# Deployment Guide for Causal Eval Bench

This document provides comprehensive information about building, containerizing, and deploying the Causal Eval Bench application.

## Quick Start

```bash
# Development environment
make dev
make run

# Production build
make build
make docker
```

## Build System Overview

The project uses a multi-layered build system optimized for different environments:

1. **Poetry**: Python dependency management and packaging
2. **Docker**: Multi-stage containerization for different environments
3. **Docker Compose**: Orchestration of services and dependencies
4. **Makefile**: Standardized build commands and automation

## Local Development

### Prerequisites

- Python 3.9+
- Poetry
- Docker & Docker Compose
- Make

### Setup Development Environment

```bash
# Install dependencies and setup pre-commit hooks
make dev

# Start all services in development mode
make run-dev

# Access services
# API: http://localhost:8000
# Documentation: http://localhost:8080
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000
```

### Development Workflow

```bash
# Format code
make format

# Run linting
make lint

# Run tests
make test

# Run specific test types
make test-unit
make test-integration
make test-performance

# View logs
make logs
make logs-app

# Access container shell
make shell
```

## Docker Architecture

### Multi-Stage Dockerfile

Our Dockerfile uses multi-stage builds for optimization:

1. **Builder Stage**: Installs dependencies and builds the application
2. **Runtime Stage**: Minimal runtime environment with only necessary files
3. **Development Stage**: Includes development tools and hot-reloading
4. **Production Stage**: Optimized for production with nginx and supervisor
5. **Testing Stage**: Pre-configured for running tests

### Build Targets

```bash
# Development image (default)
docker build --target development -t causal-eval:dev .

# Production image
docker build --target production -t causal-eval:prod .

# Testing image
docker build --target testing -t causal-eval:test .

# Runtime image (minimal)
docker build --target runtime -t causal-eval:runtime .
```

### Image Optimization

- **Base Image**: python:3.11-slim-bullseye for security and size
- **Non-root User**: Runs as appuser (UID 1000) for security
- **Multi-stage**: Separates build and runtime dependencies
- **Layer Caching**: Optimized layer ordering for build speed
- **.dockerignore**: Reduces build context size by 90%

## Docker Compose Services

### Core Services

```yaml
# Main application with hot-reloading
app:
  ports: ["8000:8000"]
  
# PostgreSQL database
postgres:
  ports: ["5432:5432"]
  
# Redis cache
redis:
  ports: ["6379:6379"]
  
# Background worker
worker:
  # Celery worker for async tasks
  
# Documentation server
docs:
  ports: ["8080:8080"]
```

### Service Profiles

Services are organized by profiles for different use cases:

```bash
# Default (development)
docker-compose up

# Testing
docker-compose --profile testing up test

# Performance testing
docker-compose --profile performance up performance-test

# Monitoring (Prometheus + Grafana)
docker-compose --profile monitoring up prometheus grafana

# Production (with nginx)
docker-compose --profile production up
```

### Networking

- **Custom Network**: `causal-eval-network` (172.20.0.0/16)
- **Service Discovery**: Services communicate by name
- **Port Mapping**: External access to key services
- **Health Checks**: Automatic health monitoring

## Build Commands Reference

### Makefile Commands

#### Setup
- `make install`: Install dependencies
- `make dev`: Setup development environment
- `make clean`: Clean build artifacts

#### Code Quality
- `make format`: Format code with Black and isort
- `make lint`: Run all linters (Ruff, MyPy, Bandit)
- `make lint-fix`: Auto-fix linting issues

#### Testing
- `make test`: Run all tests with coverage
- `make test-unit`: Unit tests only
- `make test-integration`: Integration tests only
- `make test-e2e`: End-to-end tests only
- `make test-performance`: Performance benchmarks
- `make coverage`: Open coverage report

#### Docker
- `make docker`: Build production Docker image
- `make docker-dev`: Build development Docker image
- `make run`: Start all services
- `make stop`: Stop all services
- `make logs`: View service logs
- `make shell`: Access app container shell

#### Build & Release
- `make build`: Build Python packages
- `make release-test`: Release to test PyPI
- `make release`: Release to production PyPI

## Environment Configuration

### Environment Variables

Key environment variables for deployment:

```bash
# Application
ENVIRONMENT=production
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000

# Database
DATABASE_URL=postgresql://user:pass@host:5432/db
POSTGRES_USER=causal_eval_user
POSTGRES_PASSWORD=secure_password
POSTGRES_DB=causal_eval_bench

# Redis
REDIS_URL=redis://redis:6379/0

# Security
JWT_SECRET_KEY=your-secret-key
API_REQUIRE_AUTH=true

# Model APIs
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key

# Monitoring
SENTRY_DSN=your-sentry-dsn
ENABLE_METRICS=true
```

### Configuration Files

- `.env.example`: Template for environment variables
- `docker-compose.yml`: Service orchestration
- `docker/`: Configuration files for services
  - `nginx/nginx.conf`: Reverse proxy configuration
  - `prometheus/prometheus.yml`: Metrics collection
  - `grafana/`: Dashboard configurations

## Production Deployment

### Container Registry

```bash
# Build and tag for registry
docker build -t your-registry/causal-eval:v1.0.0 .

# Push to registry
docker push your-registry/causal-eval:v1.0.0
```

### Kubernetes Deployment

Example Kubernetes deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: causal-eval-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: causal-eval
  template:
    metadata:
      labels:
        app: causal-eval
    spec:
      containers:
      - name: app
        image: your-registry/causal-eval:v1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: causal-eval-secrets
              key: database-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

### Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml causal-eval

# Scale services
docker service scale causal-eval_app=3
```

### Cloud Deployment

#### AWS ECS

```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name causal-eval

# Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create service
aws ecs create-service --cluster causal-eval --service-name causal-eval-app --task-definition causal-eval:1 --desired-count 2
```

#### Google Cloud Run

```bash
# Build and submit
gcloud builds submit --tag gcr.io/PROJECT-ID/causal-eval

# Deploy
gcloud run deploy --image gcr.io/PROJECT-ID/causal-eval --platform managed
```

#### Azure Container Instances

```bash
# Create resource group
az group create --name causal-eval-rg --location eastus

# Deploy container
az container create --resource-group causal-eval-rg --name causal-eval --image your-registry/causal-eval:latest --port 8000
```

## Performance Optimization

### Build Optimization

1. **Multi-stage Builds**: Separate build and runtime dependencies
2. **Layer Caching**: Order Dockerfile instructions for optimal caching
3. **Minimal Base Images**: Use slim/alpine variants where possible
4. **Dependency Caching**: Cache Poetry dependencies between builds

### Runtime Optimization

1. **Resource Limits**: Set appropriate CPU and memory limits
2. **Health Checks**: Configure proper health check endpoints
3. **Graceful Shutdown**: Handle SIGTERM for clean shutdowns
4. **Connection Pooling**: Configure database connection pools

### Example Resource Configuration

```yaml
# docker-compose.yml
services:
  app:
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
        reservations:
          memory: 512M
          cpus: '0.25'
```

## Monitoring and Observability

### Health Checks

All services include health checks:

```bash
# Application health
curl http://localhost:8000/health

# Database health
docker-compose exec postgres pg_isready

# Redis health
docker-compose exec redis redis-cli ping

# All services health
make health
```

### Metrics Collection

- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and alerting
- **Application Metrics**: Custom metrics via prometheus_client

### Logging

- **Structured Logging**: JSON format for production
- **Log Levels**: Configurable via LOG_LEVEL environment variable
- **Log Rotation**: Automatic log rotation in production
- **Centralized Logging**: Integration with ELK stack or similar

## Security Best Practices

### Container Security

1. **Non-root User**: Runs as unprivileged user (UID 1000)
2. **Minimal Images**: Use slim base images with fewer vulnerabilities
3. **Security Scanning**: Regular vulnerability scans
4. **Secrets Management**: Never embed secrets in images

### Network Security

1. **Custom Networks**: Isolated Docker networks
2. **Port Exposure**: Only expose necessary ports
3. **TLS/SSL**: HTTPS in production
4. **Firewall Rules**: Restrict network access

### Secrets Management

```bash
# Docker secrets
echo "secret-value" | docker secret create secret-name -

# Environment variables from secrets
docker service create --secret secret-name your-image
```

## Troubleshooting

### Common Issues

#### Build Issues

```bash
# Clear Docker cache
docker system prune -a

# Rebuild without cache
docker build --no-cache -t causal-eval .

# Check build context size
du -sh . && wc -l .dockerignore
```

#### Runtime Issues

```bash
# Check service logs
make logs-app

# Check service health
make health

# Access container for debugging
make shell

# Check resource usage
docker stats
```

#### Database Issues

```bash
# Check database connectivity
make shell-db

# Reset database (development)
make db-reset

# Run migrations
make db-migrate
```

### Performance Issues

```bash
# Monitor resource usage
docker stats

# Profile application
docker-compose exec app python -m cProfile -o profile.out your_script.py

# Analyze database performance
docker-compose exec postgres pg_stat_activity
```

## Maintenance

### Regular Tasks

1. **Update Dependencies**: `make dev && poetry update`
2. **Security Scans**: `make security`
3. **Database Backups**: Regular PostgreSQL backups
4. **Log Rotation**: Monitor and rotate log files
5. **Image Updates**: Rebuild with latest base images

### Backup and Recovery

```bash
# Database backup
docker-compose exec postgres pg_dump -U causal_eval_user causal_eval_bench > backup.sql

# Database restore
docker-compose exec -T postgres psql -U causal_eval_user causal_eval_bench < backup.sql

# Volume backup
docker run --rm -v causal-eval-bench_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_backup.tar.gz /data
```

## Contributing

### Development Workflow

1. **Setup**: `make dev`
2. **Code**: Make your changes
3. **Test**: `make test`
4. **Lint**: `make lint`
5. **Build**: `make docker-dev`
6. **Submit**: Create pull request

### Build System Changes

When modifying the build system:

1. **Update Documentation**: Keep this guide current
2. **Test All Targets**: Verify all Dockerfile stages work
3. **Check Performance**: Monitor build times and image sizes
4. **Validate Security**: Run security scans on new images

---

For more information about specific deployment scenarios or troubleshooting, please refer to the individual service documentation or open an issue.