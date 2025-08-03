# Production Deployment Guide

## Overview

This guide covers deploying Causal Eval Bench in production environments with high availability, security, and performance considerations.

## Architecture Options

### Option 1: Docker Compose (Small Scale)

Best for: Small teams, development, staging environments

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  app:
    image: causal-eval:latest
    restart: unless-stopped
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/causal_eval
      - REDIS_URL=redis://redis:6379
      - SECRET_KEY=${SECRET_KEY}
    depends_on:
      - postgres
      - redis
    ports:
      - "8080:8080"

  postgres:
    image: postgres:15-alpine
    restart: unless-stopped
    environment:
      - POSTGRES_DB=causal_eval
      - POSTGRES_USER=${DB_USER}
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app

volumes:
  postgres_data:
  redis_data:
```

### Option 2: Kubernetes (Enterprise Scale)

Best for: Large scale, high availability, multi-region deployments

```yaml
# k8s/deployment.yaml
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
        image: causal-eval:latest
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: database-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

## Environment Configuration

### Required Environment Variables

```bash
# Application Settings
SECRET_KEY=your-secret-key-here
APP_ENV=production
DEBUG=false
LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql://user:pass@host:5432/causal_eval
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Redis
REDIS_URL=redis://host:6379
REDIS_MAX_CONNECTIONS=50

# External APIs
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key

# Security
CORS_ORIGINS=https://your-domain.com
ALLOWED_HOSTS=your-domain.com,api.your-domain.com

# Monitoring
SENTRY_DSN=your-sentry-dsn
PROMETHEUS_METRICS=true
```

### Optional Environment Variables

```bash
# Performance
WORKER_TIMEOUT=300
WORKER_PROCESSES=4
MAX_CONCURRENT_EVALUATIONS=10

# Features
ENABLE_RATE_LIMITING=true
ENABLE_CACHING=true
CACHE_TTL=3600

# Storage
S3_BUCKET=causal-eval-results
S3_REGION=us-west-2
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
```

## Database Setup

### PostgreSQL Configuration

**Recommended Settings for Production:**

```sql
-- postgresql.conf
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB
max_connections = 100
random_page_cost = 1.1
effective_io_concurrency = 200
```

**Initialize Database:**

```bash
# Create database and user
createdb causal_eval
createuser causal_eval_user

# Grant privileges
psql -c "GRANT ALL PRIVILEGES ON DATABASE causal_eval TO causal_eval_user;"

# Run migrations
alembic upgrade head
```

### Database Backup Strategy

```bash
#!/bin/bash
# backup.sh - Daily database backup

BACKUP_DIR="/backups"
DB_NAME="causal_eval"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create backup
pg_dump $DB_NAME > "$BACKUP_DIR/causal_eval_$TIMESTAMP.sql"

# Compress backup
gzip "$BACKUP_DIR/causal_eval_$TIMESTAMP.sql"

# Remove backups older than 30 days
find $BACKUP_DIR -name "causal_eval_*.sql.gz" -mtime +30 -delete

# Upload to S3 (optional)
aws s3 cp "$BACKUP_DIR/causal_eval_$TIMESTAMP.sql.gz" \
    s3://your-backup-bucket/database/
```

## Load Balancing & High Availability

### Nginx Configuration

```nginx
# nginx.conf
upstream causal_eval_backend {
    least_conn;
    server app1:8080 max_fails=3 fail_timeout=30s;
    server app2:8080 max_fails=3 fail_timeout=30s;
    server app3:8080 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name api.your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.your-domain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;

    location / {
        proxy_pass http://causal_eval_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    location /health {
        access_log off;
        proxy_pass http://causal_eval_backend;
    }
}
```

## Security Hardening

### SSL/TLS Configuration

**Obtain SSL Certificate:**

```bash
# Using Let's Encrypt
certbot --nginx -d api.your-domain.com

# Or use your certificate provider
# Copy cert.pem and key.pem to /etc/nginx/ssl/
```

### Firewall Rules

```bash
# UFW rules for Ubuntu
ufw default deny incoming
ufw default allow outgoing

# SSH access
ufw allow 22/tcp

# HTTP/HTTPS
ufw allow 80/tcp
ufw allow 443/tcp

# Database (only from app servers)
ufw allow from 10.0.0.0/8 to any port 5432

# Redis (only from app servers)  
ufw allow from 10.0.0.0/8 to any port 6379

ufw enable
```

### Application Security

**secrets.yaml** (for Kubernetes):

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
type: Opaque
data:
  database-url: <base64-encoded-database-url>
  secret-key: <base64-encoded-secret-key>
  openai-api-key: <base64-encoded-openai-key>
```

**Security Checklist:**

- [ ] Use strong, unique passwords
- [ ] Enable database connection SSL
- [ ] Rotate API keys regularly
- [ ] Enable audit logging
- [ ] Use least privilege access
- [ ] Regular security scans
- [ ] Keep dependencies updated

## Monitoring & Observability

### Prometheus Metrics

The application exposes metrics at `/metrics`:

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'causal-eval'
    static_configs:
      - targets: ['app:8080']
    metrics_path: /metrics
```

**Key Metrics to Monitor:**

- `http_requests_total`: Request count by endpoint
- `http_request_duration_seconds`: Request latency
- `evaluation_duration_seconds`: Evaluation processing time
- `active_evaluations`: Number of running evaluations
- `database_connections`: Database connection pool usage

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Causal Eval Bench",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.95, http_request_duration_seconds_bucket)"
          }
        ]
      }
    ]
  }
}
```

### Logging Configuration

```python
# logging_config.py
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "json",
            "filename": "/var/log/causal_eval/app.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"]
    }
}
```

## Performance Optimization

### Application Tuning

```python
# uvicorn_config.py
UVICORN_CONFIG = {
    "host": "0.0.0.0",
    "port": 8080,
    "workers": 4,
    "worker_class": "uvicorn.workers.UvicornWorker",
    "max_requests": 1000,
    "max_requests_jitter": 100,
    "timeout": 300,
    "keepalive": 2
}
```

### Caching Strategy

```python
# Redis caching for evaluation results
CACHE_CONFIG = {
    "evaluation_results": {
        "ttl": 3600,  # 1 hour
        "key_pattern": "eval:{evaluation_id}"
    },
    "test_questions": {
        "ttl": 86400,  # 24 hours  
        "key_pattern": "questions:{task}:{domain}"
    },
    "model_responses": {
        "ttl": 7200,  # 2 hours
        "key_pattern": "response:{model}:{question_hash}"
    }
}
```

## Disaster Recovery

### Backup Strategy

1. **Database Backups**: Daily full backups, hourly incrementals
2. **File Storage**: Replicated across multiple regions
3. **Configuration**: Version controlled in Git
4. **Secrets**: Backed up in secure key management

### Recovery Procedures

**Database Recovery:**

```bash
# Restore from backup
gunzip causal_eval_20240115_120000.sql.gz
psql causal_eval < causal_eval_20240115_120000.sql

# Verify data integrity
python scripts/verify_data.py
```

**Application Recovery:**

```bash
# Deploy from known good image
docker pull causal-eval:stable
docker-compose up -d

# Verify service health
curl https://api.your-domain.com/health
```

## Scaling Considerations

### Horizontal Scaling

- **Stateless Application**: Multiple app instances behind load balancer
- **Database Scaling**: Read replicas for query load
- **Cache Scaling**: Redis Cluster for high throughput
- **Background Jobs**: Separate worker processes

### Vertical Scaling

- **CPU**: Scale based on evaluation complexity
- **Memory**: Scale based on concurrent evaluations
- **Storage**: Scale based on result storage needs

### Auto-scaling (Kubernetes)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: causal-eval-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: causal-eval-app
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Maintenance

### Regular Tasks

- **Weekly**: Update dependencies
- **Monthly**: Security patch review
- **Quarterly**: Performance review and optimization
- **Annually**: Architecture review and planning

### Health Checks

```python
# health_check.py
async def health_check():
    checks = {
        "database": await check_database(),
        "redis": await check_redis(),
        "external_apis": await check_apis(),
        "disk_space": check_disk_space(),
        "memory": check_memory_usage()
    }
    
    overall_status = "healthy" if all(checks.values()) else "unhealthy"
    return {"status": overall_status, "checks": checks}
```

## Troubleshooting

### Common Issues

**High Memory Usage:**
- Check for memory leaks in evaluation code
- Monitor Redis memory usage
- Review caching policies

**Slow Responses:**
- Check database query performance
- Monitor external API latency
- Review cache hit rates

**Database Connection Issues:**
- Check connection pool settings
- Monitor connection counts
- Review database logs

### Debugging Tools

```bash
# Application logs
docker logs causal-eval-app

# Database performance
psql -c "SELECT * FROM pg_stat_activity;"

# Redis monitoring
redis-cli --latency-history

# System resources
htop
iotop
```

## Support

- **24/7 Support**: enterprise@causal-eval-bench.com
- **Status Page**: https://status.causal-eval-bench.com
- **Documentation**: https://docs.causal-eval-bench.com
- **Emergency Hotline**: +1-800-CAUSAL-HELP