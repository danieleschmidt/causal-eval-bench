# Monitoring and Observability Guide

This document provides comprehensive information about monitoring, observability, and operational procedures for the Causal Eval Bench system.

## Overview

Our monitoring strategy follows the three pillars of observability:

1. **Metrics**: Quantitative measurements of system behavior
2. **Logs**: Detailed records of system events and errors
3. **Traces**: Request flow through distributed components

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │───▶│   Prometheus    │───▶│    Grafana      │
│   (Metrics)     │    │   (Collection)  │    │ (Visualization) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Structured    │    │  Alert Manager  │    │   Dashboards    │
│   Logging       │    │   (Alerting)    │    │   & Reports     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Start

```bash
# Start monitoring services
make monitoring

# Access dashboards
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin123)

# Check service health
make health
```

## Metrics Collection

### Application Metrics

The application exposes metrics at `/metrics` endpoint in Prometheus format:

#### Request Metrics
- `http_requests_total`: Total HTTP requests by method, status
- `http_request_duration_seconds`: Request duration histogram
- `http_request_size_bytes`: Request size histogram
- `http_response_size_bytes`: Response size histogram

#### Evaluation Metrics
- `evaluation_requests_total`: Total evaluation requests
- `evaluation_duration_seconds`: Evaluation duration histogram
- `evaluation_failures_total`: Failed evaluations by reason
- `evaluation_scores`: Distribution of evaluation scores
- `question_generation_total`: Generated questions by type/domain

#### Model API Metrics
- `model_api_requests_total`: API calls to model providers
- `model_api_duration_seconds`: API response times
- `model_api_errors_total`: API errors by provider/type
- `model_api_rate_limit_errors_total`: Rate limiting errors

#### System Metrics
- `process_cpu_seconds_total`: CPU usage
- `process_resident_memory_bytes`: Memory usage
- `process_open_fds`: Open file descriptors
- `process_max_fds`: Maximum file descriptors

#### Database Metrics
- `db_connections_active`: Active database connections
- `db_connections_max`: Maximum allowed connections
- `db_query_duration_seconds`: Query execution times
- `db_query_errors_total`: Database errors

#### Cache Metrics
- `cache_hits_total`: Cache hits
- `cache_misses_total`: Cache misses
- `cache_operations_total`: Total cache operations
- `cache_memory_usage_bytes`: Cache memory usage

### Infrastructure Metrics

Collected via exporters:

- **Node Exporter**: System metrics (CPU, memory, disk, network)
- **Postgres Exporter**: Database performance metrics
- **Redis Exporter**: Cache performance metrics

## Health Checks

### Application Health

The `/health` endpoint provides comprehensive health status:

```json
{
  "status": "healthy",
  "timestamp": "2025-01-28T10:00:00Z",
  "checks": {
    "database": {
      "status": "healthy",
      "response_time_ms": 2,
      "details": "Connected to PostgreSQL"
    },
    "redis": {
      "status": "healthy", 
      "response_time_ms": 1,
      "details": "Redis responding to ping"
    },
    "external_apis": {
      "status": "degraded",
      "details": {
        "openai": "healthy",
        "anthropic": "timeout"
      }
    }
  },
  "version": "0.1.0",
  "environment": "production"
}
```

### Health Check Configuration

Health checks are configured in `config/monitoring/health_checks.json`:

```json
{
  "health_checks": {
    "database": {
      "enabled": true,
      "timeout": 5,
      "query": "SELECT 1"
    },
    "redis": {
      "enabled": true,
      "timeout": 3,
      "command": "ping"
    },
    "external_apis": {
      "enabled": true,
      "timeout": 10,
      "endpoints": [
        "https://api.openai.com/v1/models",
        "https://api.anthropic.com/v1/messages"
      ]
    }
  }
}
```

## Alerting

### Alert Rules

Alerts are defined in `docker/prometheus/alert_rules.yml`:

#### Critical Alerts
- **ServiceDown**: Service unavailable for >1 minute
- **DatabaseConnectionPoolExhausted**: >90% connections used
- **HighMemoryUsage**: >90% memory utilization

#### Warning Alerts
- **HighErrorRate**: >10% error rate for >2 minutes
- **HighLatency**: 95th percentile >2 seconds
- **HighEvaluationFailureRate**: >5% evaluation failures
- **SlowDatabaseQueries**: 95th percentile >1 second

#### Info Alerts
- **LowThroughput**: <0.1 requests/second for >10 minutes
- **LowCacheHitRate**: <80% cache hit rate
- **UnusualScoreDistribution**: Median score deviates >30% from 0.5

### Alert Manager Configuration

```yaml
# alertmanager.yml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@causal-eval-bench.org'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
- name: 'web.hook'
  email_configs:
  - to: 'ops@causal-eval-bench.org'
    subject: 'Alert: {{ .GroupLabels.alertname }}'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      {{ end }}
```

## Dashboards

### Grafana Dashboards

Pre-configured dashboards available at `docker/grafana/dashboards/`:

#### Application Overview
- Request rate, error rate, latency (RED metrics)
- Evaluation performance metrics
- System resource utilization
- Database and cache performance

#### Evaluation Performance
- Evaluation request volume and success rate
- Duration distribution by task type and difficulty
- Model API performance and errors
- Question generation metrics

#### Infrastructure
- System metrics (CPU, memory, disk, network)  
- Database performance and connection pool status
- Cache hit rates and memory usage
- Container resource utilization

#### Business Metrics
- Model performance scores over time
- Domain-specific evaluation trends
- User engagement and API usage patterns
- Cost tracking for model API usage

### Dashboard Configuration

Dashboards are provisioned automatically via:

```yaml
# docker/grafana/provisioning/dashboards/dashboard.yml
apiVersion: 1

providers:
- name: 'default'
  orgId: 1
  folder: ''
  type: file
  disableDeletion: false
  updateIntervalSeconds: 10
  options:
    path: /etc/grafana/provisioning/dashboards
```

## Logging

### Structured Logging

All logs use structured JSON format:

```json
{
  "timestamp": "2025-01-28T10:00:00.123Z",
  "level": "INFO",
  "logger": "causal_eval.evaluation",
  "message": "Evaluation completed",
  "request_id": "req_123456",
  "evaluation_id": "eval_789012",
  "duration_ms": 1234,
  "score": 0.85,
  "task_type": "causal_attribution",
  "model": "gpt-4"
}
```

### Log Levels

- **DEBUG**: Detailed debugging information
- **INFO**: General information about system operation
- **WARNING**: Something unexpected happened but system continues
- **ERROR**: Serious problem that prevented operation completion
- **CRITICAL**: Very serious error that may cause system shutdown

### Log Configuration

```python
# Python logging configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'json': {
            '()': 'causal_eval.utils.logging.JSONFormatter',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'json',
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/app/logs/causal_eval.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'json',
        },
    },
    'loggers': {
        'causal_eval': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}
```

### Log Aggregation

For production deployments, integrate with log aggregation systems:

#### ELK Stack
```yaml
# docker-compose.yml
elasticsearch:
  image: elasticsearch:8.11.0
  environment:
    - discovery.type=single-node
    - "ES_JAVA_OPTS=-Xms512m -Xmx512m"

logstash:
  image: logstash:8.11.0
  volumes:
    - ./config/logstash.conf:/usr/share/logstash/pipeline/logstash.conf

kibana:
  image: kibana:8.11.0
  ports:
    - "5601:5601"
```

#### Fluentd
```yaml
fluentd:
  image: fluentd:latest
  volumes:
    - ./config/fluent.conf:/fluentd/etc/fluent.conf
    - /app/logs:/var/log/app
```

## Tracing

### Distributed Tracing Setup

Using OpenTelemetry for distributed tracing:

```python
# tracing.py
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)
```

### Tracing Example

```python
@tracer.start_as_current_span("evaluate_model")
def evaluate_model(model_name: str, question: dict):
    span = trace.get_current_span()
    span.set_attribute("model.name", model_name)
    span.set_attribute("question.type", question["task_type"])
    
    with tracer.start_as_current_span("call_model_api"):
        response = call_model_api(model_name, question["prompt"])
    
    with tracer.start_as_current_span("evaluate_response"):
        score = evaluate_response(response, question["ground_truth"])
    
    span.set_attribute("evaluation.score", score)
    return score
```

## Performance Monitoring

### Key Performance Indicators (KPIs)

#### Availability
- **Uptime**: Target 99.9% (43.2 minutes downtime/month)
- **Error Rate**: Target <1% of requests
- **Response Time**: 95th percentile <2 seconds

#### Performance
- **Throughput**: Target 100 evaluations/minute
- **Evaluation Duration**: Median <30 seconds
- **Database Query Time**: 95th percentile <1 second
- **Cache Hit Rate**: Target >80%

#### Business Metrics
- **Evaluation Success Rate**: Target >95%
- **Model API Success Rate**: Target >99%
- **Question Generation Rate**: Target 10 questions/minute
- **Cost per Evaluation**: Track API costs

### Performance Baselines

Establish baselines through load testing:

```bash
# Run performance tests
make test-performance-docker

# Load testing with Locust
locust -f tests/load/locustfile.py --host http://localhost:8000
```

## Runbooks

### Common Scenarios

#### High Error Rate Alert

1. **Check service status**: `make health`
2. **Review recent logs**: `make logs-app | grep ERROR`
3. **Check external API status**: Verify model API availability
4. **Scale if needed**: Increase container replicas
5. **Roll back if necessary**: Deploy previous version

#### Database Connection Issues

1. **Check database health**: `make shell-db`
2. **Review connection pool**: Check active/max connections
3. **Check for long-running queries**: Review `pg_stat_activity`
4. **Restart database if needed**: `docker-compose restart postgres`
5. **Scale database**: Consider read replicas

#### High Memory Usage

1. **Identify memory-intensive processes**: `docker stats`
2. **Check for memory leaks**: Review memory usage trends
3. **Optimize queries**: Review database query plans
4. **Scale vertically**: Increase container memory limits
5. **Scale horizontally**: Add more instances

### Incident Response

#### Severity Levels

- **P0 (Critical)**: Complete service outage
- **P1 (High)**: Major functionality unavailable
- **P2 (Medium)**: Minor functionality affected
- **P3 (Low)**: Performance degradation

#### Response Times

- **P0**: 15 minutes
- **P1**: 1 hour
- **P2**: 4 hours  
- **P3**: 24 hours

#### Communication Channels

- **Status Page**: Update public status page
- **Slack**: Internal team notifications
- **Email**: Stakeholder updates for P0/P1
- **Documentation**: Post-incident reviews

## Capacity Planning

### Resource Monitoring

Track resource usage trends:

- **CPU Utilization**: Plan scaling at 70% sustained usage
- **Memory Usage**: Plan scaling at 80% utilization
- **Disk Usage**: Monitor growth rate and plan expansion
- **Network Bandwidth**: Monitor for bottlenecks

### Scaling Triggers

Automatic scaling based on metrics:

```yaml
# Kubernetes HPA example
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
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Cost Monitoring

### API Cost Tracking

Monitor model API usage and costs:

```python
# Cost tracking example
@cost_tracker.track_api_call
def call_openai_api(prompt: str, model: str):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response
```

### Cost Optimization

- **Model Selection**: Use cheaper models for simpler tasks
- **Caching**: Cache API responses to reduce calls
- **Batch Processing**: Group requests where possible
- **Rate Limiting**: Prevent runaway costs

## Security Monitoring

### Security Metrics

- **Authentication Failures**: Track failed login attempts
- **API Abuse**: Monitor for unusual request patterns
- **Data Access**: Log sensitive data access
- **Vulnerability Scans**: Regular security assessments

### Security Alerts

- **Brute Force Attacks**: >10 failed attempts/minute
- **Unusual API Usage**: >1000% increase in requests
- **Data Breaches**: Unauthorized access to sensitive data
- **Vulnerability Discovery**: New CVEs affecting dependencies

## Maintenance

### Regular Tasks

#### Daily
- Review alert status and resolve issues
- Check system resource usage
- Monitor error rates and performance

#### Weekly
- Review capacity planning metrics
- Update dashboards and alerts
- Analyze performance trends

#### Monthly
- Performance baseline review
- Cost analysis and optimization
- Security audit and updates

#### Quarterly
- Disaster recovery testing
- Monitoring system updates
- Capacity planning review

### Monitoring System Maintenance

```bash
# Update Prometheus configuration
docker-compose restart prometheus

# Update Grafana dashboards
# Import new dashboard JSON files

# Backup monitoring data
docker run --rm -v prometheus_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/prometheus_backup.tar.gz /data

# Restore monitoring data
docker run --rm -v prometheus_data:/data -v $(pwd):/backup \
  alpine tar xzf /backup/prometheus_backup.tar.gz -C /
```

## Troubleshooting

### Common Issues

#### Metrics Not Appearing

1. Check if application is exposing metrics endpoint
2. Verify Prometheus configuration and targets
3. Check network connectivity between services
4. Review Prometheus logs for scraping errors

#### Alerts Not Firing

1. Verify alert rules syntax
2. Check if metrics are being collected
3. Confirm Alert Manager configuration
4. Test notification channels

#### Dashboard Issues

1. Check Grafana datasource configuration
2. Verify metric names and labels
3. Test queries in Prometheus directly
4. Review Grafana logs for errors

### Debug Commands

```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Query specific metric
curl 'http://localhost:9090/api/v1/query?query=up'

# Check Alert Manager status
curl http://localhost:9093/api/v1/status

# Test Grafana API
curl -u admin:admin123 http://localhost:3000/api/health
```

---

For additional monitoring questions or to contribute improvements, please open an issue or submit a pull request.