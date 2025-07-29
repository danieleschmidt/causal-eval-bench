# 🔍 Advanced Observability & Monitoring

## Overview

Comprehensive observability strategy for the causal evaluation framework, implementing modern monitoring, tracing, and alerting patterns for production-ready systems.

## 📊 Observability Stack

### Core Components
- **Metrics**: Prometheus + Grafana
- **Logging**: Structured logging with Loguru + Structured Log
- **Tracing**: OpenTelemetry with Jaeger
- **APM**: Application Performance Monitoring
- **Alerts**: Alertmanager + Slack/PagerDuty

## 🎯 Key Metrics

### Application Metrics
```python
# Evaluation performance metrics
evaluation_duration_seconds = Histogram(
    'evaluation_duration_seconds',
    'Time spent on evaluation tasks',
    ['task_type', 'difficulty', 'domain']
)

evaluation_accuracy_ratio = Gauge(
    'evaluation_accuracy_ratio',
    'Accuracy of evaluation results',
    ['model', 'task_type']
)

# System resource metrics
memory_usage_bytes = Gauge(
    'memory_usage_bytes',
    'Memory usage by component',
    ['component']
)

active_connections = Gauge(
    'active_database_connections',
    'Number of active database connections'
)
```

### Business Metrics
```python
# Usage metrics
daily_evaluations_total = Counter(
    'daily_evaluations_total',
    'Total evaluations per day',
    ['organization', 'model_type']
)

user_satisfaction_score = Histogram(
    'user_satisfaction_score',
    'User satisfaction ratings',
    ['feature', 'version']
)
```

## 🔗 Distributed Tracing

### OpenTelemetry Configuration
```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Jaeger exporter
jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)
```

### Trace Instrumentation
```python
@tracer.start_as_current_span("evaluate_causal_reasoning")
def evaluate_causal_reasoning(task_id: str, model_input: str):
    span = trace.get_current_span()
    span.set_attribute("task.id", task_id)
    span.set_attribute("input.length", len(model_input))
    
    with tracer.start_as_current_span("preprocess_input"):
        processed = preprocess_input(model_input)
    
    with tracer.start_as_current_span("model_inference"):
        result = model.infer(processed)
    
    with tracer.start_as_current_span("postprocess_result"):
        final_result = postprocess_result(result)
    
    span.set_attribute("result.score", final_result.score)
    return final_result
```

## 📈 Grafana Dashboards

### Application Dashboard
```json
{
  "dashboard": {
    "title": "Causal Evaluation Framework",
    "panels": [
      {
        "title": "Evaluation Throughput",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(evaluation_requests_total[5m])",
            "legendFormat": "Evaluations/sec"
          }
        ]
      },
      {
        "title": "Response Time Distribution",
        "type": "heatmap",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, evaluation_duration_seconds_bucket)",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(evaluation_errors_total[5m])",
            "legendFormat": "Errors/sec"
          }
        ]
      }
    ]
  }
}
```

### Infrastructure Dashboard
```json
{
  "dashboard": {
    "title": "Infrastructure Metrics",
    "panels": [
      {
        "title": "CPU Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "cpu_usage_percent",
            "legendFormat": "CPU %"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "memory_usage_bytes / memory_total_bytes * 100",
            "legendFormat": "Memory %"
          }
        ]
      },
      {
        "title": "Database Connections",
        "type": "stat",
        "targets": [
          {
            "expr": "active_database_connections",
            "legendFormat": "Active Connections"
          }
        ]
      }
    ]
  }
}
```

## 🚨 Alerting Strategy

### Critical Alerts
```yaml
# High error rate alert
- alert: HighErrorRate
  expr: rate(evaluation_errors_total[5m]) > 0.1
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "High error rate detected"
    description: "Error rate is {{ $value }} errors/sec"

# Service down alert
- alert: ServiceDown
  expr: up{job="causal-eval"} == 0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "Service is down"
    description: "Causal evaluation service is not responding"
```

### Performance Alerts
```yaml
# High response time alert
- alert: HighResponseTime
  expr: histogram_quantile(0.95, evaluation_duration_seconds_bucket) > 5
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "High response time detected"
    description: "95th percentile response time is {{ $value }}s"

# Memory usage alert
- alert: HighMemoryUsage
  expr: memory_usage_bytes / memory_total_bytes > 0.8
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "High memory usage"
    description: "Memory usage is {{ $value | humanizePercentage }}"
```

## 📝 Structured Logging

### Log Configuration
```python
import structlog
from structlog.stdlib import LoggerFactory

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
```

### Application Logging
```python
import structlog

logger = structlog.get_logger()

def evaluate_model(model_id: str, task_data: dict):
    logger.info(
        "evaluation_started",
        model_id=model_id,
        task_type=task_data.get("type"),
        difficulty=task_data.get("difficulty")
    )
    
    try:
        result = perform_evaluation(model_id, task_data)
        
        logger.info(
            "evaluation_completed",
            model_id=model_id,
            score=result.score,
            duration_ms=result.duration,
            success=True
        )
        
        return result
        
    except Exception as e:
        logger.error(
            "evaluation_failed",
            model_id=model_id,
            error=str(e),
            error_type=type(e).__name__,
            success=False
        )
        raise
```

## 🔄 Health Checks

### Application Health Endpoint
```python
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint."""
    checks = {
        "database": await check_database_connection(),
        "redis": await check_redis_connection(),
        "external_apis": await check_external_apis(),
        "disk_space": check_disk_space(),
        "memory": check_memory_usage()
    }
    
    all_healthy = all(checks.values())
    status_code = status.HTTP_200_OK if all_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "healthy" if all_healthy else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": checks,
            "version": get_application_version()
        }
    )
```

### Deep Health Checks
```python
@app.get("/health/deep")
async def deep_health_check():
    """Deep health check with performance metrics."""
    start_time = time.time()
    
    # Perform comprehensive checks
    db_latency = await measure_database_latency()
    cache_latency = await measure_cache_latency()
    model_availability = await check_model_availability()
    
    end_time = time.time()
    
    return {
        "status": "healthy",
        "latency_ms": (end_time - start_time) * 1000,
        "database_latency_ms": db_latency,
        "cache_latency_ms": cache_latency,
        "models_available": model_availability,
        "detailed_checks": {
            "can_evaluate": await test_evaluation_pipeline(),
            "can_generate": await test_generation_pipeline(),
            "can_analyze": await test_analysis_pipeline()
        }
    }
```

## 📊 SLA/SLO Monitoring

### Service Level Objectives
```yaml
slos:
  availability:
    target: 99.9%
    measurement: "uptime over 30 days"
    
  response_time:
    target: "95% of requests < 500ms"
    measurement: "p95 response time"
    
  error_rate:
    target: "< 0.1% error rate"
    measurement: "5xx errors / total requests"
    
  throughput:
    target: "> 1000 evaluations/hour"
    measurement: "successful evaluations per hour"
```

### SLI Implementation
```python
class SLIMetrics:
    def __init__(self):
        self.availability_gauge = Gauge('sli_availability_ratio', 'Service availability SLI')
        self.latency_histogram = Histogram('sli_latency_seconds', 'Request latency SLI')
        self.error_rate_gauge = Gauge('sli_error_rate_ratio', 'Error rate SLI')
        
    def record_request(self, duration: float, success: bool):
        self.latency_histogram.observe(duration)
        if not success:
            self.error_rate_gauge.inc()
    
    def update_availability(self, is_available: bool):
        self.availability_gauge.set(1 if is_available else 0)
```

## 🚀 Performance Monitoring

### APM Integration
```python
import elastic_apm
from elastic_apm.contrib.starlette import ElasticAPM

# APM configuration
apm_config = {
    'SERVICE_NAME': 'causal-eval-framework',
    'SERVER_URL': 'http://apm-server:8200',
    'ENVIRONMENT': 'production',
}

app.add_middleware(ElasticAPM, client=elastic_apm.Client(apm_config))
```

### Custom Performance Metrics
```python
@elastic_apm.capture_span()
def complex_evaluation_task(task_data):
    """Track performance of complex evaluation tasks."""
    with elastic_apm.capture_span('data_preprocessing'):
        preprocessed = preprocess_data(task_data)
    
    with elastic_apm.capture_span('model_inference', span_type='ml'):
        result = model.infer(preprocessed)
        elastic_apm.tag(model_type=model.type, input_size=len(preprocessed))
    
    return result
```

## 🔧 Implementation Guide

### 1. Metrics Collection Setup
```bash
# Install monitoring dependencies
poetry add prometheus-client grafana-client opentelemetry-api

# Configure Prometheus
cat > prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'causal-eval'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s
EOF
```

### 2. Grafana Dashboard Import
```bash
# Import pre-built dashboards
curl -X POST \
  -H "Content-Type: application/json" \
  -d @dashboard.json \
  http://admin:admin@grafana:3000/api/dashboards/db
```

### 3. Alert Configuration
```bash
# Configure Alertmanager
cat > alertmanager.yml << EOF
global:
  slack_api_url: '$SLACK_WEBHOOK_URL'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
- name: 'web.hook'
  slack_configs:
  - channel: '#alerts'
    title: 'Alert: {{ .GroupLabels.alertname }}'
    text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
EOF
```

## 📚 Best Practices

### 1. Metric Naming
- Use clear, consistent naming conventions
- Include units in metric names where applicable
- Use labels for dimensions, not metric names

### 2. Alert Fatigue Prevention
- Set appropriate thresholds and durations
- Use alert prioritization (critical, warning, info)
- Implement alert correlation and suppression

### 3. Dashboard Design
- Focus on actionable metrics
- Use appropriate visualization types
- Include context and baselines

### 4. Log Management
- Use structured logging consistently
- Include correlation IDs for tracing
- Implement log sampling for high-volume events

## 🎯 Success Metrics

- **MTTR**: Mean Time To Recovery < 15 minutes
- **MTBF**: Mean Time Between Failures > 30 days
- **Alert Precision**: > 95% actionable alerts
- **Dashboard Load Time**: < 2 seconds
- **Monitoring Coverage**: > 99% of critical paths

This comprehensive observability strategy ensures full visibility into system behavior, enabling proactive issue detection and resolution while maintaining optimal performance standards.