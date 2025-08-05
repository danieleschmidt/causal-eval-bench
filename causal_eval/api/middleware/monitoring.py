"""
Comprehensive monitoring middleware for API requests and performance.
"""

import time
import uuid
from typing import Callable, Dict, Any, Optional
from fastapi import Request, Response
import logging
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import asyncio
from dataclasses import dataclass
from contextlib import asynccontextmanager

from causal_eval.core.logging_config import get_security_logger, get_performance_logger

logger = logging.getLogger(__name__)
security_logger = get_security_logger()
performance_logger = get_performance_logger()


# Prometheus metrics
REQUEST_COUNT = Counter(
    'causal_eval_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'causal_eval_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
)

EVALUATION_COUNT = Counter(
    'causal_eval_evaluations_total',
    'Total number of evaluations',
    ['task_type', 'domain', 'model_name']
)

EVALUATION_DURATION = Histogram(
    'causal_eval_evaluation_duration_seconds',
    'Evaluation duration in seconds',
    ['task_type', 'domain'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

EVALUATION_SCORE = Histogram(
    'causal_eval_evaluation_score',
    'Evaluation scores',
    ['task_type', 'domain', 'model_name'],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

ACTIVE_REQUESTS = Gauge(
    'causal_eval_active_requests',
    'Number of active requests'
)

MODEL_API_CALLS = Counter(
    'causal_eval_model_api_calls_total',
    'Total number of model API calls',
    ['model_provider', 'model_name', 'status']
)

MODEL_API_DURATION = Histogram(
    'causal_eval_model_api_duration_seconds',
    'Model API call duration in seconds',
    ['model_provider', 'model_name'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0]
)

MODEL_API_TOKENS = Histogram(
    'causal_eval_model_api_tokens',
    'Tokens used in model API calls',
    ['model_provider', 'model_name'],
    buckets=[10, 50, 100, 500, 1000, 2000, 5000, 10000]
)

SECURITY_EVENTS = Counter(
    'causal_eval_security_events_total',
    'Security events',
    ['event_type', 'severity']
)


@dataclass
class RequestMetrics:
    """Request metrics data."""
    request_id: str
    start_time: float
    method: str
    path: str
    ip_address: str
    user_agent: str
    request_size: int


class MonitoringMiddleware:
    """Comprehensive monitoring middleware."""
    
    def __init__(self, track_detailed_metrics: bool = True):
        """Initialize monitoring middleware."""
        self.track_detailed_metrics = track_detailed_metrics
        self.active_requests: Dict[str, RequestMetrics] = {}
    
    async def __call__(self, request: Request, call_next: Callable):
        """Process request with comprehensive monitoring."""
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Extract request information
        method = request.method
        path = request.url.path
        ip_address = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Calculate request size
        content_length = request.headers.get("content-length")
        request_size = int(content_length) if content_length else 0
        
        # Store request metrics
        request_metrics = RequestMetrics(
            request_id=request_id,
            start_time=start_time,
            method=method,
            path=path,
            ip_address=ip_address,
            user_agent=user_agent,
            request_size=request_size
        )
        
        self.active_requests[request_id] = request_metrics
        ACTIVE_REQUESTS.inc()
        
        # Add request ID to request state
        request.state.request_id = request_id
        request.state.start_time = start_time
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate response time
            end_time = time.time()
            duration = end_time - start_time
            
            # Extract response information
            status_code = response.status_code
            
            # Update metrics
            endpoint = self._normalize_endpoint(path)
            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code
            ).inc()
            
            REQUEST_DURATION.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
            # Log performance metrics
            performance_logger.log_api_response_time(
                endpoint=endpoint,
                method=method,
                response_time=duration * 1000,  # Convert to milliseconds
                status_code=status_code
            )
            
            # Log request details
            logger.info(
                f"Request completed: {method} {path}",
                extra={
                    "request_id": request_id,
                    "method": method,
                    "path": path,
                    "status_code": status_code,
                    "duration_ms": duration * 1000,
                    "ip_address": ip_address,
                    "user_agent": user_agent,
                    "request_size": request_size
                }
            )
            
            # Add monitoring headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration:.3f}s"
            
            return response
            
        except Exception as e:
            # Handle errors and update metrics
            end_time = time.time()
            duration = end_time - start_time
            
            endpoint = self._normalize_endpoint(path)
            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status_code=500
            ).inc()
            
            REQUEST_DURATION.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
            # Log error
            logger.error(
                f"Request failed: {method} {path}",
                extra={
                    "request_id": request_id,
                    "method": method,
                    "path": path,
                    "duration_ms": duration * 1000,
                    "ip_address": ip_address,
                    "error": str(e)
                },
                exc_info=True
            )
            
            raise
            
        finally:
            # Cleanup
            self.active_requests.pop(request_id, None)
            ACTIVE_REQUESTS.dec()
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def _normalize_endpoint(self, path: str) -> str:
        """Normalize endpoint path for metrics."""
        # Replace IDs and other variable parts with placeholders
        import re
        
        # Replace UUIDs
        path = re.sub(r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '/{id}', path)
        
        # Replace numeric IDs
        path = re.sub(r'/\d+', '/{id}', path)
        
        # Replace session IDs
        path = re.sub(r'/sessions/[^/]+', '/sessions/{session_id}', path)
        
        return path


class EvaluationMonitoringMiddleware:
    """Specialized monitoring for evaluation endpoints."""
    
    async def track_evaluation(
        self,
        task_type: str,
        domain: str,
        model_name: Optional[str],
        duration: float,
        score: float
    ):
        """Track evaluation metrics."""
        EVALUATION_COUNT.labels(
            task_type=task_type,
            domain=domain or "unknown",
            model_name=model_name or "unknown"
        ).inc()
        
        EVALUATION_DURATION.labels(
            task_type=task_type,
            domain=domain or "unknown"
        ).observe(duration)
        
        if 0 <= score <= 1:  # Valid score range
            EVALUATION_SCORE.labels(
                task_type=task_type,
                domain=domain or "unknown",
                model_name=model_name or "unknown"
            ).observe(score)
        
        # Log evaluation performance
        performance_logger.log_evaluation_performance(
            task_type=task_type,
            model_name=model_name or "unknown",
            execution_time=duration * 1000,  # Convert to milliseconds
            token_count=0,  # TODO: Add token tracking
            score=score
        )


class ModelAPIMonitoringMixin:
    """Mixin for monitoring model API calls."""
    
    @classmethod
    def track_api_call(
        cls,
        provider: str,
        model_name: str,
        duration: float,
        tokens_used: Optional[int] = None,
        status: str = "success"
    ):
        """Track model API call metrics."""
        MODEL_API_CALLS.labels(
            model_provider=provider,
            model_name=model_name,
            status=status
        ).inc()
        
        MODEL_API_DURATION.labels(
            model_provider=provider,
            model_name=model_name
        ).observe(duration)
        
        if tokens_used:
            MODEL_API_TOKENS.labels(
                model_provider=provider,
                model_name=model_name
            ).observe(tokens_used)


class SecurityMonitoringMixin:
    """Mixin for security event monitoring."""
    
    @classmethod
    def track_security_event(cls, event_type: str, severity: str = "medium"):
        """Track security events."""
        SECURITY_EVENTS.labels(
            event_type=event_type,
            severity=severity
        ).inc()
        
        security_logger.logger.warning(
            f"Security event: {event_type}",
            extra={
                "event_type": event_type,
                "severity": severity,
                "security_event": True
            }
        )


class HealthCheckMonitoring:
    """Health check monitoring utilities."""
    
    @staticmethod
    def get_system_health() -> Dict[str, Any]:
        """Get comprehensive system health metrics."""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "active_requests": len(MonitoringMiddleware().active_requests),
            "metrics": {
                "total_requests": REQUEST_COUNT._value.sum(),
                "total_evaluations": EVALUATION_COUNT._value.sum(),
                "total_api_calls": MODEL_API_CALLS._value.sum(),
                "security_events": SECURITY_EVENTS._value.sum()
            }
        }


def create_metrics_endpoint():
    """Create Prometheus metrics endpoint."""
    def metrics_endpoint():
        """Serve Prometheus metrics."""
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
            headers={"Cache-Control": "no-cache"}
        )
    
    return metrics_endpoint


# Global monitoring instances
monitoring_middleware = MonitoringMiddleware()
evaluation_monitor = EvaluationMonitoringMiddleware()
health_monitor = HealthCheckMonitoring()