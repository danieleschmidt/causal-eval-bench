"""Comprehensive monitoring and observability system."""

import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import asyncio
import json

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Metric type definitions."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "timestamp": self.timestamp,
            "labels": self.labels
        }


@dataclass
class HealthCheck:
    """Health check result."""
    name: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """Enhanced metrics collection with monitoring capabilities."""
    
    def __init__(self):
        self.metrics: List[Metric] = []
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.health_checks: Dict[str, HealthCheck] = {}
        
        # Performance tracking
        self.request_times = deque(maxlen=1000)  # Last 1000 requests
        self.error_counts = defaultdict(int)
        self.evaluation_scores = deque(maxlen=1000)
        
        logger.info("Enhanced metrics collector initialized")
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None) -> None:
        """Increment a counter metric."""
        self.counters[name] += value
        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.COUNTER,
            timestamp=time.time(),
            labels=labels or {}
        )
        self.metrics.append(metric)
        logger.debug(f"Counter incremented: {name} += {value}")
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Set a gauge metric."""
        self.gauges[name] = value
        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            timestamp=time.time(),
            labels=labels or {}
        )
        self.metrics.append(metric)
        logger.debug(f"Gauge set: {name} = {value}")
    
    def record_timer(self, name: str, duration: float, labels: Dict[str, str] = None) -> None:
        """Record a timing measurement."""
        self.timers[name].append(duration)
        metric = Metric(
            name=name,
            value=duration,
            metric_type=MetricType.TIMER,
            timestamp=time.time(),
            labels=labels or {}
        )
        self.metrics.append(metric)
        logger.debug(f"Timer recorded: {name} = {duration:.3f}s")
    
    def record_evaluation_metrics(
        self, 
        task_type: str, 
        domain: str,
        overall_score: float,
        processing_time: float,
        success: bool = True
    ) -> None:
        """Record comprehensive evaluation metrics."""
        labels = {"task_type": task_type, "domain": domain}
        
        # Increment evaluation counter
        self.increment_counter("evaluations_total", labels=labels)
        
        # Record success/failure
        status_labels = {**labels, "status": "success" if success else "failure"}
        self.increment_counter("evaluation_status_total", labels=status_labels)
        
        # Record processing time
        self.record_timer("evaluation_duration_seconds", processing_time, labels)
        self.request_times.append(processing_time)
        
        # Record evaluation score
        if success:
            self.set_gauge("evaluation_score", overall_score, labels)
            self.evaluation_scores.append(overall_score)
        
        # Update performance gauges
        self._update_performance_gauges()
    
    def record_error(self, error_type: str, endpoint: str = "unknown") -> None:
        """Record error occurrence."""
        labels = {"error_type": error_type, "endpoint": endpoint}
        self.increment_counter("errors_total", labels=labels)
        self.error_counts[error_type] += 1
    
    def _update_performance_gauges(self) -> None:
        """Update real-time performance gauges."""
        if self.request_times:
            # Response time percentiles
            sorted_times = sorted(list(self.request_times))
            self.set_gauge("response_time_p50", sorted_times[len(sorted_times)//2])
            self.set_gauge("response_time_p95", sorted_times[int(len(sorted_times)*0.95)])
            self.set_gauge("response_time_p99", sorted_times[int(len(sorted_times)*0.99)])
            
            # Average response time
            self.set_gauge("response_time_avg", sum(self.request_times) / len(self.request_times))
        
        if self.evaluation_scores:
            # Evaluation score statistics
            scores = list(self.evaluation_scores)
            self.set_gauge("evaluation_score_avg", sum(scores) / len(scores))
            self.set_gauge("evaluation_score_min", min(scores))
            self.set_gauge("evaluation_score_max", max(scores))
    
    def register_health_check(self, check: HealthCheck) -> None:
        """Register a health check result."""
        self.health_checks[check.name] = check
        
        # Convert health status to numeric gauge
        status_value = {"healthy": 1.0, "degraded": 0.5, "unhealthy": 0.0}.get(check.status, 0.0)
        self.set_gauge(f"health_status_{check.name}", status_value)
        
        logger.info(f"Health check registered: {check.name} = {check.status}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health assessment."""
        healthy_checks = sum(1 for check in self.health_checks.values() if check.status == "healthy")
        total_checks = len(self.health_checks)
        
        overall_status = "healthy"
        if total_checks == 0:
            overall_status = "unknown"
        elif healthy_checks == 0:
            overall_status = "unhealthy"
        elif healthy_checks < total_checks:
            overall_status = "degraded"
        
        return {
            "overall_status": overall_status,
            "healthy_checks": healthy_checks,
            "total_checks": total_checks,
            "health_percentage": (healthy_checks / total_checks * 100) if total_checks > 0 else 0,
            "checks": {name: check.__dict__ for name, check in self.health_checks.items()},
            "timestamp": time.time()
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        current_time = time.time()
        recent_metrics = [m for m in self.metrics if current_time - m.timestamp < 300]  # Last 5 minutes
        
        return {
            "total_evaluations": self.counters.get("evaluations_total", 0),
            "total_errors": sum(self.error_counts.values()),
            "error_rate": sum(self.error_counts.values()) / max(self.counters.get("evaluations_total", 1), 1),
            "avg_response_time": self.gauges.get("response_time_avg", 0.0),
            "p95_response_time": self.gauges.get("response_time_p95", 0.0),
            "avg_evaluation_score": self.gauges.get("evaluation_score_avg", 0.0),
            "recent_metrics_count": len(recent_metrics),
            "timestamp": current_time
        }
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        # Export counters
        for name, value in self.counters.items():
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")
        
        # Export gauges
        for name, value in self.gauges.items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")
        
        # Export timer summaries
        for name, values in self.timers.items():
            if values:
                count = len(values)
                total = sum(values)
                lines.append(f"# TYPE {name} summary")
                lines.append(f"{name}_count {count}")
                lines.append(f"{name}_sum {total}")
        
        return "\\n".join(lines)


class HealthMonitor:
    """System health monitoring."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.monitoring_task: Optional[asyncio.Task] = None
    
    async def start_monitoring(self, interval: float = 30.0) -> None:
        """Start continuous health monitoring."""
        if self.monitoring_task:
            return
        
        self.monitoring_task = asyncio.create_task(self._monitoring_loop(interval))
        logger.info(f"Health monitoring started with {interval}s interval")
    
    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
        logger.info("Health monitoring stopped")
    
    async def _monitoring_loop(self, interval: float) -> None:
        """Main monitoring loop."""
        while True:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(interval)
    
    async def _perform_health_checks(self) -> None:
        """Perform all health checks."""
        # API responsiveness check
        api_health = await self._check_api_health()
        self.metrics_collector.register_health_check(api_health)
        
        # Evaluation engine health check
        engine_health = await self._check_evaluation_engine_health()
        self.metrics_collector.register_health_check(engine_health)
        
        # System resource checks
        resource_health = await self._check_system_resources()
        self.metrics_collector.register_health_check(resource_health)
    
    async def _check_api_health(self) -> HealthCheck:
        """Check API health."""
        try:
            # Simple health check - could ping internal endpoints
            return HealthCheck(
                name="api",
                status="healthy",
                message="API is responsive"
            )
        except Exception as e:
            return HealthCheck(
                name="api",
                status="unhealthy",
                message=f"API health check failed: {str(e)}"
            )
    
    async def _check_evaluation_engine_health(self) -> HealthCheck:
        """Check evaluation engine health."""
        try:
            # Test basic engine functionality
            from causal_eval.core.engine import EvaluationEngine
            engine = EvaluationEngine()
            
            return HealthCheck(
                name="evaluation_engine",
                status="healthy",
                message="Evaluation engine is operational"
            )
        except Exception as e:
            return HealthCheck(
                name="evaluation_engine",
                status="unhealthy",
                message=f"Evaluation engine check failed: {str(e)}"
            )
    
    async def _check_system_resources(self) -> HealthCheck:
        """Check system resource usage."""
        try:
            import psutil
            
            # Check memory usage
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            status = "healthy"
            if memory.percent > 90 or cpu_percent > 90:
                status = "degraded"
            if memory.percent > 95 or cpu_percent > 95:
                status = "unhealthy"
            
            return HealthCheck(
                name="system_resources",
                status=status,
                message=f"Memory: {memory.percent:.1f}%, CPU: {cpu_percent:.1f}%",
                details={
                    "memory_percent": memory.percent,
                    "cpu_percent": cpu_percent,
                    "available_memory_gb": memory.available / (1024**3)
                }
            )
        except ImportError:
            # psutil not available
            return HealthCheck(
                name="system_resources",
                status="healthy",
                message="Resource monitoring not available (psutil not installed)"
            )
        except Exception as e:
            return HealthCheck(
                name="system_resources",
                status="unhealthy",
                message=f"Resource check failed: {str(e)}"
            )


# Global instances
enhanced_metrics = MetricsCollector()
health_monitor = HealthMonitor(enhanced_metrics)


# Context manager for timing operations
class TimingContext:
    """Context manager for measuring operation duration."""
    
    def __init__(self, metric_name: str, labels: Dict[str, str] = None):
        self.metric_name = metric_name
        self.labels = labels or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            enhanced_metrics.record_timer(self.metric_name, duration, self.labels)


# Convenience functions
def time_operation(metric_name: str, labels: Dict[str, str] = None):
    """Decorator for timing function execution."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            with TimingContext(metric_name, labels):
                return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            with TimingContext(metric_name, labels):
                return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator