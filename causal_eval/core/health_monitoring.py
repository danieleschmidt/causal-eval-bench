"""Comprehensive health monitoring and system diagnostics."""

import asyncio
import psutil
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: str
    response_time_ms: float
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SystemMetrics:
    """System resource metrics."""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    load_average: List[float]
    process_count: int
    open_files: int
    network_connections: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


class HealthMonitor:
    """Comprehensive health monitoring system."""
    
    def __init__(self):
        """Initialize health monitor."""
        self.checks: Dict[str, HealthCheck] = {}
        self.metrics_history: List[SystemMetrics] = []
        self.max_history = 100
        self.alert_thresholds = {
            "cpu_critical": 90.0,
            "cpu_warning": 75.0,
            "memory_critical": 90.0,
            "memory_warning": 80.0,
            "disk_critical": 95.0,
            "disk_warning": 85.0,
            "response_time_critical": 5000.0,  # 5 seconds
            "response_time_warning": 1000.0    # 1 second
        }
        logger.info("Health monitor initialized")
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks and return comprehensive status."""
        start_time = time.time()
        
        # Run all checks concurrently
        check_tasks = [
            self._check_system_resources(),
            self._check_evaluation_engine(),
            self._check_cache_system(),
            self._check_api_endpoints(),
            self._check_database_connectivity(),
            self._check_external_dependencies()
        ]
        
        # Execute all checks
        check_results = await asyncio.gather(*check_tasks, return_exceptions=True)
        
        # Process results
        all_checks = []
        for result in check_results:
            if isinstance(result, HealthCheck):
                all_checks.append(result)
                self.checks[result.name] = result
            elif isinstance(result, Exception):
                logger.error(f"Health check failed: {result}")
                all_checks.append(HealthCheck(
                    name="failed_check",
                    status=HealthStatus.CRITICAL,
                    message=str(result),
                    response_time_ms=0.0
                ))
        
        # Determine overall status
        overall_status = self._determine_overall_status(all_checks)
        total_time = (time.time() - start_time) * 1000
        
        # Collect system metrics
        system_metrics = self._collect_system_metrics()
        self._store_metrics(system_metrics)
        
        return {
            "status": overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {check.name: {
                "status": check.status.value,
                "message": check.message,
                "response_time_ms": check.response_time_ms,
                "details": check.details
            } for check in all_checks},
            "system_metrics": {
                "cpu_percent": system_metrics.cpu_percent,
                "memory_percent": system_metrics.memory_percent,
                "disk_percent": system_metrics.disk_percent,
                "load_average": system_metrics.load_average,
                "process_count": system_metrics.process_count,
                "open_files": system_metrics.open_files
            },
            "total_check_time_ms": total_time,
            "alerts": self._generate_alerts(all_checks, system_metrics)
        }
    
    async def _check_system_resources(self) -> HealthCheck:
        """Check system resource utilization."""
        start_time = time.time()
        
        try:
            # CPU utilization
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory utilization
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk utilization (root partition)
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Load average
            try:
                load_avg = psutil.getloadavg()
            except AttributeError:
                load_avg = [0.0, 0.0, 0.0]  # Windows doesn't have load average
            
            # Determine status
            if cpu_percent > self.alert_thresholds["cpu_critical"] or \
               memory_percent > self.alert_thresholds["memory_critical"] or \
               disk_percent > self.alert_thresholds["disk_critical"]:
                status = HealthStatus.CRITICAL
                message = "Critical resource utilization detected"
            elif cpu_percent > self.alert_thresholds["cpu_warning"] or \
                 memory_percent > self.alert_thresholds["memory_warning"] or \
                 disk_percent > self.alert_thresholds["disk_warning"]:
                status = HealthStatus.WARNING
                message = "High resource utilization detected"
            else:
                status = HealthStatus.HEALTHY
                message = "System resources are healthy"
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheck(
                name="system_resources",
                status=status,
                message=message,
                response_time_ms=response_time,
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_percent": disk_percent,
                    "disk_free_gb": disk.free / (1024**3),
                    "load_average": load_avg
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                name="system_resources",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check system resources: {str(e)}",
                response_time_ms=response_time
            )
    
    async def _check_evaluation_engine(self) -> HealthCheck:
        """Check evaluation engine functionality."""
        start_time = time.time()
        
        try:
            # Test basic evaluation functionality
            from causal_eval.core.performance_optimizer import optimized_engine
            
            # Simple evaluation test
            test_result = await optimized_engine.evaluate_with_optimization(
                task_type="attribution",
                model_response="This appears to be a spurious correlation caused by weather.",
                domain="general",
                difficulty="easy",
                context={"health_check": True}
            )
            
            response_time = (time.time() - start_time) * 1000
            
            if test_result.get("overall_score", 0) >= 0:
                status = HealthStatus.HEALTHY
                message = "Evaluation engine is functioning correctly"
            else:
                status = HealthStatus.WARNING
                message = "Evaluation engine returned unexpected results"
            
            return HealthCheck(
                name="evaluation_engine",
                status=status,
                message=message,
                response_time_ms=response_time,
                details={
                    "test_score": test_result.get("overall_score", 0),
                    "cache_enabled": True
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                name="evaluation_engine",
                status=HealthStatus.CRITICAL,
                message=f"Evaluation engine check failed: {str(e)}",
                response_time_ms=response_time
            )
    
    async def _check_cache_system(self) -> HealthCheck:
        """Check cache system health."""
        start_time = time.time()
        
        try:
            from causal_eval.core.performance_optimizer import optimized_engine
            
            # Test cache functionality
            cache_stats = optimized_engine.cache.get_stats()
            
            response_time = (time.time() - start_time) * 1000
            
            # Determine cache health
            hit_rate = cache_stats.get("hit_rate", 0)
            memory_utilization = cache_stats.get("memory_utilization", 0)
            
            if memory_utilization > 0.95:
                status = HealthStatus.CRITICAL
                message = "Cache memory utilization critical"
            elif memory_utilization > 0.80 or hit_rate < 0.1:
                status = HealthStatus.WARNING
                message = "Cache performance suboptimal"
            else:
                status = HealthStatus.HEALTHY
                message = "Cache system healthy"
            
            return HealthCheck(
                name="cache_system",
                status=status,
                message=message,
                response_time_ms=response_time,
                details={
                    "hit_rate": hit_rate,
                    "memory_utilization": memory_utilization,
                    "entries": cache_stats.get("entries", 0),
                    "total_hits": cache_stats.get("total_hits", 0),
                    "total_misses": cache_stats.get("total_misses", 0)
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                name="cache_system",
                status=HealthStatus.CRITICAL,
                message=f"Cache system check failed: {str(e)}",
                response_time_ms=response_time
            )
    
    async def _check_api_endpoints(self) -> HealthCheck:
        """Check API endpoint responsiveness."""
        start_time = time.time()
        
        try:
            # Test internal API components
            from causal_eval.core.engine import EvaluationEngine
            
            engine = EvaluationEngine()
            task_types = engine.get_available_task_types()
            domains = engine.get_available_domains()
            
            response_time = (time.time() - start_time) * 1000
            
            if len(task_types) > 0 and len(domains) > 0:
                status = HealthStatus.HEALTHY
                message = "API endpoints are responsive"
            else:
                status = HealthStatus.WARNING
                message = "API endpoints returned incomplete data"
            
            return HealthCheck(
                name="api_endpoints",
                status=status,
                message=message,
                response_time_ms=response_time,
                details={
                    "available_task_types": len(task_types),
                    "available_domains": len(domains),
                    "task_types": task_types,
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                name="api_endpoints",
                status=HealthStatus.CRITICAL,
                message=f"API endpoint check failed: {str(e)}",
                response_time_ms=response_time
            )
    
    async def _check_database_connectivity(self) -> HealthCheck:
        """Check database connectivity (if configured)."""
        start_time = time.time()
        
        try:
            # For now, this is a placeholder since DB is optional
            # In production, this would test actual database connectivity
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheck(
                name="database",
                status=HealthStatus.HEALTHY,
                message="Database connectivity not configured (optional)",
                response_time_ms=response_time,
                details={
                    "configured": False,
                    "type": "sqlite_fallback"
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                name="database",
                status=HealthStatus.WARNING,
                message=f"Database check failed: {str(e)}",
                response_time_ms=response_time
            )
    
    async def _check_external_dependencies(self) -> HealthCheck:
        """Check external dependencies and services."""
        start_time = time.time()
        
        try:
            # Check essential Python modules
            import json
            import re
            import hashlib
            import asyncio
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheck(
                name="external_dependencies",
                status=HealthStatus.HEALTHY,
                message="All external dependencies are available",
                response_time_ms=response_time,
                details={
                    "python_version": f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}",
                    "essential_modules": ["json", "re", "hashlib", "asyncio"]
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                name="external_dependencies",
                status=HealthStatus.CRITICAL,
                message=f"External dependency check failed: {str(e)}",
                response_time_ms=response_time
            )
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Load average
            try:
                load_avg = list(psutil.getloadavg())
            except AttributeError:
                load_avg = [0.0, 0.0, 0.0]
            
            # Process information
            process_count = len(psutil.pids())
            
            # Open files (approximate)
            try:
                open_files = len(psutil.Process().open_files())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                open_files = 0
            
            # Network connections
            try:
                network_connections = len(psutil.net_connections())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                network_connections = 0
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=(disk.used / disk.total) * 100,
                load_average=load_avg,
                process_count=process_count,
                open_files=open_files,
                network_connections=network_connections
            )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_percent=0.0,
                load_average=[0.0, 0.0, 0.0],
                process_count=0,
                open_files=0,
                network_connections=0
            )
    
    def _store_metrics(self, metrics: SystemMetrics) -> None:
        """Store metrics in history."""
        self.metrics_history.append(metrics)
        
        # Keep only recent metrics
        if len(self.metrics_history) > self.max_history:
            self.metrics_history = self.metrics_history[-self.max_history:]
    
    def _determine_overall_status(self, checks: List[HealthCheck]) -> HealthStatus:
        """Determine overall system status from individual checks."""
        if not checks:
            return HealthStatus.UNKNOWN
        
        # If any check is critical, overall status is critical
        if any(check.status == HealthStatus.CRITICAL for check in checks):
            return HealthStatus.CRITICAL
        
        # If any check is warning, overall status is warning
        if any(check.status == HealthStatus.WARNING for check in checks):
            return HealthStatus.WARNING
        
        # If all checks are healthy
        if all(check.status == HealthStatus.HEALTHY for check in checks):
            return HealthStatus.HEALTHY
        
        return HealthStatus.UNKNOWN
    
    def _generate_alerts(self, checks: List[HealthCheck], metrics: SystemMetrics) -> List[Dict[str, Any]]:
        """Generate alerts based on health checks and metrics."""
        alerts = []
        
        # Check for critical/warning status
        for check in checks:
            if check.status in [HealthStatus.CRITICAL, HealthStatus.WARNING]:
                alerts.append({
                    "type": "health_check",
                    "severity": check.status.value,
                    "check_name": check.name,
                    "message": check.message,
                    "response_time_ms": check.response_time_ms
                })
        
        # Check for slow response times
        slow_checks = [check for check in checks if check.response_time_ms > self.alert_thresholds["response_time_warning"]]
        if slow_checks:
            alerts.append({
                "type": "performance",
                "severity": "warning",
                "message": f"{len(slow_checks)} checks had slow response times",
                "slow_checks": [check.name for check in slow_checks]
            })
        
        # System resource alerts
        if metrics.cpu_percent > self.alert_thresholds["cpu_warning"]:
            alerts.append({
                "type": "system_resource",
                "severity": "critical" if metrics.cpu_percent > self.alert_thresholds["cpu_critical"] else "warning",
                "resource": "cpu",
                "value": metrics.cpu_percent,
                "threshold": self.alert_thresholds["cpu_critical" if metrics.cpu_percent > self.alert_thresholds["cpu_critical"] else "cpu_warning"]
            })
        
        if metrics.memory_percent > self.alert_thresholds["memory_warning"]:
            alerts.append({
                "type": "system_resource",
                "severity": "critical" if metrics.memory_percent > self.alert_thresholds["memory_critical"] else "warning",
                "resource": "memory",
                "value": metrics.memory_percent,
                "threshold": self.alert_thresholds["memory_critical" if metrics.memory_percent > self.alert_thresholds["memory_critical"] else "memory_warning"]
            })
        
        return alerts
    
    def get_metrics_summary(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """Get metrics summary for specified duration."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=duration_minutes)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {"message": "No recent metrics available"}
        
        # Calculate averages
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        avg_disk = sum(m.disk_percent for m in recent_metrics) / len(recent_metrics)
        
        # Find peaks
        max_cpu = max(m.cpu_percent for m in recent_metrics)
        max_memory = max(m.memory_percent for m in recent_metrics)
        max_disk = max(m.disk_percent for m in recent_metrics)
        
        return {
            "duration_minutes": duration_minutes,
            "samples": len(recent_metrics),
            "averages": {
                "cpu_percent": round(avg_cpu, 2),
                "memory_percent": round(avg_memory, 2),
                "disk_percent": round(avg_disk, 2)
            },
            "peaks": {
                "cpu_percent": max_cpu,
                "memory_percent": max_memory,
                "disk_percent": max_disk
            },
            "latest": {
                "cpu_percent": recent_metrics[-1].cpu_percent,
                "memory_percent": recent_metrics[-1].memory_percent,
                "disk_percent": recent_metrics[-1].disk_percent,
                "timestamp": recent_metrics[-1].timestamp.isoformat()
            }
        }


# Global health monitor instance
health_monitor = HealthMonitor()