"""Health check endpoints with comprehensive monitoring."""

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import logging
import time

router = APIRouter()
logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str
    components: Dict[str, str]


@router.get("/", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Basic health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        components={
            "api": "healthy",
            "evaluation_engine": "healthy",
            "cache": "healthy",
            "database": "not_configured"
        }
    )


@router.get("/detailed")
async def detailed_health() -> Dict[str, Any]:
    """Detailed health check with system monitoring."""
    start_time = time.time()
    
    try:
        # Import health monitor with fallback
        try:
            from causal_eval.core.health_monitoring import health_monitor
            health_data = await health_monitor.run_all_checks()
        except ImportError:
            # Fallback health check without full monitoring
            health_data = await _basic_health_check()
        
        # Add response time
        health_data["response_time_ms"] = (time.time() - start_time) * 1000
        
        return health_data
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/metrics")
async def health_metrics(duration: int = 60) -> Dict[str, Any]:
    """Get health metrics for specified duration (in minutes)."""
    try:
        from causal_eval.core.health_monitoring import health_monitor
        metrics = health_monitor.get_metrics_summary(duration)
        return {
            "status": "success",
            "metrics": metrics,
            "timestamp": int(time.time())
        }
    except ImportError:
        return {
            "status": "limited",
            "message": "Full metrics monitoring not available",
            "basic_info": {
                "service": "causal-eval-bench",
                "status": "operational",
                "timestamp": int(time.time())
            }
        }
    except Exception as e:
        logger.error(f"Metrics collection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Metrics collection failed: {str(e)}")


@router.get("/ready")
async def readiness_check(request: Request) -> Dict[str, Any]:
    """Kubernetes-style readiness probe."""
    try:
        # Check if core services are ready
        from causal_eval.core.engine import EvaluationEngine
        
        engine = EvaluationEngine()
        task_types = engine.get_available_task_types()
        
        if len(task_types) > 0:
            return {
                "status": "ready",
                "service": "causal-eval-bench",
                "available_tasks": len(task_types),
                "timestamp": int(time.time())
            }
        else:
            raise HTTPException(status_code=503, detail="Service not ready")
            
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}")


@router.get("/live")
async def liveness_check(request: Request) -> Dict[str, Any]:
    """Kubernetes-style liveness probe."""
    # Simple liveness check - if we can respond, we're alive
    return {
        "status": "alive",
        "service": "causal-eval-bench",
        "timestamp": str(int(time.time()))
    }


async def _basic_health_check() -> Dict[str, Any]:
    """Basic health check without full monitoring dependencies."""
    try:
        # Test core functionality
        from causal_eval.core.engine import EvaluationEngine
        engine = EvaluationEngine()
        available_tasks = engine.get_available_task_types()
        available_domains = engine.get_available_domains()
        
        # Try basic system info
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            system_info = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": (disk.used / disk.total) * 100
            }
        except ImportError:
            system_info = {"note": "System monitoring not available"}
        
        return {
            "status": "healthy",
            "service": "causal-eval-bench", 
            "timestamp": int(time.time()),
            "system": system_info,
            "functionality": {
                "available_tasks": len(available_tasks),
                "available_domains": len(available_domains),
                "task_types": available_tasks
            }
        }
        
    except Exception as e:
        logger.error(f"Basic health check failed: {str(e)}")
        return {
            "status": "degraded",
            "service": "causal-eval-bench",
            "timestamp": int(time.time()),
            "error": str(e)
        }