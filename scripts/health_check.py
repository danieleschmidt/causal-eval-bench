#!/usr/bin/env python3
"""
Health Check Script for Causal Eval Bench

This script provides comprehensive health monitoring for all system components.
Can be used by orchestrators, monitoring systems, and deployment pipelines.
"""

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Union
from pathlib import Path

import httpx
import psycopg2
import redis
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class HealthCheckResult:
    """Health check result for a specific component."""
    component: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    response_time_ms: Optional[float] = None
    details: Optional[Dict] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()

@dataclass 
class SystemHealth:
    """Overall system health status."""
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: str
    components: List[HealthCheckResult]
    summary: Dict[str, Union[str, int, float]]
    
    def to_dict(self) -> Dict:
        return {
            "status": self.status,
            "timestamp": self.timestamp,
            "components": [asdict(comp) for comp in self.components],
            "summary": self.summary
        }

class HealthChecker:
    """Main health checker class."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._load_default_config()
        self.results: List[HealthCheckResult] = []
        
    def _load_default_config(self) -> Dict:
        """Load default configuration for health checks."""
        return {
            "database": {
                "url": "postgresql://causal_eval_user:causal_eval_password@localhost:5432/causal_eval_bench",
                "timeout": 5
            },
            "redis": {
                "url": "redis://localhost:6379/0",
                "timeout": 3
            },
            "api": {
                "url": "http://localhost:8000",
                "timeout": 10
            },
            "external_apis": {
                "timeout": 15
            },
            "filesystem": {
                "paths": ["/app/logs", "/app/data", "/app/cache"],
                "min_free_space_mb": 100
            }
        }
    
    async def check_all(self) -> SystemHealth:
        """Run all health checks and return system status."""
        logger.info("Starting comprehensive health check...")
        start_time = time.time()
        
        # Run all health checks concurrently
        tasks = [
            self.check_database(),
            self.check_redis(),
            self.check_api(),
            self.check_filesystem(),
            self.check_memory(),
            self.check_external_dependencies()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, Exception):
                self.results.append(
                    HealthCheckResult(
                        component="unknown",
                        status="unhealthy", 
                        message=f"Health check failed: {str(result)}"
                    )
                )
            elif isinstance(result, list):
                self.results.extend(result)
            else:
                self.results.append(result)
        
        # Calculate overall system health
        total_time = (time.time() - start_time) * 1000
        system_health = self._calculate_system_health(total_time)
        
        logger.info(f"Health check completed in {total_time:.2f}ms - Status: {system_health.status}")
        return system_health
    
    async def check_database(self) -> HealthCheckResult:
        """Check database connectivity and performance."""
        start_time = time.time()
        
        try:
            db_config = self.config["database"]
            engine = create_engine(db_config["url"], pool_timeout=db_config["timeout"])
            
            with engine.connect() as conn:
                # Test basic connectivity
                result = conn.execute(text("SELECT 1 as health_check"))
                result.fetchone()
                
                # Check database version
                version_result = conn.execute(text("SELECT version()"))
                db_version = version_result.fetchone()[0]
                
                # Check application tables exist
                tables_result = conn.execute(text("""
                    SELECT COUNT(*) FROM information_schema.tables 
                    WHERE table_schema = 'causal_eval'
                """))
                table_count = tables_result.fetchone()[0]
                
                # Check recent activity
                activity_result = conn.execute(text("""
                    SELECT COUNT(*) FROM causal_eval.evaluations 
                    WHERE created_at > NOW() - INTERVAL '24 hours'
                """))
                recent_evaluations = activity_result.fetchone()[0]
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                component="database",
                status="healthy",
                message="Database is healthy and responsive",
                response_time_ms=response_time,
                details={
                    "version": db_version.split()[1] if db_version else "unknown",
                    "table_count": table_count,
                    "recent_evaluations_24h": recent_evaluations,
                    "connection_pool": "active"
                }
            )
            
        except SQLAlchemyError as e:
            return HealthCheckResult(
                component="database",
                status="unhealthy",
                message=f"Database connection failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            return HealthCheckResult(
                component="database",
                status="unhealthy", 
                message=f"Database health check failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000
            )
    
    async def check_redis(self) -> HealthCheckResult:
        """Check Redis connectivity and performance."""
        start_time = time.time()
        
        try:
            redis_config = self.config["redis"]
            r = redis.from_url(redis_config["url"], socket_timeout=redis_config["timeout"])
            
            # Test connectivity
            pong = r.ping()
            if not pong:
                raise Exception("Redis ping failed")
            
            # Test read/write
            test_key = "health_check_test"
            test_value = f"health_check_{int(time.time())}"
            r.set(test_key, test_value, ex=60)
            stored_value = r.get(test_key).decode()
            
            if stored_value != test_value:
                raise Exception("Redis read/write test failed")
            
            # Clean up test key
            r.delete(test_key)
            
            # Get Redis info
            info = r.info()
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                component="redis",
                status="healthy",
                message="Redis is healthy and responsive",
                response_time_ms=response_time,
                details={
                    "version": info.get("redis_version"),
                    "connected_clients": info.get("connected_clients"),
                    "used_memory_human": info.get("used_memory_human"),
                    "uptime_in_seconds": info.get("uptime_in_seconds")
                }
            )
            
        except redis.RedisError as e:
            return HealthCheckResult(
                component="redis",
                status="unhealthy",
                message=f"Redis connection failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            return HealthCheckResult(
                component="redis",
                status="unhealthy",
                message=f"Redis health check failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000
            )
    
    async def check_api(self) -> HealthCheckResult:
        """Check API server health and endpoints."""
        start_time = time.time()
        
        try:
            api_config = self.config["api"]
            timeout = httpx.Timeout(api_config["timeout"])
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                # Check health endpoint
                health_response = await client.get(f"{api_config['url']}/health")
                health_response.raise_for_status()
                
                # Check version endpoint
                version_response = await client.get(f"{api_config['url']}/version")
                version_response.raise_for_status()
                version_data = version_response.json()
                
                # Check API documentation
                docs_response = await client.get(f"{api_config['url']}/docs")
                docs_response.raise_for_status()
                
                # Check metrics endpoint
                metrics_response = await client.get(f"{api_config['url']}/metrics")
                # Metrics endpoint might not exist, so don't fail on 404
                metrics_available = metrics_response.status_code == 200
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                component="api",
                status="healthy",
                message="API server is healthy and responsive",
                response_time_ms=response_time,
                details={
                    "version": version_data.get("version", "unknown"),
                    "environment": version_data.get("environment", "unknown"), 
                    "docs_available": True,
                    "metrics_available": metrics_available,
                    "endpoints_checked": ["health", "version", "docs", "metrics"]
                }
            )
            
        except httpx.HTTPError as e:
            return HealthCheckResult(
                component="api",
                status="unhealthy",
                message=f"API server request failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            return HealthCheckResult(
                component="api",
                status="unhealthy",
                message=f"API health check failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000
            )
    
    async def check_filesystem(self) -> List[HealthCheckResult]:
        """Check filesystem health and disk space."""
        results = []
        
        try:
            import shutil
            fs_config = self.config["filesystem"]
            
            for path in fs_config["paths"]:
                path_obj = Path(path)
                
                if not path_obj.exists():
                    results.append(
                        HealthCheckResult(
                            component=f"filesystem_{path}",
                            status="unhealthy",
                            message=f"Required path does not exist: {path}"
                        )
                    )
                    continue
                
                # Check disk space
                try:
                    total, used, free = shutil.disk_usage(path)
                    free_mb = free // (1024 * 1024)
                    
                    if free_mb < fs_config["min_free_space_mb"]:
                        status = "degraded"
                        message = f"Low disk space: {free_mb}MB free"
                    else:
                        status = "healthy"
                        message = f"Filesystem healthy: {free_mb}MB free"
                    
                    results.append(
                        HealthCheckResult(
                            component=f"filesystem_{path}",
                            status=status,
                            message=message,
                            details={
                                "path": str(path),
                                "total_mb": total // (1024 * 1024),
                                "used_mb": used // (1024 * 1024),
                                "free_mb": free_mb,
                                "usage_percent": round((used / total) * 100, 2)
                            }
                        )
                    )
                    
                except OSError as e:
                    results.append(
                        HealthCheckResult(
                            component=f"filesystem_{path}",
                            status="unhealthy",
                            message=f"Cannot check disk usage: {str(e)}"
                        )
                    )
        
        except Exception as e:
            results.append(
                HealthCheckResult(
                    component="filesystem",
                    status="unhealthy",
                    message=f"Filesystem check failed: {str(e)}"
                )
            )
        
        return results
    
    async def check_memory(self) -> HealthCheckResult:
        """Check system memory usage."""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            
            if memory.percent > 90:
                status = "unhealthy"
                message = f"Critical memory usage: {memory.percent}%"
            elif memory.percent > 80:
                status = "degraded"
                message = f"High memory usage: {memory.percent}%"
            else:
                status = "healthy"
                message = f"Memory usage normal: {memory.percent}%"
            
            return HealthCheckResult(
                component="memory",
                status=status,
                message=message,
                details={
                    "total_mb": memory.total // (1024 * 1024),
                    "available_mb": memory.available // (1024 * 1024),
                    "used_mb": memory.used // (1024 * 1024),
                    "usage_percent": memory.percent
                }
            )
            
        except ImportError:
            return HealthCheckResult(
                component="memory",
                status="degraded",
                message="psutil not available - cannot check memory"
            )
        except Exception as e:
            return HealthCheckResult(
                component="memory",
                status="unhealthy",
                message=f"Memory check failed: {str(e)}"
            )
    
    async def check_external_dependencies(self) -> List[HealthCheckResult]:
        """Check external API dependencies."""
        results = []
        
        # Check common external services
        external_services = [
            ("openai", "https://api.openai.com/v1/models"),
            ("anthropic", "https://api.anthropic.com/v1/messages"),
            ("huggingface", "https://huggingface.co/api/models")
        ]
        
        timeout = httpx.Timeout(self.config["external_apis"]["timeout"])
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            for service_name, url in external_services:
                start_time = time.time()
                try:
                    response = await client.head(url)
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status_code < 500:
                        # 2xx, 3xx, 4xx are acceptable (service is responding)
                        status = "healthy"
                        message = f"{service_name} API is responsive"
                    else:
                        status = "degraded"
                        message = f"{service_name} API returned {response.status_code}"
                    
                    results.append(
                        HealthCheckResult(
                            component=f"external_{service_name}",
                            status=status,
                            message=message,
                            response_time_ms=response_time,
                            details={"status_code": response.status_code}
                        )
                    )
                    
                except httpx.TimeoutException:
                    results.append(
                        HealthCheckResult(
                            component=f"external_{service_name}",
                            status="degraded",
                            message=f"{service_name} API timeout",
                            response_time_ms=(time.time() - start_time) * 1000
                        )
                    )
                except Exception as e:
                    results.append(
                        HealthCheckResult(
                            component=f"external_{service_name}",
                            status="degraded",
                            message=f"{service_name} API check failed: {str(e)}",
                            response_time_ms=(time.time() - start_time) * 1000
                        )
                    )
        
        return results
    
    def _calculate_system_health(self, total_time_ms: float) -> SystemHealth:
        """Calculate overall system health from component results."""
        healthy_count = sum(1 for r in self.results if r.status == "healthy")
        degraded_count = sum(1 for r in self.results if r.status == "degraded")
        unhealthy_count = sum(1 for r in self.results if r.status == "unhealthy")
        
        total_components = len(self.results)
        
        # Determine overall status
        if unhealthy_count > 0:
            # Any unhealthy component makes system unhealthy
            overall_status = "unhealthy"
        elif degraded_count > total_components * 0.3:
            # More than 30% degraded components makes system degraded
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        # Calculate average response time for components that have it
        response_times = [r.response_time_ms for r in self.results if r.response_time_ms is not None]
        avg_response_time = sum(response_times) / len(response_times) if response_times else None
        
        summary = {
            "total_components": total_components,
            "healthy_components": healthy_count,
            "degraded_components": degraded_count,
            "unhealthy_components": unhealthy_count,
            "health_score_percent": round((healthy_count / total_components) * 100, 2) if total_components > 0 else 0,
            "total_check_time_ms": round(total_time_ms, 2),
            "average_response_time_ms": round(avg_response_time, 2) if avg_response_time else None
        }
        
        return SystemHealth(
            status=overall_status,
            timestamp=datetime.utcnow().isoformat(),
            components=self.results,
            summary=summary
        )

async def main():
    """Main health check function."""
    try:
        checker = HealthChecker()
        health = await checker.check_all()
        
        # Output results
        if len(sys.argv) > 1 and sys.argv[1] == "--json":
            print(json.dumps(health.to_dict(), indent=2))
        else:
            print(f"System Health: {health.status.upper()}")
            print(f"Timestamp: {health.timestamp}")
            print(f"Health Score: {health.summary['health_score_percent']}%")
            print(f"Total Check Time: {health.summary['total_check_time_ms']}ms")
            print()
            
            for component in health.components:
                status_emoji = {
                    "healthy": "✅",
                    "degraded": "⚠️",
                    "unhealthy": "❌"
                }.get(component.status, "❓")
                
                print(f"{status_emoji} {component.component}: {component.message}")
                if component.response_time_ms:
                    print(f"   Response time: {component.response_time_ms:.2f}ms")
        
        # Exit with appropriate code
        exit_code = 0 if health.status == "healthy" else 1
        sys.exit(exit_code)
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        print(f"Health check failed: {str(e)}")
        sys.exit(2)

if __name__ == "__main__":
    asyncio.run(main())