"""FastAPI application factory and configuration."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from causal_eval.api.routes import evaluation, health, tasks
from causal_eval.core.engine import EvaluationEngine
from causal_eval.core.tasks import TaskRegistry
from causal_eval.core.metrics import MetricsCollector
from causal_eval.integrations.model_apis import create_default_manager
from causal_eval.api.middleware import (
    create_rate_limit_middleware, 
    ValidationMiddleware, 
    SecurityHeaders
)
from causal_eval.api.middleware.monitoring import MonitoringMiddleware, create_metrics_endpoint
from causal_eval.core.logging_config import setup_logging
from causal_eval.core.caching import initialize_cache
from causal_eval.core.concurrency import initialize_pools, shutdown_pools
import os

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Setup logging first
    setup_logging()
    logger.info("Starting Causal Evaluation Bench API")
    
    # Initialize performance systems
    redis_url = os.getenv("REDIS_URL")
    cache_manager = await initialize_cache(redis_url)
    await initialize_pools()
    
    # Initialize core components
    engine = EvaluationEngine()
    task_registry = TaskRegistry()
    metrics_collector = MetricsCollector()
    model_manager = create_default_manager()
    
    # Store in app state
    app.state.engine = engine
    app.state.task_registry = task_registry
    app.state.metrics_collector = metrics_collector
    app.state.model_manager = model_manager
    app.state.cache_manager = cache_manager
    
    yield
    
    # Cleanup resources
    await shutdown_pools()
    await cache_manager.cleanup()
    await model_manager.close_all()
    logger.info("Shutting down Causal Evaluation Bench API")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="Causal Evaluation Bench",
        description="A comprehensive evaluation framework for testing genuine causal reasoning in language models",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # Monitoring middleware (first to capture all requests)
    app.add_middleware(MonitoringMiddleware)
    
    # Security headers middleware
    app.add_middleware(SecurityHeaders)
    
    # Rate limiting middleware
    redis_url = os.getenv("REDIS_URL")
    rate_limit_middleware = create_rate_limit_middleware(redis_url)
    app.add_middleware(rate_limit_middleware.__class__, **rate_limit_middleware.__dict__)
    
    # Input validation middleware
    app.add_middleware(ValidationMiddleware)
    
    # CORS middleware (add after security middleware)
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"]
    )
    
    # Include routers
    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(tasks.router, prefix="/tasks", tags=["Tasks"])
    app.include_router(evaluation.router, prefix="/evaluation", tags=["Evaluation"])
    
    # Import leaderboard router
    try:
        from causal_eval.api.routes.leaderboard import router as leaderboard_router
        app.include_router(leaderboard_router, prefix="/leaderboard", tags=["Leaderboard"])
    except ImportError:
        logger.warning("Leaderboard router not available")
    
    # Add metrics endpoint
    app.get("/metrics", tags=["Monitoring"])(create_metrics_endpoint())
    
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "Causal Evaluation Bench",
            "version": "0.1.0",
            "description": "A comprehensive evaluation framework for testing genuine causal reasoning in language models",
            "docs_url": "/docs",
            "health_url": "/health",
            "metrics_url": "/metrics",
            "leaderboard_url": "/leaderboard"
        }
    
    return app