"""FastAPI application factory and configuration."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from causal_eval.api.routes import health
from causal_eval.api.routes import evaluation_simple as evaluation
from causal_eval.api.routes import tasks_simple as tasks
from causal_eval.core.engine import EvaluationEngine
from causal_eval.core.tasks import TaskRegistry
from causal_eval.core.metrics import MetricsCollector
from causal_eval.core.logging_config import setup_logging
import os

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management - simplified."""
    # Setup logging first
    setup_logging()
    logger.info("Starting Causal Evaluation Bench API")
    
    # Initialize core components
    engine = EvaluationEngine()
    task_registry = TaskRegistry()
    metrics_collector = MetricsCollector()
    
    # Store in app state
    app.state.engine = engine
    app.state.task_registry = task_registry
    app.state.metrics_collector = metrics_collector
    
    yield
    
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
    
    # Basic security headers
    @app.middleware("http")
    async def security_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response
    
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
    
    @app.get("/metrics", tags=["Monitoring"])
    async def metrics():
        return {"message": "Metrics endpoint - implement Prometheus integration"}
    
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