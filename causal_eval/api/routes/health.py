"""Health check endpoints."""

from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Dict, Any

router = APIRouter()


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
            "database": "not_configured",
            "cache": "not_configured"
        }
    )


@router.get("/ready")
async def readiness_check(request: Request) -> Dict[str, Any]:
    """Readiness check for Kubernetes."""
    return {"status": "ready"}


@router.get("/live")
async def liveness_check(request: Request) -> Dict[str, Any]:
    """Liveness check for Kubernetes."""
    return {"status": "alive"}