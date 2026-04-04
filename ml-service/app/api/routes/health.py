"""
SafeChat — Health Check API Route

GET /api/v1/health — Service health with model status and GPU info
"""

import time
from fastapi import APIRouter

from app.config import settings
from app.models.model_manager import model_manager

router = APIRouter(prefix="/api/v1", tags=["Health"])

_start_time = time.time()


@router.get("/health")
async def health_check():
    """
    Comprehensive health check including model status and GPU metrics.

    Used by:
      - Spring Boot backend to verify ML service availability
      - Docker health checks
      - Monitoring dashboards
    """
    health = model_manager.get_health()

    return {
        **health,
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "uptime_seconds": int(time.time() - _start_time),
    }


@router.get("/ready")
async def readiness_check():
    """
    Readiness probe — returns 200 only when models are loaded and ready.
    Used by Kubernetes / Docker for routing traffic.
    """
    if model_manager.is_ready:
        return {"ready": True}
    return {"ready": False, "detail": "Models still loading..."}
