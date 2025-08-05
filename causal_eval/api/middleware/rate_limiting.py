"""
Rate limiting middleware for API endpoints.
"""

import time
from typing import Dict, Optional
from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
import asyncio
import logging
from datetime import datetime, timedelta
import redis
from redis import Redis
import json

logger = logging.getLogger(__name__)


class RateLimiter:
    """Redis-based rate limiter with sliding window algorithm."""
    
    def __init__(self, redis_client: Optional[Redis] = None):
        """Initialize rate limiter."""
        self.redis_client = redis_client
        self.local_cache: Dict[str, Dict] = {}
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
    
    async def is_allowed(
        self,
        key: str,
        limit: int,
        window: int,
        identifier: str = "default"
    ) -> tuple[bool, Dict[str, any]]:
        """
        Check if request is allowed under rate limit.
        
        Args:
            key: Unique identifier for rate limiting (e.g., IP address)
            limit: Maximum number of requests allowed
            window: Time window in seconds
            identifier: Additional identifier (e.g., endpoint name)
            
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        current_time = time.time()
        cache_key = f"rate_limit:{identifier}:{key}"
        
        try:
            if self.redis_client:
                return await self._redis_check(cache_key, limit, window, current_time)
            else:
                return await self._local_check(cache_key, limit, window, current_time)
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            # Fail open - allow request if rate limiter fails
            return True, {
                "limit": limit,
                "remaining": limit - 1,
                "reset": int(current_time + window),
                "retry_after": None
            }
    
    async def _redis_check(
        self, 
        cache_key: str, 
        limit: int, 
        window: int, 
        current_time: float
    ) -> tuple[bool, Dict[str, any]]:
        """Redis-based sliding window rate limiting."""
        pipe = self.redis_client.pipeline()
        
        # Remove old entries outside the window
        pipe.zremrangebyscore(cache_key, 0, current_time - window)
        
        # Count current requests in window
        pipe.zcard(cache_key)
        
        # Add current request
        pipe.zadd(cache_key, {str(current_time): current_time})
        
        # Set expiration
        pipe.expire(cache_key, window + 1)
        
        results = pipe.execute()
        request_count = results[1]
        
        remaining = max(0, limit - request_count - 1)
        is_allowed = request_count < limit
        
        if not is_allowed:
            # Remove the request we just added since it's not allowed
            self.redis_client.zrem(cache_key, str(current_time))
        
        return is_allowed, {
            "limit": limit,
            "remaining": remaining,
            "reset": int(current_time + window),
            "retry_after": window if not is_allowed else None
        }
    
    async def _local_check(
        self, 
        cache_key: str, 
        limit: int, 
        window: int, 
        current_time: float
    ) -> tuple[bool, Dict[str, any]]:
        """Local memory-based sliding window rate limiting."""
        self._cleanup_local_cache(current_time)
        
        if cache_key not in self.local_cache:
            self.local_cache[cache_key] = {"requests": [], "created": current_time}
        
        entry = self.local_cache[cache_key]
        requests = entry["requests"]
        
        # Remove old requests outside the window
        entry["requests"] = [req_time for req_time in requests if req_time > current_time - window]
        requests = entry["requests"]
        
        remaining = max(0, limit - len(requests) - 1)
        is_allowed = len(requests) < limit
        
        if is_allowed:
            requests.append(current_time)
        
        return is_allowed, {
            "limit": limit,
            "remaining": remaining,
            "reset": int(current_time + window),
            "retry_after": window if not is_allowed else None
        }
    
    def _cleanup_local_cache(self, current_time: float):
        """Clean up old entries from local cache."""
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        expired_keys = []
        for key, entry in self.local_cache.items():
            if current_time - entry["created"] > 3600:  # 1 hour
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.local_cache[key]
        
        self.last_cleanup = current_time


class RateLimitMiddleware:
    """FastAPI middleware for rate limiting."""
    
    def __init__(self, rate_limiter: RateLimiter, default_limits: Dict[str, Dict[str, int]] = None):
        """Initialize rate limiting middleware."""
        self.rate_limiter = rate_limiter
        self.default_limits = default_limits or {
            "default": {"limit": 100, "window": 3600},  # 100 requests per hour
            "evaluation": {"limit": 50, "window": 3600},  # 50 evaluations per hour
            "batch": {"limit": 5, "window": 3600},  # 5 batch requests per hour
            "heavy": {"limit": 10, "window": 3600}  # 10 heavy operations per hour
        }
    
    async def __call__(self, request: Request, call_next):
        """Process request with rate limiting."""
        # Get client identifier
        client_ip = self._get_client_ip(request)
        
        # Determine rate limit category
        category = self._get_rate_limit_category(request)
        limits = self.default_limits.get(category, self.default_limits["default"])
        
        # Check rate limit
        is_allowed, rate_info = await self.rate_limiter.is_allowed(
            key=client_ip,
            limit=limits["limit"],
            window=limits["window"],
            identifier=category
        )
        
        if not is_allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Limit: {rate_info['limit']} per {limits['window']} seconds",
                    "rate_limit": rate_info
                },
                headers={
                    "X-RateLimit-Limit": str(rate_info["limit"]),
                    "X-RateLimit-Remaining": str(rate_info["remaining"]),
                    "X-RateLimit-Reset": str(rate_info["reset"]),
                    "Retry-After": str(rate_info["retry_after"]) if rate_info["retry_after"] else "3600"
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(rate_info["reset"])
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded IP (behind proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to client host
        return request.client.host if request.client else "unknown"
    
    def _get_rate_limit_category(self, request: Request) -> str:
        """Determine rate limit category based on request path."""
        path = request.url.path
        
        if "/evaluation/batch" in path:
            return "batch"
        elif "/evaluation/" in path:
            return "evaluation"
        elif any(heavy_path in path for heavy_path in ["/analyze", "/process", "/compute"]):
            return "heavy"
        else:
            return "default"


# Rate limiting decorators for specific endpoints
def rate_limit(limit: int, window: int, category: str = "default"):
    """Decorator for endpoint-specific rate limiting."""
    def decorator(func):
        func._rate_limit = {"limit": limit, "window": window, "category": category}
        return func
    return decorator


class AdaptiveRateLimiter(RateLimiter):
    """Advanced rate limiter with adaptive limits based on system load."""
    
    def __init__(self, *args, **kwargs):
        """Initialize adaptive rate limiter."""
        super().__init__(*args, **kwargs)
        self.system_load_threshold = 0.8
        self.adaptive_factor = 0.5
        
    async def get_adaptive_limit(self, base_limit: int, endpoint: str) -> int:
        """Calculate adaptive rate limit based on system metrics."""
        try:
            # In production, this would check actual system metrics
            # For now, return base limit
            system_load = 0.3  # Placeholder
            
            if system_load > self.system_load_threshold:
                # Reduce limits when system is under high load
                adaptive_limit = int(base_limit * self.adaptive_factor)
                logger.info(f"Reduced rate limit for {endpoint}: {base_limit} -> {adaptive_limit}")
                return adaptive_limit
            
            return base_limit
            
        except Exception as e:
            logger.error(f"Error calculating adaptive rate limit: {e}")
            return base_limit


def create_rate_limiter(redis_url: Optional[str] = None) -> RateLimiter:
    """Factory function to create rate limiter with optional Redis."""
    redis_client = None
    
    if redis_url:
        try:
            redis_client = redis.from_url(redis_url, decode_responses=True)
            redis_client.ping()  # Test connection
            logger.info("Connected to Redis for rate limiting")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Using local rate limiting.")
            redis_client = None
    
    return RateLimiter(redis_client)


def create_rate_limit_middleware(redis_url: Optional[str] = None) -> RateLimitMiddleware:
    """Factory function to create rate limiting middleware."""
    rate_limiter = create_rate_limiter(redis_url)
    
    # Production-ready rate limits
    limits = {
        "default": {"limit": 1000, "window": 3600},  # 1000 requests per hour
        "evaluation": {"limit": 100, "window": 3600},  # 100 evaluations per hour
        "batch": {"limit": 10, "window": 3600},  # 10 batch requests per hour
        "model_api": {"limit": 50, "window": 3600},  # 50 model API calls per hour
        "heavy": {"limit": 20, "window": 3600}  # 20 heavy operations per hour
    }
    
    return RateLimitMiddleware(rate_limiter, limits)