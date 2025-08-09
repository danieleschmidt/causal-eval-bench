"""API middleware components."""

try:
    from causal_eval.api.middleware.monitoring import MonitoringMiddleware
except ImportError:
    MonitoringMiddleware = None

try:
    from causal_eval.api.middleware.rate_limiting import create_rate_limit_middleware
except ImportError:
    def create_rate_limit_middleware(*args, **kwargs):
        return None

try:
    from causal_eval.api.middleware.validation import ValidationMiddleware
except ImportError:
    ValidationMiddleware = None

# Simple security headers middleware
class SecurityHeaders:
    """Add security headers to responses."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                headers[b"x-content-type-options"] = b"nosniff"
                headers[b"x-frame-options"] = b"DENY"
                headers[b"x-xss-protection"] = b"1; mode=block"
                message["headers"] = list(headers.items())
            await send(message)
        
        await self.app(scope, receive, send_wrapper)

__all__ = [
    "MonitoringMiddleware", 
    "create_rate_limit_middleware", 
    "ValidationMiddleware",
    "SecurityHeaders"
]