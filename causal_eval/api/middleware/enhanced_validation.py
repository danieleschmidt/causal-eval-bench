"""Enhanced input validation and security middleware."""

import re
import logging
from typing import Any, Dict, List
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)


class SecurityValidator:
    """Security validation for inputs."""
    
    SUSPICIOUS_PATTERNS = [
        r"<script[^>]*>.*?</script>",  # XSS attempts
        r"javascript:",                # JavaScript protocol
        r"data:text/html",            # HTML data URIs
        r"eval\s*\(",                 # Code evaluation
        r"exec\s*\(",                 # Code execution
        r"__import__",                # Python imports
        r"subprocess",                # System commands
        r"\bdelete\b.*\bfrom\b",     # SQL injection
        r"\bunion\b.*\bselect\b",    # SQL union attacks
        r"\.\.\/",                    # Directory traversal
    ]
    
    MAX_TEXT_LENGTH = 10000  # 10KB max for model responses
    MAX_PROMPT_LENGTH = 5000  # 5KB max for prompts
    
    @classmethod
    def validate_text_input(cls, text: str, field_name: str) -> str:
        """Validate text input for security issues."""
        if not text:
            return text
            
        # Length check
        if len(text) > cls.MAX_TEXT_LENGTH:
            raise HTTPException(
                status_code=400, 
                detail=f"{field_name} exceeds maximum length of {cls.MAX_TEXT_LENGTH} characters"
            )
        
        # Security pattern check
        for pattern in cls.SUSPICIOUS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning(f"Suspicious pattern detected in {field_name}: {pattern}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid content detected in {field_name}"
                )
        
        return text
    
    @classmethod
    def validate_task_parameters(cls, task_type: str, domain: str, difficulty: str) -> Dict[str, str]:
        """Validate task parameters."""
        # Valid values
        valid_task_types = ["attribution", "counterfactual", "intervention", "chain", "confounding"]
        valid_domains = ["general", "medical", "education", "business", "technology", 
                        "environmental", "workplace_safety", "urban_planning", 
                        "manufacturing", "recreational", "public_safety", "international"]
        valid_difficulties = ["easy", "medium", "hard"]
        
        if task_type not in valid_task_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid task_type. Must be one of: {', '.join(valid_task_types)}"
            )
        
        if domain not in valid_domains:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid domain. Must be one of: {', '.join(valid_domains)}"
            )
        
        if difficulty not in valid_difficulties:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid difficulty. Must be one of: {', '.join(valid_difficulties)}"
            )
        
        return {
            "task_type": task_type,
            "domain": domain,
            "difficulty": difficulty
        }


class RobustValidationMiddleware(BaseHTTPMiddleware):
    """Enhanced validation middleware with comprehensive security checks."""
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with enhanced validation."""
        start_time = time.time()
        
        try:
            # Rate limiting check (simplified)
            client_ip = request.client.host if request.client else "unknown"
            await self._check_rate_limit(client_ip)
            
            # Content-Type validation for POST requests
            if request.method == "POST":
                content_type = request.headers.get("content-type", "")
                if not content_type.startswith("application/json"):
                    raise HTTPException(
                        status_code=415,
                        detail="Content-Type must be application/json"
                    )
            
            # Request size validation
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > 1024 * 1024:  # 1MB limit
                raise HTTPException(
                    status_code=413,
                    detail="Request too large"
                )
            
            response = await call_next(request)
            
            # Log request metrics
            process_time = time.time() - start_time
            logger.info(
                f"Request processed: {request.method} {request.url.path} - "
                f"Status: {response.status_code} - Time: {process_time:.3f}s - IP: {client_ip}"
            )
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Validation middleware error: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    async def _check_rate_limit(self, client_ip: str) -> None:
        """Basic rate limiting check."""
        # TODO: Implement proper rate limiting with Redis
        # For now, just log the request
        logger.debug(f"Rate limit check for IP: {client_ip}")


import time