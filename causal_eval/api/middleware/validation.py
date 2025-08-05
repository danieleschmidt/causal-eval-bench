"""
Input validation and sanitization middleware.
"""

import re
import html
from typing import Any, Dict, List, Optional, Union
from fastapi import HTTPException, Request
from pydantic import BaseModel, validator
import logging
import bleach
from urllib.parse import unquote

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom validation error."""
    pass


class InputSanitizer:
    """Comprehensive input sanitization and validation."""
    
    # Allowed HTML tags and attributes for rich text fields
    ALLOWED_TAGS = ['p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li', 'blockquote', 'code']
    ALLOWED_ATTRIBUTES = {
        'a': ['href', 'title'],
        'blockquote': ['cite'],
    }
    
    # Dangerous patterns to detect
    DANGEROUS_PATTERNS = [
        # SQL injection patterns
        r"(?i)(union\s+select|drop\s+table|delete\s+from|insert\s+into)",
        r"(?i)(exec\s*\(|execute\s*\(|sp_executesql)",
        r"(?i)(script\s*>|javascript:|vbscript:|data:)",
        
        # XSS patterns
        r"(?i)(<script|</script>|<iframe|</iframe>|<object|</object>)",
        r"(?i)(on\w+\s*=|expression\s*\(|url\s*\()",
        
        # Command injection patterns
        r"(?i)(;|\||&|\$\(|\`|<\(|>\()",
        r"(?i)(rm\s+|del\s+|format\s+|shutdown\s+)",
        
        # Path traversal patterns
        r"(\.\./|\.\.\\|~\/|%2e%2e%2f|%2e%2e%5c)",
        
        # Template injection patterns
        r"(\{\{|\}\}|\{%|%\}|\${|<#|#>)",
    ]
    
    @classmethod
    def sanitize_string(cls, value: str, max_length: int = 10000, allow_html: bool = False) -> str:
        """
        Sanitize string input.
        
        Args:
            value: Input string to sanitize
            max_length: Maximum allowed length
            allow_html: Whether to allow HTML tags
            
        Returns:
            Sanitized string
            
        Raises:
            ValidationError: If input is invalid or dangerous
        """
        if not isinstance(value, str):
            raise ValidationError("Input must be a string")
        
        # URL decode to prevent encoding bypass
        decoded_value = unquote(value)
        
        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, decoded_value):
                logger.warning(f"Dangerous pattern detected: {pattern[:50]}...")
                raise ValidationError("Invalid input detected")
        
        # Length validation
        if len(value) > max_length:
            raise ValidationError(f"Input too long. Maximum length: {max_length}")
        
        # HTML sanitization
        if allow_html:
            # Use bleach to sanitize HTML
            sanitized = bleach.clean(
                value,
                tags=cls.ALLOWED_TAGS,
                attributes=cls.ALLOWED_ATTRIBUTES,
                strip=True
            )
        else:
            # Escape HTML entities
            sanitized = html.escape(value)
        
        # Remove null bytes and control characters
        sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in '\n\r\t')
        
        return sanitized.strip()
    
    @classmethod
    def sanitize_dict(cls, data: Dict[str, Any], max_depth: int = 10) -> Dict[str, Any]:
        """
        Recursively sanitize dictionary data.
        
        Args:
            data: Dictionary to sanitize
            max_depth: Maximum nesting depth allowed
            
        Returns:
            Sanitized dictionary
        """
        if max_depth <= 0:
            raise ValidationError("Maximum nesting depth exceeded")
        
        sanitized = {}
        
        for key, value in data.items():
            # Sanitize key
            if not isinstance(key, str):
                key = str(key)
            
            sanitized_key = cls.sanitize_string(key, max_length=256)
            
            # Sanitize value based on type
            if isinstance(value, str):
                sanitized_value = cls.sanitize_string(value)
            elif isinstance(value, dict):
                sanitized_value = cls.sanitize_dict(value, max_depth - 1)
            elif isinstance(value, list):
                sanitized_value = cls.sanitize_list(value, max_depth - 1)
            elif isinstance(value, (int, float, bool)) or value is None:
                sanitized_value = value
            else:
                # Convert unknown types to string and sanitize
                sanitized_value = cls.sanitize_string(str(value))
            
            sanitized[sanitized_key] = sanitized_value
        
        return sanitized
    
    @classmethod
    def sanitize_list(cls, data: List[Any], max_depth: int = 10) -> List[Any]:
        """
        Sanitize list data.
        
        Args:
            data: List to sanitize
            max_depth: Maximum nesting depth allowed
            
        Returns:
            Sanitized list
        """
        if max_depth <= 0:
            raise ValidationError("Maximum nesting depth exceeded")
        
        if len(data) > 1000:  # Prevent large lists
            raise ValidationError("List too large. Maximum size: 1000")
        
        sanitized = []
        
        for item in data:
            if isinstance(item, str):
                sanitized_item = cls.sanitize_string(item)
            elif isinstance(item, dict):
                sanitized_item = cls.sanitize_dict(item, max_depth - 1)
            elif isinstance(item, list):
                sanitized_item = cls.sanitize_list(item, max_depth - 1)
            elif isinstance(item, (int, float, bool)) or item is None:
                sanitized_item = item
            else:
                sanitized_item = cls.sanitize_string(str(item))
            
            sanitized.append(sanitized_item)
        
        return sanitized


class RequestValidator:
    """Request validation with security checks."""
    
    @staticmethod
    def validate_model_name(model_name: str) -> str:
        """Validate model name format."""
        if not model_name:
            raise ValidationError("Model name cannot be empty")
        
        # Allow alphanumeric, hyphens, underscores, and dots
        if not re.match(r'^[a-zA-Z0-9\-_.]+$', model_name):
            raise ValidationError("Invalid model name format")
        
        if len(model_name) > 100:
            raise ValidationError("Model name too long")
        
        return model_name
    
    @staticmethod
    def validate_task_type(task_type: str) -> str:
        """Validate task type."""
        allowed_tasks = ['attribution', 'counterfactual', 'intervention']
        
        if task_type not in allowed_tasks:
            raise ValidationError(f"Invalid task type. Allowed: {allowed_tasks}")
        
        return task_type
    
    @staticmethod
    def validate_domain(domain: str) -> str:
        """Validate domain."""
        allowed_domains = [
            'general', 'medical', 'education', 'business', 'technology',
            'environmental', 'workplace_safety', 'urban_planning', 
            'manufacturing', 'recreational', 'public_safety', 'international'
        ]
        
        if domain not in allowed_domains:
            raise ValidationError(f"Invalid domain. Allowed: {allowed_domains}")
        
        return domain
    
    @staticmethod
    def validate_difficulty(difficulty: str) -> str:
        """Validate difficulty level."""
        allowed_difficulties = ['easy', 'medium', 'hard']
        
        if difficulty not in allowed_difficulties:
            raise ValidationError(f"Invalid difficulty. Allowed: {allowed_difficulties}")
        
        return difficulty
    
    @staticmethod
    def validate_batch_size(batch_size: int) -> int:
        """Validate batch size."""
        if batch_size < 1:
            raise ValidationError("Batch size must be at least 1")
        
        if batch_size > 100:  # Prevent resource exhaustion
            raise ValidationError("Batch size too large. Maximum: 100")
        
        return batch_size


class ValidationMiddleware:
    """Middleware for request validation and sanitization."""
    
    def __init__(self, sanitize_input: bool = True, validate_content_type: bool = True):
        """Initialize validation middleware."""
        self.sanitize_input = sanitize_input
        self.validate_content_type = validate_content_type
        self.sanitizer = InputSanitizer()
    
    async def __call__(self, request: Request, call_next):
        """Process request with validation."""
        try:
            # Validate content type for POST/PUT requests
            if self.validate_content_type and request.method in ["POST", "PUT", "PATCH"]:
                content_type = request.headers.get("content-type", "")
                if not content_type.startswith("application/json"):
                    return JSONResponse(
                        status_code=400,
                        content={"error": "Invalid content type. Expected: application/json"}
                    )
            
            # Validate request size
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB limit
                return JSONResponse(
                    status_code=413,
                    content={"error": "Request entity too large. Maximum size: 10MB"}
                )
            
            # Process request
            response = await call_next(request)
            return response
            
        except ValidationError as e:
            logger.warning(f"Validation error: {e}")
            return JSONResponse(
                status_code=400,
                content={"error": "Validation failed", "message": str(e)}
            )
        except Exception as e:
            logger.error(f"Validation middleware error: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal validation error"}
            )


# Enhanced Pydantic models with validation
class EnhancedEvaluationRequest(BaseModel):
    """Enhanced evaluation request with comprehensive validation."""
    
    task_type: str
    model_response: Optional[str] = None
    domain: Optional[str] = "general"
    difficulty: Optional[str] = "medium"
    model_name: Optional[str] = None
    metadata: Dict[str, Any] = {}
    
    @validator('task_type')
    def validate_task_type(cls, v):
        return RequestValidator.validate_task_type(v)
    
    @validator('domain')
    def validate_domain(cls, v):
        if v:
            return RequestValidator.validate_domain(v)
        return v
    
    @validator('difficulty')
    def validate_difficulty(cls, v):
        if v:
            return RequestValidator.validate_difficulty(v)
        return v
    
    @validator('model_name')
    def validate_model_name(cls, v):
        if v:
            return RequestValidator.validate_model_name(v)
        return v
    
    @validator('model_response')
    def validate_model_response(cls, v):
        if v:
            return InputSanitizer.sanitize_string(v, max_length=50000)
        return v
    
    @validator('metadata')
    def validate_metadata(cls, v):
        if v:
            return InputSanitizer.sanitize_dict(v, max_depth=5)
        return v


class SecurityHeaders:
    """Security headers middleware."""
    
    def __init__(self):
        """Initialize security headers."""
        self.headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }
    
    async def __call__(self, request: Request, call_next):
        """Add security headers to response."""
        response = await call_next(request)
        
        for header, value in self.headers.items():
            response.headers[header] = value
        
        return response