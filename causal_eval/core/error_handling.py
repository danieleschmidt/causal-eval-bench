"""Comprehensive error handling and recovery mechanisms."""

import logging
import traceback
from typing import Any, Dict, Optional, Union
from enum import Enum
from dataclasses import dataclass
from fastapi import HTTPException
import json

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Error type classification."""
    VALIDATION_ERROR = "validation_error"
    TASK_ERROR = "task_error"
    EVALUATION_ERROR = "evaluation_error"
    PARSING_ERROR = "parsing_error"
    MODEL_ERROR = "model_error"
    SYSTEM_ERROR = "system_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    AUTHENTICATION_ERROR = "authentication_error"


@dataclass
class CausalEvalError:
    """Structured error information."""
    error_type: ErrorType
    message: str
    details: Dict[str, Any]
    recoverable: bool = True
    user_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "error_type": self.error_type.value,
            "message": self.message,
            "details": self.details,
            "recoverable": self.recoverable,
            "user_message": self.user_message or self.message
        }


class ErrorHandler:
    """Central error handling and logging."""
    
    def __init__(self):
        self.error_counts = {}
    
    def handle_evaluation_error(
        self, 
        error: Exception, 
        context: Dict[str, Any] = None
    ) -> CausalEvalError:
        """Handle evaluation-specific errors."""
        context = context or {}
        
        if isinstance(error, ValueError):
            return CausalEvalError(
                error_type=ErrorType.VALIDATION_ERROR,
                message=str(error),
                details={"context": context, "exception_type": type(error).__name__},
                recoverable=True,
                user_message="Invalid input parameters. Please check your request."
            )
        
        elif isinstance(error, KeyError):
            return CausalEvalError(
                error_type=ErrorType.TASK_ERROR,
                message=f"Missing required parameter: {str(error)}",
                details={"context": context, "missing_key": str(error)},
                recoverable=True,
                user_message="Missing required parameter in request."
            )
        
        elif "parsing" in str(error).lower():
            return CausalEvalError(
                error_type=ErrorType.PARSING_ERROR,
                message=str(error),
                details={"context": context, "response_sample": context.get("model_response", "")[:200]},
                recoverable=True,
                user_message="Unable to parse model response. The response may not follow expected format."
            )
        
        else:
            # Generic system error
            error_trace = traceback.format_exc()
            logger.error(f"Unhandled evaluation error: {error_trace}")
            
            return CausalEvalError(
                error_type=ErrorType.SYSTEM_ERROR,
                message=str(error),
                details={"context": context, "traceback": error_trace},
                recoverable=False,
                user_message="An internal error occurred. Please try again later."
            )
    
    def handle_task_creation_error(
        self, 
        error: Exception, 
        task_type: str, 
        domain: str, 
        difficulty: str
    ) -> CausalEvalError:
        """Handle task creation errors."""
        if "unknown task type" in str(error).lower():
            return CausalEvalError(
                error_type=ErrorType.TASK_ERROR,
                message=f"Unknown task type: {task_type}",
                details={
                    "task_type": task_type,
                    "domain": domain,
                    "difficulty": difficulty,
                    "available_types": ["attribution", "counterfactual", "intervention"]
                },
                recoverable=True,
                user_message=f"Task type '{task_type}' is not supported. Available types: attribution, counterfactual, intervention"
            )
        
        return self.handle_evaluation_error(error, {
            "task_type": task_type,
            "domain": domain, 
            "difficulty": difficulty
        })
    
    def log_error(self, error: CausalEvalError, request_id: str = None) -> None:
        """Log error with structured information."""
        log_data = {
            "error_type": error.error_type.value,
            "message": error.message,
            "recoverable": error.recoverable,
            "details": error.details,
            "request_id": request_id
        }
        
        if error.recoverable:
            logger.warning(f"Recoverable error: {json.dumps(log_data)}")
        else:
            logger.error(f"System error: {json.dumps(log_data)}")
        
        # Track error frequency
        error_key = f"{error.error_type.value}:{error.message[:50]}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Alert on frequent errors
        if self.error_counts[error_key] > 10:
            logger.critical(f"Frequent error detected: {error_key} - Count: {self.error_counts[error_key]}")
    
    def create_http_exception(self, error: CausalEvalError) -> HTTPException:
        """Convert CausalEvalError to HTTP exception."""
        status_code_map = {
            ErrorType.VALIDATION_ERROR: 400,
            ErrorType.TASK_ERROR: 400,
            ErrorType.EVALUATION_ERROR: 422,
            ErrorType.PARSING_ERROR: 422,
            ErrorType.MODEL_ERROR: 503,
            ErrorType.SYSTEM_ERROR: 500,
            ErrorType.RATE_LIMIT_ERROR: 429,
            ErrorType.AUTHENTICATION_ERROR: 401
        }
        
        status_code = status_code_map.get(error.error_type, 500)
        
        return HTTPException(
            status_code=status_code,
            detail={
                "error": error.user_message,
                "error_type": error.error_type.value,
                "recoverable": error.recoverable,
                "details": error.details if error.recoverable else {}
            }
        )


class CircuitBreaker:
    """Circuit breaker pattern for handling repeated failures."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failed call."""
        import time
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt to reset."""
        import time
        if self.last_failure_time is None:
            return False
        return time.time() - self.last_failure_time >= self.timeout


# Global error handler instance
error_handler = ErrorHandler()

# Helper functions
def handle_evaluation_error(error: Exception, context: Dict[str, Any] = None) -> CausalEvalError:
    """Global error handling function."""
    return error_handler.handle_evaluation_error(error, context)

def create_http_error(error: CausalEvalError) -> HTTPException:
    """Create HTTP exception from CausalEvalError."""
    return error_handler.create_http_exception(error)