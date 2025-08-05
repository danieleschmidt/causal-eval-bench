"""
Comprehensive logging configuration for production deployment.
"""

import os
import sys
import logging
import logging.config
from typing import Dict, Any, Optional
import json
from datetime import datetime
import uuid
from contextlib import contextmanager
import structlog


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id
        
        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id
        
        if hasattr(record, 'model_name'):
            log_data['model_name'] = record.model_name
        
        if hasattr(record, 'task_type'):
            log_data['task_type'] = record.task_type
        
        if hasattr(record, 'evaluation_score'):
            log_data['evaluation_score'] = record.evaluation_score
        
        if hasattr(record, 'execution_time'):
            log_data['execution_time_ms'] = record.execution_time
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add stack trace for errors
        if record.levelno >= logging.ERROR and record.stack_info:
            log_data['stack_trace'] = record.stack_info
        
        return json.dumps(log_data, ensure_ascii=False)


class RequestContextFilter(logging.Filter):
    """Filter to add request context to log records."""
    
    def filter(self, record):
        """Add request context to log record."""
        # In production, this would extract from request context
        # For now, add placeholder request ID
        if not hasattr(record, 'request_id'):
            record.request_id = getattr(self, 'current_request_id', 'no-request')
        
        return True


class SecurityAuditLogger:
    """Specialized logger for security events."""
    
    def __init__(self, logger_name: str = "security_audit"):
        """Initialize security audit logger."""
        self.logger = logging.getLogger(logger_name)
    
    def log_authentication_attempt(self, user_id: str, success: bool, ip_address: str):
        """Log authentication attempt."""
        self.logger.info(
            "Authentication attempt",
            extra={
                "event_type": "authentication",
                "user_id": user_id,
                "success": success,
                "ip_address": ip_address,
                "security_event": True
            }
        )
    
    def log_rate_limit_exceeded(self, ip_address: str, endpoint: str, limit: int):
        """Log rate limit exceeded event."""
        self.logger.warning(
            "Rate limit exceeded",
            extra={
                "event_type": "rate_limit_exceeded",
                "ip_address": ip_address,
                "endpoint": endpoint,
                "limit": limit,
                "security_event": True
            }
        )
    
    def log_suspicious_input(self, ip_address: str, input_data: str, reason: str):
        """Log suspicious input detected."""
        self.logger.warning(
            "Suspicious input detected",
            extra={
                "event_type": "suspicious_input",
                "ip_address": ip_address,
                "input_preview": input_data[:100] + "..." if len(input_data) > 100 else input_data,
                "reason": reason,
                "security_event": True
            }
        )
    
    def log_api_key_usage(self, api_key_id: str, endpoint: str, cost: float):
        """Log API key usage for billing and monitoring."""
        self.logger.info(
            "API key usage",
            extra={
                "event_type": "api_usage",
                "api_key_id": api_key_id,
                "endpoint": endpoint,
                "cost": cost,
                "billing_event": True
            }
        )


class PerformanceLogger:
    """Logger for performance metrics and monitoring."""
    
    def __init__(self, logger_name: str = "performance"):
        """Initialize performance logger."""
        self.logger = logging.getLogger(logger_name)
    
    def log_evaluation_performance(
        self, 
        task_type: str, 
        model_name: str, 
        execution_time: float,
        token_count: int,
        score: float
    ):
        """Log evaluation performance metrics."""
        self.logger.info(
            "Evaluation completed",
            extra={
                "event_type": "evaluation_performance",
                "task_type": task_type,
                "model_name": model_name,
                "execution_time": execution_time,
                "token_count": token_count,
                "evaluation_score": score,
                "performance_metric": True
            }
        )
    
    def log_api_response_time(self, endpoint: str, method: str, response_time: float, status_code: int):
        """Log API response time metrics."""
        self.logger.info(
            "API response time",
            extra={
                "event_type": "api_response_time",
                "endpoint": endpoint,
                "method": method,
                "response_time_ms": response_time,
                "status_code": status_code,
                "performance_metric": True
            }
        )
    
    def log_database_query_time(self, query_type: str, execution_time: float, row_count: int):
        """Log database query performance."""
        self.logger.info(
            "Database query executed",
            extra={
                "event_type": "database_query",
                "query_type": query_type,
                "execution_time": execution_time,
                "row_count": row_count,
                "performance_metric": True
            }
        )


class LoggingManager:
    """Central logging manager for the application."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize logging manager."""
        self.config_path = config_path
        self.security_logger = SecurityAuditLogger()
        self.performance_logger = PerformanceLogger()
        self.request_context_filter = RequestContextFilter()
    
    def setup_logging(self, environment: str = "development", log_level: str = "INFO"):
        """Setup logging configuration based on environment."""
        
        if environment == "production":
            self._setup_production_logging(log_level)
        elif environment == "development":
            self._setup_development_logging(log_level)
        elif environment == "test":
            self._setup_test_logging()
        else:
            self._setup_default_logging(log_level)
        
        # Configure structlog
        self._setup_structlog()
        
        # Set up root logger
        root_logger = logging.getLogger()
        root_logger.addFilter(self.request_context_filter)
        
        # Log startup
        logger = logging.getLogger(__name__)
        logger.info(f"Logging configured for {environment} environment")
    
    def _setup_production_logging(self, log_level: str):
        """Setup production logging with JSON format."""
        
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": JSONFormatter,
                },
                "detailed": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": log_level,
                    "formatter": "json",
                    "stream": "ext://sys.stdout"
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": log_level,
                    "formatter": "json",
                    "filename": "/var/log/causal-eval/app.log",
                    "maxBytes": 50 * 1024 * 1024,  # 50MB
                    "backupCount": 10
                },
                "security_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "INFO",
                    "formatter": "json",
                    "filename": "/var/log/causal-eval/security.log",
                    "maxBytes": 50 * 1024 * 1024,
                    "backupCount": 20
                },
                "performance_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "INFO",
                    "formatter": "json",
                    "filename": "/var/log/causal-eval/performance.log",
                    "maxBytes": 50 * 1024 * 1024,
                    "backupCount": 10
                }
            },
            "loggers": {
                "security_audit": {
                    "level": "INFO",
                    "handlers": ["security_file", "console"],
                    "propagate": False
                },
                "performance": {
                    "level": "INFO",
                    "handlers": ["performance_file", "console"],
                    "propagate": False
                },
                "causal_eval": {
                    "level": log_level,
                    "handlers": ["file", "console"],
                    "propagate": False
                },
                "uvicorn": {
                    "level": "INFO",
                    "handlers": ["file", "console"],
                    "propagate": False
                },
                "fastapi": {
                    "level": "INFO",
                    "handlers": ["file", "console"],
                    "propagate": False
                }
            },
            "root": {
                "level": log_level,
                "handlers": ["console"]
            }
        }
        
        # Create log directories if they don't exist
        os.makedirs("/var/log/causal-eval", exist_ok=True)
        
        logging.config.dictConfig(config)
    
    def _setup_development_logging(self, log_level: str):
        """Setup development logging with colored console output."""
        
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "detailed": {
                    "format": "%(asctime)s [%(levelname)8s] %(name)s: %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S"
                },
                "simple": {
                    "format": "[%(levelname)s] %(name)s: %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": log_level,
                    "formatter": "detailed",
                    "stream": "ext://sys.stdout"
                },
                "file": {
                    "class": "logging.FileHandler",
                    "level": "DEBUG",
                    "formatter": "detailed",
                    "filename": "logs/development.log"
                }
            },
            "loggers": {
                "causal_eval": {
                    "level": log_level,
                    "handlers": ["console", "file"],
                    "propagate": False
                },
                "uvicorn": {
                    "level": "INFO",
                    "handlers": ["console"],
                    "propagate": False
                }
            },
            "root": {
                "level": log_level,
                "handlers": ["console"]
            }
        }
        
        # Create logs directory
        os.makedirs("logs", exist_ok=True)
        
        logging.config.dictConfig(config)
    
    def _setup_test_logging(self):
        """Setup minimal logging for tests."""
        
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "simple": {
                    "format": "[%(levelname)s] %(name)s: %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "WARNING",
                    "formatter": "simple",
                    "stream": "ext://sys.stdout"
                }
            },
            "root": {
                "level": "WARNING",
                "handlers": ["console"]
            }
        }
        
        logging.config.dictConfig(config)
    
    def _setup_default_logging(self, log_level: str):
        """Setup default logging configuration."""
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    def _setup_structlog(self):
        """Configure structlog for structured logging."""
        
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    @contextmanager
    def request_context(self, request_id: str, user_id: Optional[str] = None):
        """Context manager for request-specific logging."""
        old_request_id = getattr(self.request_context_filter, 'current_request_id', None)
        old_user_id = getattr(self.request_context_filter, 'current_user_id', None)
        
        self.request_context_filter.current_request_id = request_id
        if user_id:
            self.request_context_filter.current_user_id = user_id
        
        try:
            yield
        finally:
            self.request_context_filter.current_request_id = old_request_id
            self.request_context_filter.current_user_id = old_user_id


# Global logging manager instance
logging_manager = LoggingManager()

# Convenience functions
def setup_logging(environment: str = None, log_level: str = None):
    """Setup logging with environment detection."""
    environment = environment or os.getenv("ENVIRONMENT", "development")
    log_level = log_level or os.getenv("LOG_LEVEL", "INFO")
    
    logging_manager.setup_logging(environment, log_level)

def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance."""
    return logging.getLogger(name)

def get_security_logger() -> SecurityAuditLogger:
    """Get the security audit logger."""
    return logging_manager.security_logger

def get_performance_logger() -> PerformanceLogger:
    """Get the performance logger."""
    return logging_manager.performance_logger