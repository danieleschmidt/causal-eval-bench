"""Comprehensive security framework for causal evaluation system."""

import hashlib
import hmac
import secrets
import time
import re
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import json

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventType(Enum):
    """Types of security events."""
    AUTHENTICATION_FAILURE = "authentication_failure"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_INPUT = "suspicious_input"
    INJECTION_ATTEMPT = "injection_attempt"
    XSS_ATTEMPT = "xss_attempt"
    DIRECTORY_TRAVERSAL = "directory_traversal"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    BRUTE_FORCE_ATTACK = "brute_force_attack"


@dataclass
class SecurityEvent:
    """Security event record."""
    event_type: SecurityEventType
    level: SecurityLevel
    source_ip: str
    user_agent: str
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    blocked: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "event_type": self.event_type.value,
            "level": self.level.value,
            "source_ip": self.source_ip,
            "user_agent": self.user_agent,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "blocked": self.blocked
        }


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    max_requests: int
    window_seconds: int
    burst_allowance: int = 0
    
    def __post_init__(self):
        """Validate configuration."""
        if self.max_requests <= 0:
            raise ValueError("max_requests must be positive")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")


class SecurityValidator:
    """Advanced input validation and security scanning."""
    
    # XSS patterns
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"data:text/html",
        r"vbscript:",
        r"onload\s*=",
        r"onerror\s*=",
        r"onclick\s*=",
        r"onfocus\s*=",
        r"onmouseover\s*=",
        r"<iframe[^>]*>",
        r"<object[^>]*>",
        r"<embed[^>]*>",
    ]
    
    # SQL injection patterns  
    SQL_INJECTION_PATTERNS = [
        r"union\s+select",
        r"or\s+1\s*=\s*1",
        r"and\s+1\s*=\s*1", 
        r"''\s*or\s*''",
        r"drop\s+table",
        r"delete\s+from",
        r"insert\s+into",
        r"update\s+\w+\s+set",
        r"exec\s*\(",
        r"sp_\w+",
        r"xp_\w+",
    ]
    
    # Code injection patterns
    CODE_INJECTION_PATTERNS = [
        r"eval\s*\(",
        r"exec\s*\(",
        r"system\s*\(",
        r"shell_exec\s*\(",
        r"passthru\s*\(",
        r"__import__",
        r"subprocess\.",
        r"os\.(system|popen|spawn)",
        r"file_get_contents",
        r"file_put_contents",
        r"fopen\s*\(",
        r"include\s*\(",
        r"require\s*\(",
    ]
    
    # Directory traversal patterns
    DIRECTORY_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.\\",
        r"%2e%2e%2f",
        r"%2e%2e%5c",
        r"..%2f",
        r"..%5c",
        r"%252e%252e%252f",
    ]
    
    # Command injection patterns
    COMMAND_INJECTION_PATTERNS = [
        r";\s*\w+",
        r"\|\s*\w+", 
        r"&&\s*\w+",
        r"\$\(\w+\)",
        r"`\w+`",
        r"nc\s+-",
        r"wget\s+",
        r"curl\s+",
        r"ping\s+",
        r"nslookup\s+",
    ]
    
    @classmethod
    def validate_input(cls, input_data: str, field_name: str, max_length: int = 10000) -> Tuple[bool, List[str]]:
        """Comprehensive input validation."""
        violations = []
        
        # Length check
        if len(input_data) > max_length:
            violations.append(f"Input exceeds maximum length of {max_length}")
        
        # XSS detection
        xss_detected = cls._check_patterns(input_data, cls.XSS_PATTERNS, "XSS")
        if xss_detected:
            violations.extend(xss_detected)
        
        # SQL injection detection
        sql_detected = cls._check_patterns(input_data, cls.SQL_INJECTION_PATTERNS, "SQL injection")
        if sql_detected:
            violations.extend(sql_detected)
        
        # Code injection detection
        code_detected = cls._check_patterns(input_data, cls.CODE_INJECTION_PATTERNS, "Code injection")
        if code_detected:
            violations.extend(code_detected)
        
        # Directory traversal detection
        traversal_detected = cls._check_patterns(input_data, cls.DIRECTORY_TRAVERSAL_PATTERNS, "Directory traversal")
        if traversal_detected:
            violations.extend(traversal_detected)
        
        # Command injection detection
        command_detected = cls._check_patterns(input_data, cls.COMMAND_INJECTION_PATTERNS, "Command injection")
        if command_detected:
            violations.extend(command_detected)
        
        # Additional suspicious patterns
        suspicious_detected = cls._check_suspicious_content(input_data)
        if suspicious_detected:
            violations.extend(suspicious_detected)
        
        is_valid = len(violations) == 0
        return is_valid, violations
    
    @classmethod
    def _check_patterns(cls, text: str, patterns: List[str], category: str) -> List[str]:
        """Check text against security patterns."""
        violations = []
        text_lower = text.lower()
        
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                violations.append(f"{category} pattern detected: {pattern}")
                break  # Only report one violation per category to avoid spam
        
        return violations
    
    @classmethod
    def _check_suspicious_content(cls, text: str) -> List[str]:
        """Check for additional suspicious content."""
        violations = []
        text_lower = text.lower()
        
        # Check for excessively long strings (potential buffer overflow)
        words = text.split()
        for word in words:
            if len(word) > 500:
                violations.append("Suspiciously long string detected")
                break
        
        # Check for repeated characters (potential DoS)
        for char in "abcdefghijklmnopqrstuvwxyz0123456789":
            if char * 100 in text_lower:
                violations.append("Repeated character pattern detected")
                break
        
        # Check for null bytes
        if '\x00' in text:
            violations.append("Null byte detected")
        
        # Check for control characters
        control_chars = set(ord(c) for c in text if ord(c) < 32 and c not in '\t\n\r')
        if control_chars:
            violations.append("Control characters detected")
        
        return violations
    
    @classmethod
    def sanitize_input(cls, input_data: str) -> str:
        """Sanitize input by removing/encoding dangerous content."""
        # Remove null bytes
        sanitized = input_data.replace('\x00', '')
        
        # Remove control characters except tab, newline, carriage return
        sanitized = ''.join(c for c in sanitized if ord(c) >= 32 or c in '\t\n\r')
        
        # Limit consecutive repeated characters
        sanitized = re.sub(r'(.)\1{50,}', r'\1' * 50, sanitized)
        
        # Basic HTML encoding for dangerous characters
        html_escape_table = {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#x27;",
            "/": "&#x2F;",
        }
        
        for char, escape in html_escape_table.items():
            sanitized = sanitized.replace(char, escape)
        
        return sanitized


class RateLimiter:
    """Advanced rate limiting with sliding window and burst protection."""
    
    def __init__(self):
        """Initialize rate limiter."""
        self.request_logs: Dict[str, List[float]] = {}
        self.blocked_ips: Dict[str, datetime] = {}
        self.configs: Dict[str, RateLimitConfig] = {
            "default": RateLimitConfig(max_requests=100, window_seconds=60),
            "evaluation": RateLimitConfig(max_requests=50, window_seconds=60),
            "batch": RateLimitConfig(max_requests=10, window_seconds=60),
            "auth": RateLimitConfig(max_requests=5, window_seconds=300),  # 5 attempts in 5 minutes
        }
    
    def check_rate_limit(self, identifier: str, endpoint: str = "default") -> Tuple[bool, Dict[str, Any]]:
        """Check if request is within rate limits."""
        current_time = time.time()
        config = self.configs.get(endpoint, self.configs["default"])
        
        # Check if IP is currently blocked
        if identifier in self.blocked_ips:
            if current_time < self.blocked_ips[identifier].timestamp():
                return False, {
                    "blocked": True,
                    "reason": "IP temporarily blocked",
                    "blocked_until": self.blocked_ips[identifier].isoformat()
                }
            else:
                # Unblock expired blocks
                del self.blocked_ips[identifier]
        
        # Initialize request log for this identifier
        if identifier not in self.request_logs:
            self.request_logs[identifier] = []
        
        # Clean old requests outside the window
        window_start = current_time - config.window_seconds
        self.request_logs[identifier] = [
            req_time for req_time in self.request_logs[identifier] 
            if req_time > window_start
        ]
        
        # Check rate limit
        request_count = len(self.request_logs[identifier])
        
        if request_count >= config.max_requests:
            # Block IP for window duration if excessive requests
            if request_count >= config.max_requests * 2:
                self.blocked_ips[identifier] = datetime.utcnow() + timedelta(seconds=config.window_seconds * 2)
            
            return False, {
                "rate_limited": True,
                "requests_in_window": request_count,
                "max_requests": config.max_requests,
                "window_seconds": config.window_seconds,
                "retry_after": config.window_seconds
            }
        
        # Record this request
        self.request_logs[identifier].append(current_time)
        
        return True, {
            "allowed": True,
            "requests_in_window": request_count + 1,
            "max_requests": config.max_requests,
            "remaining_requests": config.max_requests - request_count - 1
        }
    
    def update_config(self, endpoint: str, config: RateLimitConfig) -> None:
        """Update rate limit configuration for endpoint."""
        self.configs[endpoint] = config
        logger.info(f"Updated rate limit config for {endpoint}: {config.max_requests}/{config.window_seconds}s")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        current_time = time.time()
        active_requests = 0
        
        for identifier, requests in self.request_logs.items():
            # Count requests in last 5 minutes
            recent_requests = [r for r in requests if current_time - r < 300]
            active_requests += len(recent_requests)
        
        return {
            "tracked_identifiers": len(self.request_logs),
            "blocked_ips": len(self.blocked_ips),
            "active_requests_5min": active_requests,
            "endpoints": list(self.configs.keys()),
            "blocked_list": [
                {
                    "ip": ip,
                    "blocked_until": blocked_until.isoformat(),
                    "remaining_seconds": max(0, int((blocked_until - datetime.utcnow()).total_seconds()))
                }
                for ip, blocked_until in self.blocked_ips.items()
            ]
        }


class SecurityEventLogger:
    """Security event logging and analysis."""
    
    def __init__(self, max_events: int = 10000):
        """Initialize security event logger."""
        self.events: List[SecurityEvent] = []
        self.max_events = max_events
        self.ip_event_counts: Dict[str, int] = {}
        logger.info("Security event logger initialized")
    
    def log_event(self, event: SecurityEvent) -> None:
        """Log a security event."""
        self.events.append(event)
        
        # Track IP activity
        self.ip_event_counts[event.source_ip] = self.ip_event_counts.get(event.source_ip, 0) + 1
        
        # Trim old events
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
        
        # Log to system logger
        logger.warning(f"Security event: {event.to_dict()}")
        
        # Check for attack patterns
        self._analyze_attack_patterns(event)
    
    def _analyze_attack_patterns(self, event: SecurityEvent) -> None:
        """Analyze for potential attack patterns."""
        recent_cutoff = datetime.utcnow() - timedelta(minutes=10)
        recent_events = [e for e in self.events if e.timestamp >= recent_cutoff and e.source_ip == event.source_ip]
        
        # Brute force detection
        if len(recent_events) >= 10:
            logger.critical(f"Potential brute force attack from {event.source_ip}: {len(recent_events)} events in 10 minutes")
        
        # Multiple attack types from same IP
        event_types = set(e.event_type for e in recent_events)
        if len(event_types) >= 3:
            logger.critical(f"Multiple attack types from {event.source_ip}: {[t.value for t in event_types]}")
    
    def get_threat_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get threat summary for specified time period."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent_events = [e for e in self.events if e.timestamp >= cutoff]
        
        if not recent_events:
            return {"period_hours": hours, "total_events": 0}
        
        # Count by event type
        event_type_counts = {}
        for event in recent_events:
            event_type_counts[event.event_type.value] = event_type_counts.get(event.event_type.value, 0) + 1
        
        # Count by severity level
        level_counts = {}
        for event in recent_events:
            level_counts[event.level.value] = level_counts.get(event.level.value, 0) + 1
        
        # Top attacking IPs
        ip_counts = {}
        for event in recent_events:
            ip_counts[event.source_ip] = ip_counts.get(event.source_ip, 0) + 1
        
        top_ips = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "period_hours": hours,
            "total_events": len(recent_events),
            "event_types": event_type_counts,
            "severity_levels": level_counts,
            "top_attacking_ips": [{"ip": ip, "count": count} for ip, count in top_ips],
            "blocked_events": sum(1 for e in recent_events if e.blocked),
            "critical_events": sum(1 for e in recent_events if e.level == SecurityLevel.CRITICAL)
        }


class APIKeyManager:
    """Secure API key management with scoping and rate limits."""
    
    def __init__(self):
        """Initialize API key manager."""
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self.key_usage: Dict[str, List[float]] = {}
        logger.info("API key manager initialized")
    
    def generate_api_key(self, 
                        user_id: str, 
                        scopes: List[str] = None,
                        rate_limit: RateLimitConfig = None) -> str:
        """Generate a new API key with specified permissions."""
        # Generate secure random key
        key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        self.api_keys[key_hash] = {
            "user_id": user_id,
            "scopes": scopes or ["evaluation", "tasks"],
            "rate_limit": rate_limit or RateLimitConfig(max_requests=1000, window_seconds=3600),
            "created": datetime.utcnow(),
            "last_used": None,
            "usage_count": 0,
            "active": True
        }
        
        logger.info(f"Generated API key for user {user_id} with scopes: {scopes}")
        return key
    
    def validate_api_key(self, key: str, required_scope: str = None) -> Tuple[bool, Dict[str, Any]]:
        """Validate API key and check permissions."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        if key_hash not in self.api_keys:
            return False, {"error": "Invalid API key"}
        
        key_info = self.api_keys[key_hash]
        
        if not key_info["active"]:
            return False, {"error": "API key is disabled"}
        
        if required_scope and required_scope not in key_info["scopes"]:
            return False, {"error": f"API key lacks required scope: {required_scope}"}
        
        # Update usage
        key_info["last_used"] = datetime.utcnow()
        key_info["usage_count"] += 1
        
        # Track usage for rate limiting
        current_time = time.time()
        if key_hash not in self.key_usage:
            self.key_usage[key_hash] = []
        
        self.key_usage[key_hash].append(current_time)
        
        return True, {
            "user_id": key_info["user_id"],
            "scopes": key_info["scopes"],
            "usage_count": key_info["usage_count"]
        }
    
    def revoke_api_key(self, key: str) -> bool:
        """Revoke an API key."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        if key_hash in self.api_keys:
            self.api_keys[key_hash]["active"] = False
            logger.info(f"Revoked API key for user {self.api_keys[key_hash]['user_id']}")
            return True
        
        return False


class SecurityManager:
    """Central security manager coordinating all security components."""
    
    def __init__(self):
        """Initialize security manager."""
        self.validator = SecurityValidator()
        self.rate_limiter = RateLimiter()
        self.event_logger = SecurityEventLogger()
        self.api_key_manager = APIKeyManager()
        self.trusted_ips: Set[str] = set()
        logger.info("Security manager initialized")
    
    def validate_request(self, 
                        source_ip: str,
                        user_agent: str,
                        endpoint: str,
                        data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Comprehensive request validation."""
        violations = []
        security_events = []
        
        # Rate limiting check
        rate_allowed, rate_info = self.rate_limiter.check_rate_limit(source_ip, endpoint)
        if not rate_allowed:
            event = SecurityEvent(
                event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
                level=SecurityLevel.MEDIUM,
                source_ip=source_ip,
                user_agent=user_agent,
                details=rate_info,
                blocked=True
            )
            self.event_logger.log_event(event)
            return False, {"error": "Rate limit exceeded", "details": rate_info}
        
        # Input validation
        for field_name, field_value in data.items():
            if isinstance(field_value, str):
                is_valid, field_violations = self.validator.validate_input(field_value, field_name)
                if not is_valid:
                    violations.extend(field_violations)
                    
                    # Log security event
                    event = SecurityEvent(
                        event_type=SecurityEventType.SUSPICIOUS_INPUT,
                        level=SecurityLevel.HIGH,
                        source_ip=source_ip,
                        user_agent=user_agent,
                        details={
                            "field": field_name,
                            "violations": field_violations,
                            "value_preview": field_value[:100]
                        },
                        blocked=True
                    )
                    self.event_logger.log_event(event)
        
        if violations:
            return False, {
                "error": "Input validation failed", 
                "violations": violations[:5]  # Limit to first 5 violations
            }
        
        return True, {"allowed": True, "rate_info": rate_info}
    
    def add_trusted_ip(self, ip: str) -> None:
        """Add IP to trusted list (bypasses some security checks)."""
        self.trusted_ips.add(ip)
        logger.info(f"Added trusted IP: {ip}")
    
    def is_trusted_ip(self, ip: str) -> bool:
        """Check if IP is in trusted list."""
        return ip in self.trusted_ips
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get overall security status and statistics."""
        threat_summary = self.event_logger.get_threat_summary(24)
        rate_limit_stats = self.rate_limiter.get_stats()
        
        return {
            "status": "operational",
            "timestamp": datetime.utcnow().isoformat(),
            "threat_summary_24h": threat_summary,
            "rate_limiting": rate_limit_stats,
            "trusted_ips": len(self.trusted_ips),
            "api_keys_active": sum(1 for key in self.api_key_manager.api_keys.values() if key["active"]),
            "security_events_total": len(self.event_logger.events)
        }


# Global security manager instance
security_manager = SecurityManager()