"""Input validation and sanitization for causal evaluation."""

import re
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


class SecurityThreat(Enum):
    """Types of security threats detected in input."""
    INJECTION_ATTEMPT = "injection_attempt"
    EXCESSIVE_LENGTH = "excessive_length"
    SUSPICIOUS_PATTERNS = "suspicious_patterns"
    ENCODED_PAYLOAD = "encoded_payload"
    SCRIPT_INJECTION = "script_injection"


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    sanitized_input: str
    warnings: List[str]
    errors: List[str]
    security_threats: List[SecurityThreat]
    metadata: Dict[str, Any]


class InputValidator:
    """Validates and sanitizes user inputs for security and correctness."""
    
    def __init__(self):
        """Initialize input validator with security patterns."""
        
        # Dangerous patterns to detect
        self.injection_patterns = [
            # SQL injection patterns
            r"(?i)(union\s+select|drop\s+table|delete\s+from|insert\s+into)",
            r"(?i)(or\s+1\s*=\s*1|and\s+1\s*=\s*1)",
            r"(?i)(\'\s*or\s*\'\s*=\s*\'|\'\s*;\s*--)",
            
            # Script injection patterns
            r"(?i)(<script|</script>|javascript:|vbscript:)",
            r"(?i)(eval\s*\(|exec\s*\(|system\s*\()",
            r"(?i)(on\w+\s*=|<iframe|<object|<embed)",
            
            # Command injection patterns
            r"(?i)(;\s*rm\s+|;\s*cat\s+|;\s*ls\s+)",
            r"(?i)(\|\s*nc\s+|\|\s*curl\s+|\|\s*wget\s+)",
            
            # Path traversal
            r"(\.\./|\.\.\\)",
            
            # Template injection
            r"(\{\{.*\}\}|\{%.*%\})",
        ]
        
        # Suspicious pattern detection
        self.suspicious_patterns = [
            r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]",  # Control characters
            r"(base64|hex|url)encode",  # Encoding attempts
            r"[<>\"'&].*[<>\"'&]",  # Potential markup injection
        ]
        
        # Compile patterns for efficiency
        self.compiled_injection = [re.compile(pattern) for pattern in self.injection_patterns]
        self.compiled_suspicious = [re.compile(pattern) for pattern in self.suspicious_patterns]
    
    def validate_model_response(self, response: str, max_length: int = 10000) -> ValidationResult:
        """Validate a model response input."""
        
        if not isinstance(response, str):
            return ValidationResult(
                is_valid=False,
                sanitized_input="",
                warnings=[],
                errors=["Input must be a string"],
                security_threats=[],
                metadata={"input_type": type(response).__name__}
            )
        
        warnings = []
        errors = []
        threats = []
        sanitized = response
        
        # Length validation
        if len(response) > max_length:
            errors.append(f"Response too long: {len(response)} > {max_length} characters")
            threats.append(SecurityThreat.EXCESSIVE_LENGTH)
            sanitized = response[:max_length]
            warnings.append(f"Response truncated to {max_length} characters")
        
        # Empty input check
        if not response.strip():
            errors.append("Response cannot be empty")
        
        # Security validation
        security_result = self._check_security_threats(response)
        threats.extend(security_result.threats)
        sanitized = security_result.sanitized_input
        warnings.extend(security_result.warnings)
        
        # Content validation
        content_warnings = self._validate_content_quality(response)
        warnings.extend(content_warnings)
        
        is_valid = len(errors) == 0 and SecurityThreat.INJECTION_ATTEMPT not in threats
        
        return ValidationResult(
            is_valid=is_valid,
            sanitized_input=sanitized,
            warnings=warnings,
            errors=errors,
            security_threats=threats,
            metadata={
                "original_length": len(response),
                "sanitized_length": len(sanitized),
                "has_html": "<" in response and ">" in response,
                "has_special_chars": bool(re.search(r"[<>\"'&;]", response))
            }
        )
    
    def validate_task_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate task configuration parameters."""
        
        errors = []
        warnings = []
        threats = []
        sanitized_config = config.copy()
        
        # Required fields
        required_fields = ["task_type"]
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Validate task_type
        if "task_type" in config:
            valid_task_types = ["attribution", "counterfactual", "intervention", "chain", "confounding"]
            if config["task_type"] not in valid_task_types:
                errors.append(f"Invalid task_type: {config['task_type']}. Must be one of {valid_task_types}")
        
        # Validate domain
        if "domain" in config:
            valid_domains = [
                "general", "medical", "education", "business", "technology",
                "environmental", "workplace_safety", "urban_planning", 
                "manufacturing", "recreational", "public_safety", "international"
            ]
            if config["domain"] not in valid_domains:
                warnings.append(f"Unknown domain: {config['domain']}. Using 'general'")
                sanitized_config["domain"] = "general"
        
        # Validate difficulty
        if "difficulty" in config:
            valid_difficulties = ["easy", "medium", "hard"]
            if config["difficulty"] not in valid_difficulties:
                warnings.append(f"Invalid difficulty: {config['difficulty']}. Using 'medium'")
                sanitized_config["difficulty"] = "medium"
        
        # Check for injection attempts in string values
        for key, value in config.items():
            if isinstance(value, str):
                security_result = self._check_security_threats(value)
                if security_result.threats:
                    threats.extend(security_result.threats)
                    sanitized_config[key] = security_result.sanitized_input
                    warnings.append(f"Sanitized field: {key}")
        
        is_valid = len(errors) == 0 and SecurityThreat.INJECTION_ATTEMPT not in threats
        
        return ValidationResult(
            is_valid=is_valid,
            sanitized_input=sanitized_config,
            warnings=warnings,
            errors=errors,
            security_threats=threats,
            metadata={"field_count": len(config)}
        )
    
    def _check_security_threats(self, input_text: str) -> ValidationResult:
        """Check for security threats in input text."""
        
        threats = []
        warnings = []
        sanitized = input_text
        
        # Check for injection patterns
        for pattern in self.compiled_injection:
            if pattern.search(input_text):
                threats.append(SecurityThreat.INJECTION_ATTEMPT)
                # Remove dangerous content
                sanitized = pattern.sub("[REMOVED]", sanitized)
                warnings.append("Potential injection attempt detected and removed")
                break
        
        # Check for suspicious patterns
        for pattern in self.compiled_suspicious:
            if pattern.search(input_text):
                threats.append(SecurityThreat.SUSPICIOUS_PATTERNS)
                warnings.append("Suspicious patterns detected in input")
                break
        
        # Check for script injection
        if re.search(r"(?i)<script|javascript:|vbscript:", input_text):
            threats.append(SecurityThreat.SCRIPT_INJECTION)
            sanitized = re.sub(r"(?i)<script.*?</script>", "[SCRIPT_REMOVED]", sanitized)
            sanitized = re.sub(r"(?i)(javascript:|vbscript:)", "[SCRIPT_REMOVED]", sanitized)
            warnings.append("Script injection attempt detected and removed")
        
        # Check for encoded payloads
        if re.search(r"(%[0-9a-fA-F]{2}){3,}", input_text):
            threats.append(SecurityThreat.ENCODED_PAYLOAD)
            warnings.append("Potentially encoded payload detected")
        
        return ValidationResult(
            is_valid=len(threats) == 0,
            sanitized_input=sanitized,
            warnings=warnings,
            errors=[],
            security_threats=threats,
            metadata={}
        )
    
    def _validate_content_quality(self, content: str) -> List[str]:
        """Validate content quality and provide warnings."""
        
        warnings = []
        
        # Check for reasonable content length
        if len(content.strip()) < 10:
            warnings.append("Response appears very short")
        
        # Check for excessive repetition
        words = content.lower().split()
        if len(words) > 20:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            max_repetition = max(word_counts.values())
            if max_repetition > len(words) * 0.3:
                warnings.append("Response contains excessive repetition")
        
        # Check for reasonable structure
        if len(content) > 100 and '.' not in content and '!' not in content and '?' not in content:
            warnings.append("Response lacks sentence structure")
        
        return warnings
    
    def sanitize_for_display(self, text: str) -> str:
        """Sanitize text for safe display in web interfaces."""
        
        # HTML escape
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        text = text.replace('"', "&quot;")
        text = text.replace("'", "&#x27;")
        
        # Remove control characters
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
        
        return text
    
    def validate_batch_request(self, batch_data: List[Dict[str, Any]], max_batch_size: int = 100) -> ValidationResult:
        """Validate a batch evaluation request."""
        
        errors = []
        warnings = []
        threats = []
        sanitized_batch = []
        
        # Check batch size
        if len(batch_data) > max_batch_size:
            errors.append(f"Batch too large: {len(batch_data)} > {max_batch_size} items")
            threats.append(SecurityThreat.EXCESSIVE_LENGTH)
        
        # Validate each item in batch
        for i, item in enumerate(batch_data[:max_batch_size]):
            if not isinstance(item, dict):
                errors.append(f"Batch item {i} must be a dictionary")
                continue
            
            # Validate model response if present
            if "model_response" in item:
                response_validation = self.validate_model_response(item["model_response"])
                if not response_validation.is_valid:
                    errors.extend([f"Item {i}: {error}" for error in response_validation.errors])
                threats.extend(response_validation.security_threats)
                warnings.extend([f"Item {i}: {warning}" for warning in response_validation.warnings])
                
                item["model_response"] = response_validation.sanitized_input
            
            # Validate task config if present
            if "task_config" in item:
                config_validation = self.validate_task_config(item["task_config"])
                if not config_validation.is_valid:
                    errors.extend([f"Item {i} config: {error}" for error in config_validation.errors])
                threats.extend(config_validation.security_threats)
                warnings.extend([f"Item {i} config: {warning}" for warning in config_validation.warnings])
                
                item["task_config"] = config_validation.sanitized_input
            
            sanitized_batch.append(item)
        
        is_valid = len(errors) == 0 and SecurityThreat.INJECTION_ATTEMPT not in threats
        
        return ValidationResult(
            is_valid=is_valid,
            sanitized_input=sanitized_batch,
            warnings=warnings,
            errors=errors,
            security_threats=threats,
            metadata={
                "original_batch_size": len(batch_data),
                "processed_batch_size": len(sanitized_batch)
            }
        )


# Global validator instance
validator = InputValidator()


def validate_input(input_data: Any, input_type: str = "model_response") -> ValidationResult:
    """Convenience function for input validation."""
    
    if input_type == "model_response":
        return validator.validate_model_response(input_data)
    elif input_type == "task_config":
        return validator.validate_task_config(input_data)
    elif input_type == "batch_request":
        return validator.validate_batch_request(input_data)
    else:
        raise ValueError(f"Unknown input type: {input_type}")


def require_valid_input(validation_result: ValidationResult) -> None:
    """Raise ValidationError if input is not valid."""
    
    if not validation_result.is_valid:
        error_msg = "; ".join(validation_result.errors)
        if validation_result.security_threats:
            threat_names = [threat.value for threat in validation_result.security_threats]
            error_msg += f" (Security threats: {', '.join(threat_names)})"
        
        raise ValidationError(error_msg)