"""
Custom assertion helpers for causal evaluation testing.
"""

from typing import Any, Dict, List, Union, Optional
import json
import pytest


def assert_evaluation_response_valid(response: Dict[str, Any]) -> None:
    """Assert that an evaluation response has the expected structure."""
    required_fields = ["id", "score", "reasoning", "timestamp"]
    
    for field in required_fields:
        assert field in response, f"Missing required field: {field}"
    
    assert isinstance(response["score"], (int, float)), "Score must be numeric"
    assert 0 <= response["score"] <= 1, "Score must be between 0 and 1"
    assert isinstance(response["reasoning"], str), "Reasoning must be a string"
    assert len(response["reasoning"]) > 0, "Reasoning cannot be empty"


def assert_question_structure_valid(question: Dict[str, Any]) -> None:
    """Assert that a question has the expected structure."""
    required_fields = ["id", "text", "context", "options", "domain"]
    
    for field in required_fields:
        assert field in question, f"Missing required field: {field}"
    
    assert isinstance(question["text"], str), "Question text must be a string"
    assert len(question["text"]) > 10, "Question text too short"
    assert isinstance(question["options"], list), "Options must be a list"
    assert len(question["options"]) >= 2, "Must have at least 2 options"


def assert_api_response_format(response: Dict[str, Any], expected_status: str = "success") -> None:
    """Assert that an API response follows the expected format."""
    assert "status" in response, "Response must include status"
    assert "data" in response, "Response must include data"
    assert response["status"] == expected_status, f"Expected status {expected_status}, got {response['status']}"
    
    if response["status"] == "error":
        assert "error" in response, "Error responses must include error details"
        assert "message" in response["error"], "Error must include message"


def assert_performance_metrics_valid(metrics: Dict[str, Any]) -> None:
    """Assert that performance metrics are valid."""
    required_metrics = ["response_time", "throughput", "accuracy"]
    
    for metric in required_metrics:
        assert metric in metrics, f"Missing performance metric: {metric}"
    
    assert metrics["response_time"] > 0, "Response time must be positive"
    assert metrics["throughput"] > 0, "Throughput must be positive"
    assert 0 <= metrics["accuracy"] <= 1, "Accuracy must be between 0 and 1"


def assert_causal_reasoning_valid(reasoning: Dict[str, Any]) -> None:
    """Assert that causal reasoning analysis is valid."""
    required_fields = ["causal_chain", "confounders", "interventions", "confidence"]
    
    for field in required_fields:
        assert field in reasoning, f"Missing reasoning field: {field}"
    
    assert isinstance(reasoning["causal_chain"], list), "Causal chain must be a list"
    assert len(reasoning["causal_chain"]) > 0, "Causal chain cannot be empty"
    assert isinstance(reasoning["confounders"], list), "Confounders must be a list"
    assert 0 <= reasoning["confidence"] <= 1, "Confidence must be between 0 and 1"


def assert_database_record_valid(record: Dict[str, Any], table: str) -> None:
    """Assert that a database record has required fields for the table."""
    common_fields = ["id", "created_at", "updated_at"]
    
    for field in common_fields:
        assert field in record, f"Missing common field: {field}"
    
    # Table-specific validations
    if table == "evaluations":
        eval_fields = ["question_id", "model_id", "response", "score"]
        for field in eval_fields:
            assert field in record, f"Missing evaluation field: {field}"
    
    elif table == "questions":
        question_fields = ["text", "domain", "difficulty", "options"]
        for field in question_fields:
            assert field in record, f"Missing question field: {field}"


def assert_json_schema_valid(data: Any, schema: Dict[str, Any]) -> None:
    """Assert that data matches a JSON schema structure."""
    if "type" in schema:
        expected_type = schema["type"]
        if expected_type == "object":
            assert isinstance(data, dict), f"Expected object, got {type(data)}"
        elif expected_type == "array":
            assert isinstance(data, list), f"Expected array, got {type(data)}"
        elif expected_type == "string":
            assert isinstance(data, str), f"Expected string, got {type(data)}"
        elif expected_type == "number":
            assert isinstance(data, (int, float)), f"Expected number, got {type(data)}"
    
    if "required" in schema and isinstance(data, dict):
        for field in schema["required"]:
            assert field in data, f"Missing required field: {field}"


def assert_file_exists_with_content(file_path: str, min_size: int = 1) -> None:
    """Assert that a file exists and has minimum content."""
    import os
    
    assert os.path.exists(file_path), f"File does not exist: {file_path}"
    
    file_size = os.path.getsize(file_path)
    assert file_size >= min_size, f"File too small: {file_size} bytes, expected at least {min_size}"


def assert_environment_variable_set(var_name: str, required: bool = True) -> None:
    """Assert that an environment variable is set."""
    import os
    
    value = os.getenv(var_name)
    
    if required:
        assert value is not None, f"Required environment variable not set: {var_name}"
        assert len(value.strip()) > 0, f"Environment variable is empty: {var_name}"
    
    return value


def assert_docker_service_healthy(service_name: str, timeout: int = 30) -> None:
    """Assert that a Docker service is healthy."""
    import subprocess
    import time
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            result = subprocess.run(
                ["docker-compose", "ps", "-q", service_name],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and result.stdout.strip():
                # Service is running, check health
                health_result = subprocess.run(
                    ["docker", "inspect", "--format='{{.State.Health.Status}}'", result.stdout.strip()],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if "healthy" in health_result.stdout:
                    return
                    
        except subprocess.TimeoutExpired:
            pass
        
        time.sleep(2)
    
    pytest.fail(f"Docker service {service_name} is not healthy after {timeout} seconds")


class CustomAssertions:
    """Container for custom assertion methods."""
    
    @staticmethod
    def assert_scores_within_range(scores: List[float], min_score: float, max_score: float) -> None:
        """Assert all scores are within expected range."""
        for i, score in enumerate(scores):
            assert min_score <= score <= max_score, \
                f"Score {i} ({score}) not in range [{min_score}, {max_score}]"
    
    @staticmethod
    def assert_model_performance_acceptable(metrics: Dict[str, float], thresholds: Dict[str, float]) -> None:
        """Assert model performance meets minimum thresholds."""
        for metric, threshold in thresholds.items():
            assert metric in metrics, f"Missing performance metric: {metric}"
            assert metrics[metric] >= threshold, \
                f"Performance metric {metric} ({metrics[metric]}) below threshold ({threshold})"
    
    @staticmethod
    def assert_causal_consistency(causal_graph: Dict[str, List[str]]) -> None:
        """Assert causal graph has no cycles (DAG property)."""
        def has_cycle(graph: Dict[str, List[str]]) -> bool:
            visited = set()
            rec_stack = set()
            
            def dfs(node: str) -> bool:
                visited.add(node)
                rec_stack.add(node)
                
                for neighbor in graph.get(node, []):
                    if neighbor not in visited:
                        if dfs(neighbor):
                            return True
                    elif neighbor in rec_stack:
                        return True
                
                rec_stack.remove(node)
                return False
            
            for node in graph:
                if node not in visited:
                    if dfs(node):
                        return True
            return False
        
        assert not has_cycle(causal_graph), "Causal graph contains cycles (not a DAG)"
    
    @staticmethod
    def assert_statistical_significance(p_value: float, alpha: float = 0.05) -> None:
        """Assert statistical significance."""
        assert 0 <= p_value <= 1, f"Invalid p-value: {p_value}"
        assert p_value < alpha, f"Result not statistically significant (p={p_value}, α={alpha})"


# Convenience functions for common assertions
def assert_valid_evaluation(response: Dict[str, Any]) -> None:
    """Convenience function for evaluation response validation."""
    assert_evaluation_response_valid(response)


def assert_valid_question(question: Dict[str, Any]) -> None:
    """Convenience function for question validation."""
    assert_question_structure_valid(question)


def assert_valid_api_response(response: Dict[str, Any], expected_status: str = "success") -> None:
    """Convenience function for API response validation."""
    assert_api_response_format(response, expected_status)