"""Custom assertion helpers for Causal Eval Bench tests."""

import json
import time
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse


def assert_valid_question_format(question: Dict[str, Any]) -> None:
    """Assert that a question follows the expected format."""
    required_fields = ['id', 'prompt', 'ground_truth', 'task_type', 'domain', 'difficulty']
    
    for field in required_fields:
        assert field in question, f"Question missing required field: {field}"
    
    # Validate field types
    assert isinstance(question['id'], str), "Question ID must be string"
    assert isinstance(question['prompt'], str), "Question prompt must be string"
    assert isinstance(question['ground_truth'], dict), "Ground truth must be dict"
    assert isinstance(question['task_type'], str), "Task type must be string"
    assert isinstance(question['domain'], str), "Domain must be string"
    assert isinstance(question['difficulty'], str), "Difficulty must be string"
    
    # Validate enums
    valid_task_types = ['causal_attribution', 'counterfactual', 'intervention', 'causal_chain', 'confounding']
    assert question['task_type'] in valid_task_types, f"Invalid task type: {question['task_type']}"
    
    valid_difficulties = ['easy', 'medium', 'hard', 'expert']
    assert question['difficulty'] in valid_difficulties, f"Invalid difficulty: {question['difficulty']}"
    
    # Validate ground truth content
    ground_truth = question['ground_truth']
    assert 'answer' in ground_truth, "Ground truth missing 'answer' field"
    assert 'explanation' in ground_truth, "Ground truth missing 'explanation' field"


def assert_valid_evaluation_result(result: Dict[str, Any]) -> None:
    """Assert that an evaluation result follows the expected format."""
    required_fields = ['overall_score', 'task_scores', 'questions_answered']
    
    for field in required_fields:
        assert field in result, f"Evaluation result missing required field: {field}"
    
    # Validate score ranges
    assert 0 <= result['overall_score'] <= 1, "Overall score must be between 0 and 1"
    
    if 'task_scores' in result:
        for task, score in result['task_scores'].items():
            assert 0 <= score <= 1, f"Task score for {task} must be between 0 and 1"
    
    # Validate question count
    assert isinstance(result['questions_answered'], int), "Questions answered must be integer"
    assert result['questions_answered'] >= 0, "Questions answered must be non-negative"


def assert_valid_response_format(response: Dict[str, Any]) -> None:
    """Assert that a model response follows the expected format."""
    required_fields = ['question_id', 'response_text']
    
    for field in required_fields:
        assert field in response, f"Response missing required field: {field}"
    
    assert isinstance(response['question_id'], str), "Question ID must be string"
    assert isinstance(response['response_text'], str), "Response text must be string"
    
    if 'confidence' in response:
        assert 0 <= response['confidence'] <= 1, "Confidence must be between 0 and 1"
    
    if 'response_time' in response:
        assert response['response_time'] >= 0, "Response time must be non-negative"


def assert_api_response_structure(response_data: Dict[str, Any], expected_keys: List[str]) -> None:
    """Assert that an API response contains expected keys."""
    for key in expected_keys:
        assert key in response_data, f"API response missing key: {key}"


def assert_performance_metrics(metrics: Dict[str, float], thresholds: Dict[str, float]) -> None:
    """Assert that performance metrics meet specified thresholds."""
    for metric, threshold in thresholds.items():
        assert metric in metrics, f"Performance metrics missing: {metric}"
        
        if metric.endswith('_time') or metric.endswith('_latency'):
            # For time-based metrics, assert below threshold
            assert metrics[metric] <= threshold, f"{metric} ({metrics[metric]}) exceeds threshold ({threshold})"
        elif metric.endswith('_rate') or metric.endswith('_score'):
            # For rate/score metrics, assert above threshold
            assert metrics[metric] >= threshold, f"{metric} ({metrics[metric]}) below threshold ({threshold})"
        else:
            # For other metrics, assume above threshold is better
            assert metrics[metric] >= threshold, f"{metric} ({metrics[metric]}) below threshold ({threshold})"


def assert_causal_reasoning_quality(response: str, expected_elements: List[str]) -> None:
    """Assert that a causal reasoning response contains expected elements."""
    response_lower = response.lower()
    
    for element in expected_elements:
        assert element.lower() in response_lower, f"Response missing expected element: {element}"


def assert_evaluation_consistency(evaluations: List[Dict[str, Any]], tolerance: float = 0.1) -> None:
    """Assert that repeated evaluations are consistent within tolerance."""
    if len(evaluations) < 2:
        return
    
    scores = [eval_result['overall_score'] for eval_result in evaluations]
    mean_score = sum(scores) / len(scores)
    
    for score in scores:
        deviation = abs(score - mean_score)
        assert deviation <= tolerance, f"Evaluation inconsistency: {deviation} > {tolerance}"


def assert_valid_url(url: str, require_https: bool = False) -> None:
    """Assert that a URL is valid and optionally requires HTTPS."""
    parsed = urlparse(url)
    
    assert parsed.scheme, "URL must have a scheme"
    assert parsed.netloc, "URL must have a network location"
    
    if require_https:
        assert parsed.scheme == 'https', "URL must use HTTPS"


def assert_json_serializable(data: Any) -> None:
    """Assert that data is JSON serializable."""
    try:
        json.dumps(data)
    except (TypeError, ValueError) as e:
        assert False, f"Data is not JSON serializable: {e}"


def assert_execution_time(func, max_time: float, *args, **kwargs) -> Any:
    """Assert that function execution completes within specified time."""
    start_time = time.time()
    result = func(*args, **kwargs)
    execution_time = time.time() - start_time
    
    assert execution_time <= max_time, f"Execution time ({execution_time:.3f}s) exceeds limit ({max_time}s)"
    return result


def assert_memory_usage(func, max_memory_mb: float, *args, **kwargs) -> Any:
    """Assert that function execution stays within memory limits."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    result = func(*args, **kwargs)
    
    peak_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = peak_memory - initial_memory
    
    assert memory_increase <= max_memory_mb, f"Memory usage ({memory_increase:.2f} MB) exceeds limit ({max_memory_mb} MB)"
    return result


def assert_no_data_leakage(train_questions: List[Dict], test_questions: List[Dict]) -> None:
    """Assert that there's no data leakage between training and test sets."""
    train_ids = {q['id'] for q in train_questions}
    test_ids = {q['id'] for q in test_questions}
    
    overlap = train_ids.intersection(test_ids)
    assert not overlap, f"Data leakage detected: {len(overlap)} questions appear in both sets"
    
    # Check for near-duplicate prompts
    train_prompts = {q['prompt'] for q in train_questions}
    test_prompts = {q['prompt'] for q in test_questions}
    
    prompt_overlap = train_prompts.intersection(test_prompts)
    assert not prompt_overlap, f"Prompt duplication detected: {len(prompt_overlap)} identical prompts"


def assert_balanced_dataset(questions: List[Dict], field: str, tolerance: float = 0.2) -> None:
    """Assert that a dataset is reasonably balanced across a categorical field."""
    if not questions:
        return
    
    field_counts = {}
    for question in questions:
        value = question.get(field)
        field_counts[value] = field_counts.get(value, 0) + 1
    
    total_count = len(questions)
    expected_count = total_count / len(field_counts)
    
    for value, count in field_counts.items():
        ratio = count / total_count
        expected_ratio = 1.0 / len(field_counts)
        deviation = abs(ratio - expected_ratio)
        
        assert deviation <= tolerance, f"Dataset imbalanced for {field}={value}: {ratio:.2%} vs expected {expected_ratio:.2%}"


def assert_valid_confidence_scores(responses: List[Dict[str, Any]]) -> None:
    """Assert that confidence scores are valid and reasonable."""
    confidences = [r.get('confidence') for r in responses if 'confidence' in r]
    
    if not confidences:
        return
    
    # All confidence scores should be between 0 and 1
    for conf in confidences:
        assert 0 <= conf <= 1, f"Invalid confidence score: {conf}"
    
    # Check for reasonable distribution (not all the same)
    unique_confidences = set(confidences)
    if len(confidences) > 10:
        assert len(unique_confidences) > 1, "Confidence scores show no variation"


def assert_evaluation_reproducibility(eval_func, question: Dict, num_runs: int = 3, tolerance: float = 0.05) -> None:
    """Assert that evaluation function produces consistent results."""
    scores = []
    
    for _ in range(num_runs):
        result = eval_func(question)
        scores.append(result.get('score', 0))
    
    if len(scores) < 2:
        return
    
    score_range = max(scores) - min(scores)
    assert score_range <= tolerance, f"Evaluation not reproducible: score range {score_range} > tolerance {tolerance}"


class AssertionContext:
    """Context manager for grouped assertions with detailed error reporting."""
    
    def __init__(self, description: str):
        self.description = description
        self.errors = []
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.errors:
            error_msg = f"\n{self.description} failed with {len(self.errors)} errors:\n"
            error_msg += "\n".join(f"  - {error}" for error in self.errors)
            raise AssertionError(error_msg)
    
    def check(self, condition: bool, message: str) -> None:
        """Add a conditional assertion."""
        if not condition:
            self.errors.append(message)
    
    def check_equal(self, actual: Any, expected: Any, message: str = None) -> None:
        """Add an equality assertion."""
        if actual != expected:
            msg = message or f"Expected {expected}, got {actual}"
            self.errors.append(msg)
    
    def check_in_range(self, value: float, min_val: float, max_val: float, message: str = None) -> None:
        """Add a range assertion."""
        if not (min_val <= value <= max_val):
            msg = message or f"Value {value} not in range [{min_val}, {max_val}]"
            self.errors.append(msg)