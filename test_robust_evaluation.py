#!/usr/bin/env python3
"""Test robust evaluation system with error handling and validation."""

import asyncio
import time
from typing import Dict, Any
from dataclasses import dataclass


# Minimal validation for testing
@dataclass 
class ValidationResult:
    is_valid: bool
    sanitized_input: str
    warnings: list
    errors: list
    security_threats: list


class SimpleValidator:
    """Simplified validator for testing."""
    
    def validate_model_response(self, response: str) -> ValidationResult:
        errors = []
        warnings = []
        threats = []
        
        if not response or not response.strip():
            errors.append("Response cannot be empty")
        
        if len(response) > 10000:
            errors.append("Response too long")
            response = response[:10000]
            warnings.append("Response truncated")
        
        # Check for suspicious content
        if "<script>" in response.lower():
            threats.append("script_injection")
            response = response.replace("<script>", "[SCRIPT_REMOVED]")
            warnings.append("Script injection detected and removed")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_input=response,
            warnings=warnings,
            errors=errors,
            security_threats=threats
        )
    
    def validate_task_config(self, config: Dict[str, Any]) -> ValidationResult:
        errors = []
        warnings = []
        
        if "task_type" not in config:
            errors.append("Missing task_type")
        
        valid_types = ["attribution", "counterfactual", "intervention"]
        if config.get("task_type") not in valid_types:
            errors.append(f"Invalid task_type: {config.get('task_type')}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_input=config,
            warnings=warnings,
            errors=errors,
            security_threats=[]
        )


class RobustEvaluationEngine:
    """Robust evaluation engine with error handling."""
    
    def __init__(self):
        self.validator = SimpleValidator()
        self.error_count = 0
    
    async def evaluate(self, model_response: str, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """Robust evaluation with validation and error handling."""
        start_time = time.time()
        
        try:
            # Input validation
            response_validation = self.validator.validate_model_response(model_response)
            config_validation = self.validator.validate_task_config(task_config)
            
            # Check validation results
            if not response_validation.is_valid:
                raise ValueError(f"Invalid response: {'; '.join(response_validation.errors)}")
            
            if not config_validation.is_valid:
                raise ValueError(f"Invalid config: {'; '.join(config_validation.errors)}")
            
            # Use sanitized inputs
            sanitized_response = response_validation.sanitized_input
            sanitized_config = config_validation.sanitized_input
            
            # Simulate evaluation with timeout protection
            try:
                result = await asyncio.wait_for(
                    self._perform_evaluation(sanitized_response, sanitized_config),
                    timeout=30.0
                )
                
                # Add metadata
                execution_time = time.time() - start_time
                result["metadata"]["execution_time_seconds"] = execution_time
                result["metadata"]["validation_warnings"] = response_validation.warnings + config_validation.warnings
                result["metadata"]["security_threats_detected"] = response_validation.security_threats + config_validation.security_threats
                
                return result
                
            except asyncio.TimeoutError:
                raise ValueError("Evaluation timed out")
        
        except Exception as e:
            # Error handling and recovery
            self.error_count += 1
            execution_time = time.time() - start_time
            
            return {
                "task_id": task_config.get("task_id", "error"),
                "domain": task_config.get("domain", "unknown"),
                "score": 0.0,
                "reasoning_quality": 0.0,
                "explanation": f"Evaluation failed: {str(e)}",
                "metadata": {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "execution_time_seconds": execution_time,
                    "error_count": self.error_count,
                    "recovery_attempted": True
                }
            }
    
    async def _perform_evaluation(self, response: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform the actual evaluation."""
        
        # Simulate task execution
        task_type = config["task_type"]
        
        if task_type == "attribution":
            return await self._evaluate_attribution(response, config)
        else:
            raise ValueError(f"Task type {task_type} not implemented")
    
    async def _evaluate_attribution(self, response: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Simple attribution evaluation."""
        
        # Basic scoring based on response content
        response_lower = response.lower()
        
        score = 0.0
        if "spurious" in response_lower:
            score += 0.5
        if "causal" in response_lower:
            score += 0.3
        if "weather" in response_lower or "temperature" in response_lower:
            score += 0.2
        
        return {
            "task_id": config.get("task_id", "attribution_test"),
            "domain": config.get("domain", "general"),
            "score": min(score, 1.0),
            "reasoning_quality": min(score * 0.8, 1.0),
            "explanation": f"Attribution evaluation completed with score {score:.2f}",
            "metadata": {
                "task_type": config["task_type"],
                "response_length": len(response),
                "evaluation_method": "simple_keyword_matching"
            }
        }


async def test_robust_validation():
    """Test input validation and sanitization."""
    print("=== Testing Robust Validation ===")
    
    engine = RobustEvaluationEngine()
    
    # Test 1: Valid input
    print("\nTest 1: Valid input")
    valid_response = "This relationship is spurious due to weather conditions."
    valid_config = {"task_type": "attribution", "domain": "recreational"}
    
    result1 = await engine.evaluate(valid_response, valid_config)
    print(f"✓ Score: {result1['score']:.3f}")
    print(f"✓ No errors: {result1['metadata'].get('error') is None}")
    
    # Test 2: Invalid task type
    print("\nTest 2: Invalid task type (should be handled gracefully)")
    invalid_config = {"task_type": "invalid_type", "domain": "general"}
    
    result2 = await engine.evaluate(valid_response, invalid_config)
    print(f"✓ Error handled: {'error' in result2['metadata']}")
    print(f"✓ Recovery attempted: {result2['metadata'].get('recovery_attempted', False)}")
    
    # Test 3: Suspicious content
    print("\nTest 3: Suspicious content (should be sanitized)")
    suspicious_response = "This is spurious <script>alert('hack')</script> due to weather."
    
    result3 = await engine.evaluate(suspicious_response, valid_config)
    print(f"✓ Score: {result3['score']:.3f}")
    print(f"✓ Security threats detected: {len(result3['metadata'].get('security_threats_detected', []))}")
    print(f"✓ Warnings: {len(result3['metadata'].get('validation_warnings', []))}")
    
    # Test 4: Empty response
    print("\nTest 4: Empty response (should fail gracefully)")
    
    result4 = await engine.evaluate("", valid_config)
    print(f"✓ Error handled: {'error' in result4['metadata']}")
    print(f"✓ Score is zero: {result4['score'] == 0.0}")
    
    return all([
        result1['score'] > 0,
        'error' in result2['metadata'],
        result3['score'] > 0,
        'error' in result4['metadata']
    ])


async def test_error_recovery():
    """Test error recovery mechanisms."""
    print("\n=== Testing Error Recovery ===")
    
    engine = RobustEvaluationEngine()
    
    # Test batch processing with mixed valid/invalid inputs
    test_cases = [
        {"response": "Spurious relationship due to weather", "config": {"task_type": "attribution"}},
        {"response": "Invalid content", "config": {"task_type": "invalid_task"}},
        {"response": "", "config": {"task_type": "attribution"}},
        {"response": "Causal relationship exists", "config": {"task_type": "attribution"}}
    ]
    
    results = []
    for i, case in enumerate(test_cases):
        print(f"\nProcessing case {i+1}: {case['config']['task_type']}")
        
        result = await engine.evaluate(case["response"], case["config"])
        results.append(result)
        
        if "error" in result["metadata"]:
            print(f"  ✓ Error handled gracefully: {result['metadata']['error'][:50]}...")
        else:
            print(f"  ✓ Success: Score = {result['score']:.3f}")
    
    # Verify recovery statistics
    successful_evaluations = sum(1 for r in results if "error" not in r["metadata"])
    failed_but_recovered = sum(1 for r in results if "error" in r["metadata"] and r["metadata"].get("recovery_attempted"))
    
    print(f"\n✓ Successful evaluations: {successful_evaluations}")
    print(f"✓ Failed but recovered: {failed_but_recovered}")
    print(f"✓ Total processed: {len(results)}")
    
    return len(results) == 4 and failed_but_recovered > 0


async def test_performance_monitoring():
    """Test performance monitoring and execution time tracking."""
    print("\n=== Testing Performance Monitoring ===")
    
    engine = RobustEvaluationEngine()
    
    # Test execution time tracking
    response = "This is a spurious relationship caused by weather conditions."
    config = {"task_type": "attribution", "domain": "recreational"}
    
    start = time.time()
    result = await engine.evaluate(response, config)
    end = time.time()
    
    actual_time = end - start
    reported_time = result["metadata"]["execution_time_seconds"]
    
    print(f"✓ Actual execution time: {actual_time:.3f}s")
    print(f"✓ Reported execution time: {reported_time:.3f}s")
    print(f"✓ Time tracking accurate: {abs(actual_time - reported_time) < 0.1}")
    
    # Test metadata completeness
    expected_metadata = ["execution_time_seconds", "validation_warnings", "security_threats_detected"]
    metadata_complete = all(key in result["metadata"] for key in expected_metadata)
    
    print(f"✓ Metadata complete: {metadata_complete}")
    
    return metadata_complete and result["metadata"]["execution_time_seconds"] > 0


def main():
    """Run all robust evaluation tests."""
    print("=== Causal Evaluation Bench - Robust System Test ===")
    
    async def run_tests():
        test_results = []
        
        # Test input validation
        try:
            result1 = await test_robust_validation()
            test_results.append(("Robust Validation", result1))
            print(f"\n✓ Robust Validation: {'PASS' if result1 else 'FAIL'}")
        except Exception as e:
            test_results.append(("Robust Validation", False))
            print(f"\n✗ Robust Validation: FAIL ({e})")
        
        # Test error recovery
        try:
            result2 = await test_error_recovery()
            test_results.append(("Error Recovery", result2))
            print(f"✓ Error Recovery: {'PASS' if result2 else 'FAIL'}")
        except Exception as e:
            test_results.append(("Error Recovery", False))
            print(f"✗ Error Recovery: FAIL ({e})")
        
        # Test performance monitoring
        try:
            result3 = await test_performance_monitoring()
            test_results.append(("Performance Monitoring", result3))
            print(f"✓ Performance Monitoring: {'PASS' if result3 else 'FAIL'}")
        except Exception as e:
            test_results.append(("Performance Monitoring", False))
            print(f"✗ Performance Monitoring: FAIL ({e})")
        
        # Summary
        passed = sum(1 for _, result in test_results if result)
        total = len(test_results)
        
        print(f"\n=== Test Summary ===")
        print(f"Passed: {passed}/{total}")
        
        if passed == total:
            print("🎉 All robust evaluation tests passed!")
            print("✅ System demonstrates resilient error handling")
            print("✅ Input validation and sanitization working")
            print("✅ Performance monitoring operational")
        else:
            print("❌ Some robust evaluation tests failed")
        
        return passed == total
    
    return asyncio.run(run_tests())


if __name__ == "__main__":
    main()
