#!/usr/bin/env python3
"""Simple test runner to verify core functionality."""

import asyncio
import sys
import os
sys.path.insert(0, '/root/repo')

from causal_eval.core.engine import EvaluationEngine, CausalEvaluationRequest
from causal_eval.core.tasks import TaskConfig
from causal_eval.tasks.attribution import CausalAttribution


async def test_core_functionality():
    """Test the core evaluation engine."""
    print("🧪 Testing Core Causal Evaluation Engine...")
    
    # Test 1: Initialize engine
    print("\n1. Initializing Evaluation Engine...")
    engine = EvaluationEngine()
    assert engine is not None
    print("✅ Engine initialized successfully")
    
    # Test 2: Test task types
    print("\n2. Testing available task types...")
    task_types = engine.get_available_task_types()
    print(f"Available task types: {task_types}")
    assert "attribution" in task_types
    print("✅ Task types available")
    
    # Test 3: Generate prompt
    print("\n3. Testing prompt generation...")
    prompt = await engine.generate_task_prompt("attribution", "medical", "medium")
    print(f"Generated prompt preview: {prompt[:200]}...")
    assert len(prompt) > 100
    print("✅ Prompt generation successful")
    
    # Test 4: Evaluate sample response
    print("\n4. Testing response evaluation...")
    sample_response = """
    The relationship between regular exercise and cardiovascular disease risk is causal.
    Exercise directly improves heart health through multiple mechanisms including strengthening the heart muscle,
    improving circulation, and reducing blood pressure. Confounding factors might include diet and genetics.
    Confidence: 0.8
    """
    
    request = CausalEvaluationRequest(
        task_type="attribution",
        model_response=sample_response,
        domain="medical",
        difficulty="medium"
    )
    
    result = await engine.evaluate_request(request)
    print(f"Evaluation result: {result}")
    assert "overall_score" in result
    print("✅ Response evaluation successful")
    
    # Test 5: Direct task instantiation
    print("\n5. Testing direct task instantiation...")
    config = TaskConfig(
        task_id="test_attribution",
        domain="medical",
        difficulty="medium",
        description="Test causal attribution",
        expected_reasoning_type="attribution"
    )
    
    task = CausalAttribution(config)
    task_prompt = await task.generate_prompt()
    task_result = await task.evaluate_response(sample_response)
    
    print(f"Task evaluation: {task_result}")
    assert "overall_score" in task_result
    print("✅ Direct task instantiation successful")
    
    print("\n🎉 All core functionality tests passed!")
    return True


if __name__ == "__main__":
    try:
        result = asyncio.run(test_core_functionality())
        print("✅ GENERATION 1 CORE FUNCTIONALITY VERIFIED")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)