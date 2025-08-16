#!/usr/bin/env python3
"""Simple test of basic functionality without external dependencies."""

import sys
import os
import asyncio

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_basic_engine():
    """Test basic evaluation engine functionality."""
    print("Testing causal evaluation engine...")
    
    try:
        # Test basic imports
        from causal_eval.core.tasks import TaskConfig
        print("✓ TaskConfig import successful")
        
        # Test task config creation
        config = TaskConfig(
            task_id="test_attribution",
            domain="general",
            difficulty="medium",
            description="Test attribution task",
            expected_reasoning_type="attribution"
        )
        print("✓ TaskConfig creation successful")
        
        # Test attribution task
        from causal_eval.tasks.attribution import CausalAttribution
        task = CausalAttribution(config)
        print("✓ Attribution task creation successful")
        
        # Test prompt generation
        prompt = await task.generate_prompt()
        print(f"✓ Prompt generation successful (length: {len(prompt)} chars)")
        
        # Test evaluation with sample response
        sample_response = """
        1. Relationship Type: spurious
        2. Confidence Level: 0.8
        3. Reasoning: Both variables are likely caused by a third factor such as warm weather conditions.
        4. Potential Confounders: weather, temperature, season
        """
        
        result = await task.evaluate_response(sample_response)
        print(f"✓ Evaluation successful (score: {result.get('overall_score', 0):.2f})")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_counterfactual_task():
    """Test counterfactual reasoning task."""
    print("\nTesting counterfactual reasoning...")
    
    try:
        from causal_eval.core.tasks import TaskConfig
        from causal_eval.tasks.counterfactual import CounterfactualReasoning
        
        config = TaskConfig(
            task_id="test_counterfactual",
            domain="education",
            difficulty="medium",
            description="Test counterfactual task",
            expected_reasoning_type="counterfactual"
        )
        
        task = CounterfactualReasoning(config)
        prompt = await task.generate_prompt()
        print(f"✓ Counterfactual prompt generation successful (length: {len(prompt)} chars)")
        
        sample_response = """
        1. Predicted Outcome: Lower exam score, likely around 70%
        2. Confidence Level: 0.75
        3. Reasoning: Less study time typically leads to reduced knowledge retention and performance
        4. Causal Chain: Less study time -> reduced knowledge -> lower exam performance
        5. Key Assumptions: Same exam difficulty, same student motivation
        """
        
        result = await task.evaluate_response(sample_response)
        print(f"✓ Counterfactual evaluation successful (score: {result.get('overall_score', 0):.2f})")
        
        return True
        
    except Exception as e:
        print(f"✗ Counterfactual test failed: {str(e)}")
        return False


async def test_intervention_task():
    """Test intervention analysis task."""
    print("\nTesting intervention analysis...")
    
    try:
        from causal_eval.core.tasks import TaskConfig
        from causal_eval.tasks.intervention import CausalIntervention
        
        config = TaskConfig(
            task_id="test_intervention",
            domain="home_automation",
            difficulty="medium",
            description="Test intervention task",
            expected_reasoning_type="intervention"
        )
        
        task = CausalIntervention(config)
        prompt = await task.generate_prompt()
        print(f"✓ Intervention prompt generation successful (length: {len(prompt)} chars)")
        
        sample_response = """
        1. Predicted Effect: decrease
        2. Effect Magnitude: 2-4°C decrease
        3. Time Frame: 15-30 minutes
        4. Confidence Level: 0.9
        5. Reasoning: Thermostat directly controls heating system
        6. Potential Side Effects: Energy savings, comfort change
        7. Key Assumptions: Normal insulation, no external heat sources
        """
        
        result = await task.evaluate_response(sample_response)
        print(f"✓ Intervention evaluation successful (score: {result.get('overall_score', 0):.2f})")
        
        return True
        
    except Exception as e:
        print(f"✗ Intervention test failed: {str(e)}")
        return False


async def main():
    """Run all basic functionality tests."""
    print("=== Causal Evaluation Bench - Basic Functionality Test ===\n")
    
    tests = [
        test_basic_engine,
        test_counterfactual_task,
        test_intervention_task,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if await test():
            passed += 1
        print()
    
    print(f"=== Test Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 All basic functionality tests passed!")
        print("Generation 1 (Simple) implementation is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    asyncio.run(main())