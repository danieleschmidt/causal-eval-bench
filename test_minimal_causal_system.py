#!/usr/bin/env python3
"""Minimal test of causal evaluation system without external dependencies."""

import asyncio
import json
import re
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class TaskConfig:
    """Simple task configuration."""
    task_id: str
    domain: str
    difficulty: str
    description: str
    expected_reasoning_type: str


@dataclass 
class CausalScenario:
    """A causal scenario with variables and relationships."""
    context: str
    variable_a: str
    variable_b: str
    actual_relationship: str  # "causal", "correlation", "spurious", "reverse_causal"
    confounders: List[str]
    domain: str = "general"
    explanation: str = ""


class SimpleCausalAttribution:
    """Minimal causal attribution task without external dependencies."""
    
    def __init__(self, config: TaskConfig):
        self.config = config
        self.scenarios = [
            CausalScenario(
                context="During summer months, both ice cream sales and swimming pool accidents increase.",
                variable_a="ice cream sales",
                variable_b="swimming pool accidents", 
                actual_relationship="spurious",
                confounders=["summer weather", "outdoor activity"],
                domain="recreational",
                explanation="Both variables are caused by a third factor (warm weather) but don't cause each other."
            ),
            CausalScenario(
                context="Students who study more hours tend to get better grades in their courses.",
                variable_a="study hours",
                variable_b="course grades",
                actual_relationship="causal",
                confounders=["student motivation", "prior knowledge"],
                domain="education",
                explanation="Study time directly impacts learning and comprehension, leading to better grades."
            )
        ]
    
    async def generate_prompt(self) -> str:
        """Generate a causal attribution prompt."""
        scenario = self.scenarios[0]  # Use first scenario for simplicity
        
        prompt = f"""
Analyze the following scenario and determine the nature of the relationship between the two variables.

Scenario: {scenario.context}

Variables:
- Variable A: {scenario.variable_a}
- Variable B: {scenario.variable_b}

Question: What is the relationship between Variable A and Variable B?

Please provide your analysis in the following format:
1. Relationship Type: [Choose: causal, correlation, spurious, reverse_causal]
2. Confidence Level: [0.0 to 1.0]
3. Reasoning: [Explain your analysis]
4. Potential Confounders: [List any confounding variables you identify]

Definitions:
- causal: A directly causes B
- correlation: A and B are correlated but no direct causation
- spurious: A and B appear related but are caused by a third factor
- reverse_causal: B causes A (reverse of what might be expected)
"""
        
        self._current_scenario = scenario
        return prompt.strip()
    
    async def evaluate_response(self, response: str) -> Dict[str, Any]:
        """Evaluate the model's causal attribution response."""
        scenario = self._current_scenario
        
        # Simple parsing
        response_lower = response.lower()
        
        if "spurious" in response_lower:
            relationship_type = "spurious"
        elif "causal" in response_lower and "reverse" not in response_lower:
            relationship_type = "causal"
        elif "reverse" in response_lower:
            relationship_type = "reverse_causal"
        elif "correlation" in response_lower:
            relationship_type = "correlation"
        else:
            relationship_type = "unknown"
        
        # Score relationship accuracy
        relationship_score = 1.0 if relationship_type == scenario.actual_relationship else 0.0
        
        # Simple reasoning quality check
        reasoning_score = 0.5
        if "weather" in response_lower or "temperature" in response_lower:
            reasoning_score = 0.8
        
        overall_score = (relationship_score * 0.7) + (reasoning_score * 0.3)
        
        return {
            "overall_score": overall_score,
            "relationship_score": relationship_score,
            "reasoning_score": reasoning_score,
            "expected_relationship": scenario.actual_relationship,
            "predicted_relationship": relationship_type,
            "scenario_domain": scenario.domain,
            "model_reasoning": response[:200] + "..." if len(response) > 200 else response
        }


class SimpleEvaluationEngine:
    """Minimal evaluation engine."""
    
    def __init__(self):
        self._tasks = {}
    
    async def evaluate(self, model_response: str, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a model's response on a causal reasoning task."""
        task_type = task_config.get("task_type", "attribution")
        domain = task_config.get("domain", "general")
        difficulty = task_config.get("difficulty", "medium")
        
        # Create task config
        config = TaskConfig(
            task_id=f"{task_type}_{domain}_{difficulty}",
            domain=domain,
            difficulty=difficulty,
            description=f"Causal {task_type} task",
            expected_reasoning_type=task_type
        )
        
        # Create task
        if task_type == "attribution":
            task = SimpleCausalAttribution(config)
        else:
            raise ValueError(f"Task type {task_type} not implemented")
        
        # Generate prompt and evaluate
        prompt = await task.generate_prompt()
        result = await task.evaluate_response(model_response)
        
        # Add metadata
        result.update({
            "task_id": config.task_id,
            "task_type": task_type,
            "domain": domain,
            "difficulty": difficulty,
            "generated_prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt
        })
        
        return result


async def test_causal_attribution():
    """Test causal attribution functionality."""
    print("=== Testing Causal Attribution ===")
    
    engine = SimpleEvaluationEngine()
    
    # Test case 1: Correct spurious relationship identification
    print("\nTest 1: Ice cream and swimming accidents (spurious relationship)")
    
    task_config = {
        "task_type": "attribution",
        "domain": "recreational",
        "difficulty": "medium"
    }
    
    # Simulate a good model response
    model_response = """
1. Relationship Type: spurious
2. Confidence Level: 0.9
3. Reasoning: Both ice cream sales and swimming accidents increase during summer due to warm weather. The weather is the common cause - people buy more ice cream when it's hot, and they also use pools more often, leading to more accidents. There's no direct causal link between eating ice cream and having pool accidents.
4. Potential Confounders: summer weather, outdoor temperature, seasonal activities
"""
    
    result = await engine.evaluate(model_response, task_config)
    print(f"✓ Overall Score: {result['overall_score']:.3f}")
    print(f"✓ Relationship Score: {result['relationship_score']:.3f}")
    print(f"✓ Expected: {result['expected_relationship']}")
    print(f"✓ Predicted: {result['predicted_relationship']}")
    
    # Test case 2: Incorrect relationship identification
    print("\nTest 2: Same scenario but with incorrect response")
    
    bad_response = """
1. Relationship Type: causal
2. Confidence Level: 0.7
3. Reasoning: Eating ice cream makes people more likely to have swimming accidents because sugar makes them hyperactive.
4. Potential Confounders: sugar content
"""
    
    result2 = await engine.evaluate(bad_response, task_config)
    print(f"✓ Overall Score: {result2['overall_score']:.3f}")
    print(f"✓ Relationship Score: {result2['relationship_score']:.3f}")
    print(f"✓ Expected: {result2['expected_relationship']}")
    print(f"✓ Predicted: {result2['predicted_relationship']}")
    
    return result['overall_score'] > 0.5 and result2['overall_score'] < 0.5


async def test_api_simulation():
    """Test API-like functionality."""
    print("\n=== Testing API Simulation ===")
    
    engine = SimpleEvaluationEngine()
    
    # Simulate multiple evaluations
    evaluations = [
        {
            "model_response": "Relationship Type: spurious\nReasoning: Weather is the common cause.",
            "task_config": {"task_type": "attribution", "domain": "recreational"}
        },
        {
            "model_response": "Relationship Type: causal\nReasoning: Direct causation exists.",
            "task_config": {"task_type": "attribution", "domain": "recreational"}
        }
    ]
    
    results = []
    for i, evaluation in enumerate(evaluations):
        print(f"\nEvaluation {i+1}:")
        result = await engine.evaluate(
            evaluation["model_response"],
            evaluation["task_config"]
        )
        results.append(result)
        print(f"  Score: {result['overall_score']:.3f}")
        print(f"  Relationship: {result['predicted_relationship']}")
    
    # Calculate aggregate statistics
    avg_score = sum(r['overall_score'] for r in results) / len(results)
    print(f"\n✓ Average Score: {avg_score:.3f}")
    print(f"✓ Total Evaluations: {len(results)}")
    
    return len(results) == 2


def main():
    """Run all tests."""
    print("=== Causal Evaluation Bench - Minimal System Test ===\n")
    
    async def run_tests():
        test_results = []
        
        # Test causal attribution
        try:
            result1 = await test_causal_attribution()
            test_results.append(("Causal Attribution", result1))
            print(f"✓ Causal Attribution: {'PASS' if result1 else 'FAIL'}")
        except Exception as e:
            test_results.append(("Causal Attribution", False))
            print(f"✗ Causal Attribution: FAIL ({e})")
        
        # Test API simulation
        try:
            result2 = await test_api_simulation()
            test_results.append(("API Simulation", result2))
            print(f"✓ API Simulation: {'PASS' if result2 else 'FAIL'}")
        except Exception as e:
            test_results.append(("API Simulation", False))
            print(f"✗ API Simulation: FAIL ({e})")
        
        # Summary
        passed = sum(1 for _, result in test_results if result)
        total = len(test_results)
        
        print(f"\n=== Test Summary ===")
        print(f"Passed: {passed}/{total}")
        
        if passed == total:
            print("🎉 All tests passed! Core causal evaluation system is working.")
        else:
            print("❌ Some tests failed. Check implementation.")
        
        return passed == total
    
    return asyncio.run(run_tests())


if __name__ == "__main__":
    main()