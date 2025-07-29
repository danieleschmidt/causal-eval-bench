#!/usr/bin/env python3
"""
Performance testing script for causal evaluation framework.
Used by modernization workflows for memory profiling and benchmarking.
"""

import asyncio
import time
from typing import List, Dict, Any
import random
import json


class MockCausalEvaluator:
    """Mock evaluator for performance testing."""
    
    def __init__(self):
        self.cache = {}
        self.evaluation_count = 0
    
    def generate_test_case(self) -> Dict[str, Any]:
        """Generate a mock test case."""
        scenarios = [
            "Ice cream sales and drowning incidents both increase in summer",
            "Students who eat breakfast perform better on tests",
            "Companies with more employees tend to have higher revenue",
            "People who exercise regularly have lower blood pressure"
        ]
        
        return {
            "id": f"test_{self.evaluation_count}",
            "scenario": random.choice(scenarios),
            "task_type": random.choice(["attribution", "counterfactual", "intervention"]),
            "difficulty": random.choice(["easy", "medium", "hard"]),
            "domain": random.choice(["medical", "social", "economic", "scientific"])
        }
    
    def evaluate_response(self, test_case: Dict[str, Any], response: str) -> float:
        """Mock evaluation with realistic processing time."""
        # Simulate some computation time
        time.sleep(random.uniform(0.001, 0.01))
        
        # Cache simulation
        cache_key = f"{test_case['id']}_{hash(response)}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Mock scoring based on response length and content
        score = min(1.0, len(response) / 200 + random.uniform(0.1, 0.3))
        self.cache[cache_key] = score
        self.evaluation_count += 1
        
        return score
    
    async def async_evaluate_batch(self, test_cases: List[Dict[str, Any]]) -> List[float]:
        """Async batch evaluation for performance testing."""
        tasks = []
        for test_case in test_cases:
            response = f"Mock response for {test_case['scenario']}"
            task = asyncio.create_task(self.async_evaluate_single(test_case, response))
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
    
    async def async_evaluate_single(self, test_case: Dict[str, Any], response: str) -> float:
        """Single async evaluation."""
        # Simulate async processing
        await asyncio.sleep(random.uniform(0.001, 0.005))
        return self.evaluate_response(test_case, response)


def memory_intensive_operation():
    """Memory-intensive operation for profiling."""
    # Create large data structures
    data = []
    for i in range(10000):
        record = {
            "id": i,
            "scenario": f"Test scenario {i}" * 10,
            "features": [random.random() for _ in range(100)],
            "metadata": {
                "timestamp": time.time(),
                "version": "1.0.0",
                "tags": [f"tag_{j}" for j in range(10)]
            }
        }
        data.append(record)
    
    # Process data (simulating evaluation pipeline)
    processed = []
    for record in data:
        # Simulate complex processing
        result = {
            "original_id": record["id"],
            "processed_features": [x * 2 for x in record["features"]],
            "aggregated_score": sum(record["features"]) / len(record["features"]),
            "complexity_measure": len(record["scenario"]) * len(record["features"])
        }
        processed.append(result)
    
    return processed


async def async_performance_test():
    """Async performance testing scenario."""
    evaluator = MockCausalEvaluator()
    
    # Generate test cases
    test_cases = [evaluator.generate_test_case() for _ in range(1000)]
    
    print("Starting async performance test...")
    start_time = time.time()
    
    # Batch processing
    batch_size = 50
    results = []
    
    for i in range(0, len(test_cases), batch_size):
        batch = test_cases[i:i + batch_size]
        batch_results = await evaluator.async_evaluate_batch(batch)
        results.extend(batch_results)
        
        if i % 200 == 0:
            print(f"Processed {i + len(batch)} test cases...")
    
    end_time = time.time()
    
    print(f"Async processing completed in {end_time - start_time:.2f} seconds")
    print(f"Average score: {sum(results) / len(results):.3f}")
    print(f"Throughput: {len(results) / (end_time - start_time):.1f} evaluations/second")
    
    return results


def cpu_intensive_benchmark():
    """CPU-intensive benchmark for performance analysis."""
    print("Running CPU-intensive benchmark...")
    
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n - 1) + fibonacci(n - 2)
    
    def prime_check(n):
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    start_time = time.time()
    
    # Calculate fibonacci numbers
    fib_results = [fibonacci(i) for i in range(25)]
    
    # Find prime numbers
    prime_results = [n for n in range(1000, 2000) if prime_check(n)]
    
    end_time = time.time()
    
    print(f"CPU benchmark completed in {end_time - start_time:.2f} seconds")
    print(f"Fibonacci results count: {len(fib_results)}")
    print(f"Prime numbers found: {len(prime_results)}")
    
    return {
        "fibonacci_count": len(fib_results),
        "prime_count": len(prime_results),
        "execution_time": end_time - start_time
    }


def main():
    """Main performance testing function."""
    print("🚀 Starting Performance Testing Suite")
    print("=" * 50)
    
    # Memory-intensive test
    print("\n1. Memory-intensive operations test...")
    memory_start = time.time()
    memory_data = memory_intensive_operation()
    memory_end = time.time()
    print(f"Memory test completed in {memory_end - memory_start:.2f} seconds")
    print(f"Processed {len(memory_data)} records")
    
    # CPU-intensive test
    print("\n2. CPU-intensive benchmark...")
    cpu_results = cpu_intensive_benchmark()
    
    # Async test
    print("\n3. Async performance test...")
    async_results = asyncio.run(async_performance_test())
    
    # Generate performance report
    report = {
        "timestamp": time.time(),
        "memory_test": {
            "records_processed": len(memory_data),
            "execution_time": memory_end - memory_start
        },
        "cpu_test": cpu_results,
        "async_test": {
            "evaluations_completed": len(async_results),
            "average_score": sum(async_results) / len(async_results)
        }
    }
    
    # Save report
    with open("performance_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n✅ Performance testing completed!")
    print("📊 Report saved to performance_report.json")


if __name__ == "__main__":
    main()