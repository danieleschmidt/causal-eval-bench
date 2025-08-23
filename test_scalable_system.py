#!/usr/bin/env python3
"""Test scalable causal evaluation system with performance optimizations."""

import asyncio
import time
import random
from typing import Dict, Any, List
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    operation_name: str
    execution_time: float
    throughput_ops_per_sec: float
    cache_hit_rate: float = 0.0
    memory_usage_mb: float = 0.0


class SimpleCache:
    """Basic cache for testing."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}
        self.hits = 0
        self.misses = 0
        self.access_order = []
    
    def get(self, key: str) -> Any:
        if key in self.cache:
            self.hits += 1
            # Move to end for LRU
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def set(self, key: str, value: Any):
        # Evict if at capacity
        while len(self.cache) >= self.max_size and self.access_order:
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
        
        # Remove existing if present
        if key in self.cache:
            self.access_order.remove(key)
        
        self.cache[key] = value
        self.access_order.append(key)
    
    def get_stats(self) -> Dict[str, Any]:
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "entries": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }


class ConcurrencyManager:
    """Manages concurrent operations."""
    
    def __init__(self, max_workers: int = 8):
        self.max_workers = max_workers
        self.semaphore = asyncio.Semaphore(max_workers)
        self.active_operations = 0
    
    async def execute_concurrent(self, operations: List[Any]) -> List[Any]:
        """Execute operations concurrently."""
        
        async def execute_with_semaphore(operation):
            async with self.semaphore:
                self.active_operations += 1
                try:
                    if asyncio.iscoroutinefunction(operation):
                        return await operation()
                    else:
                        return operation()
                finally:
                    self.active_operations -= 1
        
        tasks = [execute_with_semaphore(op) for op in operations]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "max_workers": self.max_workers,
            "active_operations": self.active_operations
        }


class ScalableEvaluationEngine:
    """Scalable evaluation engine with performance optimizations."""
    
    def __init__(self):
        self.cache = SimpleCache(max_size=500)
        self.concurrency_manager = ConcurrencyManager(max_workers=8)
        self.metrics_history = []
        self.operation_counts = defaultdict(int)
    
    def _generate_cache_key(self, task_type: str, response: str, domain: str) -> str:
        """Generate cache key."""
        return f"{task_type}:{domain}:{hash(response) % 100000}"
    
    async def evaluate_cached(self, task_type: str, model_response: str, 
                             domain: str = "general", difficulty: str = "medium") -> Dict[str, Any]:
        """Cached evaluation with performance tracking."""
        start_time = time.time()
        
        # Check cache first
        cache_key = self._generate_cache_key(task_type, model_response, domain)
        cached_result = self.cache.get(cache_key)
        
        if cached_result is not None:
            execution_time = time.time() - start_time
            self._record_metrics("cached_evaluation", execution_time, 
                               cache_hit=True, operation_count=1)
            return cached_result
        
        # Perform evaluation
        try:
            result = await self._perform_evaluation(task_type, model_response, domain, difficulty)
            
            # Cache the result
            self.cache.set(cache_key, result)
            
            execution_time = time.time() - start_time
            self._record_metrics("evaluation", execution_time, 
                               cache_hit=False, operation_count=1)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._record_metrics("failed_evaluation", execution_time, 
                               cache_hit=False, operation_count=1)
            
            # Return error result
            return {
                "task_id": f"{task_type}_{domain}",
                "domain": domain,
                "score": 0.0,
                "reasoning_quality": 0.0,
                "explanation": f"Evaluation failed: {str(e)}",
                "metadata": {"error": str(e), "cached": False}
            }
    
    async def _perform_evaluation(self, task_type: str, response: str, 
                                 domain: str, difficulty: str) -> Dict[str, Any]:
        """Simulate evaluation with varying complexity."""
        
        # Simulate different evaluation times based on difficulty
        if difficulty == "easy":
            await asyncio.sleep(0.01 + random.uniform(0, 0.02))
        elif difficulty == "medium":
            await asyncio.sleep(0.02 + random.uniform(0, 0.05))
        else:  # hard
            await asyncio.sleep(0.05 + random.uniform(0, 0.1))
        
        # Simple scoring based on response content
        response_lower = response.lower()
        
        score = 0.0
        if task_type == "attribution":
            if "spurious" in response_lower:
                score += 0.5
            if "causal" in response_lower:
                score += 0.3
            if "weather" in response_lower:
                score += 0.2
        elif task_type == "counterfactual":
            if "would" in response_lower or "if" in response_lower:
                score += 0.4
            if "because" in response_lower:
                score += 0.3
            if "likely" in response_lower:
                score += 0.3
        
        return {
            "task_id": f"{task_type}_{domain}_{difficulty}",
            "domain": domain,
            "score": min(score, 1.0),
            "reasoning_quality": min(score * 0.8, 1.0),
            "explanation": f"{task_type} evaluation completed with score {score:.2f}",
            "metadata": {
                "task_type": task_type,
                "difficulty": difficulty,
                "response_length": len(response),
                "cached": False
            }
        }
    
    async def batch_evaluate_optimized(self, evaluations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimized batch evaluation with concurrency."""
        start_time = time.time()
        
        # Create evaluation operations
        operations = []
        for eval_data in evaluations:
            operation = lambda ed=eval_data: self.evaluate_cached(
                ed.get("task_type", "attribution"),
                ed.get("model_response", ""),
                ed.get("domain", "general"),
                ed.get("difficulty", "medium")
            )
            operations.append(operation)
        
        # Execute concurrently
        results = await self.concurrency_manager.execute_concurrent(operations)
        
        # Filter out exceptions and convert to proper results
        valid_results = []
        for result in results:
            if not isinstance(result, Exception):
                valid_results.append(result)
            else:
                # Create error result
                valid_results.append({
                    "task_id": "batch_error",
                    "domain": "unknown",
                    "score": 0.0,
                    "reasoning_quality": 0.0,
                    "explanation": f"Batch evaluation failed: {str(result)}",
                    "metadata": {"error": str(result)}
                })
        
        execution_time = time.time() - start_time
        throughput = len(evaluations) / execution_time if execution_time > 0 else 0
        
        self._record_metrics("batch_evaluation", execution_time, 
                           cache_hit=False, operation_count=len(evaluations),
                           throughput=throughput)
        
        return valid_results
    
    def _record_metrics(self, operation_name: str, execution_time: float,
                       cache_hit: bool = False, operation_count: int = 1,
                       throughput: float = None):
        """Record performance metrics."""
        
        cache_stats = self.cache.get_stats()
        
        if throughput is None:
            throughput = operation_count / execution_time if execution_time > 0 else 0
        
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            execution_time=execution_time,
            throughput_ops_per_sec=throughput,
            cache_hit_rate=cache_stats["hit_rate"]
        )
        
        self.metrics_history.append(metrics)
        self.operation_counts[operation_name] += operation_count
        
        # Keep only recent metrics
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.metrics_history:
            return {"message": "No metrics available"}
        
        recent_metrics = self.metrics_history[-100:]  # Last 100 operations
        
        # Calculate averages
        avg_execution_time = sum(m.execution_time for m in recent_metrics) / len(recent_metrics)
        avg_throughput = sum(m.throughput_ops_per_sec for m in recent_metrics) / len(recent_metrics)
        avg_cache_hit_rate = sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics)
        
        return {
            "total_operations": sum(self.operation_counts.values()),
            "average_execution_time_seconds": avg_execution_time,
            "average_throughput_ops_per_sec": avg_throughput,
            "average_cache_hit_rate": avg_cache_hit_rate,
            "operation_breakdown": dict(self.operation_counts),
            "cache_stats": self.cache.get_stats(),
            "concurrency_stats": self.concurrency_manager.get_stats()
        }


async def test_caching_performance():
    """Test caching performance and hit rates."""
    print("=== Testing Caching Performance ===")
    
    engine = ScalableEvaluationEngine()
    
    # Test repeated evaluations to trigger cache hits
    test_responses = [
        "This relationship is spurious due to weather conditions",
        "The causal relationship is clear and direct",
        "No clear causation can be established here"
    ] * 5  # Repeat to trigger cache hits
    
    results = []
    for i, response in enumerate(test_responses):
        result = await engine.evaluate_cached(
            "attribution", 
            response, 
            "recreational", 
            "medium"
        )
        results.append(result)
        print(f"Evaluation {i+1}: Score={result['score']:.3f}, Cached={result['metadata'].get('cached', 'N/A')}")
    
    # Check cache performance
    cache_stats = engine.cache.get_stats()
    print(f"\n✓ Total evaluations: {len(results)}")
    print(f"✓ Cache entries: {cache_stats['entries']}")
    print(f"✓ Cache hit rate: {cache_stats['hit_rate']:.3f}")
    print(f"✓ Cache hits: {cache_stats['hits']}")
    print(f"✓ Cache misses: {cache_stats['misses']}")
    
    return cache_stats['hit_rate'] > 0.5  # Expect some cache hits


async def test_concurrent_processing():
    """Test concurrent evaluation processing."""
    print("\n=== Testing Concurrent Processing ===")
    
    engine = ScalableEvaluationEngine()
    
    # Create batch of evaluations
    batch_size = 20
    evaluations = []
    for i in range(batch_size):
        evaluations.append({
            "task_type": "attribution" if i % 2 == 0 else "counterfactual",
            "model_response": f"Test response {i} with spurious correlation and weather patterns",
            "domain": "recreational" if i % 3 == 0 else "general",
            "difficulty": ["easy", "medium", "hard"][i % 3]
        })
    
    # Test sequential processing
    print(f"Processing {batch_size} evaluations sequentially...")
    start_time = time.time()
    sequential_results = []
    for eval_data in evaluations:
        result = await engine.evaluate_cached(
            eval_data["task_type"],
            eval_data["model_response"],
            eval_data["domain"],
            eval_data["difficulty"]
        )
        sequential_results.append(result)
    sequential_time = time.time() - start_time
    
    # Reset engine for fair comparison
    engine = ScalableEvaluationEngine()
    
    # Test concurrent processing
    print(f"Processing {batch_size} evaluations concurrently...")
    start_time = time.time()
    concurrent_results = await engine.batch_evaluate_optimized(evaluations)
    concurrent_time = time.time() - start_time
    
    # Calculate performance improvement
    speedup = sequential_time / concurrent_time if concurrent_time > 0 else 1
    sequential_throughput = batch_size / sequential_time
    concurrent_throughput = batch_size / concurrent_time
    
    print(f"\n✓ Sequential time: {sequential_time:.3f}s")
    print(f"✓ Concurrent time: {concurrent_time:.3f}s")
    print(f"✓ Speedup: {speedup:.2f}x")
    print(f"✓ Sequential throughput: {sequential_throughput:.1f} ops/sec")
    print(f"✓ Concurrent throughput: {concurrent_throughput:.1f} ops/sec")
    print(f"✓ Results count (sequential): {len(sequential_results)}")
    print(f"✓ Results count (concurrent): {len(concurrent_results)}")
    
    return speedup > 1.5  # Expect significant speedup


async def test_scalability_limits():
    """Test system behavior under high load."""
    print("\n=== Testing Scalability Limits ===")
    
    engine = ScalableEvaluationEngine()
    
    # Test increasing batch sizes
    batch_sizes = [10, 50, 100]
    performance_results = []
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        
        # Create evaluation batch
        evaluations = []
        for i in range(batch_size):
            evaluations.append({
                "task_type": "attribution",
                "model_response": f"Response {i} testing scalability with various patterns",
                "domain": "general",
                "difficulty": "medium"
            })
        
        # Measure performance
        start_time = time.time()
        results = await engine.batch_evaluate_optimized(evaluations)
        execution_time = time.time() - start_time
        
        throughput = len(results) / execution_time if execution_time > 0 else 0
        
        performance_results.append({
            "batch_size": batch_size,
            "execution_time": execution_time,
            "throughput": throughput,
            "results_count": len(results)
        })
        
        print(f"  ✓ Execution time: {execution_time:.3f}s")
        print(f"  ✓ Throughput: {throughput:.1f} ops/sec")
        print(f"  ✓ Results count: {len(results)}")
    
    # Analyze scalability
    print(f"\n=== Scalability Analysis ===")
    for result in performance_results:
        print(f"Batch {result['batch_size']}: {result['throughput']:.1f} ops/sec")
    
    # Check if throughput scales reasonably
    throughputs = [r["throughput"] for r in performance_results]
    scales_well = all(throughputs[i] >= throughputs[i-1] * 0.8 for i in range(1, len(throughputs)))
    
    return scales_well and all(r["results_count"] == r["batch_size"] for r in performance_results)


async def test_memory_efficiency():
    """Test memory usage and efficiency."""
    print("\n=== Testing Memory Efficiency ===")
    
    engine = ScalableEvaluationEngine()
    
    # Process many evaluations to test memory management
    total_evaluations = 200
    evaluations = []
    
    for i in range(total_evaluations):
        evaluations.append({
            "task_type": ["attribution", "counterfactual"][i % 2],
            "model_response": f"Memory test response {i} " + "x" * (i % 100),  # Varying sizes
            "domain": "general",
            "difficulty": "medium"
        })
    
    # Process in batches to test memory management
    batch_size = 25
    all_results = []
    
    for i in range(0, total_evaluations, batch_size):
        batch = evaluations[i:i + batch_size]
        batch_results = await engine.batch_evaluate_optimized(batch)
        all_results.extend(batch_results)
        
        # Check cache behavior
        cache_stats = engine.cache.get_stats()
        print(f"Processed {i + len(batch)}/{total_evaluations}: "
              f"Cache entries={cache_stats['entries']}, "
              f"Hit rate={cache_stats['hit_rate']:.3f}")
    
    # Final statistics
    final_cache_stats = engine.cache.get_stats()
    performance_summary = engine.get_performance_summary()
    
    print(f"\n✓ Total evaluations processed: {len(all_results)}")
    print(f"✓ Final cache entries: {final_cache_stats['entries']}")
    print(f"✓ Final cache hit rate: {final_cache_stats['hit_rate']:.3f}")
    print(f"✓ Average throughput: {performance_summary['average_throughput_ops_per_sec']:.1f} ops/sec")
    
    # Check memory efficiency
    cache_within_limits = final_cache_stats['entries'] <= engine.cache.max_size
    good_hit_rate = final_cache_stats['hit_rate'] > 0.1
    
    return cache_within_limits and good_hit_rate and len(all_results) == total_evaluations


def main():
    """Run all scalability tests."""
    print("=== Causal Evaluation Bench - Scalable System Test ===")
    
    async def run_tests():
        test_results = []
        
        # Test caching performance
        try:
            result1 = await test_caching_performance()
            test_results.append(("Caching Performance", result1))
            print(f"\n✓ Caching Performance: {'PASS' if result1 else 'FAIL'}")
        except Exception as e:
            test_results.append(("Caching Performance", False))
            print(f"\n✗ Caching Performance: FAIL ({e})")
        
        # Test concurrent processing
        try:
            result2 = await test_concurrent_processing()
            test_results.append(("Concurrent Processing", result2))
            print(f"✓ Concurrent Processing: {'PASS' if result2 else 'FAIL'}")
        except Exception as e:
            test_results.append(("Concurrent Processing", False))
            print(f"✗ Concurrent Processing: FAIL ({e})")
        
        # Test scalability limits
        try:
            result3 = await test_scalability_limits()
            test_results.append(("Scalability Limits", result3))
            print(f"✓ Scalability Limits: {'PASS' if result3 else 'FAIL'}")
        except Exception as e:
            test_results.append(("Scalability Limits", False))
            print(f"✗ Scalability Limits: FAIL ({e})")
        
        # Test memory efficiency
        try:
            result4 = await test_memory_efficiency()
            test_results.append(("Memory Efficiency", result4))
            print(f"✓ Memory Efficiency: {'PASS' if result4 else 'FAIL'}")
        except Exception as e:
            test_results.append(("Memory Efficiency", False))
            print(f"✗ Memory Efficiency: FAIL ({e})")
        
        # Summary
        passed = sum(1 for _, result in test_results if result)
        total = len(test_results)
        
        print(f"\n=== Test Summary ===")
        print(f"Passed: {passed}/{total}")
        
        if passed == total:
            print("🎉 All scalability tests passed!")
            print("✅ System demonstrates high performance")
            print("✅ Caching system working effectively") 
            print("✅ Concurrent processing provides speedup")
            print("✅ System scales well with load")
            print("✅ Memory usage is efficient")
        else:
            print("❌ Some scalability tests failed")
        
        return passed == total
    
    return asyncio.run(run_tests())


if __name__ == "__main__":
    main()
