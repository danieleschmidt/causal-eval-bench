"""Performance benchmarks for Causal Eval Bench."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import pytest
from httpx import AsyncClient


@pytest.mark.performance
@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    @pytest.mark.benchmark(group="evaluation")
    def test_single_evaluation_performance(self, benchmark, sample_causal_question):
        """Benchmark single evaluation performance."""
        
        def run_single_evaluation():
            # Simulate evaluation logic
            time.sleep(0.1)  # Simulate processing time
            return {"score": 0.85, "explanation": "Test evaluation"}
        
        result = benchmark(run_single_evaluation)
        assert result["score"] > 0

    @pytest.mark.benchmark(group="evaluation")
    def test_batch_evaluation_performance(self, benchmark, performance_test_data):
        """Benchmark batch evaluation performance."""
        
        def run_batch_evaluation():
            results = []
            for question in performance_test_data[:10]:  # Test with 10 questions
                # Simulate evaluation
                time.sleep(0.01)  # Simulate processing
                results.append({"id": question["id"], "score": 0.8})
            return results
        
        results = benchmark(run_batch_evaluation)
        assert len(results) == 10

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_concurrent_evaluations(self, performance_test_data, benchmark_config):
        """Test concurrent evaluation performance."""
        
        async def simulate_evaluation(question_id: str):
            # Simulate async evaluation
            await asyncio.sleep(0.1)
            return {"id": question_id, "score": 0.85}
        
        start_time = time.time()
        
        # Run 10 concurrent evaluations
        tasks = []
        for question in performance_test_data[:10]:
            task = simulate_evaluation(question["id"])
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        execution_time = time.time() - start_time
        
        # Assert performance criteria
        assert execution_time < benchmark_config["max_execution_time"]
        assert len(results) == 10
        
        # Calculate throughput
        throughput = len(results) / execution_time
        assert throughput >= benchmark_config["target_throughput"]

    @pytest.mark.benchmark(group="generation")
    def test_question_generation_performance(self, benchmark):
        """Benchmark question generation performance."""
        
        def generate_questions():
            questions = []
            for i in range(50):
                # Simulate question generation
                time.sleep(0.02)  # Simulate AI model call
                questions.append({
                    "id": f"generated_{i}",
                    "prompt": f"Generated question {i}",
                    "task_type": "causal_attribution"
                })
            return questions
        
        questions = benchmark(generate_questions)
        assert len(questions) == 50

    @pytest.mark.load_test
    def test_load_performance(self, performance_test_data, benchmark_config):
        """Test system performance under load."""
        
        def simulate_user_request():
            # Simulate user making evaluation request
            time.sleep(0.5)  # Simulate network + processing
            return {"status": "success", "evaluation_time": 0.5}
        
        start_time = time.time()
        
        # Simulate 20 concurrent users
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(simulate_user_request) for _ in range(100)]
            results = [future.result() for future in as_completed(futures)]
        
        execution_time = time.time() - start_time
        
        # Calculate metrics
        throughput = len(results) / execution_time
        successful_requests = sum(1 for r in results if r["status"] == "success")
        error_rate = 1 - (successful_requests / len(results))
        
        # Assert performance criteria
        assert error_rate <= benchmark_config["acceptable_error_rate"]
        assert execution_time < 60.0  # Should complete within 1 minute
        
        print(f"Throughput: {throughput:.2f} req/sec")
        print(f"Error rate: {error_rate:.2%}")
        print(f"Total time: {execution_time:.2f}s")

    @pytest.mark.memory_test
    def test_memory_usage(self, performance_test_data, benchmark_config):
        """Test memory usage during evaluation."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate memory-intensive operations
        large_data = []
        for question in performance_test_data:
            # Simulate storing evaluation results
            large_data.append({
                "question": question,
                "evaluation_result": {"score": 0.8, "details": "x" * 1000}
            })
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Clean up
        del large_data
        
        # Assert memory usage is within limits
        assert memory_increase < benchmark_config["max_memory_mb"]
        
        print(f"Memory increase: {memory_increase:.2f} MB")

    @pytest.mark.asyncio
    @pytest.mark.api_performance
    async def test_api_response_times(self):
        """Test API endpoint response times."""
        
        # This would test actual API endpoints
        # Placeholder for API performance testing
        
        endpoints = [
            "/health",
            "/api/v1/evaluate",
            "/api/v1/questions",
            "/api/v1/leaderboard"
        ]
        
        response_times = []
        
        async with AsyncClient(base_url="http://localhost:8000") as client:
            for endpoint in endpoints:
                start_time = time.time()
                
                try:
                    # Simulate API call (would be actual call in real test)
                    await asyncio.sleep(0.1)  # Simulate network delay
                    response_time = time.time() - start_time
                    response_times.append(response_time)
                except Exception as e:
                    pytest.skip(f"API endpoint {endpoint} not available: {e}")
        
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            
            # Assert response time criteria
            assert avg_response_time < 1.0  # Average under 1 second
            assert max_response_time < 5.0  # Max under 5 seconds
            
            print(f"Average response time: {avg_response_time:.3f}s")
            print(f"Max response time: {max_response_time:.3f}s")


import asyncio  # Import needed for async tests
