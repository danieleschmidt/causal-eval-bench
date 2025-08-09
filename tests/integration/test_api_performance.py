"""Integration tests for API performance and optimization."""

import pytest
import asyncio
import time
import httpx
from unittest.mock import Mock, patch

# Test configuration
API_BASE_URL = "http://localhost:8000"


class TestAPIPerformance:
    """Integration tests for API performance optimization."""
    
    @pytest.fixture
    async def client(self):
        """Create async HTTP client for testing."""
        async with httpx.AsyncClient(base_url=API_BASE_URL) as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_evaluation_caching_performance(self, client):
        """Test that identical requests are cached and faster."""
        
        evaluation_request = {
            "task_type": "attribution",
            "model_response": "Ice cream sales and drowning incidents are both caused by hot weather, making their relationship spurious rather than causal.",
            "domain": "recreational",
            "difficulty": "easy"
        }
        
        # First request - should be computed
        start_time = time.time()
        response1 = await client.post("/evaluation/evaluate", json=evaluation_request)
        first_duration = time.time() - start_time
        
        assert response1.status_code == 200
        result1 = response1.json()
        
        # Second request - should be cached
        start_time = time.time()
        response2 = await client.post("/evaluation/evaluate", json=evaluation_request)
        second_duration = time.time() - start_time
        
        assert response2.status_code == 200
        result2 = response2.json()
        
        # Results should be identical
        assert result1["overall_score"] == result2["overall_score"]
        assert result1["metadata"]["expected_relationship"] == result2["metadata"]["expected_relationship"]
        
        # Second request should be faster (cached)
        assert second_duration < first_duration * 0.8  # At least 20% faster
        
        # Check processing times in response
        assert result2["processing_time"] < result1["processing_time"] * 0.5  # At least 50% faster
    
    @pytest.mark.asyncio
    async def test_batch_evaluation_performance(self, client):
        """Test batch evaluation performance and optimization."""
        
        batch_requests = [
            {
                "task_type": "attribution",
                "model_response": f"Test response {i} for causal attribution analysis.",
                "domain": "general",
                "difficulty": "medium"
            } for i in range(10)
        ]
        
        start_time = time.time()
        response = await client.post("/evaluation/batch", json=batch_requests)
        total_duration = time.time() - start_time
        
        assert response.status_code == 200
        result = response.json()
        
        # Check batch results
        assert result["summary"]["total_evaluations"] == 10
        assert result["summary"]["success_rate"] == 1.0
        assert len(result["results"]) == 10
        
        # Batch should be more efficient than individual requests
        avg_time_per_eval = result["summary"]["processing_time"] / 10
        assert avg_time_per_eval < 0.01  # Less than 10ms per evaluation on average
        
        # Total batch time should be reasonable
        assert total_duration < 2.0  # Less than 2 seconds for 10 evaluations
    
    @pytest.mark.asyncio
    async def test_concurrent_evaluation_performance(self, client):
        """Test API performance under concurrent load."""
        
        evaluation_request = {
            "task_type": "attribution",
            "model_response": "Regular exercise directly improves cardiovascular health through multiple physiological mechanisms.",
            "domain": "medical",
            "difficulty": "medium"
        }
        
        # Create multiple concurrent requests
        async def make_request():
            response = await client.post("/evaluation/evaluate", json=evaluation_request)
            return response.status_code, response.json()
        
        # Execute 20 concurrent requests
        start_time = time.time()
        tasks = [make_request() for _ in range(20)]
        results = await asyncio.gather(*tasks)
        total_duration = time.time() - start_time
        
        # All requests should succeed
        successful_requests = [r for r in results if r[0] == 200]
        assert len(successful_requests) == 20
        
        # Should handle concurrent load efficiently
        assert total_duration < 5.0  # Less than 5 seconds for 20 concurrent requests
        avg_response_time = total_duration / 20
        assert avg_response_time < 0.25  # Less than 250ms average per request
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_endpoint(self, client):
        """Test performance monitoring and statistics endpoint."""
        
        # Make a few requests to generate some stats
        for i in range(3):
            await client.post("/evaluation/evaluate", json={
                "task_type": "attribution",
                "model_response": f"Test response {i}",
                "domain": "general"
            })
        
        # Get performance stats
        response = await client.get("/evaluation/performance")
        assert response.status_code == 200
        
        stats = response.json()
        assert "optimization_stats" in stats
        
        opt_stats = stats["optimization_stats"]
        assert "cache_stats" in opt_stats
        assert "performance_analysis" in opt_stats
        
        # Cache should show activity
        cache_stats = opt_stats["cache_stats"]
        assert cache_stats["total_hits"] + cache_stats["total_misses"] > 0
        assert "hit_rate" in cache_stats
        assert "memory_usage_mb" in cache_stats
    
    @pytest.mark.asyncio
    async def test_cache_clear_functionality(self, client):
        """Test cache clearing functionality."""
        
        # Make a request to populate cache
        evaluation_request = {
            "task_type": "attribution",
            "model_response": "Test response for cache clearing",
            "domain": "general"
        }
        
        await client.post("/evaluation/evaluate", json=evaluation_request)
        
        # Check cache has entries
        perf_response = await client.get("/evaluation/performance")
        cache_stats_before = perf_response.json()["optimization_stats"]["cache_stats"]
        assert cache_stats_before["entries"] > 0
        
        # Clear cache
        clear_response = await client.post("/evaluation/cache/clear")
        assert clear_response.status_code == 200
        assert clear_response.json()["status"] == "success"
        
        # Check cache is cleared
        perf_response = await client.get("/evaluation/performance")
        cache_stats_after = perf_response.json()["optimization_stats"]["cache_stats"]
        assert cache_stats_after["entries"] == 0
    
    @pytest.mark.asyncio
    async def test_error_handling_performance(self, client):
        """Test that error handling doesn't significantly impact performance."""
        
        invalid_requests = [
            {
                "task_type": "invalid_task",
                "model_response": "Test response",
                "domain": "general"
            },
            {
                "task_type": "attribution",
                "model_response": "",  # Empty response
                "domain": "general"
            },
            {
                "task_type": "attribution",
                "model_response": "Valid response",
                "domain": "invalid_domain"
            }
        ]
        
        start_time = time.time()
        for request in invalid_requests:
            response = await client.post("/evaluation/evaluate", json=request)
            assert response.status_code in [400, 422]  # Should handle errors gracefully
        
        total_duration = time.time() - start_time
        
        # Error handling should be fast
        assert total_duration < 1.0  # Less than 1 second for all error cases
        
        # System should still be responsive after errors
        valid_response = await client.post("/evaluation/evaluate", json={
            "task_type": "attribution",
            "model_response": "Valid response after errors",
            "domain": "general"
        })
        assert valid_response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_mixed_workload_performance(self, client):
        """Test performance with mixed workload of different operations."""
        
        async def single_evaluation():
            return await client.post("/evaluation/evaluate", json={
                "task_type": "attribution",
                "model_response": f"Single evaluation at {time.time()}",
                "domain": "general"
            })
        
        async def batch_evaluation():
            return await client.post("/evaluation/batch", json=[
                {
                    "task_type": "attribution",
                    "model_response": f"Batch item {i} at {time.time()}",
                    "domain": "medical"
                } for i in range(3)
            ])
        
        async def get_performance():
            return await client.get("/evaluation/performance")
        
        # Mix of different operations
        tasks = []
        tasks.extend([single_evaluation() for _ in range(5)])
        tasks.extend([batch_evaluation() for _ in range(2)])
        tasks.extend([get_performance() for _ in range(3)])
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_duration = time.time() - start_time
        
        # Check that most operations succeeded
        successful_results = [r for r in results if not isinstance(r, Exception) and hasattr(r, 'status_code') and r.status_code == 200]
        success_rate = len(successful_results) / len(tasks)
        assert success_rate >= 0.9  # At least 90% success rate
        
        # Mixed workload should complete in reasonable time
        assert total_duration < 10.0  # Less than 10 seconds for mixed workload


class TestAPIScalability:
    """Tests for API scalability characteristics."""
    
    @pytest.mark.asyncio
    async def test_memory_usage_stability(self, client):
        """Test that memory usage remains stable under load."""
        
        # Get baseline performance stats
        baseline_response = await client.get("/evaluation/performance")
        baseline_memory = baseline_response.json()["optimization_stats"]["cache_stats"]["memory_usage_mb"]
        
        # Generate load with many different requests
        for batch in range(5):
            batch_requests = [
                {
                    "task_type": "attribution",
                    "model_response": f"Unique response {batch}_{i} for memory test",
                    "domain": "general",
                    "difficulty": "medium"
                } for i in range(20)
            ]
            
            response = await client.post("/evaluation/batch", json=batch_requests)
            assert response.status_code == 200
        
        # Check memory usage after load
        after_response = await client.get("/evaluation/performance")
        after_memory = after_response.json()["optimization_stats"]["cache_stats"]["memory_usage_mb"]
        
        # Memory usage should not grow excessively
        memory_increase = after_memory - baseline_memory
        assert memory_increase < 50.0  # Less than 50MB increase
    
    @pytest.mark.asyncio
    async def test_response_time_consistency(self, client):
        """Test that response times remain consistent under load."""
        
        response_times = []
        
        # Make 50 requests and measure response times
        for i in range(50):
            start_time = time.time()
            response = await client.post("/evaluation/evaluate", json={
                "task_type": "attribution",
                "model_response": f"Consistency test {i}",
                "domain": "general"
            })
            duration = time.time() - start_time
            response_times.append(duration)
            
            assert response.status_code == 200
        
        # Calculate response time statistics
        avg_time = sum(response_times) / len(response_times)
        max_time = max(response_times)
        min_time = min(response_times)
        
        # Response times should be consistent
        assert avg_time < 0.5  # Average less than 500ms
        assert max_time < avg_time * 3  # Max not more than 3x average
        assert max_time < 2.0  # No request takes more than 2 seconds
        
        # Most requests should be fast
        fast_requests = [t for t in response_times if t < 0.1]  # Less than 100ms
        assert len(fast_requests) / len(response_times) >= 0.7  # At least 70% fast requests


@pytest.mark.asyncio
async def test_health_check_performance():
    """Test that health checks are fast and don't impact performance."""
    
    async with httpx.AsyncClient(base_url=API_BASE_URL) as client:
        # Health checks should be very fast
        start_time = time.time()
        response = await client.get("/health/")
        duration = time.time() - start_time
        
        assert response.status_code == 200
        assert duration < 0.1  # Less than 100ms
        
        # Health checks shouldn't impact other operations
        eval_start = time.time()
        eval_response = await client.post("/evaluation/evaluate", json={
            "task_type": "attribution",
            "model_response": "Health check impact test",
            "domain": "general"
        })
        eval_duration = time.time() - eval_start
        
        assert eval_response.status_code == 200
        assert eval_duration < 0.5  # Normal evaluation performance maintained