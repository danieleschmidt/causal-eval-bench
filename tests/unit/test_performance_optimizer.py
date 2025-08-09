"""Unit tests for performance optimization components."""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch

from causal_eval.core.performance_optimizer import (
    IntelligentCache,
    CacheEntry,
    PerformanceProfiler,
    OptimizedEvaluationEngine,
    cached_evaluation
)


class TestIntelligentCache:
    """Test the intelligent caching system."""
    
    @pytest.fixture
    def cache(self):
        return IntelligentCache(max_size=10, max_memory_mb=1)
    
    @pytest.mark.asyncio
    async def test_basic_cache_operations(self, cache):
        """Test basic cache set/get operations."""
        # Test set and get
        await cache.set("key1", "value1")
        result = await cache.get("key1")
        assert result == "value1"
        
        # Test cache miss
        result = await cache.get("nonexistent")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_expiration(self, cache):
        """Test cache TTL expiration."""
        # Set with short TTL
        await cache.set("expiring_key", "value", ttl=0.1)
        
        # Should exist immediately
        result = await cache.get("expiring_key")
        assert result == "value"
        
        # Wait for expiration
        await asyncio.sleep(0.2)
        
        # Should be expired
        result = await cache.get("expiring_key")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_eviction(self, cache):
        """Test intelligent cache eviction."""
        # Fill cache to max size
        for i in range(12):  # More than max_size of 10
            await cache.set(f"key{i}", f"value{i}")
        
        # Should have evicted some entries
        assert len(cache.cache) <= 10
        
        # Most recent entries should still be there
        result = await cache.get("key11")
        assert result == "value11"
    
    @pytest.mark.asyncio
    async def test_hit_count_tracking(self, cache):
        """Test hit count and access pattern tracking."""
        await cache.set("popular_key", "value")
        
        # Access multiple times
        for _ in range(5):
            await cache.get("popular_key")
        
        entry = cache.cache["popular_key"]
        assert entry.hit_count == 5
    
    @pytest.mark.asyncio
    async def test_cache_stats(self, cache):
        """Test cache statistics generation."""
        await cache.set("key1", "value1")
        await cache.get("key1")
        await cache.get("nonexistent")
        
        stats = cache.get_stats()
        
        assert stats["entries"] == 1
        assert stats["total_hits"] == 1
        assert stats["total_misses"] == 1
        assert stats["hit_rate"] == 0.5
    
    @pytest.mark.asyncio
    async def test_memory_tracking(self, cache):
        """Test memory usage tracking."""
        large_value = "x" * 1024  # 1KB value
        await cache.set("large_key", large_value)
        
        stats = cache.get_stats()
        assert stats["memory_usage_bytes"] > 1000
        assert stats["memory_usage_mb"] > 0
    
    @pytest.mark.asyncio
    async def test_clear_cache(self, cache):
        """Test cache clearing functionality."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        
        assert len(cache.cache) == 2
        
        await cache.clear()
        
        assert len(cache.cache) == 0
        assert cache.current_memory_usage == 0


class TestPerformanceProfiler:
    """Test the performance profiling system."""
    
    @pytest.fixture
    def profiler(self):
        return PerformanceProfiler()
    
    def test_operation_recording(self, profiler):
        """Test recording operation timings."""
        profiler.record_operation("test_op", 1.5)
        profiler.record_operation("test_op", 0.8)
        profiler.record_operation("test_op", 2.1)
        
        times = profiler.operation_times["test_op"]
        assert len(times) == 3
        assert 0.8 in times
        assert 2.1 in times
    
    def test_slow_operation_detection(self, profiler):
        """Test detection of slow operations."""
        # Record a slow operation
        profiler.record_operation("slow_op", 2.5, {"context": "test"})
        
        assert len(profiler.slow_operations) == 1
        assert profiler.slow_operations[0]["operation"] == "slow_op"
        assert profiler.slow_operations[0]["duration"] == 2.5
    
    def test_performance_analysis(self, profiler):
        """Test comprehensive performance analysis."""
        # Record various operations
        for i in range(10):
            profiler.record_operation("fast_op", 0.1 + i * 0.01)
        
        analysis = profiler.get_performance_analysis()
        
        assert "operations" in analysis
        assert "fast_op" in analysis["operations"]
        
        fast_op_stats = analysis["operations"]["fast_op"]
        assert fast_op_stats["count"] == 10
        assert fast_op_stats["min"] == 0.1
        assert fast_op_stats["max"] == 0.19
        assert 0.1 <= fast_op_stats["avg"] <= 0.2
    
    def test_recommendation_generation(self, profiler):
        """Test optimization recommendation generation."""
        # Create scenario that should generate recommendations
        for _ in range(15):
            profiler.record_operation("evaluation", 0.8)  # Somewhat slow
        
        analysis = profiler.get_performance_analysis()
        recommendations = analysis["recommendations"]
        
        # Should have some recommendations
        assert isinstance(recommendations, list)


class TestOptimizedEvaluationEngine:
    """Test the optimized evaluation engine."""
    
    @pytest.fixture
    def engine(self):
        return OptimizedEvaluationEngine()
    
    def test_cache_key_generation(self, engine):
        """Test cache key generation."""
        key1 = engine._generate_cache_key("attribution", "test response", "medical", "hard")
        key2 = engine._generate_cache_key("attribution", "test response", "medical", "hard")
        key3 = engine._generate_cache_key("attribution", "different response", "medical", "hard")
        
        # Same inputs should generate same key
        assert key1 == key2
        
        # Different inputs should generate different keys
        assert key1 != key3
        
        # Keys should be properly formatted
        assert key1.startswith("eval:attribution:medical:hard:")
    
    def test_cache_ttl_calculation(self, engine):
        """Test intelligent TTL calculation."""
        # High confidence, clear result (high score)
        result1 = {"confidence": 0.9, "overall_score": 0.9}
        ttl1 = engine._calculate_cache_ttl(result1)
        
        # Low confidence result
        result2 = {"confidence": 0.2, "overall_score": 0.5}
        ttl2 = engine._calculate_cache_ttl(result2)
        
        # High confidence should have longer TTL
        assert ttl1 > ttl2
        assert ttl1 == 7200.0  # 2 hours
        assert ttl2 == 1800.0  # 30 minutes
    
    def test_evaluation_grouping(self, engine):
        """Test evaluation grouping for batch optimization."""
        evaluations = [
            {"task_type": "attribution", "domain": "medical", "model_response": "test1"},
            {"task_type": "attribution", "domain": "medical", "model_response": "test2"},
            {"task_type": "counterfactual", "domain": "general", "model_response": "test3"},
            {"task_type": "attribution", "domain": "business", "model_response": "test4"},
        ]
        
        groups = engine._group_similar_evaluations(evaluations)
        
        # Should create 3 groups: attribution+medical, counterfactual+general, attribution+business
        assert len(groups) == 3
        
        # Find attribution+medical group
        medical_group = None
        for group in groups:
            if (group[0]["task_type"] == "attribution" and 
                group[0]["domain"] == "medical"):
                medical_group = group
                break
        
        assert medical_group is not None
        assert len(medical_group) == 2
    
    @pytest.mark.asyncio
    async def test_optimization_stats(self, engine):
        """Test optimization statistics retrieval."""
        stats = await engine.get_optimization_stats()
        
        assert "cache_stats" in stats
        assert "performance_analysis" in stats
        assert "concurrent_evaluations_limit" in stats
        
        # Cache stats should have expected structure
        cache_stats = stats["cache_stats"]
        assert "entries" in cache_stats
        assert "hit_rate" in cache_stats
        assert "memory_usage_mb" in cache_stats


class TestCachedEvaluationDecorator:
    """Test the cached evaluation decorator."""
    
    @pytest.mark.asyncio
    async def test_cached_function(self):
        """Test function caching with decorator."""
        call_count = 0
        
        @cached_evaluation(ttl=1.0)
        async def test_function(arg1, arg2):
            nonlocal call_count
            call_count += 1
            return f"{arg1}_{arg2}_{call_count}"
        
        # First call should execute function
        result1 = await test_function("a", "b")
        assert result1 == "a_b_1"
        assert call_count == 1
        
        # Second call with same args should use cache
        result2 = await test_function("a", "b")
        assert result2 == "a_b_1"  # Same result
        assert call_count == 1  # Function not called again
        
        # Different args should execute function
        result3 = await test_function("c", "d")
        assert result3 == "c_d_2"
        assert call_count == 2


@pytest.mark.asyncio
async def test_cache_entry_expiration():
    """Test cache entry expiration logic."""
    entry = CacheEntry(
        value="test_value",
        timestamp=time.time() - 7200,  # 2 hours ago
        ttl=3600.0  # 1 hour TTL
    )
    
    assert entry.is_expired is True
    assert entry.age > 7000  # More than 7000 seconds old


@pytest.mark.asyncio
async def test_cache_entry_touch():
    """Test cache entry touch functionality."""
    entry = CacheEntry(
        value="test_value",
        timestamp=time.time(),
        ttl=3600.0
    )
    
    initial_access = entry.last_access
    initial_hits = entry.hit_count
    
    entry.touch()
    
    assert entry.last_access > initial_access
    assert entry.hit_count == initial_hits + 1


class TestIntegration:
    """Integration tests for performance optimization components."""
    
    @pytest.mark.asyncio
    async def test_cache_performance_under_load(self):
        """Test cache performance under high load."""
        cache = IntelligentCache(max_size=100, max_memory_mb=10)
        
        # Simulate high load
        tasks = []
        for i in range(1000):
            if i % 3 == 0:  # Set operations
                task = cache.set(f"key_{i}", f"value_{i}")
            else:  # Get operations
                task = cache.get(f"key_{i % 300}")  # Some keys will exist
            tasks.append(task)
        
        # Execute all operations concurrently
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Cache should still be functional
        stats = cache.get_stats()
        assert stats["entries"] <= 100  # Shouldn't exceed max size
        assert stats["total_hits"] + stats["total_misses"] > 0
    
    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self):
        """Test cache behavior under memory pressure."""
        cache = IntelligentCache(max_size=1000, max_memory_mb=1)  # Small memory limit
        
        # Add large values to trigger memory-based eviction
        large_value = "x" * (50 * 1024)  # 50KB per value
        
        for i in range(50):  # Try to add 2.5MB of data
            await cache.set(f"large_key_{i}", large_value)
        
        stats = cache.get_stats()
        # Should have evicted entries to stay under memory limit
        assert stats["memory_usage_mb"] <= 1.5  # Some tolerance for overhead
        assert stats["eviction_count"] > 0