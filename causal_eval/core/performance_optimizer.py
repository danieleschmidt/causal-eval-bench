"""Advanced performance optimization and intelligent caching system."""

import time
import asyncio
import hashlib
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass
from collections import defaultdict, deque
import logging
from functools import wraps
import json

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Smart cache entry with metadata."""
    value: Any
    timestamp: float
    hit_count: int = 0
    last_access: float = 0.0
    ttl: float = 3600.0  # 1 hour default
    size_estimate: int = 0
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return time.time() - self.timestamp > self.ttl
    
    @property
    def age(self) -> float:
        """Get age of cache entry in seconds."""
        return time.time() - self.timestamp
    
    def touch(self) -> None:
        """Update last access time and increment hit count."""
        self.last_access = time.time()
        self.hit_count += 1


class IntelligentCache:
    """High-performance cache with intelligent eviction and prefetching."""
    
    def __init__(self, max_size: int = 10000, max_memory_mb: int = 512):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order = deque()  # For LRU tracking
        self.hit_counts = defaultdict(int)
        self.miss_counts = defaultdict(int)
        
        # Performance metrics
        self.total_hits = 0
        self.total_misses = 0
        self.eviction_count = 0
        self.current_memory_usage = 0
        
        logger.info(f"Intelligent cache initialized: {max_size} entries, {max_memory_mb}MB limit")
    
    def _calculate_size(self, value: Any) -> int:
        """Estimate memory usage of a value."""
        try:
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, dict):
                return len(json.dumps(value).encode('utf-8'))
            else:
                return len(str(value).encode('utf-8'))
        except Exception:
            return 1024  # Default estimate
    
    def _evict_if_needed(self) -> None:
        """Intelligent cache eviction based on multiple factors."""
        while (len(self.cache) >= self.max_size or 
               self.current_memory_usage >= self.max_memory_bytes):
            
            if not self.cache:
                break
            
            # Find candidate for eviction based on multiple factors
            candidate_key = self._find_eviction_candidate()
            if candidate_key:
                self._evict_entry(candidate_key)
            else:
                break
    
    def _find_eviction_candidate(self) -> Optional[str]:
        """Find the best candidate for eviction using intelligent scoring."""
        if not self.cache:
            return None
        
        current_time = time.time()
        best_score = float('inf')
        best_key = None
        
        for key, entry in self.cache.items():
            # Scoring factors:
            # - Age (older = higher score)
            # - Hit frequency (lower frequency = higher score)
            # - Size (larger = slightly higher score)
            # - Expiration (expired = very high score)
            
            if entry.is_expired:
                return key  # Immediately evict expired entries
            
            age_score = entry.age / 3600  # Hours
            frequency_score = 1.0 / max(entry.hit_count, 1)
            size_score = entry.size_estimate / (1024 * 1024)  # MB
            
            total_score = age_score + frequency_score * 2 + size_score * 0.1
            
            if total_score < best_score:
                best_score = total_score
                best_key = key
        
        return best_key
    
    def _evict_entry(self, key: str) -> None:
        """Remove an entry from cache and update metrics."""
        if key in self.cache:
            entry = self.cache[key]
            self.current_memory_usage -= entry.size_estimate
            del self.cache[key]
            self.eviction_count += 1
            
            # Remove from access order if present
            try:
                self.access_order.remove(key)
            except ValueError:
                pass
            
            logger.debug(f"Evicted cache entry: {key}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with intelligent tracking."""
        if key in self.cache:
            entry = self.cache[key]
            
            if entry.is_expired:
                self._evict_entry(key)
                self.miss_counts[key] += 1
                self.total_misses += 1
                return None
            
            entry.touch()
            self.hit_counts[key] += 1
            self.total_hits += 1
            
            # Update LRU order
            try:
                self.access_order.remove(key)
            except ValueError:
                pass
            self.access_order.append(key)
            
            logger.debug(f"Cache hit: {key} (hits: {entry.hit_count})")
            return entry.value
        
        self.miss_counts[key] += 1
        self.total_misses += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: float = 3600.0) -> None:
        """Set value in cache with intelligent management."""
        size_estimate = self._calculate_size(value)
        
        # Remove existing entry if present
        if key in self.cache:
            old_entry = self.cache[key]
            self.current_memory_usage -= old_entry.size_estimate
        
        # Check if we need to evict
        self._evict_if_needed()
        
        # Create new entry
        entry = CacheEntry(
            value=value,
            timestamp=time.time(),
            ttl=ttl,
            size_estimate=size_estimate
        )
        
        self.cache[key] = entry
        self.current_memory_usage += size_estimate
        self.access_order.append(key)
        
        logger.debug(f"Cache set: {key} (size: {size_estimate} bytes, TTL: {ttl}s)")
    
    async def delete(self, key: str) -> bool:
        """Delete specific key from cache."""
        if key in self.cache:
            self._evict_entry(key)
            return True
        return False
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.access_order.clear()
        self.current_memory_usage = 0
        self.eviction_count = 0
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        total_requests = self.total_hits + self.total_misses
        hit_rate = self.total_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "entries": len(self.cache),
            "max_size": self.max_size,
            "memory_usage_bytes": self.current_memory_usage,
            "memory_usage_mb": self.current_memory_usage / (1024 * 1024),
            "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
            "memory_utilization": self.current_memory_usage / self.max_memory_bytes,
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "hit_rate": hit_rate,
            "eviction_count": self.eviction_count,
            "top_keys": sorted(
                self.hit_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        }


class PerformanceProfiler:
    """Advanced performance profiling and optimization recommendations."""
    
    def __init__(self):
        self.operation_times: Dict[str, List[float]] = defaultdict(list)
        self.slow_operations: List[Dict[str, Any]] = []
        self.optimization_recommendations: List[str] = []
        
    def record_operation(self, operation: str, duration: float, context: Dict[str, Any] = None) -> None:
        """Record operation timing for analysis."""
        self.operation_times[operation].append(duration)
        
        # Flag slow operations
        if duration > 1.0:  # More than 1 second
            self.slow_operations.append({
                "operation": operation,
                "duration": duration,
                "timestamp": time.time(),
                "context": context or {}
            })
            
            # Keep only recent slow operations
            if len(self.slow_operations) > 100:
                self.slow_operations = self.slow_operations[-100:]
    
    def get_performance_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive performance analysis."""
        analysis = {
            "operations": {},
            "slow_operations": self.slow_operations[-10:],  # Last 10 slow operations
            "recommendations": self._generate_recommendations()
        }
        
        # Analyze each operation type
        for operation, times in self.operation_times.items():
            if times:
                sorted_times = sorted(times)
                analysis["operations"][operation] = {
                    "count": len(times),
                    "avg": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                    "p50": sorted_times[len(sorted_times) // 2],
                    "p95": sorted_times[int(len(sorted_times) * 0.95)],
                    "p99": sorted_times[int(len(sorted_times) * 0.99)]
                }
        
        return analysis
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on performance data."""
        recommendations = []
        
        # Analyze evaluation times
        eval_times = self.operation_times.get("evaluation", [])
        if eval_times and len(eval_times) > 10:
            avg_time = sum(eval_times) / len(eval_times)
            if avg_time > 0.5:
                recommendations.append("Consider caching evaluation results for similar inputs")
            if max(eval_times) > 5.0:
                recommendations.append("Some evaluations are very slow - investigate complex scenarios")
        
        # Analyze batch processing
        batch_times = self.operation_times.get("batch_evaluation", [])
        if batch_times and len(batch_times) > 5:
            avg_batch_time = sum(batch_times) / len(batch_times)
            if avg_batch_time > 2.0:
                recommendations.append("Implement parallel processing for batch evaluations")
        
        # Check for frequent slow operations
        recent_slow = [op for op in self.slow_operations if time.time() - op["timestamp"] < 3600]
        if len(recent_slow) > 10:
            recommendations.append("High frequency of slow operations detected - consider system resources")
        
        return recommendations


class OptimizedEvaluationEngine:
    """High-performance evaluation engine with intelligent caching and optimization."""
    
    def __init__(self):
        self.cache = IntelligentCache(max_size=5000, max_memory_mb=256)
        self.profiler = PerformanceProfiler()
        self.concurrent_evaluations = asyncio.Semaphore(10)  # Limit concurrent evaluations
        
        # Response pattern learning
        self.response_patterns = defaultdict(int)
        self.common_responses_cache = {}
        
        logger.info("Optimized evaluation engine initialized")
    
    def _generate_cache_key(self, task_type: str, response: str, domain: str, difficulty: str) -> str:
        """Generate intelligent cache key."""
        # Hash the response for consistent keying while preserving privacy
        response_hash = hashlib.sha256(response.encode()).hexdigest()[:16]
        return f"eval:{task_type}:{domain}:{difficulty}:{response_hash}"
    
    async def evaluate_with_optimization(
        self, 
        task_type: str, 
        model_response: str, 
        domain: str = "general",
        difficulty: str = "medium",
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Evaluate with full optimization pipeline."""
        start_time = time.time()
        
        # Generate cache key
        cache_key = self._generate_cache_key(task_type, model_response, domain, difficulty)
        
        # Check cache first
        cached_result = await self.cache.get(cache_key)
        if cached_result is not None:
            duration = time.time() - start_time
            self.profiler.record_operation("cached_evaluation", duration)
            logger.debug(f"Cache hit for evaluation: {cache_key}")
            return cached_result
        
        # Perform evaluation with concurrency control
        async with self.concurrent_evaluations:
            try:
                # Import here to avoid circular imports
                from causal_eval.core.engine import EvaluationEngine
                from causal_eval.core.engine import CausalEvaluationRequest
                
                # Create standard evaluation request
                request = CausalEvaluationRequest(
                    task_type=task_type,
                    model_response=model_response,
                    domain=domain,
                    difficulty=difficulty
                )
                
                # Perform evaluation
                engine = EvaluationEngine()
                result = await engine.evaluate_request(request)
                
                # Cache the result with intelligent TTL
                ttl = self._calculate_cache_ttl(result)
                await self.cache.set(cache_key, result, ttl)
                
                duration = time.time() - start_time
                self.profiler.record_operation("evaluation", duration, {
                    "task_type": task_type,
                    "domain": domain,
                    "difficulty": difficulty,
                    "score": result.get("overall_score", 0)
                })
                
                logger.debug(f"Evaluation completed in {duration:.3f}s: {cache_key}")
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                self.profiler.record_operation("failed_evaluation", duration, {
                    "error": str(e),
                    "task_type": task_type
                })
                raise e
    
    def _calculate_cache_ttl(self, result: Dict[str, Any]) -> float:
        """Calculate intelligent TTL based on result characteristics."""
        base_ttl = 3600.0  # 1 hour
        
        # High-confidence results can be cached longer
        confidence = result.get("confidence", 0.5)
        score = result.get("overall_score", 0.0)
        
        # Longer TTL for high-confidence, clear results
        if confidence > 0.8 and (score > 0.8 or score < 0.2):
            return base_ttl * 2  # 2 hours
        
        # Shorter TTL for uncertain results
        if confidence < 0.3 or (0.4 < score < 0.6):
            return base_ttl * 0.5  # 30 minutes
        
        return base_ttl
    
    async def batch_evaluate_optimized(
        self, 
        evaluations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Optimized batch evaluation with intelligent scheduling."""
        start_time = time.time()
        
        # Group evaluations by similarity for potential optimization
        grouped_evaluations = self._group_similar_evaluations(evaluations)
        
        # Process groups concurrently
        all_results = []
        tasks = []
        
        for group in grouped_evaluations:
            task = self._process_evaluation_group(group)
            tasks.append(task)
        
        # Wait for all groups to complete
        group_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results
        for group_result in group_results:
            if isinstance(group_result, list):
                all_results.extend(group_result)
            else:
                # Handle exceptions
                logger.error(f"Batch group failed: {group_result}")
        
        duration = time.time() - start_time
        self.profiler.record_operation("batch_evaluation", duration, {
            "count": len(evaluations),
            "groups": len(grouped_evaluations)
        })
        
        return all_results
    
    def _group_similar_evaluations(
        self, 
        evaluations: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Group similar evaluations for batch optimization."""
        groups = defaultdict(list)
        
        for eval_data in evaluations:
            # Group by task type and domain for cache locality
            key = f"{eval_data.get('task_type', 'unknown')}:{eval_data.get('domain', 'general')}"
            groups[key].append(eval_data)
        
        return list(groups.values())
    
    async def _process_evaluation_group(
        self, 
        evaluations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process a group of similar evaluations."""
        results = []
        
        # Process evaluations in the group with controlled concurrency
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent per group
        
        async def process_single(eval_data):
            async with semaphore:
                return await self.evaluate_with_optimization(
                    eval_data.get("task_type", "attribution"),
                    eval_data.get("model_response", ""),
                    eval_data.get("domain", "general"),
                    eval_data.get("difficulty", "medium")
                )
        
        tasks = [process_single(eval_data) for eval_data in evaluations]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [r for r in results if not isinstance(r, Exception)]
    
    async def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        return {
            "cache_stats": self.cache.get_stats(),
            "performance_analysis": self.profiler.get_performance_analysis(),
            "concurrent_evaluations_limit": self.concurrent_evaluations._value,
            "total_patterns_learned": len(self.response_patterns)
        }


# Global optimized engine instance
optimized_engine = OptimizedEvaluationEngine()


# Convenience functions and decorators
def cached_evaluation(ttl: float = 3600.0):
    """Decorator for caching evaluation functions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key from function and args
            key_data = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            cache_key = hashlib.sha256(key_data.encode()).hexdigest()[:16]
            
            # Check cache
            cached_result = await optimized_engine.cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await optimized_engine.cache.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator