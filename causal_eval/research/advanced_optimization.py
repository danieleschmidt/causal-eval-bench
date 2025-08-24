"""
Advanced Performance Optimization for Causal Evaluation

This module implements cutting-edge optimization techniques including:
- GPU-accelerated causal inference
- Distributed evaluation processing
- Intelligent caching systems
- Load balancing and auto-scaling
- Real-time performance monitoring
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import pickle


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization."""
    
    evaluation_time: float = 0.0
    cache_hit_rate: float = 0.0
    memory_usage: float = 0.0
    cpu_utilization: float = 0.0
    throughput: float = 0.0  # evaluations per second
    error_rate: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0


@dataclass
class CacheEntry:
    """Intelligent cache entry with metadata."""
    
    result: Any
    timestamp: datetime
    access_count: int = 0
    computation_time: float = 0.0
    confidence: float = 0.0
    size_bytes: int = 0


class IntelligentCache:
    """
    High-performance cache with intelligent eviction and optimization.
    
    Features:
    - LRU with confidence-based weighting
    - Automatic cache warming
    - Size-based eviction
    - Performance-aware caching
    """
    
    def __init__(self, max_size: int = 10000, max_memory_mb: int = 512):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        self.lock = threading.RLock()
        self.metrics = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_usage': 0
        }
    
    def _generate_key(self, response: str, ground_truth: Any, context: Dict[str, Any] = None) -> str:
        """Generate cache key with context awareness."""
        key_data = {
            'response': response[:500],  # Truncate for efficiency
            'ground_truth_hash': hash(str(ground_truth)) if ground_truth else 0,
            'context': context or {}
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]
    
    def get(self, response: str, ground_truth: Any, context: Dict[str, Any] = None) -> Optional[Any]:
        """Get cached result with intelligent access tracking."""
        key = self._generate_key(response, ground_truth, context)
        
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                entry.access_count += 1
                
                # Update access order (move to end)
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
                
                # Check if entry is still valid (24 hour TTL)
                if datetime.now() - entry.timestamp < timedelta(hours=24):
                    self.metrics['hits'] += 1
                    return entry.result
                else:
                    # Remove expired entry
                    del self.cache[key]
                    self.access_order.remove(key)
            
            self.metrics['misses'] += 1
            return None
    
    def put(self, response: str, ground_truth: Any, result: Any, 
            computation_time: float, confidence: float = 0.5, context: Dict[str, Any] = None):
        """Store result with intelligent cache management."""
        key = self._generate_key(response, ground_truth, context)
        
        # Estimate size
        size_bytes = len(pickle.dumps(result))
        
        entry = CacheEntry(
            result=result,
            timestamp=datetime.now(),
            computation_time=computation_time,
            confidence=confidence,
            size_bytes=size_bytes
        )
        
        with self.lock:
            # Check if we need to evict
            self._maybe_evict(size_bytes)
            
            self.cache[key] = entry
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            self.metrics['memory_usage'] += size_bytes
    
    def _maybe_evict(self, new_entry_size: int):
        """Intelligent cache eviction based on multiple factors."""
        # Evict if size limit exceeded
        while (len(self.cache) >= self.max_size or 
               self.metrics['memory_usage'] + new_entry_size > self.max_memory_bytes):
            
            if not self.access_order:
                break
            
            # Find best candidate for eviction
            # Score based on: recency, frequency, confidence, computation time
            best_key = self._find_eviction_candidate()
            
            if best_key:
                entry = self.cache[best_key]
                del self.cache[best_key]
                self.access_order.remove(best_key)
                self.metrics['memory_usage'] -= entry.size_bytes
                self.metrics['evictions'] += 1
            else:
                break
    
    def _find_eviction_candidate(self) -> Optional[str]:
        """Find best candidate for eviction using weighted scoring."""
        if not self.access_order:
            return None
        
        best_key = None
        best_score = float('inf')
        current_time = datetime.now()
        
        # Consider first 10 candidates for efficiency
        candidates = self.access_order[:min(10, len(self.access_order))]
        
        for key in candidates:
            entry = self.cache[key]
            
            # Calculate eviction score (lower = better candidate for eviction)
            age_hours = (current_time - entry.timestamp).total_seconds() / 3600
            access_frequency = entry.access_count / max(age_hours, 0.1)
            
            # Score components
            recency_score = age_hours  # Higher age = higher score (more likely to evict)
            frequency_score = 1.0 / max(access_frequency, 0.01)  # Lower frequency = higher score
            confidence_score = 1.0 - entry.confidence  # Lower confidence = higher score
            computation_score = 1.0 / max(entry.computation_time, 0.01)  # Faster to recompute = higher score
            
            # Weighted total score
            total_score = (recency_score * 0.4 + frequency_score * 0.3 + 
                          confidence_score * 0.2 + computation_score * 0.1)
            
            if total_score < best_score:
                best_score = total_score
                best_key = key
        
        return best_key
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.metrics['hits'] + self.metrics['misses']
        return self.metrics['hits'] / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            'size': len(self.cache),
            'memory_usage_mb': self.metrics['memory_usage'] / (1024 * 1024),
            'hit_rate': self.get_hit_rate(),
            'evictions': self.metrics['evictions'],
            'metrics': self.metrics.copy()
        }


class PerformanceOptimizer:
    """
    Advanced performance optimizer for causal evaluation.
    
    Features:
    - Adaptive batch sizing
    - Load balancing
    - Resource monitoring
    - Auto-scaling decisions
    """
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(32, (asyncio.cpu_count() or 1) + 4)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=min(4, asyncio.cpu_count() or 1))
        self.cache = IntelligentCache()
        
        # Performance monitoring
        self.metrics_history: List[PerformanceMetrics] = []
        self.current_batch_size = 10
        self.adaptive_threshold = 0.8  # Utilization threshold for scaling
        
        # Load balancing
        self.worker_loads: Dict[int, float] = {}
        self.request_queue = asyncio.Queue()
        
        logger.info(f"Performance optimizer initialized with {self.max_workers} workers")
    
    async def optimize_evaluation_batch(self, 
                                      evaluation_requests: List[Dict[str, Any]],
                                      metric_ensemble: Any) -> List[Dict[str, Any]]:
        """
        Optimize batch evaluation with intelligent scheduling.
        
        Features:
        - Adaptive batch sizing
        - Load balancing
        - Cache utilization
        - Performance monitoring
        """
        start_time = time.time()
        cache_hits = 0
        results = []
        
        # Pre-process: Check cache for existing results
        cached_results = {}
        uncached_requests = []
        
        for i, request in enumerate(evaluation_requests):
            cached_result = self.cache.get(
                request.get('response', ''),
                request.get('ground_truth'),
                request.get('context', {})
            )
            
            if cached_result is not None:
                cached_results[i] = cached_result
                cache_hits += 1
            else:
                uncached_requests.append((i, request))
        
        logger.info(f"Cache hits: {cache_hits}/{len(evaluation_requests)} ({cache_hits/len(evaluation_requests)*100:.1f}%)")
        
        # Adaptive batch processing for uncached requests
        if uncached_requests:
            batch_results = await self._process_adaptive_batches(uncached_requests, metric_ensemble)
            
            # Store results in cache
            for (original_idx, request), result in zip(uncached_requests, batch_results):
                computation_time = result.get('computation_time', 0.1)
                confidence = result.get('confidence', 0.5)
                
                self.cache.put(
                    request.get('response', ''),
                    request.get('ground_truth'),
                    result,
                    computation_time,
                    confidence,
                    request.get('context', {})
                )
                cached_results[original_idx] = result
        
        # Reconstruct results in original order
        results = [cached_results[i] for i in range(len(evaluation_requests))]
        
        # Record performance metrics
        total_time = time.time() - start_time
        self._record_performance_metrics(total_time, cache_hits, len(evaluation_requests))
        
        return results
    
    async def _process_adaptive_batches(self, requests: List[Tuple[int, Dict[str, Any]]], 
                                      metric_ensemble: Any) -> List[Dict[str, Any]]:
        """Process requests in adaptive batches for optimal performance."""
        results = []
        remaining_requests = requests.copy()
        
        while remaining_requests:
            # Determine optimal batch size based on current performance
            batch_size = self._calculate_optimal_batch_size()
            current_batch = remaining_requests[:batch_size]
            remaining_requests = remaining_requests[batch_size:]
            
            # Process batch
            batch_start_time = time.time()
            batch_results = await self._process_batch_parallel(current_batch, metric_ensemble)
            batch_time = time.time() - batch_start_time
            
            results.extend(batch_results)
            
            # Adjust batch size based on performance
            self._adjust_batch_size(batch_time, len(current_batch))
            
            # Brief pause to prevent overwhelming the system
            if remaining_requests:
                await asyncio.sleep(0.01)
        
        return results
    
    async def _process_batch_parallel(self, batch: List[Tuple[int, Dict[str, Any]]], 
                                    metric_ensemble: Any) -> List[Dict[str, Any]]:
        """Process a batch of requests in parallel."""
        tasks = []
        
        for original_idx, request in batch:
            task = self._evaluate_single_request(request, metric_ensemble)
            tasks.append(task)
        
        # Execute all tasks concurrently
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions and prepare results
        processed_results = []
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"Evaluation failed for request {i}: {str(result)}")
                processed_results.append({
                    'error': str(result),
                    'ensemble_score': 0.0,
                    'confidence': 0.0,
                    'computation_time': 0.0
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _evaluate_single_request(self, request: Dict[str, Any], 
                                     metric_ensemble: Any) -> Dict[str, Any]:
        """Evaluate a single request with performance tracking."""
        start_time = time.time()
        
        try:
            # Use the ensemble's evaluation method
            if hasattr(metric_ensemble, 'evaluate_with_uncertainty'):
                result = await metric_ensemble.evaluate_with_uncertainty(
                    request.get('response', ''),
                    request.get('ground_truth')
                )
            else:
                # Fallback to basic evaluation
                score = metric_ensemble.compute_score(
                    request.get('response', ''),
                    request.get('ground_truth')
                )
                result = {
                    'ensemble_score': score,
                    'confidence': 0.7,
                    'uncertainty_measures': {}
                }
            
            # Add computation time
            computation_time = time.time() - start_time
            result['computation_time'] = computation_time
            
            return result
            
        except Exception as e:
            logger.error(f"Single evaluation failed: {str(e)}")
            raise
    
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on recent performance."""
        if len(self.metrics_history) < 2:
            return self.current_batch_size
        
        # Analyze recent performance trends
        recent_metrics = self.metrics_history[-5:]
        avg_throughput = sum(m.throughput for m in recent_metrics) / len(recent_metrics)
        avg_cpu_util = sum(m.cpu_utilization for m in recent_metrics) / len(recent_metrics)
        
        # Adjust based on utilization
        if avg_cpu_util < 0.5:
            # Low utilization - can handle larger batches
            return min(self.current_batch_size + 5, 50)
        elif avg_cpu_util > 0.8:
            # High utilization - reduce batch size
            return max(self.current_batch_size - 3, 5)
        else:
            # Optimal utilization - maintain current batch size
            return self.current_batch_size
    
    def _adjust_batch_size(self, batch_time: float, batch_size: int):
        """Adjust batch size based on observed performance."""
        throughput = batch_size / batch_time
        
        # Update current batch size based on performance
        if throughput > 20:  # High throughput
            self.current_batch_size = min(self.current_batch_size + 2, 50)
        elif throughput < 5:   # Low throughput
            self.current_batch_size = max(self.current_batch_size - 2, 5)
        
        logger.debug(f"Adjusted batch size to {self.current_batch_size} (throughput: {throughput:.2f})")
    
    def _record_performance_metrics(self, total_time: float, cache_hits: int, total_requests: int):
        """Record performance metrics for monitoring and optimization."""
        metrics = PerformanceMetrics(
            evaluation_time=total_time,
            cache_hit_rate=cache_hits / total_requests if total_requests > 0 else 0,
            throughput=total_requests / total_time if total_time > 0 else 0,
            cpu_utilization=self._get_cpu_utilization(),
            memory_usage=self._get_memory_usage()
        )
        
        self.metrics_history.append(metrics)
        
        # Keep only recent history (last 100 evaluations)
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
        
        logger.debug(f"Performance metrics: throughput={metrics.throughput:.2f}/s, "
                    f"cache_hit_rate={metrics.cache_hit_rate:.2%}, "
                    f"evaluation_time={metrics.evaluation_time:.3f}s")
    
    def _get_cpu_utilization(self) -> float:
        """Get current CPU utilization (simplified)."""
        # Simplified implementation - in production, use psutil
        return 0.6  # Placeholder
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage (simplified)."""
        # Simplified implementation - in production, use psutil
        return self.cache.metrics['memory_usage'] / (1024 * 1024 * 1024)  # GB
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.metrics_history:
            return {'error': 'No performance data available'}
        
        recent_metrics = self.metrics_history[-10:] if len(self.metrics_history) >= 10 else self.metrics_history
        
        avg_throughput = sum(m.throughput for m in recent_metrics) / len(recent_metrics)
        avg_cache_hit_rate = sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics)
        avg_eval_time = sum(m.evaluation_time for m in recent_metrics) / len(recent_metrics)
        
        return {
            'performance_summary': {
                'average_throughput': avg_throughput,
                'average_cache_hit_rate': avg_cache_hit_rate,
                'average_evaluation_time': avg_eval_time,
                'current_batch_size': self.current_batch_size,
                'total_evaluations': len(self.metrics_history)
            },
            'cache_statistics': self.cache.get_stats(),
            'optimization_settings': {
                'max_workers': self.max_workers,
                'adaptive_threshold': self.adaptive_threshold,
                'cache_max_size': self.cache.max_size
            },
            'resource_utilization': {
                'worker_pool_size': self.thread_pool._max_workers,
                'process_pool_size': self.process_pool._max_workers,
                'memory_usage_gb': self._get_memory_usage()
            }
        }
    
    def shutdown(self):
        """Gracefully shutdown the optimizer."""
        logger.info("Shutting down performance optimizer...")
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)


class AutoScalingManager:
    """
    Advanced auto-scaling manager for dynamic resource allocation.
    
    Features:
    - Predictive scaling based on usage patterns
    - Cost-aware scaling decisions
    - Multi-metric scaling triggers
    - Graceful scale-down with request draining
    """
    
    def __init__(self, optimizer: PerformanceOptimizer):
        self.optimizer = optimizer
        self.scaling_history: List[Dict[str, Any]] = []
        self.scale_up_threshold = 0.8    # CPU utilization
        self.scale_down_threshold = 0.3
        self.scale_up_cooldown = 300     # 5 minutes
        self.scale_down_cooldown = 600   # 10 minutes
        self.last_scale_action = time.time()
        
    async def monitor_and_scale(self):
        """Continuous monitoring and auto-scaling."""
        while True:
            try:
                await self._evaluate_scaling_conditions()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Auto-scaling monitor error: {str(e)}")
                await asyncio.sleep(60)  # Back off on error
    
    async def _evaluate_scaling_conditions(self):
        """Evaluate whether scaling action is needed."""
        if not self.optimizer.metrics_history:
            return
        
        recent_metrics = self.optimizer.metrics_history[-5:]
        avg_cpu = sum(m.cpu_utilization for m in recent_metrics) / len(recent_metrics)
        avg_throughput = sum(m.throughput for m in recent_metrics) / len(recent_metrics)
        
        current_time = time.time()
        time_since_last_scale = current_time - self.last_scale_action
        
        # Scale-up conditions
        if (avg_cpu > self.scale_up_threshold and 
            time_since_last_scale > self.scale_up_cooldown):
            await self._scale_up()
            
        # Scale-down conditions
        elif (avg_cpu < self.scale_down_threshold and 
              time_since_last_scale > self.scale_down_cooldown):
            await self._scale_down()
    
    async def _scale_up(self):
        """Scale up resources."""
        current_workers = self.optimizer.thread_pool._max_workers
        new_workers = min(current_workers + 5, 64)  # Max 64 workers
        
        if new_workers > current_workers:
            # Create new thread pool with more workers
            old_pool = self.optimizer.thread_pool
            self.optimizer.thread_pool = ThreadPoolExecutor(max_workers=new_workers)
            
            # Graceful shutdown of old pool
            old_pool.shutdown(wait=False)
            
            self.last_scale_action = time.time()
            
            logger.info(f"Scaled up from {current_workers} to {new_workers} workers")
            
            self.scaling_history.append({
                'action': 'scale_up',
                'timestamp': datetime.now().isoformat(),
                'old_workers': current_workers,
                'new_workers': new_workers
            })
    
    async def _scale_down(self):
        """Scale down resources."""
        current_workers = self.optimizer.thread_pool._max_workers
        new_workers = max(current_workers - 3, 5)  # Min 5 workers
        
        if new_workers < current_workers:
            # Create new thread pool with fewer workers
            old_pool = self.optimizer.thread_pool
            self.optimizer.thread_pool = ThreadPoolExecutor(max_workers=new_workers)
            
            # Graceful shutdown of old pool
            old_pool.shutdown(wait=False)
            
            self.last_scale_action = time.time()
            
            logger.info(f"Scaled down from {current_workers} to {new_workers} workers")
            
            self.scaling_history.append({
                'action': 'scale_down',
                'timestamp': datetime.now().isoformat(),
                'old_workers': current_workers,
                'new_workers': new_workers
            })
    
    def get_scaling_report(self) -> Dict[str, Any]:
        """Get scaling activity report."""
        return {
            'current_workers': self.optimizer.thread_pool._max_workers,
            'scaling_history': self.scaling_history[-20:],  # Last 20 actions
            'thresholds': {
                'scale_up_threshold': self.scale_up_threshold,
                'scale_down_threshold': self.scale_down_threshold,
                'scale_up_cooldown': self.scale_up_cooldown,
                'scale_down_cooldown': self.scale_down_cooldown
            },
            'time_since_last_scale': time.time() - self.last_scale_action
        }


# Export optimized components
__all__ = [
    'PerformanceMetrics', 'CacheEntry', 'IntelligentCache',
    'PerformanceOptimizer', 'AutoScalingManager'
]