"""
Advanced concurrency and resource pooling for high-performance evaluation.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from contextlib import asynccontextmanager
import threading
from queue import Queue, Empty
import weakref

logger = logging.getLogger(__name__)


class PoolType(Enum):
    """Resource pool types."""
    THREAD = "thread"
    PROCESS = "process" 
    ASYNC = "async"
    HYBRID = "hybrid"


@dataclass
class PoolConfig:
    """Pool configuration."""
    max_workers: int = 4
    max_concurrent_tasks: int = 100
    task_timeout: float = 300.0  # 5 minutes
    queue_timeout: float = 30.0
    pool_type: PoolType = PoolType.ASYNC
    auto_scale: bool = True
    min_workers: int = 1
    scale_threshold: float = 0.8  # Scale when utilization > 80%


@dataclass
class TaskResult:
    """Task execution result."""
    task_id: str
    result: Any
    execution_time: float
    error: Optional[Exception] = None
    retries: int = 0


class ResourcePool:
    """Base resource pool for managing concurrent execution."""
    
    def __init__(self, config: PoolConfig):
        """Initialize resource pool."""
        self.config = config
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.task_queue = asyncio.Queue(maxsize=config.max_concurrent_tasks)
        self.semaphore = asyncio.Semaphore(config.max_workers)
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "average_execution_time": 0.0,
            "current_active": 0,
            "peak_concurrent": 0
        }
        self._shutdown = False
        logger.info(f"Initialized {config.pool_type.value} pool with {config.max_workers} workers")
    
    async def submit_task(self, task_id: str, coro_func: Callable, *args, **kwargs) -> asyncio.Task:
        """Submit task for async execution."""
        if self._shutdown:
            raise RuntimeError("Pool is shutting down")
        
        async def execute_task():
            start_time = time.time()
            self.stats["total_tasks"] += 1
            self.stats["current_active"] += 1
            self.stats["peak_concurrent"] = max(self.stats["peak_concurrent"], self.stats["current_active"])
            
            try:
                async with self.semaphore:
                    if asyncio.iscoroutinefunction(coro_func):
                        result = await asyncio.wait_for(
                            coro_func(*args, **kwargs),
                            timeout=self.config.task_timeout
                        )
                    else:
                        # Run sync function in thread pool
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(None, coro_func, *args, **kwargs)
                    
                    execution_time = time.time() - start_time
                    self.stats["completed_tasks"] += 1
                    self._update_average_time(execution_time)
                    
                    return TaskResult(
                        task_id=task_id,
                        result=result,
                        execution_time=execution_time
                    )
                    
            except Exception as e:
                execution_time = time.time() - start_time
                self.stats["failed_tasks"] += 1
                logger.error(f"Task {task_id} failed: {e}")
                
                return TaskResult(
                    task_id=task_id,
                    result=None,
                    execution_time=execution_time,
                    error=e
                )
            finally:
                self.stats["current_active"] -= 1
                self.active_tasks.pop(task_id, None)
        
        task = asyncio.create_task(execute_task())
        self.active_tasks[task_id] = task
        return task
    
    async def submit_batch(self, tasks: List[Tuple[str, Callable, tuple, dict]]) -> List[TaskResult]:
        """Submit batch of tasks for concurrent execution."""
        submitted_tasks = []
        
        for task_id, func, args, kwargs in tasks:
            task = await self.submit_task(task_id, func, *args, **kwargs)
            submitted_tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*submitted_tasks, return_exceptions=True)
        
        return [result if isinstance(result, TaskResult) else 
                TaskResult(task_id=f"batch_{i}", result=None, execution_time=0.0, error=result)
                for i, result in enumerate(results)]
    
    def _update_average_time(self, execution_time: float):
        """Update average execution time."""
        completed = self.stats["completed_tasks"]
        current_avg = self.stats["average_execution_time"]
        self.stats["average_execution_time"] = (current_avg * (completed - 1) + execution_time) / completed
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            **self.stats,
            "config": {
                "max_workers": self.config.max_workers,
                "max_concurrent_tasks": self.config.max_concurrent_tasks,
                "pool_type": self.config.pool_type.value,
                "auto_scale": self.config.auto_scale
            },
            "utilization": self.stats["current_active"] / self.config.max_workers if self.config.max_workers > 0 else 0
        }
    
    async def shutdown(self, timeout: float = 30.0):
        """Shutdown pool gracefully."""
        logger.info("Shutting down resource pool...")
        self._shutdown = True
        
        # Wait for active tasks to complete
        if self.active_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.active_tasks.values(), return_exceptions=True),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning("Some tasks didn't complete within shutdown timeout")
                
                # Cancel remaining tasks
                for task in self.active_tasks.values():
                    task.cancel()
        
        logger.info("Resource pool shutdown complete")


class EvaluationPool(ResourcePool):
    """Specialized pool for evaluation tasks."""
    
    def __init__(self, config: PoolConfig):
        """Initialize evaluation pool."""
        super().__init__(config)
        self.evaluation_cache = {}
        self.model_locks = {}  # Per-model locks to prevent overloading
    
    async def evaluate_single(
        self, 
        task_id: str, 
        engine, 
        model_response: str, 
        task_config: Dict[str, Any]
    ) -> TaskResult:
        """Evaluate single model response."""
        model_name = task_config.get("model_name", "unknown")
        
        # Get or create model-specific lock
        if model_name not in self.model_locks:
            self.model_locks[model_name] = asyncio.Semaphore(2)  # Max 2 concurrent per model
        
        async with self.model_locks[model_name]:
            task = await self.submit_task(
                task_id,
                engine.evaluate,
                model_response,
                task_config
            )
            return await task
    
    async def evaluate_batch(
        self, 
        engine, 
        evaluations: List[Dict[str, Any]]
    ) -> List[TaskResult]:
        """Evaluate batch of model responses concurrently."""
        tasks = []
        
        for i, eval_data in enumerate(evaluations):
            task_id = eval_data.get("task_id", f"eval_{i}")
            model_response = eval_data["model_response"]
            task_config = eval_data["task_config"]
            
            tasks.append((
                task_id,
                self.evaluate_single,
                (task_id, engine, model_response, task_config),
                {}
            ))
        
        return await self.submit_batch(tasks)


class ModelAPIPool(ResourcePool):
    """Specialized pool for model API calls with rate limiting."""
    
    def __init__(self, config: PoolConfig):
        """Initialize model API pool."""
        super().__init__(config)
        self.api_limits = {}  # Per-API rate limits
        self.api_usage = {}   # Track usage per API
    
    async def call_model_api(
        self, 
        task_id: str, 
        client, 
        prompt: str, 
        **kwargs
    ) -> TaskResult:
        """Call model API with rate limiting."""
        model_name = kwargs.get("model_name", "unknown")
        
        # Implement basic rate limiting per model
        if model_name not in self.api_limits:
            self.api_limits[model_name] = asyncio.Semaphore(5)  # Max 5 concurrent per model
        
        async with self.api_limits[model_name]:
            task = await self.submit_task(
                task_id,
                client.generate_response,
                prompt,
                **kwargs
            )
            return await task
    
    async def batch_model_calls(
        self, 
        requests: List[Dict[str, Any]]
    ) -> List[TaskResult]:
        """Execute batch model API calls."""
        tasks = []
        
        for i, request in enumerate(requests):
            task_id = request.get("task_id", f"api_{i}")
            client = request["client"]
            prompt = request["prompt"]
            kwargs = request.get("kwargs", {})
            
            tasks.append((
                task_id,
                self.call_model_api,
                (task_id, client, prompt),
                kwargs
            ))
        
        return await self.submit_batch(tasks)


class AdaptivePoolManager:
    """Adaptive pool manager that scales resources based on load."""
    
    def __init__(self):
        """Initialize adaptive pool manager."""
        self.pools: Dict[str, ResourcePool] = {}
        self.monitoring_task: Optional[asyncio.Task] = None
        self._monitoring = False
        
    def create_pool(self, name: str, config: PoolConfig) -> ResourcePool:
        """Create and register a resource pool."""
        if config.pool_type == PoolType.ASYNC:
            if name == "evaluation":
                pool = EvaluationPool(config)
            elif name == "model_api":
                pool = ModelAPIPool(config)
            else:
                pool = ResourcePool(config)
        else:
            pool = ResourcePool(config)
        
        self.pools[name] = pool
        logger.info(f"Created pool '{name}' with config: {config}")
        return pool
    
    def get_pool(self, name: str) -> Optional[ResourcePool]:
        """Get pool by name."""
        return self.pools.get(name)
    
    async def start_monitoring(self, interval: float = 30.0):
        """Start adaptive monitoring and scaling."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitor_pools(interval))
        logger.info("Started adaptive pool monitoring")
    
    async def stop_monitoring(self):
        """Stop adaptive monitoring."""
        self._monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped adaptive pool monitoring")
    
    async def _monitor_pools(self, interval: float):
        """Monitor pools and adjust resources."""
        while self._monitoring:
            try:
                for name, pool in self.pools.items():
                    stats = await pool.get_stats()
                    utilization = stats["utilization"]
                    
                    if pool.config.auto_scale:
                        await self._scale_pool(name, pool, utilization)
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in pool monitoring: {e}")
                await asyncio.sleep(interval)
    
    async def _scale_pool(self, name: str, pool: ResourcePool, utilization: float):
        """Scale pool based on utilization."""
        config = pool.config
        
        if utilization > config.scale_threshold and config.max_workers < 20:
            # Scale up
            new_workers = min(config.max_workers + 2, 20)
            config.max_workers = new_workers
            pool.semaphore = asyncio.Semaphore(new_workers)
            logger.info(f"Scaled up pool '{name}' to {new_workers} workers (utilization: {utilization:.2f})")
            
        elif utilization < 0.3 and config.max_workers > config.min_workers:
            # Scale down
            new_workers = max(config.max_workers - 1, config.min_workers)
            config.max_workers = new_workers
            pool.semaphore = asyncio.Semaphore(new_workers)
            logger.info(f"Scaled down pool '{name}' to {new_workers} workers (utilization: {utilization:.2f})")
    
    async def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all pools."""
        stats = {}
        for name, pool in self.pools.items():
            stats[name] = await pool.get_stats()
        return stats
    
    async def shutdown_all(self, timeout: float = 30.0):
        """Shutdown all pools."""
        await self.stop_monitoring()
        
        shutdown_tasks = [
            pool.shutdown(timeout) for pool in self.pools.values()
        ]
        
        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        self.pools.clear()
        logger.info("All pools shutdown complete")


# Global pool manager
pool_manager = AdaptivePoolManager()


# Convenience functions
def create_evaluation_pool(max_workers: int = 4) -> EvaluationPool:
    """Create evaluation pool with default configuration."""
    config = PoolConfig(
        max_workers=max_workers,
        max_concurrent_tasks=max_workers * 10,
        pool_type=PoolType.ASYNC,
        auto_scale=True
    )
    return pool_manager.create_pool("evaluation", config)


def create_model_api_pool(max_workers: int = 3) -> ModelAPIPool:
    """Create model API pool with default configuration."""
    config = PoolConfig(
        max_workers=max_workers,
        max_concurrent_tasks=max_workers * 5,
        task_timeout=60.0,  # Shorter timeout for API calls
        pool_type=PoolType.ASYNC,
        auto_scale=True
    )
    return pool_manager.create_pool("model_api", config)


def get_pool(name: str) -> Optional[ResourcePool]:
    """Get pool by name."""
    return pool_manager.get_pool(name)


async def initialize_pools():
    """Initialize default pools."""
    # Create default pools
    create_evaluation_pool()
    create_model_api_pool()
    
    # Start monitoring
    await pool_manager.start_monitoring()
    
    logger.info("Concurrency pools initialized")


async def shutdown_pools():
    """Shutdown all pools."""
    await pool_manager.shutdown_all()


@asynccontextmanager
async def managed_pools():
    """Context manager for pool lifecycle."""
    try:
        await initialize_pools()
        yield pool_manager
    finally:
        await shutdown_pools()