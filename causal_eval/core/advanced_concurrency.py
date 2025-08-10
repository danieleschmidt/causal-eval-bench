"""Advanced concurrency management with intelligent load balancing and resource optimization."""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Callable, Awaitable, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import deque, defaultdict
import weakref
import json

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels for intelligent scheduling."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class ResourceType(Enum):
    """Types of system resources to manage."""
    CPU_BOUND = "cpu_bound"
    IO_BOUND = "io_bound"
    MEMORY_INTENSIVE = "memory_intensive"
    NETWORK_INTENSIVE = "network_intensive"


@dataclass
class ConcurrentTask:
    """Represents a task in the concurrent execution system."""
    id: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    resource_type: ResourceType = ResourceType.IO_BOUND
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    estimated_duration: float = 1.0
    dependencies: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[Exception] = None


@dataclass
class WorkerStats:
    """Statistics for worker performance."""
    worker_id: str
    tasks_completed: int = 0
    total_execution_time: float = 0.0
    average_task_time: float = 0.0
    errors_encountered: int = 0
    last_active: float = field(default_factory=time.time)
    cpu_utilization: float = 0.0
    memory_usage_mb: float = 0.0


class IntelligentScheduler:
    """Advanced task scheduler with load balancing and resource optimization."""
    
    def __init__(self, max_workers: int = None):
        """Initialize intelligent scheduler."""
        self.max_workers = max_workers or min(32, (asyncio.get_event_loop()._thread_pool_executor.max_workers if hasattr(asyncio.get_event_loop(), '_thread_pool_executor') else 8))
        
        # Task queues by priority
        self.task_queues: Dict[TaskPriority, deque] = {
            priority: deque() for priority in TaskPriority
        }
        
        # Resource pools
        self.cpu_pool = ProcessPoolExecutor(max_workers=min(8, self.max_workers // 2))
        self.io_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.memory_semaphore = asyncio.Semaphore(min(10, self.max_workers))
        self.network_semaphore = asyncio.Semaphore(min(20, self.max_workers))
        
        # Worker management
        self.active_tasks: Dict[str, ConcurrentTask] = {}
        self.completed_tasks: Dict[str, ConcurrentTask] = {}
        self.worker_stats: Dict[str, WorkerStats] = {}
        
        # Performance tracking
        self.total_tasks_scheduled = 0
        self.total_tasks_completed = 0
        self.total_execution_time = 0.0
        self.queue_wait_times: List[float] = []
        
        # Load balancing
        self.resource_utilization: Dict[ResourceType, float] = {
            ResourceType.CPU_BOUND: 0.0,
            ResourceType.IO_BOUND: 0.0,
            ResourceType.MEMORY_INTENSIVE: 0.0,
            ResourceType.NETWORK_INTENSIVE: 0.0
        }
        
        logger.info(f"Intelligent scheduler initialized with {self.max_workers} max workers")
    
    async def schedule_task(self, task: ConcurrentTask) -> str:
        """Schedule a task for execution with intelligent prioritization."""
        self.total_tasks_scheduled += 1
        
        # Add to appropriate priority queue
        self.task_queues[task.priority].append(task)
        
        logger.debug(f"Scheduled task {task.id} with priority {task.priority.name}")
        
        # Trigger task execution
        asyncio.create_task(self._execute_next_task())
        
        return task.id
    
    async def _execute_next_task(self) -> None:
        """Execute the next highest priority task with resource optimization."""
        # Find highest priority task
        task = None
        for priority in reversed(list(TaskPriority)):
            if self.task_queues[priority]:
                task = self.task_queues[priority].popleft()
                break
        
        if not task:
            return
        
        # Check dependencies
        if task.dependencies and not all(dep in self.completed_tasks for dep in task.dependencies):
            # Re-queue task if dependencies not met
            self.task_queues[task.priority].append(task)
            return
        
        # Select appropriate execution method based on resource type
        task.started_at = time.time()
        queue_wait_time = task.started_at - task.created_at
        self.queue_wait_times.append(queue_wait_time)
        
        self.active_tasks[task.id] = task
        
        try:
            if task.resource_type == ResourceType.CPU_BOUND:
                await self._execute_cpu_bound_task(task)
            elif task.resource_type == ResourceType.MEMORY_INTENSIVE:
                await self._execute_memory_intensive_task(task)
            elif task.resource_type == ResourceType.NETWORK_INTENSIVE:
                await self._execute_network_intensive_task(task)
            else:  # IO_BOUND (default)
                await self._execute_io_bound_task(task)
                
        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            task.error = e
            
            # Retry if possible
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.started_at = None
                self.task_queues[task.priority].append(task)
                logger.info(f"Retrying task {task.id} (attempt {task.retry_count + 1}/{task.max_retries + 1})")
            else:
                self._complete_task(task)
        
        finally:
            if task.id in self.active_tasks:
                del self.active_tasks[task.id]
    
    async def _execute_cpu_bound_task(self, task: ConcurrentTask) -> None:
        """Execute CPU-bound task using process pool."""
        async with self.memory_semaphore:  # Limit concurrent CPU tasks
            loop = asyncio.get_event_loop()
            
            # Execute in process pool for true parallelism
            if asyncio.iscoroutinefunction(task.func):
                # Convert coroutine to regular function for process pool
                result = await self._run_async_in_executor(task.func, *task.args, **task.kwargs)
            else:
                result = await loop.run_in_executor(self.cpu_pool, task.func, *task.args, **task.kwargs)
            
            task.result = result
            self._complete_task(task)
    
    async def _execute_io_bound_task(self, task: ConcurrentTask) -> None:
        """Execute I/O-bound task using thread pool."""
        loop = asyncio.get_event_loop()
        
        if asyncio.iscoroutinefunction(task.func):
            # Direct async execution
            result = await task.func(*task.args, **task.kwargs)
        else:
            # Execute in thread pool
            result = await loop.run_in_executor(self.io_pool, task.func, *task.args, **task.kwargs)
        
        task.result = result
        self._complete_task(task)
    
    async def _execute_memory_intensive_task(self, task: ConcurrentTask) -> None:
        """Execute memory-intensive task with resource limits."""
        async with self.memory_semaphore:
            await self._execute_io_bound_task(task)
    
    async def _execute_network_intensive_task(self, task: ConcurrentTask) -> None:
        """Execute network-intensive task with connection limits."""
        async with self.network_semaphore:
            await self._execute_io_bound_task(task)
    
    async def _run_async_in_executor(self, async_func: Callable, *args, **kwargs) -> Any:
        """Run async function in thread pool executor."""
        def run_async_func():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(async_func(*args, **kwargs))
            finally:
                loop.close()
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.io_pool, run_async_func)
    
    def _complete_task(self, task: ConcurrentTask) -> None:
        """Mark task as completed and update statistics."""
        task.completed_at = time.time()
        execution_time = task.completed_at - (task.started_at or task.created_at)
        
        self.completed_tasks[task.id] = task
        self.total_tasks_completed += 1
        self.total_execution_time += execution_time
        
        # Update worker stats
        worker_id = f"worker_{task.resource_type.value}"
        if worker_id not in self.worker_stats:
            self.worker_stats[worker_id] = WorkerStats(worker_id=worker_id)
        
        stats = self.worker_stats[worker_id]
        stats.tasks_completed += 1
        stats.total_execution_time += execution_time
        stats.average_task_time = stats.total_execution_time / stats.tasks_completed
        stats.last_active = time.time()
        
        if task.error:
            stats.errors_encountered += 1
        
        logger.debug(f"Completed task {task.id} in {execution_time:.3f}s")
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Wait for specific task to complete."""
        start_time = time.time()
        
        while task_id not in self.completed_tasks:
            if timeout and (time.time() - start_time) > timeout:
                raise asyncio.TimeoutError(f"Task {task_id} timed out after {timeout}s")
            
            await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
        
        task = self.completed_tasks[task_id]
        if task.error:
            raise task.error
        
        return task.result
    
    async def wait_for_all_tasks(self, task_ids: List[str], timeout: Optional[float] = None) -> List[Any]:
        """Wait for all specified tasks to complete."""
        results = []
        
        for task_id in task_ids:
            result = await self.wait_for_task(task_id, timeout)
            results.append(result)
        
        return results
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get comprehensive scheduler statistics."""
        current_time = time.time()
        
        # Queue statistics
        queue_lengths = {
            priority.name: len(queue) for priority, queue in self.task_queues.items()
        }
        total_queued = sum(queue_lengths.values())
        
        # Performance metrics
        avg_execution_time = (self.total_execution_time / self.total_tasks_completed) if self.total_tasks_completed > 0 else 0.0
        avg_queue_wait = sum(self.queue_wait_times) / len(self.queue_wait_times) if self.queue_wait_times else 0.0
        
        completion_rate = self.total_tasks_completed / max(1, self.total_tasks_scheduled)
        
        # Resource utilization
        active_by_resource = defaultdict(int)
        for task in self.active_tasks.values():
            active_by_resource[task.resource_type.value] += 1
        
        return {
            "total_scheduled": self.total_tasks_scheduled,
            "total_completed": self.total_tasks_completed,
            "currently_active": len(self.active_tasks),
            "currently_queued": total_queued,
            "completion_rate": completion_rate,
            "queue_lengths": queue_lengths,
            "performance": {
                "avg_execution_time_s": avg_execution_time,
                "avg_queue_wait_s": avg_queue_wait,
                "total_execution_time_s": self.total_execution_time
            },
            "resource_utilization": {
                "cpu_pool_active": active_by_resource["cpu_bound"],
                "io_pool_active": active_by_resource["io_bound"],
                "memory_tasks_active": active_by_resource["memory_intensive"],
                "network_tasks_active": active_by_resource["network_intensive"]
            },
            "worker_stats": {
                worker_id: {
                    "tasks_completed": stats.tasks_completed,
                    "average_task_time": stats.average_task_time,
                    "errors": stats.errors_encountered,
                    "last_active_ago": current_time - stats.last_active
                }
                for worker_id, stats in self.worker_stats.items()
            }
        }


class LoadBalancer:
    """Intelligent load balancer for distributing evaluation requests."""
    
    def __init__(self, max_concurrent_evaluations: int = 100):
        """Initialize load balancer."""
        self.max_concurrent_evaluations = max_concurrent_evaluations
        self.scheduler = IntelligentScheduler(max_workers=max_concurrent_evaluations)
        
        # Request routing
        self.active_evaluations = 0
        self.evaluation_semaphore = asyncio.Semaphore(max_concurrent_evaluations)
        
        # Performance tracking
        self.request_counts_by_type: Dict[str, int] = defaultdict(int)
        self.average_processing_times: Dict[str, float] = defaultdict(float)
        self.error_rates: Dict[str, float] = defaultdict(float)
        
        logger.info(f"Load balancer initialized with {max_concurrent_evaluations} max concurrent evaluations")
    
    async def distribute_evaluation(self, 
                                  task_type: str,
                                  model_response: str,
                                  domain: str = "general",
                                  difficulty: str = "medium",
                                  priority: TaskPriority = TaskPriority.NORMAL,
                                  context: Dict[str, Any] = None) -> str:
        """Distribute evaluation request with intelligent load balancing."""
        
        # Create task
        task_id = f"eval_{task_type}_{int(time.time() * 1000)}"
        
        # Determine resource type based on task characteristics
        resource_type = self._classify_resource_type(task_type, len(model_response), domain, difficulty)
        
        # Estimate duration based on historical data
        estimated_duration = self._estimate_task_duration(task_type, domain, difficulty, len(model_response))
        
        # Create concurrent task
        from causal_eval.core.performance_optimizer import optimized_engine
        
        concurrent_task = ConcurrentTask(
            id=task_id,
            func=optimized_engine.evaluate_with_optimization,
            kwargs={
                "task_type": task_type,
                "model_response": model_response,
                "domain": domain,
                "difficulty": difficulty,
                "context": context or {}
            },
            priority=priority,
            resource_type=resource_type,
            estimated_duration=estimated_duration,
            timeout=30.0,  # 30 second timeout
            max_retries=2
        )
        
        # Schedule task
        await self.scheduler.schedule_task(concurrent_task)
        
        # Update tracking
        self.request_counts_by_type[task_type] += 1
        
        return task_id
    
    def _classify_resource_type(self, task_type: str, response_length: int, domain: str, difficulty: str) -> ResourceType:
        """Classify resource type based on task characteristics."""
        
        # Large responses are memory intensive
        if response_length > 5000:
            return ResourceType.MEMORY_INTENSIVE
        
        # Complex tasks are CPU bound
        if difficulty == "hard" or task_type in ["chain", "confounding"]:
            return ResourceType.CPU_BOUND
        
        # Medical domain requires more processing
        if domain == "medical":
            return ResourceType.CPU_BOUND
        
        # Default to I/O bound (most evaluations)
        return ResourceType.IO_BOUND
    
    def _estimate_task_duration(self, task_type: str, domain: str, difficulty: str, response_length: int) -> float:
        """Estimate task duration based on characteristics and historical data."""
        
        # Base time by task type
        base_times = {
            "attribution": 0.5,
            "counterfactual": 0.8,
            "intervention": 0.6,
            "chain": 1.2,
            "confounding": 1.0
        }
        
        base_time = base_times.get(task_type, 0.7)
        
        # Adjust for difficulty
        difficulty_multipliers = {
            "easy": 0.7,
            "medium": 1.0,
            "hard": 1.5
        }
        base_time *= difficulty_multipliers.get(difficulty, 1.0)
        
        # Adjust for domain complexity
        domain_multipliers = {
            "medical": 1.3,
            "business": 1.1,
            "general": 1.0,
            "education": 0.9
        }
        base_time *= domain_multipliers.get(domain, 1.0)
        
        # Adjust for response length
        if response_length > 2000:
            base_time *= 1.2
        elif response_length > 1000:
            base_time *= 1.1
        
        # Add historical adjustment if available
        if task_type in self.average_processing_times:
            historical_avg = self.average_processing_times[task_type]
            base_time = (base_time + historical_avg) / 2
        
        return max(0.1, base_time)  # Minimum 0.1 seconds
    
    async def get_evaluation_result(self, task_id: str, timeout: float = 30.0) -> Dict[str, Any]:
        """Get evaluation result with timeout handling."""
        start_time = time.time()
        
        try:
            result = await self.scheduler.wait_for_task(task_id, timeout)
            
            # Update performance metrics
            duration = time.time() - start_time
            task = self.scheduler.completed_tasks.get(task_id)
            if task and 'task_type' in task.kwargs:
                task_type = task.kwargs['task_type']
                current_avg = self.average_processing_times.get(task_type, 0.0)
                self.average_processing_times[task_type] = (current_avg + duration) / 2
            
            return result
            
        except Exception as e:
            # Update error rates
            task = self.scheduler.completed_tasks.get(task_id) or self.scheduler.active_tasks.get(task_id)
            if task and 'task_type' in task.kwargs:
                task_type = task.kwargs['task_type']
                self.error_rates[task_type] = self.error_rates.get(task_type, 0.0) + 0.1
            
            raise e
    
    async def batch_distribute_evaluations(self, evaluations: List[Dict[str, Any]]) -> List[str]:
        """Distribute batch evaluations with intelligent scheduling."""
        
        task_ids = []
        
        # Sort evaluations by estimated processing time (shortest first)
        sorted_evaluations = sorted(evaluations, key=lambda x: self._estimate_task_duration(
            x.get("task_type", "attribution"),
            x.get("domain", "general"), 
            x.get("difficulty", "medium"),
            len(x.get("model_response", ""))
        ))
        
        # Distribute with priority adjustment
        for i, eval_request in enumerate(sorted_evaluations):
            # Higher priority for later items in batch to balance load
            priority = TaskPriority.NORMAL if i < len(sorted_evaluations) // 2 else TaskPriority.HIGH
            
            task_id = await self.distribute_evaluation(
                eval_request.get("task_type", "attribution"),
                eval_request.get("model_response", ""),
                eval_request.get("domain", "general"),
                eval_request.get("difficulty", "medium"),
                priority,
                {"batch_index": i, "batch_size": len(evaluations)}
            )
            
            task_ids.append(task_id)
        
        return task_ids
    
    async def get_batch_results(self, task_ids: List[str], timeout: float = 60.0) -> List[Dict[str, Any]]:
        """Get batch evaluation results with optimized waiting."""
        
        # Use scheduler's batch waiting for efficiency
        results = await self.scheduler.wait_for_all_tasks(task_ids, timeout)
        return results
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get comprehensive load balancer statistics."""
        scheduler_stats = self.scheduler.get_scheduler_stats()
        
        return {
            "load_balancer": {
                "max_concurrent_evaluations": self.max_concurrent_evaluations,
                "active_evaluations": len(self.scheduler.active_tasks),
                "request_counts": dict(self.request_counts_by_type),
                "average_processing_times": dict(self.average_processing_times),
                "error_rates": dict(self.error_rates)
            },
            "scheduler": scheduler_stats,
            "performance_summary": {
                "total_requests_processed": scheduler_stats["total_completed"],
                "current_load_percentage": (len(self.scheduler.active_tasks) / self.max_concurrent_evaluations) * 100,
                "average_response_time": scheduler_stats["performance"]["avg_execution_time_s"],
                "completion_rate": scheduler_stats["completion_rate"]
            }
        }


# Global load balancer instance
load_balancer = LoadBalancer(max_concurrent_evaluations=100)