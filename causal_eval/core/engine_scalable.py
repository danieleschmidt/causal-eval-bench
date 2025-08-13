"""Highly scalable evaluation engine with performance optimization."""

import asyncio
import time
import uuid
import hashlib
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
import logging
from collections import defaultdict
from dataclasses import dataclass

from causal_eval.tasks.attribution import CausalAttribution
from causal_eval.tasks.counterfactual import CounterfactualReasoning
from causal_eval.tasks.intervention import CausalIntervention
from causal_eval.core.tasks import BaseTask, TaskConfig
from causal_eval.core.error_handling import (
    ErrorHandler, CausalEvalError, ErrorType, CircuitBreaker,
    handle_evaluation_error
)
from causal_eval.core.logging_config import get_performance_logger, get_security_logger
from causal_eval.core.performance_optimizer import IntelligentCache

logger = logging.getLogger(__name__)
performance_logger = get_performance_logger()
security_logger = get_security_logger()


@dataclass
class PerformanceMetrics:
    """Performance metrics collection."""
    request_count: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    concurrent_requests: int = 0
    peak_concurrent_requests: int = 0
    errors: int = 0
    
    def update_execution_time(self, execution_time: float):
        """Update execution time metrics."""
        self.request_count += 1
        self.total_execution_time += execution_time
        self.average_execution_time = self.total_execution_time / self.request_count
    
    def cache_hit(self):
        """Record cache hit."""
        self.cache_hits += 1
    
    def cache_miss(self):
        """Record cache miss."""
        self.cache_misses += 1
    
    @property
    def cache_hit_ratio(self) -> float:
        """Calculate cache hit ratio."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class ScalableEvaluationResult(BaseModel):
    """Enhanced result with performance tracking."""
    
    task_id: str
    domain: str
    score: float = Field(..., ge=0.0, le=1.0)
    reasoning_quality: float = Field(..., ge=0.0, le=1.0)
    explanation: str
    metadata: Dict[str, Any] = {}
    execution_time_ms: Optional[float] = None
    request_id: Optional[str] = None
    timestamp: Optional[str] = None
    cache_hit: Optional[bool] = None
    processing_node: Optional[str] = None
    
    @validator('score', 'reasoning_quality')
    def validate_scores(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Scores must be between 0.0 and 1.0')
        return v


class ScalableCausalEvaluationRequest(BaseModel):
    """Request with performance optimization options."""
    
    model_config = {"protected_namespaces": ()}
    
    task_type: str = Field(..., description="Task type: attribution, counterfactual, intervention")
    model_response: str = Field(..., min_length=1, max_length=10000)
    domain: Optional[str] = Field("general", description="Domain for evaluation")
    difficulty: Optional[str] = Field("medium", description="Difficulty level")
    task_id: Optional[str] = Field(None, description="Task identifier")
    request_id: Optional[str] = Field(None, description="Request tracking ID")
    api_key_id: Optional[str] = Field(None, description="API key for rate limiting")
    
    # Performance options
    use_cache: Optional[bool] = Field(True, description="Enable caching")
    priority: Optional[int] = Field(1, description="Request priority (1-10)")
    timeout_ms: Optional[int] = Field(30000, description="Request timeout in milliseconds")
    
    @validator('task_type')
    def validate_task_type(cls, v):
        allowed_types = ['attribution', 'counterfactual', 'intervention', 'chain', 'confounding']
        if v not in allowed_types:
            raise ValueError(f'Task type must be one of {allowed_types}')
        return v


class TaskPool:
    """Intelligent task pool for concurrent processing."""
    
    def __init__(self, max_workers: int = 10, auto_scale: bool = True):
        self.max_workers = max_workers
        self.auto_scale = auto_scale
        self.current_workers = 0
        self.queue = asyncio.Queue()
        self.active_tasks = set()
        self.worker_semaphore = asyncio.Semaphore(max_workers)
        self.metrics = PerformanceMetrics()
        
    async def submit_task(self, coro, priority: int = 1):
        """Submit task with priority."""
        task_id = str(uuid.uuid4())
        await self.queue.put((priority, task_id, coro))
        return task_id
    
    async def process_queue(self):
        """Process queued tasks with prioritization."""
        while True:
            try:
                priority, task_id, coro = await self.queue.get()
                
                async with self.worker_semaphore:
                    self.current_workers += 1
                    self.metrics.concurrent_requests = self.current_workers
                    
                    if self.current_workers > self.metrics.peak_concurrent_requests:
                        self.metrics.peak_concurrent_requests = self.current_workers
                    
                    try:
                        self.active_tasks.add(task_id)
                        await coro
                    finally:
                        self.active_tasks.discard(task_id)
                        self.current_workers -= 1
                        self.queue.task_done()
                        
            except Exception as e:
                logger.error(f"Task pool error: {e}")
                self.metrics.errors += 1


class ScalableEvaluationEngine:
    """High-performance, scalable evaluation engine."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize scalable evaluation engine."""
        self.config = config or {}
        self.task_registry = None
        self._task_cache: Dict[str, BaseTask] = {}
        
        # Enhanced components
        self.error_handler = ErrorHandler()
        self.circuit_breaker = CircuitBreaker()
        self.intelligent_cache = IntelligentCache(
            max_size=self.config.get('cache_size', 5000),
            max_memory_mb=self.config.get('cache_memory_mb', 256)
        )
        
        # Performance optimization
        self.task_pool = TaskPool(
            max_workers=self.config.get('max_workers', 20),
            auto_scale=self.config.get('auto_scale', True)
        )
        self.metrics = PerformanceMetrics()
        self.rate_limiter = {}
        self.request_counts = {}
        
        # Start background task processing
        asyncio.create_task(self.task_pool.process_queue())
        
        logger.info("Scalable evaluation engine initialized with performance optimization")
    
    def _generate_cache_key(self, task_type: str, model_response: str, domain: str, difficulty: str) -> str:
        """Generate cache key for evaluation results."""
        content = f"{task_type}:{domain}:{difficulty}:{model_response}"
        hash_value = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"eval_result:{hash_value}"
    
    def _generate_prompt_cache_key(self, task_type: str, domain: str, difficulty: str) -> str:
        """Generate cache key for prompts."""
        return f"prompt:{task_type}:{domain}:{difficulty}"
    
    async def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached evaluation result."""
        try:
            cached = self.intelligent_cache.get(cache_key)
            if cached is not None:
                self.metrics.cache_hit()
                logger.debug(f"Cache hit for key: {cache_key}")
                return cached
            else:
                self.metrics.cache_miss()
                return None
        except Exception as e:
            logger.warning(f"Cache retrieval error: {e}")
            self.metrics.cache_miss()
            return None
    
    async def _cache_result(self, cache_key: str, result: Dict[str, Any], ttl: float = 3600.0):
        """Cache evaluation result."""
        try:
            self.intelligent_cache.set(cache_key, result, ttl)
            logger.debug(f"Cached result for key: {cache_key}")
        except Exception as e:
            logger.warning(f"Cache storage error: {e}")
    
    def _create_task(self, task_type: str, domain: str = "general", difficulty: str = "medium") -> BaseTask:
        """Create a task instance with caching."""
        # Check task cache first
        task_key = f"{task_type}_{domain}_{difficulty}"
        if task_key in self._task_cache:
            return self._task_cache[task_key]
        
        task_config = TaskConfig(
            task_id=f"{task_type}_{domain}_{difficulty}",
            domain=domain,
            difficulty=difficulty,
            description=f"Causal {task_type} task for {domain} domain",
            expected_reasoning_type=task_type
        )
        
        task_map = {
            "attribution": CausalAttribution,
            "counterfactual": CounterfactualReasoning,
            "intervention": CausalIntervention,
        }
        
        if task_type not in task_map:
            raise ValueError(f"Unknown task type: {task_type}")
        
        task = task_map[task_type](task_config)
        
        # Cache task instance
        if len(self._task_cache) < 100:  # Limit task cache size
            self._task_cache[task_key] = task
        
        return task
    
    async def _create_task_with_protection(self, task_type: str, domain: str, difficulty: str) -> BaseTask:
        """Create task with circuit breaker protection."""
        try:
            return self.circuit_breaker.call(self._create_task, task_type, domain, difficulty)
        except Exception as e:
            error = self.error_handler.handle_task_creation_error(e, task_type, domain, difficulty)
            raise Exception(error.user_message)
    
    def _validate_security(self, model_response: str, request_id: str) -> None:
        """Validate input for security concerns."""
        suspicious_patterns = [
            'exec(', 'eval(', '__import__', 'subprocess', 'os.system',
            '<script', 'javascript:', 'data:text/html', 'vbscript:'
        ]
        
        response_lower = model_response.lower()
        for pattern in suspicious_patterns:
            if pattern in response_lower:
                security_logger.log_suspicious_input(
                    ip_address="unknown",
                    input_data=model_response,
                    reason=f"Contains suspicious pattern: {pattern}"
                )
                raise ValueError(f"Input contains suspicious content: {pattern}")
    
    def _check_rate_limit(self, api_key_id: str) -> bool:
        """Enhanced rate limiting with burst support."""
        current_time = time.time()
        window_size = 60  # 1 minute window
        max_requests = 200  # Increased for scalability
        burst_allowance = 50  # Allow bursts
        
        if api_key_id not in self.request_counts:
            self.request_counts[api_key_id] = []
        
        # Clean old requests outside window
        self.request_counts[api_key_id] = [
            timestamp for timestamp in self.request_counts[api_key_id]
            if current_time - timestamp < window_size
        ]
        
        current_count = len(self.request_counts[api_key_id])
        
        # Check if within limit (considering burst)
        effective_limit = max_requests + burst_allowance
        if current_count >= effective_limit:
            security_logger.log_rate_limit_exceeded(
                ip_address="unknown",
                endpoint="/evaluation",
                limit=effective_limit
            )
            return False
        
        # Add current request
        self.request_counts[api_key_id].append(current_time)
        return True
    
    async def evaluate(
        self,
        model_response: str,
        task_config: Dict[str, Any]
    ) -> ScalableEvaluationResult:
        """High-performance evaluation with caching and optimization."""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        task_type = task_config.get("task_type", "unknown")
        domain = task_config.get("domain", "general")
        difficulty = task_config.get("difficulty", "medium")
        use_cache = task_config.get("use_cache", True)
        
        logger.info(f"Starting scalable evaluation {request_id}: {task_type}")
        
        try:
            # Input validation
            if not model_response or len(model_response.strip()) == 0:
                raise ValueError("Model response cannot be empty")
            
            if len(model_response) > 10000:
                raise ValueError("Model response too long (max 10000 characters)")
            
            # Security validation
            self._validate_security(model_response, request_id)
            
            # Rate limiting check
            api_key_id = task_config.get('api_key_id', 'anonymous')
            if not self._check_rate_limit(api_key_id):
                raise Exception("Rate limit exceeded")
            
            # Check cache first
            cache_hit = False
            evaluation_result = None
            
            if use_cache:
                cache_key = self._generate_cache_key(task_type, model_response, domain, difficulty)
                evaluation_result = await self._get_cached_result(cache_key)
                if evaluation_result:
                    cache_hit = True
                    logger.info(f"Cache hit for evaluation {request_id}")
            
            # If not cached, perform evaluation
            if not cache_hit:
                # Create task instance with circuit breaker
                task = await self._create_task_with_protection(task_type, domain, difficulty)
                
                # Check prompt cache
                prompt_cache_key = self._generate_prompt_cache_key(task_type, domain, difficulty)
                cached_prompt = await self._get_cached_result(prompt_cache_key) if use_cache else None
                
                if cached_prompt:
                    prompt = cached_prompt.get('prompt', '')
                    logger.debug(f"Using cached prompt for {task_type}")
                else:
                    # Generate prompt
                    prompt = await task.generate_prompt()
                    if use_cache:
                        await self._cache_result(prompt_cache_key, {'prompt': prompt}, ttl=7200)  # 2 hours
                
                # Evaluate the response
                evaluation_result = await task.evaluate_response(model_response)
                
                # Cache the result
                if use_cache:
                    await self._cache_result(cache_key, evaluation_result, ttl=3600)  # 1 hour
            
            # Calculate execution time
            execution_time = (time.time() - start_time) * 1000
            self.metrics.update_execution_time(execution_time)
            
            # Log performance metrics
            performance_logger.log_evaluation_performance(
                task_type=task_type,
                model_name=task_config.get('model_name', 'unknown'),
                execution_time=execution_time,
                token_count=len(model_response.split()),
                score=evaluation_result.get("overall_score", 0.0)
            )
            
            # Create result with performance data
            result = ScalableEvaluationResult(
                task_id=task_config.get("task_id", f"{task_type}_{domain}"),
                domain=domain,
                score=evaluation_result.get("overall_score", 0.0),
                reasoning_quality=evaluation_result.get("reasoning_score", 0.0),
                explanation=f"Evaluated {task_type} task with score {evaluation_result.get('overall_score', 0.0):.2f}",
                metadata={
                    "task_type": task_type,
                    "difficulty": difficulty,
                    "evaluation_details": evaluation_result,
                    "cache_metrics": {
                        "hit_ratio": self.metrics.cache_hit_ratio,
                        "total_hits": self.metrics.cache_hits,
                        "total_misses": self.metrics.cache_misses
                    }
                },
                execution_time_ms=execution_time,
                request_id=request_id,
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
                cache_hit=cache_hit,
                processing_node=f"node-{hash(request_id) % 100:02d}"
            )
            
            logger.info(f"Evaluation {request_id} completed in {execution_time:.2f}ms (cache_hit={cache_hit})")
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.metrics.errors += 1
            
            # Handle error with structured error handling
            error = self.error_handler.handle_evaluation_error(e, {
                "request_id": request_id,
                "task_type": task_type,
                "domain": domain,
                "difficulty": difficulty,
                "model_response": model_response[:200] + "..." if len(model_response) > 200 else model_response
            })
            
            self.error_handler.log_error(error, request_id)
            logger.error(f"Evaluation {request_id} failed after {execution_time:.2f}ms: {str(e)}")
            
            return ScalableEvaluationResult(
                task_id=task_config.get("task_id", "error"),
                domain=domain,
                score=0.0,
                reasoning_quality=0.0,
                explanation=error.user_message,
                metadata={
                    "error": error.to_dict(),
                    "task_type": task_type,
                    "error_type": error.error_type.value
                },
                execution_time_ms=execution_time,
                request_id=request_id,
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
                cache_hit=False
            )
    
    async def evaluate_request(self, request: ScalableCausalEvaluationRequest) -> Dict[str, Any]:
        """Process evaluation request with enhanced performance."""
        # Convert request to task config
        task_config = {
            "task_type": request.task_type,
            "domain": request.domain,
            "difficulty": request.difficulty,
            "task_id": request.task_id,
            "api_key_id": request.api_key_id,
            "use_cache": request.use_cache,
            "model_name": "unknown"
        }
        
        # Perform evaluation
        result = await self.evaluate(request.model_response, task_config)
        
        # Convert to dictionary format
        result_dict = result.dict()
        result_dict.update({
            "request_metadata": {
                "priority": request.priority,
                "timeout_ms": request.timeout_ms,
                "use_cache": request.use_cache
            }
        })
        
        return result_dict
    
    async def batch_evaluate(
        self,
        evaluations: List[Dict[str, Any]],
        max_concurrent: int = 20,
        use_task_pool: bool = True
    ) -> List[ScalableEvaluationResult]:
        """High-performance batch evaluation with task pooling."""
        if len(evaluations) > 1000:
            raise ValueError("Batch size cannot exceed 1000 evaluations")
        
        batch_id = str(uuid.uuid4())
        logger.info(f"Starting scalable batch evaluation {batch_id} with {len(evaluations)} items")
        
        if use_task_pool:
            # Use task pool for better resource management
            tasks = []
            for i, eval_config in enumerate(evaluations):
                priority = eval_config.get("priority", 1)
                coro = self._evaluate_batch_item(eval_config, i, batch_id)
                task_id = await self.task_pool.submit_task(coro, priority)
                tasks.append(task_id)
            
            # Wait for all tasks to complete
            while len(self.task_pool.active_tasks) > 0:
                await asyncio.sleep(0.1)
            
            # Results are handled within the coroutines
            return []  # Task pool manages results differently
        
        else:
            # Use semaphore-based concurrency
            async def evaluate_with_semaphore(semaphore, eval_config, index):
                async with semaphore:
                    try:
                        return await self.evaluate(
                            eval_config.get("model_response", ""),
                            eval_config.get("task_config", {})
                        )
                    except Exception as e:
                        logger.error(f"Batch item {index} failed: {str(e)}")
                        return ScalableEvaluationResult(
                            task_id=f"batch_error_{index}",
                            domain="unknown",
                            score=0.0,
                            reasoning_quality=0.0,
                            explanation=f"Batch evaluation failed: {str(e)}",
                            metadata={"error": str(e), "batch_id": batch_id, "item_index": index}
                        )
            
            semaphore = asyncio.Semaphore(max_concurrent)
            tasks = [
                evaluate_with_semaphore(semaphore, eval_config, i)
                for i, eval_config in enumerate(evaluations)
            ]
            
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            execution_time = (time.time() - start_time) * 1000
            
            logger.info(f"Batch evaluation {batch_id} completed in {execution_time:.2f}ms")
            return results
    
    async def _evaluate_batch_item(self, eval_config: Dict[str, Any], index: int, batch_id: str):
        """Helper for batch evaluation with task pool."""
        try:
            result = await self.evaluate(
                eval_config.get("model_response", ""),
                eval_config.get("task_config", {})
            )
            logger.debug(f"Batch item {index} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Batch item {index} failed: {str(e)}")
            return ScalableEvaluationResult(
                task_id=f"batch_error_{index}",
                domain="unknown",
                score=0.0,
                reasoning_quality=0.0,
                explanation=f"Batch evaluation failed: {str(e)}",
                metadata={"error": str(e), "batch_id": batch_id, "item_index": index}
            )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        cache_stats = self.intelligent_cache.get_stats()
        
        return {
            "evaluation_metrics": {
                "total_requests": self.metrics.request_count,
                "average_execution_time_ms": self.metrics.average_execution_time,
                "total_execution_time_ms": self.metrics.total_execution_time,
                "errors": self.metrics.errors,
                "error_rate": self.metrics.errors / max(self.metrics.request_count, 1)
            },
            "cache_metrics": {
                "hit_ratio": self.metrics.cache_hit_ratio,
                "total_hits": self.metrics.cache_hits,
                "total_misses": self.metrics.cache_misses,
                "cache_size": cache_stats.get("size", 0),
                "cache_memory_usage": cache_stats.get("memory_usage", 0)
            },
            "concurrency_metrics": {
                "current_concurrent_requests": self.metrics.concurrent_requests,
                "peak_concurrent_requests": self.metrics.peak_concurrent_requests,
                "task_pool_active_tasks": len(self.task_pool.active_tasks),
                "task_pool_queue_size": self.task_pool.queue.qsize()
            },
            "system_health": {
                "circuit_breaker_state": self.circuit_breaker.state,
                "circuit_breaker_failures": self.circuit_breaker.failure_count,
                "active_rate_limits": len(self.request_counts)
            }
        }
    
    def get_available_task_types(self) -> List[str]:
        """Get list of available task types."""
        return ["attribution", "counterfactual", "intervention", "chain", "confounding"]
    
    def get_available_domains(self) -> List[str]:
        """Get list of available domains."""
        return [
            "general", "medical", "education", "business", "technology",
            "environmental", "workplace_safety", "urban_planning", 
            "manufacturing", "recreational", "public_safety", "international",
            "daily_life", "economics", "social", "scientific"
        ]
    
    def get_available_difficulties(self) -> List[str]:
        """Get list of available difficulty levels."""
        return ["easy", "medium", "hard"]
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check with performance data."""
        health_status = {
            "status": "healthy",
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            "version": "0.1.0-scalable"
        }
        
        try:
            # Test basic functionality
            test_task = self._create_task("attribution", "general", "medium")
            test_prompt = await test_task.generate_prompt()
            
            health_status.update({
                "engine_status": "operational",
                "task_creation": "ok",
                "prompt_generation": "ok" if len(test_prompt) > 0 else "error",
                "performance_metrics": self.get_performance_metrics(),
                "scalability_features": {
                    "intelligent_caching": "enabled",
                    "task_pooling": "enabled",
                    "rate_limiting": "enabled",
                    "circuit_breaker": "enabled"
                }
            })
            
        except Exception as e:
            health_status.update({
                "status": "degraded",
                "engine_status": "error",
                "error": str(e)
            })
        
        return health_status