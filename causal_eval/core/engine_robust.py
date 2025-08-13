"""Enhanced evaluation engine with robustness features."""

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
import logging

from causal_eval.tasks.attribution import CausalAttribution
from causal_eval.tasks.counterfactual import CounterfactualReasoning
from causal_eval.tasks.intervention import CausalIntervention
from causal_eval.core.tasks import BaseTask, TaskConfig
from causal_eval.core.error_handling import (
    ErrorHandler, CausalEvalError, ErrorType, CircuitBreaker,
    handle_evaluation_error
)
from causal_eval.core.logging_config import get_performance_logger, get_security_logger

logger = logging.getLogger(__name__)
performance_logger = get_performance_logger()
security_logger = get_security_logger()


class RobustEvaluationResult(BaseModel):
    """Enhanced result of a causal evaluation task."""
    
    task_id: str
    domain: str
    score: float = Field(..., ge=0.0, le=1.0, description="Evaluation score between 0 and 1")
    reasoning_quality: float = Field(..., ge=0.0, le=1.0, description="Quality of reasoning between 0 and 1")
    explanation: str
    metadata: Dict[str, Any] = {}
    execution_time_ms: Optional[float] = None
    request_id: Optional[str] = None
    timestamp: Optional[str] = None
    
    @validator('score', 'reasoning_quality')
    def validate_scores(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Scores must be between 0.0 and 1.0')
        return v


class RobustCausalEvaluationRequest(BaseModel):
    """Request for causal evaluation with validation."""
    
    model_config = {"protected_namespaces": ()}
    
    task_type: str = Field(..., description="Task type: attribution, counterfactual, intervention")
    model_response: str = Field(..., min_length=1, max_length=10000, description="Model response to evaluate")
    domain: Optional[str] = Field("general", description="Domain for evaluation")
    difficulty: Optional[str] = Field("medium", description="Difficulty level: easy, medium, hard")
    task_id: Optional[str] = Field(None, description="Optional task identifier")
    request_id: Optional[str] = Field(None, description="Request tracking ID")
    api_key_id: Optional[str] = Field(None, description="API key for rate limiting")
    
    @validator('task_type')
    def validate_task_type(cls, v):
        allowed_types = ['attribution', 'counterfactual', 'intervention', 'chain', 'confounding']
        if v not in allowed_types:
            raise ValueError(f'Task type must be one of {allowed_types}')
        return v
    
    @validator('difficulty')
    def validate_difficulty(cls, v):
        allowed_difficulties = ['easy', 'medium', 'hard']
        if v not in allowed_difficulties:
            raise ValueError(f'Difficulty must be one of {allowed_difficulties}')
        return v
    
    @validator('model_response')
    def validate_response_content(cls, v):
        # Basic security validation
        suspicious_patterns = ['<script', 'javascript:', 'eval(', 'exec(']
        v_lower = v.lower()
        for pattern in suspicious_patterns:
            if pattern in v_lower:
                raise ValueError('Response contains potentially malicious content')
        return v


class RobustEvaluationEngine:
    """Core engine for running causal evaluation tasks with robustness features."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the evaluation engine."""
        self.config = config or {}
        self.task_registry = None  # Will be injected
        self._task_cache: Dict[str, BaseTask] = {}
        self.error_handler = ErrorHandler()
        self.circuit_breaker = CircuitBreaker()
        self.rate_limiter = {}
        self.request_counts = {}
        logger.info("Robust evaluation engine initialized with enhanced features")
    
    def _create_task(self, task_type: str, domain: str = "general", difficulty: str = "medium") -> BaseTask:
        """Create a task instance of the specified type."""
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
        
        return task_map[task_type](task_config)
    
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
        """Check if request is within rate limits."""
        current_time = time.time()
        window_size = 60  # 1 minute window
        max_requests = 100  # Max requests per minute
        
        if api_key_id not in self.request_counts:
            self.request_counts[api_key_id] = []
        
        # Clean old requests outside window
        self.request_counts[api_key_id] = [
            timestamp for timestamp in self.request_counts[api_key_id]
            if current_time - timestamp < window_size
        ]
        
        # Check if within limit
        if len(self.request_counts[api_key_id]) >= max_requests:
            security_logger.log_rate_limit_exceeded(
                ip_address="unknown",
                endpoint="/evaluation",
                limit=max_requests
            )
            return False
        
        # Add current request
        self.request_counts[api_key_id].append(current_time)
        return True
    
    async def evaluate(
        self,
        model_response: str,
        task_config: Dict[str, Any]
    ) -> RobustEvaluationResult:
        """Evaluate a model's response with comprehensive error handling."""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        task_type = task_config.get("task_type", task_config.get("task_id", "unknown"))
        domain = task_config.get("domain", "general")
        difficulty = task_config.get("difficulty", "medium")
        
        logger.info(f"Starting evaluation {request_id}: {task_type} task for domain: {domain}")
        
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
            
            # Create task instance with circuit breaker
            task = await self._create_task_with_protection(task_type, domain, difficulty)
            
            # Generate prompt (to get scenario context)
            prompt = await task.generate_prompt()
            
            # Evaluate the response
            evaluation_result = await task.evaluate_response(model_response)
            
            # Calculate execution time
            execution_time = (time.time() - start_time) * 1000
            
            # Log performance metrics
            performance_logger.log_evaluation_performance(
                task_type=task_type,
                model_name=task_config.get('model_name', 'unknown'),
                execution_time=execution_time,
                token_count=len(model_response.split()),
                score=evaluation_result.get("overall_score", 0.0)
            )
            
            # Convert to standardized format
            result = RobustEvaluationResult(
                task_id=task_config.get("task_id", f"{task_type}_{domain}"),
                domain=domain,
                score=evaluation_result.get("overall_score", 0.0),
                reasoning_quality=evaluation_result.get("reasoning_score", 0.0),
                explanation=f"Evaluated {task_type} task with score {evaluation_result.get('overall_score', 0.0):.2f}",
                metadata={
                    "task_type": task_type,
                    "difficulty": difficulty,
                    "evaluation_details": evaluation_result,
                    "prompt_generated": prompt[:200] + "..." if len(prompt) > 200 else prompt
                },
                execution_time_ms=execution_time,
                request_id=request_id,
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
            )
            
            logger.info(f"Evaluation {request_id} completed successfully in {execution_time:.2f}ms")
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
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
            
            return RobustEvaluationResult(
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
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
            )
    
    async def evaluate_request(self, request: RobustCausalEvaluationRequest) -> Dict[str, Any]:
        """Evaluate a causal evaluation request with comprehensive validation."""
        start_time = time.time()
        request_id = request.request_id or str(uuid.uuid4())
        
        try:
            # Validate request
            if not request.model_response or len(request.model_response.strip()) == 0:
                raise ValueError("Model response cannot be empty")
            
            # Security validation
            self._validate_security(request.model_response, request_id)
            
            # Rate limiting
            if request.api_key_id and not self._check_rate_limit(request.api_key_id):
                raise Exception("Rate limit exceeded")
            
            # Create task instance with protection
            task = await self._create_task_with_protection(
                request.task_type, 
                request.domain or "general", 
                request.difficulty or "medium"
            )
            
            # Generate prompt for context
            prompt = await task.generate_prompt()
            
            # Evaluate the response
            evaluation_result = await task.evaluate_response(request.model_response)
            
            # Calculate execution time
            execution_time = (time.time() - start_time) * 1000
            
            # Log performance
            performance_logger.log_evaluation_performance(
                task_type=request.task_type,
                model_name="unknown",
                execution_time=execution_time,
                token_count=len(request.model_response.split()),
                score=evaluation_result.get("overall_score", 0.0)
            )
            
            # Add request metadata
            evaluation_result.update({
                "task_id": request.task_id or f"{request.task_type}_{request.domain}",
                "task_type": request.task_type,
                "domain": request.domain,
                "difficulty": request.difficulty,
                "model_response": request.model_response,
                "generated_prompt": prompt,
                "execution_time_ms": execution_time,
                "request_id": request_id,
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
            })
            
            logger.info(f"Request {request_id} processed successfully in {execution_time:.2f}ms")
            return evaluation_result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            # Handle with structured error handling
            error = self.error_handler.handle_evaluation_error(e, {
                "request_id": request_id,
                "task_type": request.task_type,
                "domain": request.domain,
                "model_response": request.model_response[:200] if request.model_response else ""
            })
            
            self.error_handler.log_error(error, request_id)
            
            logger.error(f"Request {request_id} failed after {execution_time:.2f}ms: {str(e)}")
            
            return {
                "error": error.user_message,
                "error_type": error.error_type.value,
                "task_type": request.task_type,
                "overall_score": 0.0,
                "execution_time_ms": execution_time,
                "request_id": request_id,
                "recoverable": error.recoverable
            }
    
    async def batch_evaluate(
        self,
        evaluations: List[Dict[str, Any]],
        max_concurrent: int = 10
    ) -> List[RobustEvaluationResult]:
        """Run multiple evaluations in batch with rate limiting."""
        if len(evaluations) > 100:
            raise ValueError("Batch size cannot exceed 100 evaluations")
        
        # Split into smaller chunks to prevent overload
        batch_id = str(uuid.uuid4())
        logger.info(f"Starting batch evaluation {batch_id} with {len(evaluations)} items")
        
        async def evaluate_with_semaphore(semaphore, eval_config, index):
            async with semaphore:
                try:
                    return await self.evaluate(
                        eval_config.get("model_response", ""),
                        eval_config.get("task_config", {})
                    )
                except Exception as e:
                    logger.error(f"Batch item {index} failed: {str(e)}")
                    return RobustEvaluationResult(
                        task_id=f"batch_error_{index}",
                        domain="unknown",
                        score=0.0,
                        reasoning_quality=0.0,
                        explanation=f"Batch evaluation failed: {str(e)}",
                        metadata={"error": str(e), "batch_id": batch_id, "item_index": index}
                    )
        
        # Use semaphore to limit concurrent evaluations
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Create tasks with rate limiting
        tasks = [
            evaluate_with_semaphore(semaphore, eval_config, i)
            for i, eval_config in enumerate(evaluations)
        ]
        
        # Execute all tasks
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        execution_time = (time.time() - start_time) * 1000
        
        logger.info(f"Batch evaluation {batch_id} completed in {execution_time:.2f}ms")
        
        # Log batch performance
        performance_logger.log_evaluation_performance(
            task_type="batch",
            model_name="batch_evaluation",
            execution_time=execution_time,
            token_count=sum(len(e.get("model_response", "").split()) for e in evaluations),
            score=sum(r.score for r in results) / len(results) if results else 0.0
        )
        
        return results
    
    async def generate_task_prompt(self, task_type: str, domain: str = "general", difficulty: str = "medium") -> str:
        """Generate a prompt for a specific task type with validation."""
        try:
            # Validate inputs
            if task_type not in self.get_available_task_types():
                raise ValueError(f"Invalid task type: {task_type}")
            
            if domain not in self.get_available_domains():
                raise ValueError(f"Invalid domain: {domain}")
            
            if difficulty not in self.get_available_difficulties():
                raise ValueError(f"Invalid difficulty: {difficulty}")
            
            # Create task with protection
            task = await self._create_task_with_protection(task_type, domain, difficulty)
            prompt = await task.generate_prompt()
            
            logger.info(f"Generated prompt for {task_type}/{domain}/{difficulty}")
            return prompt
            
        except Exception as e:
            error = self.error_handler.handle_evaluation_error(e, {
                "task_type": task_type,
                "domain": domain,
                "difficulty": difficulty
            })
            self.error_handler.log_error(error)
            raise Exception(error.user_message)
    
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
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics."""
        return {
            "circuit_breaker_state": self.circuit_breaker.state,
            "circuit_breaker_failures": self.circuit_breaker.failure_count,
            "error_counts": dict(self.error_handler.error_counts),
            "active_rate_limits": len(self.request_counts),
            "cache_size": len(self._task_cache)
        }
    
    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker for maintenance."""
        self.circuit_breaker.state = "CLOSED"
        self.circuit_breaker.failure_count = 0
        logger.info("Circuit breaker reset")
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check."""
        health_status = {
            "status": "healthy",
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            "version": "0.1.0"
        }
        
        try:
            # Test basic functionality
            test_task = self._create_task("attribution", "general", "medium")
            test_prompt = await test_task.generate_prompt()
            
            health_status.update({
                "engine_status": "operational",
                "task_creation": "ok",
                "prompt_generation": "ok" if len(test_prompt) > 0 else "error",
                "system_metrics": self.get_system_health()
            })
            
        except Exception as e:
            health_status.update({
                "status": "degraded",
                "engine_status": "error",
                "error": str(e)
            })
        
        return health_status