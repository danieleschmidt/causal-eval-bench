"""
Advanced benchmarking framework for causal reasoning evaluation.
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class BenchmarkStatus(Enum):
    """Status of benchmark execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    
    name: str
    description: str
    models: List[str]
    tasks: List[str] = field(default_factory=list)  # Empty means all tasks
    domains: List[str] = field(default_factory=list)  # Empty means all domains
    difficulties: List[str] = field(default_factory=list)  # Empty means all difficulties
    n_samples_per_task: int = 50
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout_seconds: int = 300
    parallel_execution: bool = True
    max_concurrent: int = 5
    include_statistical_analysis: bool = True
    include_error_analysis: bool = True
    include_profiling: bool = True
    save_responses: bool = True
    output_directory: Optional[str] = None


@dataclass
class TaskResult:
    """Result from a single task execution."""
    
    model_name: str
    task_id: str
    task_type: str
    domain: str
    difficulty: str
    prompt: str
    response: str
    evaluation_scores: Dict[str, Any]
    execution_time_ms: int
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class BenchmarkResult:
    """Complete results from benchmark execution."""
    
    config: BenchmarkConfig
    status: BenchmarkStatus = BenchmarkStatus.PENDING
    task_results: List[TaskResult] = field(default_factory=list)
    model_summaries: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    comparative_analysis: Dict[str, Any] = field(default_factory=dict)
    statistical_analysis: Dict[str, Any] = field(default_factory=dict)
    error_analysis: Dict[str, Any] = field(default_factory=dict)
    execution_metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: float = 0.0
    end_time: float = 0.0
    
    @property
    def execution_time_seconds(self) -> float:
        """Get total execution time in seconds."""
        return self.end_time - self.start_time if self.end_time > 0 else 0.0
    
    @property
    def successful_tasks(self) -> List[TaskResult]:
        """Get tasks that completed successfully."""
        return [task for task in self.task_results if task.error is None]
    
    @property
    def failed_tasks(self) -> List[TaskResult]:
        """Get tasks that failed."""
        return [task for task in self.task_results if task.error is not None]
    
    @property
    def success_rate(self) -> float:
        """Get overall success rate."""
        if not self.task_results:
            return 0.0
        return len(self.successful_tasks) / len(self.task_results)


class BenchmarkRunner:
    """Advanced benchmark runner for causal reasoning evaluation."""
    
    def __init__(
        self,
        model_manager=None,
        evaluation_engine=None,
        statistical_analyzer=None,
        error_analyzer=None,
        profiler=None
    ):
        """Initialize the benchmark runner."""
        self.model_manager = model_manager
        self.evaluation_engine = evaluation_engine
        self.statistical_analyzer = statistical_analyzer
        self.error_analyzer = error_analyzer
        self.profiler = profiler
        self.benchmark_history: List[BenchmarkResult] = []
        logger.info("Benchmark runner initialized")
    
    async def run_benchmark(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Execute a complete benchmark."""
        
        logger.info(f"Starting benchmark: {config.name}")
        
        result = BenchmarkResult(
            config=config,
            status=BenchmarkStatus.RUNNING,
            start_time=time.time()
        )
        
        try:
            # Validate configuration
            self._validate_config(config)
            
            # Generate task list
            tasks = await self._generate_task_list(config)
            logger.info(f"Generated {len(tasks)} tasks for benchmark")
            
            # Execute tasks
            if config.parallel_execution:
                task_results = await self._execute_tasks_parallel(tasks, config)
            else:
                task_results = await self._execute_tasks_sequential(tasks, config)
            
            result.task_results = task_results
            
            # Generate summaries and analyses
            await self._generate_analyses(result)
            
            result.status = BenchmarkStatus.COMPLETED
            result.end_time = time.time()
            
            # Save results if requested
            if config.output_directory:
                await self._save_results(result, config.output_directory)
            
            # Add to history
            self.benchmark_history.append(result)
            
            logger.info(f"Benchmark completed: {config.name} ({result.execution_time_seconds:.1f}s)")
            
        except Exception as e:
            logger.error(f"Benchmark failed: {str(e)}")
            result.status = BenchmarkStatus.FAILED
            result.execution_metadata["error"] = str(e)
            result.end_time = time.time()
        
        return result
    
    def _validate_config(self, config: BenchmarkConfig) -> None:
        """Validate benchmark configuration."""
        if not config.models:
            raise ValueError("At least one model must be specified")
        
        if config.n_samples_per_task < 1:
            raise ValueError("n_samples_per_task must be at least 1")
        
        if config.max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        
        # Validate models exist
        if self.model_manager:
            available_models = self.model_manager.list_clients()
            for model in config.models:
                if model not in available_models:
                    raise ValueError(f"Model '{model}' not available. Available: {available_models}")
    
    async def _generate_task_list(self, config: BenchmarkConfig) -> List[Dict[str, Any]]:
        """Generate list of tasks to execute."""
        
        # Default task configurations
        default_tasks = ["attribution", "counterfactual", "intervention", "chain", "confounding"]
        default_domains = ["general", "medical", "education", "business", "technology"]
        default_difficulties = ["easy", "medium", "hard"]
        
        tasks = config.tasks or default_tasks
        domains = config.domains or default_domains
        difficulties = config.difficulties or default_difficulties
        
        task_list = []
        task_id = 0
        
        for model in config.models:
            for task_type in tasks:
                for domain in domains:
                    for difficulty in difficulties:
                        for sample_idx in range(config.n_samples_per_task):
                            task_list.append({
                                "task_id": f"{model}_{task_type}_{domain}_{difficulty}_{task_id:04d}",
                                "model_name": model,
                                "task_type": task_type,
                                "domain": domain,
                                "difficulty": difficulty,
                                "sample_index": sample_idx
                            })
                            task_id += 1
        
        return task_list
    
    async def _execute_tasks_parallel(
        self,
        tasks: List[Dict[str, Any]],
        config: BenchmarkConfig
    ) -> List[TaskResult]:
        """Execute tasks in parallel."""
        
        semaphore = asyncio.Semaphore(config.max_concurrent)
        
        async def execute_single_task(task_spec):
            async with semaphore:
                return await self._execute_single_task(task_spec, config)
        
        task_coroutines = [execute_single_task(task) for task in tasks]
        results = await asyncio.gather(*task_coroutines, return_exceptions=True)
        
        # Handle exceptions
        task_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create failed task result
                task_spec = tasks[i]
                task_results.append(TaskResult(
                    model_name=task_spec["model_name"],
                    task_id=task_spec["task_id"],
                    task_type=task_spec["task_type"],
                    domain=task_spec["domain"],
                    difficulty=task_spec["difficulty"],
                    prompt="",
                    response="",
                    evaluation_scores={},
                    execution_time_ms=0,
                    error=str(result)
                )
            else:
                task_results.append(result)
        
        return task_results
    
    async def _execute_tasks_sequential(
        self,
        tasks: List[Dict[str, Any]],
        config: BenchmarkConfig
    ) -> List[TaskResult]:
        """Execute tasks sequentially."""
        
        results = []
        for i, task in enumerate(tasks):
            logger.info(f"Executing task {i+1}/{len(tasks)}: {task['task_id']}")
            result = await self._execute_single_task(task, config)
            results.append(result)
            
            # Brief pause between tasks to avoid rate limiting
            await asyncio.sleep(0.1)
        
        return results
    
    async def _execute_single_task(
        self,
        task_spec: Dict[str, Any],
        config: BenchmarkConfig
    ) -> TaskResult:
        """Execute a single task."""
        
        start_time = time.time()
        
        try:
            # Generate prompt using evaluation engine
            if not self.evaluation_engine:
                raise ValueError("Evaluation engine required for task execution")
            
            prompt = await self.evaluation_engine.generate_task_prompt(
                task_spec["task_type"],
                task_spec["domain"],
                task_spec["difficulty"]
            )
            
            # Get model response
            if not self.model_manager:
                raise ValueError("Model manager required for task execution")
            
            response = await self.model_manager.generate_response(
                task_spec["model_name"],
                prompt,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout_seconds
            )
            
            # Evaluate response
            evaluation_scores = await self.evaluation_engine.evaluate(
                response.content,
                {
                    "task_type": task_spec["task_type"],
                    "domain": task_spec["domain"],
                    "difficulty": task_spec["difficulty"],
                    "task_id": task_spec["task_id"]
                }
            )
            
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            return TaskResult(
                model_name=task_spec["model_name"],
                task_id=task_spec["task_id"],
                task_type=task_spec["task_type"],
                domain=task_spec["domain"],
                difficulty=task_spec["difficulty"],
                prompt=prompt,
                response=response.content,
                evaluation_scores=evaluation_scores.dict() if hasattr(evaluation_scores, 'dict') else evaluation_scores,
                execution_time_ms=execution_time_ms,
                tokens_used=response.tokens_used,
                cost=response.cost
            )
            
        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            return TaskResult(
                model_name=task_spec["model_name"],
                task_id=task_spec["task_id"],
                task_type=task_spec["task_type"],
                domain=task_spec["domain"],
                difficulty=task_spec["difficulty"],
                prompt="",
                response="",
                evaluation_scores={},
                execution_time_ms=execution_time_ms,
                error=str(e)
            )
    
    async def _generate_analyses(self, result: BenchmarkResult) -> None:
        """Generate comprehensive analyses of benchmark results."""
        
        successful_tasks = result.successful_tasks
        
        if not successful_tasks:
            logger.warning("No successful tasks to analyze")
            return
        
        # Model summaries
        result.model_summaries = self._generate_model_summaries(successful_tasks)
        
        # Comparative analysis
        result.comparative_analysis = self._generate_comparative_analysis(successful_tasks)
        
        # Statistical analysis
        if result.config.include_statistical_analysis and self.statistical_analyzer:
            result.statistical_analysis = await self._generate_statistical_analysis(successful_tasks)
        
        # Error analysis
        if result.config.include_error_analysis and self.error_analyzer:
            result.error_analysis = await self._generate_error_analysis(result.task_results)
        
        # Execution metadata
        result.execution_metadata = {
            "total_tasks": len(result.task_results),
            "successful_tasks": len(successful_tasks),
            "failed_tasks": len(result.failed_tasks),
            "success_rate": result.success_rate,
            "average_execution_time_ms": np.mean([t.execution_time_ms for t in successful_tasks]),
            "total_tokens_used": sum(t.tokens_used or 0 for t in successful_tasks),
            "total_cost": sum(t.cost or 0 for t in successful_tasks)
        }