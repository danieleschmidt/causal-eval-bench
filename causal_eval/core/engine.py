"""Main evaluation engine for causal reasoning assessment."""

import asyncio
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel
import logging

from causal_eval.tasks.attribution import CausalAttribution
from causal_eval.tasks.counterfactual import CounterfactualReasoning
from causal_eval.tasks.intervention import CausalIntervention
from causal_eval.core.tasks import BaseTask, TaskConfig

logger = logging.getLogger(__name__)


class EvaluationResult(BaseModel):
    """Result of a causal evaluation task."""
    
    task_id: str
    domain: str
    score: float
    reasoning_quality: float
    explanation: str
    metadata: Dict[str, Any] = {}


class CausalEvaluationRequest(BaseModel):
    """Request for causal evaluation."""
    
    task_type: str  # "attribution", "counterfactual", "intervention", etc.
    model_response: str
    domain: Optional[str] = "general"
    difficulty: Optional[str] = "medium"
    task_id: Optional[str] = None


class EvaluationEngine:
    """Core engine for running causal evaluation tasks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the evaluation engine."""
        self.config = config or {}
        self.task_registry = None  # Will be injected
        self._task_cache: Dict[str, BaseTask] = {}
        logger.info("Evaluation engine initialized")
    
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
    
    async def evaluate(
        self,
        model_response: str,
        task_config: Dict[str, Any]
    ) -> EvaluationResult:
        """Evaluate a model's response on a causal reasoning task."""
        task_type = task_config.get("task_type", task_config.get("task_id", "unknown"))
        domain = task_config.get("domain", "general")
        difficulty = task_config.get("difficulty", "medium")
        
        logger.info(f"Evaluating {task_type} task for domain: {domain}")
        
        try:
            # Create task instance
            task = self._create_task(task_type, domain, difficulty)
            
            # Generate prompt (to get scenario context)
            prompt = await task.generate_prompt()
            
            # Evaluate the response
            evaluation_result = await task.evaluate_response(model_response)
            
            # Convert to standardized format
            return EvaluationResult(
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
                }
            )
            
        except Exception as e:
            logger.error(f"Error evaluating task {task_type}: {str(e)}")
            return EvaluationResult(
                task_id=task_config.get("task_id", "error"),
                domain=domain,
                score=0.0,
                reasoning_quality=0.0,
                explanation=f"Evaluation failed: {str(e)}",
                metadata={"error": str(e), "task_type": task_type}
            )
    
    async def evaluate_request(self, request: CausalEvaluationRequest) -> Dict[str, Any]:
        """Evaluate a causal evaluation request and return detailed results."""
        try:
            # Create task instance
            task = self._create_task(request.task_type, request.domain or "general", request.difficulty or "medium")
            
            # Generate prompt for context
            prompt = await task.generate_prompt()
            
            # Evaluate the response
            evaluation_result = await task.evaluate_response(request.model_response)
            
            # Add request metadata
            evaluation_result.update({
                "task_id": request.task_id or f"{request.task_type}_{request.domain}",
                "task_type": request.task_type,
                "domain": request.domain,
                "difficulty": request.difficulty,
                "model_response": request.model_response,
                "generated_prompt": prompt
            })
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Error processing evaluation request: {str(e)}")
            return {
                "error": str(e),
                "task_type": request.task_type,
                "overall_score": 0.0
            }
    
    async def batch_evaluate(
        self,
        evaluations: List[Dict[str, Any]]
    ) -> List[EvaluationResult]:
        """Run multiple evaluations in batch."""
        results = []
        
        # Process evaluations concurrently for better performance
        tasks = []
        for eval_config in evaluations:
            task = self.evaluate(
                eval_config.get("model_response", ""),
                eval_config.get("task_config", {})
            )
            tasks.append(task)
        
        # Wait for all evaluations to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch evaluation {i} failed: {str(result)}")
                final_results.append(EvaluationResult(
                    task_id=f"batch_error_{i}",
                    domain="unknown",
                    score=0.0,
                    reasoning_quality=0.0,
                    explanation=f"Batch evaluation failed: {str(result)}",
                    metadata={"error": str(result)}
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    async def generate_task_prompt(self, task_type: str, domain: str = "general", difficulty: str = "medium") -> str:
        """Generate a prompt for a specific task type."""
        task = self._create_task(task_type, domain, difficulty)
        return await task.generate_prompt()
    
    def get_available_task_types(self) -> List[str]:
        """Get list of available task types."""
        return ["attribution", "counterfactual", "intervention"]
    
    def get_available_domains(self) -> List[str]:
        """Get list of available domains."""
        return [
            "general", "medical", "education", "business", "technology",
            "environmental", "workplace_safety", "urban_planning", 
            "manufacturing", "recreational", "public_safety", "international"
        ]
    
    def get_available_difficulties(self) -> List[str]:
        """Get list of available difficulty levels."""
        return ["easy", "medium", "hard"]