"""Simple evaluation endpoints without database dependencies."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, field_validator
from typing import Dict, Any, List, Optional
import logging
import time

from causal_eval.core.engine import EvaluationEngine, CausalEvaluationRequest
from causal_eval.core.error_handling import error_handler, handle_evaluation_error, create_http_error
from causal_eval.api.middleware.enhanced_validation import SecurityValidator
from causal_eval.core.performance_optimizer import optimized_engine

logger = logging.getLogger(__name__)
router = APIRouter()


class EvaluationResponse(BaseModel):
    """Response model for evaluations."""
    task_id: str
    overall_score: float
    reasoning_score: float
    explanation: str
    metadata: Dict[str, Any] = {}
    processing_time: Optional[float] = None
    warnings: List[str] = []

class EnhancedEvaluationRequest(BaseModel):
    """Enhanced evaluation request with validation."""
    task_type: str
    model_response: str
    domain: Optional[str] = "general"
    difficulty: Optional[str] = "medium"
    task_id: Optional[str] = None
    
    @field_validator('model_response')
    @classmethod
    def validate_model_response(cls, v):
        return SecurityValidator.validate_text_input(v, "model_response")
    
    @field_validator('task_type', 'domain', 'difficulty')
    @classmethod
    def validate_parameters(cls, v):
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_response(request: EnhancedEvaluationRequest) -> EvaluationResponse:
    """Evaluate a model's causal reasoning response with enhanced error handling."""
    start_time = time.time()
    request_id = f"eval_{int(time.time() * 1000)}"
    
    logger.info(
        f"Processing evaluation request - ID: {request_id}, Task: {request.task_type}, "
        f"Domain: {request.domain}, Difficulty: {request.difficulty}"
    )
    
    try:
        # Validate task parameters
        validated_params = SecurityValidator.validate_task_parameters(
            request.task_type, request.domain, request.difficulty
        )
        
        # Create evaluation request
        eval_request = CausalEvaluationRequest(
            task_type=validated_params["task_type"],
            model_response=request.model_response,
            domain=validated_params["domain"],
            difficulty=validated_params["difficulty"],
            task_id=request.task_id or request_id
        )
        
        # Perform optimized evaluation with caching
        result = await optimized_engine.evaluate_with_optimization(
            task_type=validated_params["task_type"],
            model_response=request.model_response,
            domain=validated_params["domain"],
            difficulty=validated_params["difficulty"],
            context={"request_id": request_id}
        )
        
        processing_time = time.time() - start_time
        
        # Check for warnings
        warnings = []
        if result.get("overall_score", 0) < 0.1:
            warnings.append("Very low evaluation score - check response format")
        if processing_time > 5.0:
            warnings.append("Evaluation took longer than expected")
        
        logger.info(
            f"Evaluation completed - ID: {request_id}, Score: {result.get('overall_score', 0):.3f}, "
            f"Time: {processing_time:.3f}s"
        )
        
        return EvaluationResponse(
            task_id=result.get("task_id", request_id),
            overall_score=result.get("overall_score", 0.0),
            reasoning_score=result.get("reasoning_score", 0.0),
            explanation=f"Evaluated {request.task_type} task for {request.domain} domain",
            metadata=result,
            processing_time=processing_time,
            warnings=warnings
        )
    
    except HTTPException:
        raise  # Re-raise HTTP exceptions from validation
    except Exception as e:
        # Handle with comprehensive error processing
        context = {
            "request_id": request_id,
            "task_type": request.task_type,
            "domain": request.domain,
            "difficulty": request.difficulty,
            "model_response": request.model_response[:200] + "..." if len(request.model_response) > 200 else request.model_response
        }
        
        causal_error = handle_evaluation_error(e, context)
        error_handler.log_error(causal_error, request_id)
        
        raise create_http_error(causal_error)


@router.get("/tasks")
async def get_available_tasks():
    """Get list of available evaluation tasks."""
    engine = EvaluationEngine()
    return {
        "task_types": engine.get_available_task_types(),
        "domains": engine.get_available_domains(),
        "difficulties": engine.get_available_difficulties()
    }


@router.post("/prompt/{task_type}")
async def generate_prompt(task_type: str, domain: str = "general", difficulty: str = "medium"):
    """Generate a prompt for a specific task type."""
    logger.info(f"Generating prompt for {task_type}")
    
    engine = EvaluationEngine()
    
    try:
        prompt = await engine.generate_task_prompt(task_type, domain, difficulty)
        return {
            "task_type": task_type,
            "domain": domain,
            "difficulty": difficulty,
            "prompt": prompt
        }
    
    except Exception as e:
        logger.error(f"Prompt generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prompt generation failed: {str(e)}")


@router.post("/batch")
async def batch_evaluate(evaluations: List[EnhancedEvaluationRequest]):
    """Evaluate multiple responses in batch with enhanced error handling."""
    start_time = time.time()
    batch_id = f"batch_{int(time.time() * 1000)}"
    
    logger.info(f"Processing batch evaluation - ID: {batch_id}, Count: {len(evaluations)}")
    
    if len(evaluations) > 100:  # Prevent abuse
        raise HTTPException(
            status_code=400,
            detail="Batch size too large. Maximum 100 evaluations per batch."
        )
    
    try:
        results = []
        errors = []
        
        # Process each evaluation individually with error isolation
        for i, eval_request in enumerate(evaluations):
            try:
                # Create individual evaluation request
                request_id = f"{batch_id}_{i}"
                
                validated_params = SecurityValidator.validate_task_parameters(
                    eval_request.task_type, eval_request.domain, eval_request.difficulty
                )
                
                individual_request = CausalEvaluationRequest(
                    task_type=validated_params["task_type"],
                    model_response=eval_request.model_response,
                    domain=validated_params["domain"],
                    difficulty=validated_params["difficulty"],
                    task_id=eval_request.task_id or request_id
                )
                
                # Perform optimized evaluation
                result = await optimized_engine.evaluate_with_optimization(
                    task_type=validated_params["task_type"],
                    model_response=eval_request.model_response,
                    domain=validated_params["domain"],
                    difficulty=validated_params["difficulty"],
                    context={"batch_id": batch_id, "index": i}
                )
                
                results.append({
                    "task_id": result.get("task_id", request_id),
                    "overall_score": result.get("overall_score", 0.0),
                    "reasoning_score": result.get("reasoning_score", 0.0),
                    "explanation": f"Evaluated {eval_request.task_type} task for {eval_request.domain} domain",
                    "metadata": result,
                    "index": i,
                    "status": "success"
                })
                
            except Exception as e:
                error_context = {
                    "batch_id": batch_id,
                    "index": i,
                    "task_type": eval_request.task_type,
                    "domain": eval_request.domain
                }
                
                causal_error = handle_evaluation_error(e, error_context)
                error_handler.log_error(causal_error, f"{batch_id}_{i}")
                
                errors.append({
                    "index": i,
                    "task_type": eval_request.task_type,
                    "error": causal_error.user_message,
                    "error_type": causal_error.error_type.value,
                    "recoverable": causal_error.recoverable
                })
                
                # Add placeholder result for failed evaluation
                results.append({
                    "task_id": f"{batch_id}_{i}",
                    "overall_score": 0.0,
                    "reasoning_score": 0.0,
                    "explanation": f"Evaluation failed: {causal_error.user_message}",
                    "metadata": {"error": True},
                    "index": i,
                    "status": "failed"
                })
        
        processing_time = time.time() - start_time
        successful_results = [r for r in results if r["status"] == "success"]
        
        summary = {
            "total_evaluations": len(evaluations),
            "successful_evaluations": len(successful_results),
            "failed_evaluations": len(errors),
            "success_rate": len(successful_results) / len(evaluations) if evaluations else 0.0,
            "average_score": sum(r["overall_score"] for r in successful_results) / len(successful_results) if successful_results else 0.0,
            "processing_time": processing_time,
            "batch_id": batch_id
        }
        
        logger.info(
            f"Batch evaluation completed - ID: {batch_id}, Success: {len(successful_results)}/{len(evaluations)}, "
            f"Avg Score: {summary['average_score']:.3f}, Time: {processing_time:.3f}s"
        )
        
        response = {
            "results": results,
            "summary": summary
        }
        
        if errors:
            response["errors"] = errors
        
        return response
    
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Batch evaluation system error: {str(e)}")
        raise HTTPException(status_code=500, detail="Batch evaluation system error")


@router.get("/performance")
async def get_performance_stats():
    """Get comprehensive performance and optimization statistics."""
    try:
        stats = await optimized_engine.get_optimization_stats()
        return {
            "status": "success",
            "timestamp": time.time(),
            "optimization_stats": stats
        }
    except Exception as e:
        logger.error(f"Performance stats error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get performance statistics")


@router.post("/cache/clear")
async def clear_cache():
    """Clear the evaluation cache (admin endpoint)."""
    try:
        await optimized_engine.cache.clear()
        logger.info("Evaluation cache cleared via API")
        return {
            "status": "success",
            "message": "Cache cleared successfully",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Cache clear error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")