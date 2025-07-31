"""Evaluation endpoints."""

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from causal_eval.core.engine import EvaluationResult

router = APIRouter()


class EvaluationRequest(BaseModel):
    """Request model for single evaluation."""
    task_id: str
    model_response: str
    metadata: Dict[str, Any] = {}


class BatchEvaluationRequest(BaseModel):
    """Request model for batch evaluation."""
    evaluations: List[EvaluationRequest]


@router.post("/single", response_model=EvaluationResult)
async def evaluate_single(
    request: Request,
    eval_request: EvaluationRequest
) -> EvaluationResult:
    """Evaluate a single model response."""
    engine = request.app.state.engine
    task_registry = request.app.state.task_registry
    metrics_collector = request.app.state.metrics_collector
    
    # Verify task exists
    task = task_registry.get_task(eval_request.task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Prepare task config
    task_config = {
        "task_id": eval_request.task_id,
        "domain": task.config.domain,
        **eval_request.metadata
    }
    
    # Run evaluation
    result = await engine.evaluate(eval_request.model_response, task_config)
    
    # Collect metrics
    metrics_collector.add_result(result.dict())
    
    return result


@router.post("/batch", response_model=List[EvaluationResult])
async def evaluate_batch(
    request: Request,
    batch_request: BatchEvaluationRequest
) -> List[EvaluationResult]:
    """Evaluate multiple model responses in batch."""
    engine = request.app.state.engine
    task_registry = request.app.state.task_registry
    metrics_collector = request.app.state.metrics_collector
    
    # Prepare evaluations
    evaluations = []
    for eval_req in batch_request.evaluations:
        # Verify task exists
        task = task_registry.get_task(eval_req.task_id)
        if not task:
            raise HTTPException(
                status_code=404, 
                detail=f"Task not found: {eval_req.task_id}"
            )
        
        task_config = {
            "task_id": eval_req.task_id,
            "domain": task.config.domain,
            **eval_req.metadata
        }
        
        evaluations.append({
            "model_response": eval_req.model_response,
            "task_config": task_config
        })
    
    # Run batch evaluation
    results = await engine.batch_evaluate(evaluations)
    
    # Collect metrics
    metrics_collector.add_batch_results([r.dict() for r in results])
    
    return results


@router.get("/metrics")
async def get_metrics(request: Request) -> dict:
    """Get evaluation metrics summary."""
    metrics_collector = request.app.state.metrics_collector
    summary = metrics_collector.calculate_summary()
    
    return {
        "metrics_summary": summary.dict(),
        "collection_timestamp": "2025-07-31T00:00:00Z"  # TODO: Add actual timestamp
    }