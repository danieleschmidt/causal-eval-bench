"""Evaluation endpoints."""

import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Request, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from causal_eval.core.engine import EvaluationResult, CausalEvaluationRequest
from causal_eval.database.connection import get_session
from causal_eval.repositories.evaluation import EvaluationRepository

router = APIRouter()


class EvaluationRequest(BaseModel):
    """Request model for single evaluation."""
    task_type: str = Field(..., description="Type of causal reasoning task")
    model_response: str = Field(..., description="Model's response to evaluate")
    domain: Optional[str] = Field("general", description="Domain for the task")
    difficulty: Optional[str] = Field("medium", description="Difficulty level")
    model_name: Optional[str] = Field(None, description="Name of the model being evaluated")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BatchEvaluationRequest(BaseModel):
    """Request model for batch evaluation."""
    evaluations: List[EvaluationRequest] = Field(..., description="List of evaluations to run")
    session_config: Dict[str, Any] = Field(default_factory=dict, description="Session configuration")


class EvaluationSessionRequest(BaseModel):
    """Request model for creating evaluation session."""
    model_name: str = Field(..., description="Name of the model")
    model_version: Optional[str] = Field(None, description="Version of the model")
    config: Dict[str, Any] = Field(default_factory=dict, description="Session configuration")


class TaskPromptRequest(BaseModel):
    """Request model for generating task prompts."""
    task_type: str = Field(..., description="Type of causal reasoning task")
    domain: Optional[str] = Field("general", description="Domain for the task")
    difficulty: Optional[str] = Field("medium", description="Difficulty level")


@router.post("/evaluate", response_model=Dict[str, Any])
async def evaluate_single(
    request: Request,
    eval_request: EvaluationRequest,
    background_tasks: BackgroundTasks,
    session = Depends(get_session)
) -> Dict[str, Any]:
    """Evaluate a single model response."""
    engine = request.app.state.engine
    metrics_collector = request.app.state.metrics_collector
    
    try:
        # Create causal evaluation request
        causal_request = CausalEvaluationRequest(
            task_type=eval_request.task_type,
            model_response=eval_request.model_response,
            domain=eval_request.domain,
            difficulty=eval_request.difficulty,
            task_id=f"{eval_request.task_type}_{eval_request.domain}_{eval_request.difficulty}"
        )
        
        # Run evaluation
        result = await engine.evaluate_request(causal_request)
        
        # Collect metrics in background
        background_tasks.add_task(
            metrics_collector.add_evaluation_result,
            result
        )
        
        # Store in database if session provided
        if eval_request.model_name:
            background_tasks.add_task(
                _store_evaluation_result,
                session,
                eval_request.model_name,
                result
            )
        
        return {
            "evaluation_id": result.get("task_id"),
            "overall_score": result.get("overall_score", 0.0),
            "detailed_scores": {
                k: v for k, v in result.items() 
                if k.endswith("_score") and k != "overall_score"
            },
            "confidence": result.get("confidence"),
            "task_type": result.get("task_type"),
            "domain": result.get("domain"),
            "difficulty": result.get("difficulty"),
            "evaluation_metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "model_name": eval_request.model_name,
                "prompt_length": len(result.get("generated_prompt", "")),
                "response_length": len(eval_request.model_response)
            },
            "explanation": result.get("correct_explanation", ""),
            "model_reasoning": result.get("model_reasoning", "")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@router.post("/batch", response_model=Dict[str, Any])
async def evaluate_batch(
    request: Request,
    batch_request: BatchEvaluationRequest,
    background_tasks: BackgroundTasks,
    session = Depends(get_session)
) -> Dict[str, Any]:
    """Evaluate multiple model responses in batch."""
    engine = request.app.state.engine
    metrics_collector = request.app.state.metrics_collector
    
    try:
        # Create evaluation session if model name provided
        evaluation_session = None
        if batch_request.evaluations and batch_request.evaluations[0].model_name:
            repo = EvaluationRepository(session)
            model_name = batch_request.evaluations[0].model_name
            evaluation_session = await repo.create_evaluation_session(
                model_name=model_name,
                config=batch_request.session_config
            )
        
        # Prepare evaluation requests
        causal_requests = []
        for i, eval_req in enumerate(batch_request.evaluations):
            causal_request = CausalEvaluationRequest(
                task_type=eval_req.task_type,
                model_response=eval_req.model_response,
                domain=eval_req.domain,
                difficulty=eval_req.difficulty,
                task_id=f"batch_{i}_{eval_req.task_type}_{eval_req.domain}"
            )
            causal_requests.append(causal_request)
        
        # Run evaluations concurrently
        evaluation_tasks = [
            engine.evaluate_request(req) for req in causal_requests
        ]
        results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
        
        # Process results
        successful_results = []
        failed_results = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_results.append({
                    "index": i,
                    "error": str(result),
                    "task_type": causal_requests[i].task_type
                })
            else:
                successful_results.append(result)
                # Collect metrics in background
                background_tasks.add_task(
                    metrics_collector.add_evaluation_result,
                    result
                )
        
        # Calculate aggregate scores
        if successful_results:
            overall_scores = [r.get("overall_score", 0.0) for r in successful_results]
            aggregate_score = sum(overall_scores) / len(overall_scores)
            
            # Task type breakdown
            task_type_scores = {}
            for result in successful_results:
                task_type = result.get("task_type", "unknown")
                if task_type not in task_type_scores:
                    task_type_scores[task_type] = []
                task_type_scores[task_type].append(result.get("overall_score", 0.0))
            
            # Average by task type
            task_type_averages = {
                task_type: sum(scores) / len(scores)
                for task_type, scores in task_type_scores.items()
            }
        else:
            aggregate_score = 0.0
            task_type_averages = {}
        
        # Complete evaluation session
        if evaluation_session:
            background_tasks.add_task(
                _complete_evaluation_session,
                session,
                evaluation_session.id,
                successful_results,
                failed_results
            )
        
        return {
            "session_id": evaluation_session.session_id if evaluation_session else None,
            "total_evaluations": len(batch_request.evaluations),
            "successful_evaluations": len(successful_results),
            "failed_evaluations": len(failed_results),
            "aggregate_score": aggregate_score,
            "task_type_scores": task_type_averages,
            "detailed_results": successful_results,
            "failed_results": failed_results,
            "evaluation_metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "batch_size": len(batch_request.evaluations),
                "session_config": batch_request.session_config
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch evaluation failed: {str(e)}")


@router.post("/sessions", response_model=Dict[str, Any])
async def create_evaluation_session(
    session_request: EvaluationSessionRequest,
    session = Depends(get_session)
) -> Dict[str, Any]:
    """Create a new evaluation session."""
    try:
        repo = EvaluationRepository(session)
        
        evaluation_session = await repo.create_evaluation_session(
            model_name=session_request.model_name,
            model_version=session_request.model_version,
            config=session_request.config
        )
        
        return {
            "session_id": evaluation_session.session_id,
            "id": evaluation_session.id,
            "model_name": evaluation_session.model_name,
            "model_version": evaluation_session.model_version,
            "status": evaluation_session.status,
            "created_at": evaluation_session.created_at.isoformat(),
            "config": evaluation_session.config
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session creation failed: {str(e)}")


@router.get("/sessions/{session_id}", response_model=Dict[str, Any])
async def get_evaluation_session(
    session_id: str,
    session = Depends(get_session)
) -> Dict[str, Any]:
    """Get evaluation session details."""
    try:
        repo = EvaluationRepository(session)
        
        # Get session by session_id (UUID)
        eval_session = await repo.evaluation_sessions.get_by_field("session_id", session_id)
        
        if not eval_session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get session statistics
        statistics = await repo.get_session_statistics(eval_session.id)
        
        return {
            "session_info": statistics.get("session_info", {}),
            "task_statistics": statistics.get("task_statistics", {}),
            "evaluation_statistics": statistics.get("evaluation_statistics", {}),
            "task_type_breakdown": statistics.get("task_type_breakdown", {}),
            "domain_breakdown": statistics.get("domain_breakdown", {})
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")


@router.post("/prompts", response_model=Dict[str, Any])
async def generate_task_prompt(
    request: Request,
    prompt_request: TaskPromptRequest
) -> Dict[str, Any]:
    """Generate a prompt for a specific task type."""
    engine = request.app.state.engine
    
    try:
        prompt = await engine.generate_task_prompt(
            prompt_request.task_type,
            prompt_request.domain,
            prompt_request.difficulty
        )
        
        return {
            "task_type": prompt_request.task_type,
            "domain": prompt_request.domain,
            "difficulty": prompt_request.difficulty,
            "prompt": prompt,
            "generated_at": datetime.utcnow().isoformat(),
            "prompt_length": len(prompt)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prompt generation failed: {str(e)}")


@router.get("/tasks/types", response_model=List[str])
async def get_task_types(request: Request) -> List[str]:
    """Get available task types."""
    engine = request.app.state.engine
    return engine.get_available_task_types()


@router.get("/tasks/domains", response_model=List[str])
async def get_domains(request: Request) -> List[str]:
    """Get available domains."""
    engine = request.app.state.engine
    return engine.get_available_domains()


@router.get("/tasks/difficulties", response_model=List[str])
async def get_difficulties(request: Request) -> List[str]:
    """Get available difficulty levels."""
    engine = request.app.state.engine
    return engine.get_available_difficulties()


@router.post("/evaluate-with-model", response_model=Dict[str, Any])
async def evaluate_with_model(
    request: Request,
    eval_request: EvaluationRequest,
    background_tasks: BackgroundTasks,
    session = Depends(get_session)
) -> Dict[str, Any]:
    """Generate prompt, get model response, and evaluate in one endpoint."""
    engine = request.app.state.engine
    model_manager = getattr(request.app.state, 'model_manager', None)
    metrics_collector = request.app.state.metrics_collector
    
    try:
        # Generate prompt for the specified task
        prompt = await engine.generate_task_prompt(
            eval_request.task_type,
            eval_request.domain or "general",
            eval_request.difficulty or "medium"
        )
        
        # Get model response if model_name provided
        if eval_request.model_name and model_manager:
            try:
                model_response_obj = await model_manager.generate_response(
                    eval_request.model_name,
                    prompt,
                    temperature=eval_request.metadata.get("temperature", 0.7),
                    max_tokens=eval_request.metadata.get("max_tokens", 1000)
                )
                model_response = model_response_obj.content
                api_metadata = {
                    "tokens_used": model_response_obj.tokens_used,
                    "cost": model_response_obj.cost,
                    "latency_ms": model_response_obj.latency_ms,
                    "model_metadata": model_response_obj.metadata
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Model API error: {str(e)}")
        else:
            # Use provided model response or return prompt for manual evaluation
            if eval_request.model_response:
                model_response = eval_request.model_response
                api_metadata = {}
            else:
                return {
                    "prompt": prompt,
                    "task_type": eval_request.task_type,
                    "domain": eval_request.domain,
                    "difficulty": eval_request.difficulty,
                    "message": "Provide model_response in request body or model_name for automatic generation"
                }
        
        # Create causal evaluation request
        causal_request = CausalEvaluationRequest(
            task_type=eval_request.task_type,
            model_response=model_response,
            domain=eval_request.domain,
            difficulty=eval_request.difficulty,
            task_id=f"{eval_request.task_type}_{eval_request.domain}_{eval_request.difficulty}"
        )
        
        # Run evaluation
        result = await engine.evaluate_request(causal_request)
        
        # Collect metrics in background
        background_tasks.add_task(
            metrics_collector.add_evaluation_result,
            result
        )
        
        # Store in database if session provided
        if eval_request.model_name:
            background_tasks.add_task(
                _store_evaluation_result,
                session,
                eval_request.model_name,
                result
            )
        
        return {
            "evaluation_id": result.get("task_id"),
            "overall_score": result.get("overall_score", 0.0),
            "detailed_scores": {
                k: v for k, v in result.items() 
                if k.endswith("_score") and k != "overall_score"
            },
            "confidence": result.get("confidence"),
            "task_type": result.get("task_type"),
            "domain": result.get("domain"),
            "difficulty": result.get("difficulty"),
            "prompt": prompt,
            "model_response": model_response,
            "model_api_metadata": api_metadata,
            "evaluation_metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "model_name": eval_request.model_name,
                "prompt_length": len(prompt),
                "response_length": len(model_response)
            },
            "explanation": result.get("correct_explanation", ""),
            "model_reasoning": result.get("model_reasoning", "")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"End-to-end evaluation failed: {str(e)}")


@router.get("/metrics", response_model=Dict[str, Any])
async def get_metrics(request: Request) -> Dict[str, Any]:
    """Get evaluation metrics summary."""
    metrics_collector = request.app.state.metrics_collector
    
    try:
        # Get both legacy and new metrics
        summary = metrics_collector.calculate_summary()
        aggregate_metrics = metrics_collector.calculate_aggregate_metrics()
        causal_profile = metrics_collector.calculate_causal_reasoning_profile()
        
        return {
            "basic_metrics": summary.dict(),
            "aggregate_metrics": {
                "overall_score": aggregate_metrics.overall_score,
                "task_scores": aggregate_metrics.task_scores,
                "domain_scores": aggregate_metrics.domain_scores,
                "difficulty_scores": aggregate_metrics.difficulty_scores,
                "confidence_analysis": aggregate_metrics.confidence_scores,
                "error_analysis": aggregate_metrics.error_analysis,
                "statistical_significance": aggregate_metrics.statistical_significance
            },
            "causal_reasoning_profile": causal_profile,
            "collection_timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")


# Background task functions
async def _store_evaluation_result(session, model_name: str, result: Dict[str, Any]):
    """Store evaluation result in database."""
    try:
        repo = EvaluationRepository(session)
        
        # Create evaluation session if needed
        eval_session = await repo.create_evaluation_session(
            model_name=model_name,
            config={"single_evaluation": True}
        )
        
        # Create task execution
        task_execution = await repo.create_task_execution(
            session_id=eval_session.id,
            task_type=result.get("task_type", "unknown"),
            domain=result.get("domain", "general"),
            difficulty=result.get("difficulty", "medium"),
            prompt=result.get("generated_prompt", ""),
            task_metadata=result
        )
        
        # Record evaluation result
        await repo.record_evaluation_result(
            task_execution_id=task_execution.id,
            overall_score=result.get("overall_score", 0.0),
            evaluation_metadata=result,
            confidence=result.get("confidence"),
            explanation=result.get("explanation")
        )
        
        # Complete session
        await repo.complete_evaluation_session(eval_session.id)
        
    except Exception as e:
        # Log error but don't fail the main request
        print(f"Failed to store evaluation result: {e}")


async def _complete_evaluation_session(
    session, 
    session_id: int, 
    successful_results: List[Dict[str, Any]], 
    failed_results: List[Dict[str, Any]]
):
    """Complete evaluation session with results."""
    try:
        repo = EvaluationRepository(session)
        
        status = "completed" if not failed_results else "completed_with_errors"
        await repo.complete_evaluation_session(session_id, status)
        
    except Exception as e:
        # Log error but don't fail the main request
        print(f"Failed to complete evaluation session: {e}")