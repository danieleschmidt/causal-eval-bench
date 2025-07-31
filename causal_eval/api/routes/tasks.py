"""Task management endpoints."""

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from causal_eval.core.tasks import TaskConfig

router = APIRouter()


class TaskListResponse(BaseModel):
    """Response model for task listing."""
    tasks: List[TaskConfig]
    total: int


@router.get("/", response_model=TaskListResponse)
async def list_tasks(
    request: Request,
    domain: Optional[str] = None
) -> TaskListResponse:
    """List available evaluation tasks."""
    task_registry = request.app.state.task_registry
    tasks = task_registry.list_tasks(domain=domain)
    
    return TaskListResponse(
        tasks=tasks,
        total=len(tasks)
    )


@router.get("/{task_id}", response_model=TaskConfig)
async def get_task(request: Request, task_id: str) -> TaskConfig:
    """Get details for a specific task."""
    task_registry = request.app.state.task_registry
    
    # Get task configuration
    if task_id not in task_registry._configs:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task_registry._configs[task_id]


@router.post("/{task_id}/prompt")
async def generate_task_prompt(request: Request, task_id: str) -> dict:
    """Generate a prompt for the specified task."""
    task_registry = request.app.state.task_registry
    task = task_registry.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    prompt = await task.generate_prompt()
    
    return {
        "task_id": task_id,
        "prompt": prompt,
        "instructions": "Provide your response to evaluate causal reasoning"
    }