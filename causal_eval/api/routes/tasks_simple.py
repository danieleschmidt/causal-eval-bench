"""Simple task endpoints without database dependencies."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import logging

from causal_eval.tasks.attribution import CausalAttribution
from causal_eval.core.tasks import TaskConfig

logger = logging.getLogger(__name__)
router = APIRouter()


class TaskInfo(BaseModel):
    """Task information response model."""
    task_type: str
    description: str
    domains: List[str]
    difficulties: List[str]


@router.get("/", response_model=List[TaskInfo])
async def list_available_tasks():
    """List all available causal reasoning tasks."""
    
    tasks = [
        TaskInfo(
            task_type="attribution",
            description="Test ability to identify true causes vs. mere correlations",
            domains=["general", "medical", "education", "business", "recreational"],
            difficulties=["easy", "medium", "hard"]
        ),
        TaskInfo(
            task_type="counterfactual",
            description="Assess understanding of 'what if' scenarios",
            domains=["general", "medical", "education", "business"],
            difficulties=["easy", "medium", "hard"]
        ),
        TaskInfo(
            task_type="intervention",
            description="Test understanding of the effects of causal interventions",
            domains=["general", "medical", "business"],
            difficulties=["easy", "medium", "hard"]
        ),
    ]
    
    return tasks


@router.get("/{task_type}")
async def get_task_info(task_type: str):
    """Get detailed information about a specific task type."""
    
    task_descriptions = {
        "attribution": {
            "name": "Causal Attribution",
            "description": "Evaluate ability to distinguish genuine causation from correlation",
            "example": "Determining whether ice cream sales cause drowning incidents or if both are correlated with warm weather",
            "scoring": {
                "relationship_accuracy": 0.5,
                "reasoning_quality": 0.3,
                "confounder_identification": 0.2
            }
        },
        "counterfactual": {
            "name": "Counterfactual Reasoning", 
            "description": "Assess understanding of alternative scenarios and their outcomes",
            "example": "What would have happened if a student had not studied for an exam?",
            "scoring": {
                "logical_consistency": 0.4,
                "causal_understanding": 0.4,
                "scenario_plausibility": 0.2
            }
        },
        "intervention": {
            "name": "Causal Intervention",
            "description": "Test understanding of direct causal manipulations and their effects",
            "example": "What happens when a thermostat is manually set to a specific temperature?",
            "scoring": {
                "intervention_prediction": 0.6,
                "mechanism_understanding": 0.4
            }
        }
    }
    
    if task_type not in task_descriptions:
        raise HTTPException(status_code=404, detail=f"Task type '{task_type}' not found")
    
    return task_descriptions[task_type]


@router.post("/{task_type}/generate")
async def generate_task_instance(task_type: str, domain: str = "general", difficulty: str = "medium"):
    """Generate a specific instance of a task."""
    
    logger.info(f"Generating {task_type} task for domain: {domain}, difficulty: {difficulty}")
    
    try:
        if task_type == "attribution":
            config = TaskConfig(
                task_id=f"{task_type}_{domain}_{difficulty}",
                domain=domain,
                difficulty=difficulty,
                description=f"Causal attribution task for {domain} domain",
                expected_reasoning_type="attribution"
            )
            
            task = CausalAttribution(config)
            prompt = await task.generate_prompt()
            
            return {
                "task_type": task_type,
                "domain": domain,
                "difficulty": difficulty,
                "prompt": prompt,
                "instructions": [
                    "Analyze the relationship between the two variables",
                    "Choose from: causal, correlation, spurious, reverse_causal",
                    "Provide confidence level (0.0 to 1.0)",
                    "Explain your reasoning",
                    "Identify any confounding variables"
                ]
            }
        else:
            return {
                "task_type": task_type,
                "domain": domain, 
                "difficulty": difficulty,
                "prompt": f"Sample {task_type} task prompt for {domain} domain at {difficulty} difficulty.",
                "note": f"Full implementation for {task_type} tasks coming soon!"
            }
    
    except Exception as e:
        logger.error(f"Task generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Task generation failed: {str(e)}")