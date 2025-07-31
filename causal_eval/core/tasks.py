"""Task registry and management for causal evaluation."""

from typing import Any, Dict, List, Optional, Type
from abc import ABC, abstractmethod
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


class TaskConfig(BaseModel):
    """Configuration for an evaluation task."""
    
    task_id: str
    domain: str
    difficulty: str  # easy, medium, hard
    description: str
    expected_reasoning_type: str
    metadata: Dict[str, Any] = {}


class BaseTask(ABC):
    """Base class for all causal evaluation tasks."""
    
    def __init__(self, config: TaskConfig):
        self.config = config
    
    @abstractmethod
    async def generate_prompt(self) -> str:
        """Generate the task prompt for the model."""
        pass
    
    @abstractmethod
    async def evaluate_response(self, response: str) -> Dict[str, Any]:
        """Evaluate the model's response."""
        pass


class TaskRegistry:
    """Registry for managing evaluation tasks."""
    
    def __init__(self):
        self._tasks: Dict[str, Type[BaseTask]] = {}
        self._configs: Dict[str, TaskConfig] = {}
        logger.info("Task registry initialized")
    
    def register_task(self, task_class: Type[BaseTask], config: TaskConfig) -> None:
        """Register a new task type."""
        self._tasks[config.task_id] = task_class
        self._configs[config.task_id] = config
        logger.info(f"Registered task: {config.task_id}")
    
    def get_task(self, task_id: str) -> Optional[BaseTask]:
        """Get a task instance by ID."""
        if task_id in self._tasks and task_id in self._configs:
            task_class = self._tasks[task_id]
            config = self._configs[task_id]
            return task_class(config)
        return None
    
    def list_tasks(self, domain: Optional[str] = None) -> List[TaskConfig]:
        """List available tasks, optionally filtered by domain."""
        tasks = list(self._configs.values())
        if domain:
            tasks = [task for task in tasks if task.domain == domain]
        return tasks