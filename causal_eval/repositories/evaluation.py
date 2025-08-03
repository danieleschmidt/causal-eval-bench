"""
Repository for evaluation-related database operations.
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc, distinct
from sqlalchemy.orm import selectinload
import logging

from causal_eval.database.models import (
    EvaluationSession, TaskExecution, ModelResponse, 
    EvaluationResult, TestCase, AuditLog
)
from causal_eval.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class EvaluationRepository:
    """Repository for evaluation-related operations."""
    
    def __init__(self, session: AsyncSession):
        """Initialize with async session."""
        self.session = session
        self.evaluation_sessions = BaseRepository(EvaluationSession, session)
        self.task_executions = BaseRepository(TaskExecution, session)
        self.model_responses = BaseRepository(ModelResponse, session)
        self.evaluation_results = BaseRepository(EvaluationResult, session)
        self.test_cases = BaseRepository(TestCase, session)
        self.audit_logs = BaseRepository(AuditLog, session)
    
    async def create_evaluation_session(
        self,
        model_name: str,
        model_version: Optional[str] = None,
        user_id: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> EvaluationSession:
        """Create a new evaluation session."""
        session = await self.evaluation_sessions.create(
            model_name=model_name,
            model_version=model_version,
            user_id=user_id,
            config=config or {},
            status="running"
        )
        
        # Log session creation
        await self.audit_logs.create(
            event_type="evaluation_session_created",
            entity_type="session",
            entity_id=session.id,
            user_id=user_id,
            event_data={
                "model_name": model_name,
                "model_version": model_version,
                "session_id": session.session_id
            }
        )
        
        return session
    
    async def complete_evaluation_session(
        self,
        session_id: int,
        status: str = "completed"
    ) -> Optional[EvaluationSession]:
        """Mark evaluation session as completed."""
        session = await self.evaluation_sessions.update(
            session_id,
            status=status,
            completed_at=datetime.utcnow()
        )
        
        if session:
            # Update task counts
            task_counts = await self._get_session_task_counts(session_id)
            await self.evaluation_sessions.update(
                session_id,
                total_tasks=task_counts["total"],
                completed_tasks=task_counts["completed"],
                failed_tasks=task_counts["failed"]
            )
            
            # Log completion
            await self.audit_logs.create(
                event_type="evaluation_session_completed",
                entity_type="session",
                entity_id=session_id,
                event_data={
                    "status": status,
                    "task_counts": task_counts
                }
            )
        
        return session
    
    async def create_task_execution(
        self,
        session_id: int,
        task_type: str,
        domain: str,
        difficulty: str,
        prompt: str,
        expected_outcome: Optional[str] = None,
        task_metadata: Optional[Dict[str, Any]] = None
    ) -> TaskExecution:
        """Create a new task execution."""
        task = await self.task_executions.create(
            session_id=session_id,
            task_type=task_type,
            domain=domain,
            difficulty=difficulty,
            prompt=prompt,
            expected_outcome=expected_outcome,
            task_metadata=task_metadata or {},
            status="pending",
            started_at=datetime.utcnow()
        )
        
        # Log task creation
        await self.audit_logs.create(
            event_type="task_execution_created",
            entity_type="task",
            entity_id=task.id,
            event_data={
                "session_id": session_id,
                "task_type": task_type,
                "domain": domain,
                "difficulty": difficulty
            }
        )
        
        return task
    
    async def record_model_response(
        self,
        task_execution_id: int,
        raw_response: str,
        parsed_response: Optional[Dict[str, Any]] = None,
        tokens_used: Optional[int] = None,
        api_cost: Optional[float] = None,
        response_time_ms: Optional[int] = None,
        model_params: Optional[Dict[str, Any]] = None
    ) -> ModelResponse:
        """Record model response for a task execution."""
        response = await self.model_responses.create(
            task_execution_id=task_execution_id,
            raw_response=raw_response,
            parsed_response=parsed_response,
            tokens_used=tokens_used,
            api_cost=api_cost,
            response_time_ms=response_time_ms,
            model_params=model_params
        )
        
        # Update task execution status
        await self.task_executions.update(
            task_execution_id,
            status="running"
        )
        
        return response
    
    async def record_evaluation_result(
        self,
        task_execution_id: int,
        overall_score: float,
        evaluation_metadata: Dict[str, Any],
        confidence: Optional[float] = None,
        explanation: Optional[str] = None,
        evaluation_duration_ms: Optional[int] = None,
        **component_scores
    ) -> EvaluationResult:
        """Record evaluation result for a task execution."""
        result = await self.evaluation_results.create(
            task_execution_id=task_execution_id,
            overall_score=overall_score,
            confidence=confidence,
            explanation=explanation,
            evaluation_metadata=evaluation_metadata,
            evaluation_duration_ms=evaluation_duration_ms,
            **component_scores
        )
        
        # Update task execution status
        duration = None
        task = await self.task_executions.get_by_id(task_execution_id)
        if task and task.started_at:
            duration = (datetime.utcnow() - task.started_at).total_seconds()
        
        await self.task_executions.update(
            task_execution_id,
            status="completed",
            completed_at=datetime.utcnow(),
            duration_seconds=duration
        )
        
        # Log evaluation completion
        await self.audit_logs.create(
            event_type="task_evaluation_completed",
            entity_type="task",
            entity_id=task_execution_id,
            event_data={
                "overall_score": overall_score,
                "confidence": confidence,
                "duration_seconds": duration
            }
        )
        
        return result
    
    async def get_session_with_results(self, session_id: int) -> Optional[EvaluationSession]:
        """Get evaluation session with all task executions and results."""
        result = await self.session.execute(
            select(EvaluationSession)
            .options(
                selectinload(EvaluationSession.task_executions)
                .selectinload(TaskExecution.evaluation_result),
                selectinload(EvaluationSession.task_executions)
                .selectinload(TaskExecution.model_response)
            )
            .where(EvaluationSession.id == session_id)
        )
        return result.scalar_one_or_none()
    
    async def get_session_statistics(self, session_id: int) -> Dict[str, Any]:
        """Get comprehensive statistics for an evaluation session."""
        # Get basic session info
        session = await self.evaluation_sessions.get_by_id(session_id)
        if not session:
            return {}
        
        # Get task execution statistics
        task_stats_query = select(
            func.count(TaskExecution.id).label("total_tasks"),
            func.count().filter(TaskExecution.status == "completed").label("completed_tasks"),
            func.count().filter(TaskExecution.status == "failed").label("failed_tasks"),
            func.avg(TaskExecution.duration_seconds).label("avg_duration"),
            func.count(distinct(TaskExecution.task_type)).label("unique_task_types"),
            func.count(distinct(TaskExecution.domain)).label("unique_domains")
        ).where(TaskExecution.session_id == session_id)
        
        task_stats_result = await self.session.execute(task_stats_query)
        task_stats = task_stats_result.first()
        
        # Get evaluation result statistics
        eval_stats_query = select(
            func.avg(EvaluationResult.overall_score).label("avg_score"),
            func.min(EvaluationResult.overall_score).label("min_score"),
            func.max(EvaluationResult.overall_score).label("max_score"),
            func.stddev(EvaluationResult.overall_score).label("score_std"),
            func.avg(EvaluationResult.confidence).label("avg_confidence")
        ).join(TaskExecution).where(TaskExecution.session_id == session_id)
        
        eval_stats_result = await self.session.execute(eval_stats_query)
        eval_stats = eval_stats_result.first()
        
        # Get task type breakdown
        task_type_query = select(
            TaskExecution.task_type,
            func.count(TaskExecution.id).label("count"),
            func.avg(EvaluationResult.overall_score).label("avg_score")
        ).join(EvaluationResult).where(
            TaskExecution.session_id == session_id
        ).group_by(TaskExecution.task_type)
        
        task_type_result = await self.session.execute(task_type_query)
        task_type_breakdown = {
            row.task_type: {"count": row.count, "avg_score": float(row.avg_score or 0)}
            for row in task_type_result
        }
        
        # Get domain breakdown
        domain_query = select(
            TaskExecution.domain,
            func.count(TaskExecution.id).label("count"),
            func.avg(EvaluationResult.overall_score).label("avg_score")
        ).join(EvaluationResult).where(
            TaskExecution.session_id == session_id
        ).group_by(TaskExecution.domain)
        
        domain_result = await self.session.execute(domain_query)
        domain_breakdown = {
            row.domain: {"count": row.count, "avg_score": float(row.avg_score or 0)}
            for row in domain_result
        }
        
        return {
            "session_info": {
                "id": session.id,
                "session_id": session.session_id,
                "model_name": session.model_name,
                "model_version": session.model_version,
                "status": session.status,
                "started_at": session.started_at,
                "completed_at": session.completed_at
            },
            "task_statistics": {
                "total_tasks": task_stats.total_tasks or 0,
                "completed_tasks": task_stats.completed_tasks or 0,
                "failed_tasks": task_stats.failed_tasks or 0,
                "success_rate": (task_stats.completed_tasks or 0) / max(task_stats.total_tasks or 1, 1),
                "avg_duration_seconds": float(task_stats.avg_duration or 0),
                "unique_task_types": task_stats.unique_task_types or 0,
                "unique_domains": task_stats.unique_domains or 0
            },
            "evaluation_statistics": {
                "avg_score": float(eval_stats.avg_score or 0),
                "min_score": float(eval_stats.min_score or 0),
                "max_score": float(eval_stats.max_score or 0),
                "score_std": float(eval_stats.score_std or 0),
                "avg_confidence": float(eval_stats.avg_confidence or 0)
            },
            "task_type_breakdown": task_type_breakdown,
            "domain_breakdown": domain_breakdown
        }
    
    async def get_recent_sessions(
        self,
        days: int = 7,
        limit: int = 50,
        model_name: Optional[str] = None
    ) -> List[EvaluationSession]:
        """Get recent evaluation sessions."""
        since_date = datetime.utcnow() - timedelta(days=days)
        
        filters = {"status": "completed"}
        if model_name:
            filters["model_name"] = model_name
        
        sessions = await self.evaluation_sessions.get_all(
            limit=limit,
            order_by="-started_at",
            **filters
        )
        
        # Filter by date
        return [s for s in sessions if s.started_at >= since_date]
    
    async def get_model_performance_history(
        self,
        model_name: str,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get performance history for a specific model."""
        since_date = datetime.utcnow() - timedelta(days=days)
        
        query = select(
            EvaluationSession.started_at,
            func.avg(EvaluationResult.overall_score).label("avg_score"),
            func.count(EvaluationResult.id).label("evaluation_count")
        ).join(TaskExecution).join(EvaluationResult).where(
            and_(
                EvaluationSession.model_name == model_name,
                EvaluationSession.started_at >= since_date,
                EvaluationSession.status == "completed"
            )
        ).group_by(
            func.date(EvaluationSession.started_at)
        ).order_by(EvaluationSession.started_at)
        
        result = await self.session.execute(query)
        return [
            {
                "date": row.started_at.date(),
                "avg_score": float(row.avg_score),
                "evaluation_count": row.evaluation_count
            }
            for row in result
        ]
    
    async def _get_session_task_counts(self, session_id: int) -> Dict[str, int]:
        """Get task counts for a session."""
        query = select(
            func.count(TaskExecution.id).label("total"),
            func.count().filter(TaskExecution.status == "completed").label("completed"),
            func.count().filter(TaskExecution.status == "failed").label("failed")
        ).where(TaskExecution.session_id == session_id)
        
        result = await self.session.execute(query)
        row = result.first()
        
        return {
            "total": row.total or 0,
            "completed": row.completed or 0,
            "failed": row.failed or 0
        }