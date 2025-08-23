"""Metrics collection and monitoring for causal evaluation."""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import logging

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Metrics for a single evaluation."""
    
    task_type: str
    domain: str
    score: float
    duration: float
    timestamp: float = field(default_factory=time.time)
    model_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects and manages evaluation metrics."""
    
    def __init__(self):
        """Initialize metrics collector with Prometheus metrics."""
        self.registry = CollectorRegistry()
        
        # Prometheus metrics
        self.evaluation_counter = Counter(
            'causal_eval_evaluations_total',
            'Total number of evaluations performed',
            ['task_type', 'domain', 'model_name'],
            registry=self.registry
        )
        
        self.evaluation_duration = Histogram(
            'causal_eval_duration_seconds',
            'Time spent on evaluations',
            ['task_type', 'domain'],
            registry=self.registry
        )
        
        self.evaluation_score = Histogram(
            'causal_eval_score',
            'Evaluation scores distribution',
            ['task_type', 'domain'],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )
        
        self.active_evaluations = Gauge(
            'causal_eval_active_evaluations',
            'Number of currently active evaluations',
            registry=self.registry
        )
        
        # In-memory storage for recent metrics
        self.recent_evaluations: List[EvaluationMetrics] = []
        self.max_recent_size = 1000
        
        logger.info("Metrics collector initialized")
    
    def record_evaluation(
        self,
        task_type: str,
        domain: str,
        score: float,
        duration: float,
        model_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record metrics for a completed evaluation."""
        
        # Create metrics object
        metrics = EvaluationMetrics(
            task_type=task_type,
            domain=domain,
            score=score,
            duration=duration,
            model_name=model_name or "unknown",
            metadata=metadata or {}
        )
        
        # Record to Prometheus
        self.evaluation_counter.labels(
            task_type=task_type,
            domain=domain,
            model_name=model_name or "unknown"
        ).inc()
        
        self.evaluation_duration.labels(
            task_type=task_type,
            domain=domain
        ).observe(duration)
        
        self.evaluation_score.labels(
            task_type=task_type,
            domain=domain
        ).observe(score)
        
        # Store in memory
        self.recent_evaluations.append(metrics)
        
        # Maintain size limit
        if len(self.recent_evaluations) > self.max_recent_size:
            self.recent_evaluations = self.recent_evaluations[-self.max_recent_size:]
        
        logger.debug(f"Recorded evaluation metrics: {task_type}/{domain} score={score:.3f}")
    
    def increment_active_evaluations(self) -> None:
        """Increment the active evaluations counter."""
        self.active_evaluations.inc()
    
    def decrement_active_evaluations(self) -> None:
        """Decrement the active evaluations counter."""
        self.active_evaluations.dec()
    
    def get_summary_stats(self, 
                         task_type: Optional[str] = None,
                         domain: Optional[str] = None,
                         time_window_hours: Optional[float] = None) -> Dict[str, Any]:
        """Get summary statistics for evaluations."""
        
        # Filter evaluations
        evaluations = self.recent_evaluations
        
        if time_window_hours:
            cutoff_time = time.time() - (time_window_hours * 3600)
            evaluations = [e for e in evaluations if e.timestamp > cutoff_time]
        
        if task_type:
            evaluations = [e for e in evaluations if e.task_type == task_type]
        
        if domain:
            evaluations = [e for e in evaluations if e.domain == domain]
        
        if not evaluations:
            return {"message": "No evaluations found for criteria"}
        
        # Calculate statistics
        scores = [e.score for e in evaluations]
        durations = [e.duration for e in evaluations]
        
        stats = {
            "count": len(evaluations),
            "score_stats": {
                "mean": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
                "median": sorted(scores)[len(scores)//2]
            },
            "duration_stats": {
                "mean": sum(durations) / len(durations),
                "min": min(durations),
                "max": max(durations),
                "total": sum(durations)
            },
            "task_types": list(set(e.task_type for e in evaluations)),
            "domains": list(set(e.domain for e in evaluations)),
            "models": list(set(e.model_name for e in evaluations if e.model_name))
        }
        
        return stats
    
    def get_leaderboard(self, 
                       task_type: Optional[str] = None,
                       domain: Optional[str] = None,
                       limit: int = 10) -> List[Dict[str, Any]]:
        """Get top performing models/evaluations."""
        
        # Filter evaluations
        evaluations = self.recent_evaluations
        
        if task_type:
            evaluations = [e for e in evaluations if e.task_type == task_type]
        
        if domain:
            evaluations = [e for e in evaluations if e.domain == domain]
        
        # Group by model and calculate average scores
        model_scores = {}
        for evaluation in evaluations:
            model = evaluation.model_name or "unknown"
            if model not in model_scores:
                model_scores[model] = []
            model_scores[model].append(evaluation.score)
        
        # Calculate averages and sort
        leaderboard = []
        for model, scores in model_scores.items():
            avg_score = sum(scores) / len(scores)
            leaderboard.append({
                "model_name": model,
                "average_score": avg_score,
                "evaluation_count": len(scores),
                "task_type": task_type,
                "domain": domain
            })
        
        # Sort by average score descending
        leaderboard.sort(key=lambda x: x["average_score"], reverse=True)
        
        return leaderboard[:limit]
    
    def get_recent_evaluations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get most recent evaluations."""
        recent = self.recent_evaluations[-limit:] if limit else self.recent_evaluations
        
        return [
            {
                "task_type": e.task_type,
                "domain": e.domain,
                "score": e.score,
                "duration": e.duration,
                "timestamp": e.timestamp,
                "model_name": e.model_name,
                "metadata": e.metadata
            }
            for e in reversed(recent)
        ]
    
    def clear_recent_evaluations(self) -> None:
        """Clear recent evaluations cache."""
        self.recent_evaluations.clear()
        logger.info("Cleared recent evaluations cache")