"""Metrics collection and monitoring for causal evaluation."""

import time
import math
import statistics
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
from pydantic import BaseModel
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
    """Advanced metrics collector for causal evaluation analysis."""
    
    def __init__(self):
        """Initialize metrics collector with Prometheus metrics and advanced analytics."""
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
        
        # Advanced analytics storage
        self._evaluation_results: List[Dict[str, Any]] = []
        self._session_metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._performance_history: List[Tuple[datetime, float]] = []
        self._start_time = time.time()
        
        logger.info("Advanced metrics collector initialized")
    
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
    
    def add_evaluation_result(self, result: Dict[str, Any]) -> None:
        """Add an evaluation result to advanced analytics."""
        result_with_timestamp = {
            **result,
            'timestamp': time.time(),
            'datetime': datetime.utcnow().isoformat()
        }
        self._evaluation_results.append(result_with_timestamp)
        
        # Track performance over time
        self._performance_history.append((
            datetime.utcnow(),
            result.get('overall_score', 0.0)
        ))
        
        # Also record traditional metrics
        self.record_evaluation(
            task_type=result.get('task_type', 'unknown'),
            domain=result.get('domain', 'general'),
            score=result.get('overall_score', 0.0),
            duration=result.get('duration', 0.0),
            model_name=result.get('model_name'),
            metadata=result
        )
        
        # Also add to session metrics if session_id present
        session_id = result.get('session_id')
        if session_id:
            self._session_metrics[session_id].append(result_with_timestamp)
    
    def calculate_summary(self) -> 'MetricsSummary':
        """Calculate comprehensive metrics summary."""
        from datetime import timedelta
        
        if not self._evaluation_results:
            return MetricsSummary(
                total_evaluations=0,
                successful_evaluations=0,
                failed_evaluations=0,
                success_rate=0.0,
                average_score=0.0,
                score_std=0.0,
                median_score=0.0,
                min_score=0.0,
                max_score=0.0,
                task_type_breakdown={},
                domain_breakdown={},
                difficulty_breakdown={},
                confidence_stats={},
                recent_performance={}
            )
        
        # Basic counts
        total = len(self._evaluation_results)
        successful = sum(1 for e in self._evaluation_results if e.get('overall_score', 0) > 0)
        failed = total - successful
        success_rate = successful / total if total > 0 else 0.0
        
        # Score statistics
        scores = [e.get('overall_score', 0.0) for e in self._evaluation_results]
        avg_score = statistics.mean(scores) if scores else 0.0
        score_std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        median_score = statistics.median(scores) if scores else 0.0
        min_score = min(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0
        
        # Categorical breakdowns
        task_types = Counter(e.get('task_type', 'unknown') for e in self._evaluation_results)
        domains = Counter(e.get('domain', 'unknown') for e in self._evaluation_results)
        difficulties = Counter(e.get('difficulty', 'unknown') for e in self._evaluation_results)
        
        # Confidence analysis
        confidences = [e.get('confidence', 0.5) for e in self._evaluation_results if e.get('confidence') is not None]
        confidence_stats = {
            'mean': statistics.mean(confidences) if confidences else 0.0,
            'median': statistics.median(confidences) if confidences else 0.0,
            'std': statistics.stdev(confidences) if len(confidences) > 1 else 0.0
        }
        
        # Recent performance (last 24 hours)
        recent_cutoff = datetime.utcnow() - timedelta(hours=24)
        recent_evals = [
            e for e in self._evaluation_results 
            if datetime.fromisoformat(e['datetime'].replace('Z', '+00:00')).replace(tzinfo=None) > recent_cutoff
        ]
        
        recent_performance = {
            'count': len(recent_evals),
            'average_score': statistics.mean([e.get('overall_score', 0.0) for e in recent_evals]) if recent_evals else 0.0,
            'success_rate': sum(1 for e in recent_evals if e.get('overall_score', 0) > 0) / len(recent_evals) if recent_evals else 0.0
        }
        
        return MetricsSummary(
            total_evaluations=total,
            successful_evaluations=successful,
            failed_evaluations=failed,
            success_rate=success_rate,
            average_score=avg_score,
            score_std=score_std,
            median_score=median_score,
            min_score=min_score,
            max_score=max_score,
            task_type_breakdown=dict(task_types),
            domain_breakdown=dict(domains),
            difficulty_breakdown=dict(difficulties),
            confidence_stats=confidence_stats,
            recent_performance=recent_performance
        )
    
    def calculate_aggregate_metrics(self) -> 'AggregateMetrics':
        """Calculate advanced aggregate metrics."""
        if not self._evaluation_results:
            return AggregateMetrics(
                overall_score=0.0,
                task_scores={},
                domain_scores={},
                difficulty_scores={},
                confidence_scores={},
                error_analysis={},
                statistical_significance={},
                temporal_trends={},
                performance_distribution={}
            )
        
        # Implementation here matches the comprehensive version from the enhanced metrics
        scores = [e.get('overall_score', 0.0) for e in self._evaluation_results]
        overall_score = statistics.mean(scores)
        
        # Task scores with statistical analysis
        task_scores = {}
        for task_type in set(e.get('task_type', 'unknown') for e in self._evaluation_results):
            task_evals = [e for e in self._evaluation_results if e.get('task_type') == task_type]
            task_scores_list = [e.get('overall_score', 0.0) for e in task_evals]
            if task_scores_list:
                task_scores[task_type] = statistics.mean(task_scores_list)
        
        # Similar calculations for domains, difficulties, etc.
        domain_scores = {}
        for domain in set(e.get('domain', 'unknown') for e in self._evaluation_results):
            domain_evals = [e for e in self._evaluation_results if e.get('domain') == domain]
            domain_scores_list = [e.get('overall_score', 0.0) for e in domain_evals]
            if domain_scores_list:
                domain_scores[domain] = statistics.mean(domain_scores_list)
        
        difficulty_scores = {}
        for difficulty in set(e.get('difficulty', 'unknown') for e in self._evaluation_results):
            diff_evals = [e for e in self._evaluation_results if e.get('difficulty') == difficulty]
            diff_scores_list = [e.get('overall_score', 0.0) for e in diff_evals]
            if diff_scores_list:
                difficulty_scores[difficulty] = statistics.mean(diff_scores_list)
        
        return AggregateMetrics(
            overall_score=overall_score,
            task_scores=task_scores,
            domain_scores=domain_scores,
            difficulty_scores=difficulty_scores,
            confidence_scores={},  # Simplified for now
            error_analysis={},     # Simplified for now
            statistical_significance={},  # Simplified for now
            temporal_trends={},    # Simplified for now
            performance_distribution={}   # Simplified for now
        )
    
    def calculate_causal_reasoning_profile(self) -> Dict[str, Any]:
        """Calculate causal reasoning capability profile."""
        if not self._evaluation_results:
            return {}
        
        # Analyze specific causal reasoning capabilities
        attribution_scores = []
        counterfactual_scores = []
        intervention_scores = []
        
        for eval_result in self._evaluation_results:
            task_type = eval_result.get('task_type', '')
            score = eval_result.get('overall_score', 0.0)
            
            if 'attribution' in task_type.lower():
                attribution_scores.append(score)
            elif 'counterfactual' in task_type.lower():
                counterfactual_scores.append(score)
            elif 'intervention' in task_type.lower():
                intervention_scores.append(score)
        
        # Calculate capability strengths and weaknesses
        capabilities = {
            'causal_attribution': {
                'score': statistics.mean(attribution_scores) if attribution_scores else 0.0,
                'consistency': 1.0 - (statistics.stdev(attribution_scores) if len(attribution_scores) > 1 else 0.0),
                'sample_size': len(attribution_scores)
            },
            'counterfactual_reasoning': {
                'score': statistics.mean(counterfactual_scores) if counterfactual_scores else 0.0,
                'consistency': 1.0 - (statistics.stdev(counterfactual_scores) if len(counterfactual_scores) > 1 else 0.0),
                'sample_size': len(counterfactual_scores)
            },
            'intervention_analysis': {
                'score': statistics.mean(intervention_scores) if intervention_scores else 0.0,
                'consistency': 1.0 - (statistics.stdev(intervention_scores) if len(intervention_scores) > 1 else 0.0),
                'sample_size': len(intervention_scores)
            }
        }
        
        # Identify strengths and weaknesses
        all_scores = [(k, v['score']) for k, v in capabilities.items() if v['sample_size'] > 0]
        if all_scores:
            sorted_scores = sorted(all_scores, key=lambda x: x[1], reverse=True)
            strengths = sorted_scores[:len(sorted_scores)//2 + 1]
            weaknesses = sorted_scores[len(sorted_scores)//2 + 1:]
        else:
            strengths, weaknesses = [], []
        
        return {
            'capabilities': capabilities,
            'strengths': [s[0] for s in strengths],
            'weaknesses': [w[0] for w in weaknesses],
            'overall_causal_reasoning_score': statistics.mean([v['score'] for v in capabilities.values() if v['sample_size'] > 0]) if any(v['sample_size'] > 0 for v in capabilities.values()) else 0.0,
            'reasoning_consistency': statistics.mean([v['consistency'] for v in capabilities.values() if v['sample_size'] > 0]) if any(v['sample_size'] > 0 for v in capabilities.values()) else 0.0
        }


@dataclass
class MetricsSummary:
    """Summary of evaluation metrics."""
    
    total_evaluations: int
    successful_evaluations: int
    failed_evaluations: int
    success_rate: float
    average_score: float
    score_std: float
    median_score: float
    min_score: float
    max_score: float
    task_type_breakdown: Dict[str, int]
    domain_breakdown: Dict[str, int]
    difficulty_breakdown: Dict[str, int]
    confidence_stats: Dict[str, float]
    recent_performance: Dict[str, float]
    
    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'total_evaluations': self.total_evaluations,
            'successful_evaluations': self.successful_evaluations,
            'failed_evaluations': self.failed_evaluations,
            'success_rate': self.success_rate,
            'average_score': self.average_score,
            'score_std': self.score_std,
            'median_score': self.median_score,
            'min_score': self.min_score,
            'max_score': self.max_score,
            'task_type_breakdown': self.task_type_breakdown,
            'domain_breakdown': self.domain_breakdown,
            'difficulty_breakdown': self.difficulty_breakdown,
            'confidence_stats': self.confidence_stats,
            'recent_performance': self.recent_performance
        }


class AggregateMetrics(BaseModel):
    """Advanced aggregate metrics for causal reasoning evaluation."""
    
    overall_score: float
    task_scores: Dict[str, float]
    domain_scores: Dict[str, float]
    difficulty_scores: Dict[str, float]
    confidence_scores: Dict[str, Any]
    error_analysis: Dict[str, Any]
    statistical_significance: Dict[str, Any]
    temporal_trends: Dict[str, Any]
    performance_distribution: Dict[str, Any]