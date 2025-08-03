"""Metrics collection and calculation for causal evaluation."""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pydantic import BaseModel
import logging
import statistics
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """A single metric calculation result."""
    
    name: str
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationSummary:
    """Summary of evaluation results across tasks."""
    
    overall_score: float
    task_scores: Dict[str, float]
    domain_scores: Dict[str, float]
    difficulty_scores: Dict[str, float]
    confidence_scores: Dict[str, float]
    error_analysis: Dict[str, Any]
    statistical_significance: Dict[str, float]


class MetricsSummary(BaseModel):
    """Summary of evaluation metrics."""
    
    total_evaluations: int
    average_score: float
    average_reasoning_quality: float
    domain_breakdown: Dict[str, Dict[str, float]]
    difficulty_breakdown: Dict[str, Dict[str, float]]


class MetricsCollector:
    """Collects and calculates evaluation metrics."""
    
    def __init__(self):
        """Initialize the metrics collector."""
        self.metrics: List[MetricResult] = []
        self.raw_results: List[Dict[str, Any]] = []
        self.results: List[Dict[str, Any]] = []  # Legacy compatibility
        logger.info("Metrics collector initialized")
    
    def add_metric(self, name: str, value: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a metric result."""
        metric = MetricResult(
            name=name,
            value=value,
            metadata=metadata or {}
        )
        self.metrics.append(metric)
        logger.info(f"Added metric: {name} = {value}")
    
    def add_evaluation_result(self, result: Dict[str, Any]) -> None:
        """Add a complete evaluation result for analysis."""
        self.raw_results.append(result)
    
    def add_result(self, result: Dict[str, Any]) -> None:
        """Add an evaluation result to the collection (legacy compatibility)."""
        self.results.append(result)
        self.raw_results.append(result)
        logger.debug(f"Added result for task: {result.get('task_id', 'unknown')}")
    
    def add_batch_results(self, results: List[Dict[str, Any]]) -> None:
        """Add multiple evaluation results."""
        self.results.extend(results)
        self.raw_results.extend(results)
        logger.info(f"Added {len(results)} results to collection")
    
    def calculate_aggregate_metrics(self) -> EvaluationSummary:
        """Calculate comprehensive aggregate metrics from collected results."""
        if not self.raw_results:
            return self._empty_summary()
        
        # Extract scores by category
        overall_scores = [r.get("overall_score", 0.0) for r in self.raw_results]
        task_results = defaultdict(list)
        domain_results = defaultdict(list)
        difficulty_results = defaultdict(list)
        confidence_scores = [r.get("confidence", 0.5) for r in self.raw_results]
        
        for result in self.raw_results:
            # Task-specific scores
            if "task_id" in result:
                task_results[result["task_id"]].append(result.get("overall_score", 0.0))
            
            # Domain-specific scores
            domain = result.get("scenario_domain", result.get("domain", "unknown"))
            domain_results[domain].append(result.get("overall_score", 0.0))
            
            # Difficulty-specific scores
            difficulty = result.get("scenario_difficulty", result.get("difficulty", "unknown"))
            difficulty_results[difficulty].append(result.get("overall_score", 0.0))
        
        return EvaluationSummary(
            overall_score=statistics.mean(overall_scores) if overall_scores else 0.0,
            task_scores={task: statistics.mean(scores) for task, scores in task_results.items()},
            domain_scores={domain: statistics.mean(scores) for domain, scores in domain_results.items()},
            difficulty_scores={diff: statistics.mean(scores) for diff, scores in difficulty_results.items()},
            confidence_scores=self._analyze_confidence(confidence_scores, overall_scores),
            error_analysis=self._analyze_errors(),
            statistical_significance=self._calculate_statistical_significance()
        )
    
    def calculate_summary(self) -> MetricsSummary:
        """Calculate summary statistics from collected results (legacy compatibility)."""
        if not self.results:
            return MetricsSummary(
                total_evaluations=0,
                average_score=0.0,
                average_reasoning_quality=0.0,
                domain_breakdown={},
                difficulty_breakdown={}
            )
        
        # Calculate overall averages
        scores = [r.get("score", r.get("overall_score", 0.0)) for r in self.results]
        reasoning_scores = [r.get("reasoning_quality", r.get("reasoning_score", 0.0)) for r in self.results]
        
        avg_score = statistics.mean(scores) if scores else 0.0
        avg_reasoning = statistics.mean(reasoning_scores) if reasoning_scores else 0.0
        
        # Domain breakdown
        domain_stats = defaultdict(list)
        for result in self.results:
            domain = result.get("domain", result.get("scenario_domain", "unknown"))
            domain_stats[domain].append({
                "score": result.get("score", result.get("overall_score", 0.0)),
                "reasoning_quality": result.get("reasoning_quality", result.get("reasoning_score", 0.0))
            })
        
        domain_breakdown = {}
        for domain, stats in domain_stats.items():
            domain_breakdown[domain] = {
                "average_score": statistics.mean([s["score"] for s in stats]),
                "average_reasoning_quality": statistics.mean([s["reasoning_quality"] for s in stats]),
                "count": len(stats)
            }
        
        return MetricsSummary(
            total_evaluations=len(self.results),
            average_score=avg_score,
            average_reasoning_quality=avg_reasoning,
            domain_breakdown=domain_breakdown,
            difficulty_breakdown={}  # TODO: Implement difficulty breakdown
        )
    
    def _empty_summary(self) -> EvaluationSummary:
        """Return empty evaluation summary."""
        return EvaluationSummary(
            overall_score=0.0,
            task_scores={},
            domain_scores={},
            difficulty_scores={},
            confidence_scores={},
            error_analysis={},
            statistical_significance={}
        )
    
    def _analyze_confidence(self, confidence_scores: List[float], overall_scores: List[float]) -> Dict[str, float]:
        """Analyze relationship between confidence and performance."""
        if not confidence_scores or not overall_scores:
            return {}
        
        # Calculate correlation between confidence and performance
        try:
            correlation = np.corrcoef(confidence_scores, overall_scores)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        except Exception:
            correlation = 0.0
        
        # Calibration analysis (how well confidence predicts performance)
        confidence_bins = np.linspace(0, 1, 6)  # 5 bins
        calibration_scores = []
        
        for i in range(len(confidence_bins) - 1):
            bin_mask = (np.array(confidence_scores) >= confidence_bins[i]) & (np.array(confidence_scores) < confidence_bins[i + 1])
            if np.any(bin_mask):
                bin_scores = np.array(overall_scores)[bin_mask]
                bin_conf = np.array(confidence_scores)[bin_mask]
                calibration_scores.append({
                    "confidence_range": f"{confidence_bins[i]:.1f}-{confidence_bins[i+1]:.1f}",
                    "avg_confidence": float(np.mean(bin_conf)),
                    "avg_performance": float(np.mean(bin_scores)),
                    "count": int(np.sum(bin_mask))
                })
        
        return {
            "confidence_performance_correlation": float(correlation),
            "average_confidence": float(np.mean(confidence_scores)),
            "confidence_std": float(np.std(confidence_scores)),
            "calibration_analysis": calibration_scores
        }
    
    def _analyze_errors(self) -> Dict[str, Any]:
        """Analyze common error patterns."""
        if not self.raw_results:
            return {}
        
        error_patterns = defaultdict(int)
        low_score_results = [r for r in self.raw_results if r.get("overall_score", 0) < 0.5]
        
        # Analyze low-scoring results for patterns
        for result in low_score_results:
            # Task-specific error patterns
            task_id = result.get("task_id", "unknown")
            error_patterns[f"low_score_{task_id}"] += 1
            
            # Domain-specific patterns
            domain = result.get("scenario_domain", result.get("domain", "unknown"))
            error_patterns[f"domain_difficulty_{domain}"] += 1
            
            # Specific error types
            if result.get("relationship_score", 1.0) < 0.3:
                error_patterns["relationship_misidentification"] += 1
            
            if result.get("reasoning_score", 1.0) < 0.3:
                error_patterns["poor_reasoning_quality"] += 1
            
            if result.get("confidence", 0.5) > 0.8 and result.get("overall_score", 0) < 0.3:
                error_patterns["overconfident_errors"] += 1
        
        # Calculate error rates
        total_results = len(self.raw_results)
        error_rates = {pattern: count / total_results for pattern, count in error_patterns.items()}
        
        return {
            "total_evaluations": total_results,
            "low_score_count": len(low_score_results),
            "low_score_rate": len(low_score_results) / total_results if total_results > 0 else 0,
            "error_patterns": dict(error_patterns),
            "error_rates": error_rates,
            "most_common_errors": sorted(error_rates.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def _calculate_statistical_significance(self) -> Dict[str, float]:
        """Calculate statistical significance of results."""
        if len(self.raw_results) < 10:
            return {"note": "Insufficient data for statistical analysis"}
        
        overall_scores = [r.get("overall_score", 0.0) for r in self.raw_results]
        
        # Basic statistical measures
        stats = {
            "mean": float(np.mean(overall_scores)),
            "std": float(np.std(overall_scores)),
            "sem": float(np.std(overall_scores) / np.sqrt(len(overall_scores))),  # Standard error of mean
            "min": float(np.min(overall_scores)),
            "max": float(np.max(overall_scores)),
            "median": float(np.median(overall_scores)),
            "q25": float(np.percentile(overall_scores, 25)),
            "q75": float(np.percentile(overall_scores, 75)),
            "sample_size": len(overall_scores)
        }
        
        # Confidence intervals (95%)
        stats["ci_95_lower"] = stats["mean"] - 1.96 * stats["sem"]
        stats["ci_95_upper"] = stats["mean"] + 1.96 * stats["sem"]
        
        # Test against random performance (0.5)
        if stats["sem"] > 0:
            t_stat = (stats["mean"] - 0.5) / stats["sem"]
            stats["t_statistic_vs_random"] = float(t_stat)
        
        return stats
    
    def calculate_causal_reasoning_profile(self) -> Dict[str, Any]:
        """Calculate a comprehensive causal reasoning capability profile."""
        if not self.raw_results:
            return {}
        
        profile = {
            "causal_attribution": self._calculate_task_profile("attribution"),
            "counterfactual_reasoning": self._calculate_task_profile("counterfactual"),
            "intervention_analysis": self._calculate_task_profile("intervention"),
            "causal_chain_reasoning": self._calculate_task_profile("chain"),
            "confounding_analysis": self._calculate_task_profile("confounding")
        }
        
        # Overall causal reasoning strengths and weaknesses
        task_scores = [(task, metrics.get("average_score", 0)) for task, metrics in profile.items() if metrics]
        
        if task_scores:
            sorted_tasks = sorted(task_scores, key=lambda x: x[1], reverse=True)
            profile["strengths"] = [task for task, score in sorted_tasks[:2]]
            profile["weaknesses"] = [task for task, score in sorted_tasks[-2:]]
            profile["causal_reasoning_score"] = statistics.mean([score for _, score in task_scores])
        
        return profile
    
    def _calculate_task_profile(self, task_type: str) -> Dict[str, Any]:
        """Calculate performance profile for a specific task type."""
        task_results = [r for r in self.raw_results if task_type in r.get("task_id", "").lower()]
        
        if not task_results:
            return {}
        
        scores = [r.get("overall_score", 0.0) for r in task_results]
        confidences = [r.get("confidence", 0.5) for r in task_results]
        
        # Domain breakdown
        domain_performance = defaultdict(list)
        for result in task_results:
            domain = result.get("scenario_domain", result.get("domain", "unknown"))
            domain_performance[domain].append(result.get("overall_score", 0.0))
        
        return {
            "average_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "average_confidence": float(np.mean(confidences)),
            "sample_size": len(task_results),
            "domain_breakdown": {domain: float(np.mean(scores)) for domain, scores in domain_performance.items()},
            "best_domain": max(domain_performance.items(), key=lambda x: np.mean(x[1]))[0] if domain_performance else None,
            "worst_domain": min(domain_performance.items(), key=lambda x: np.mean(x[1]))[0] if domain_performance else None
        }
    
    def export_results(self, format_type: str = "json") -> str:
        """Export results in specified format."""
        if format_type == "json":
            import json
            return json.dumps(self.results, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def clear_results(self) -> None:
        """Clear all collected results."""
        self.results.clear()
        self.raw_results.clear()
        self.metrics.clear()
        logger.info("Cleared all collected results")