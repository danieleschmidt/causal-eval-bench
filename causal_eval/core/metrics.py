"""Metrics collection and analysis for evaluation results."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from collections import defaultdict
import statistics
import logging

logger = logging.getLogger(__name__)


class MetricsSummary(BaseModel):
    """Summary of evaluation metrics."""
    
    total_evaluations: int
    average_score: float
    average_reasoning_quality: float
    domain_breakdown: Dict[str, Dict[str, float]]
    difficulty_breakdown: Dict[str, Dict[str, float]]


class MetricsCollector:
    """Collects and analyzes evaluation metrics."""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        logger.info("Metrics collector initialized")
    
    def add_result(self, result: Dict[str, Any]) -> None:
        """Add an evaluation result to the collection."""
        self.results.append(result)
        logger.debug(f"Added result for task: {result.get('task_id', 'unknown')}")
    
    def add_batch_results(self, results: List[Dict[str, Any]]) -> None:
        """Add multiple evaluation results."""
        self.results.extend(results)
        logger.info(f"Added {len(results)} results to collection")
    
    def calculate_summary(self) -> MetricsSummary:
        """Calculate summary statistics from collected results."""
        if not self.results:
            return MetricsSummary(
                total_evaluations=0,
                average_score=0.0,
                average_reasoning_quality=0.0,
                domain_breakdown={},
                difficulty_breakdown={}
            )
        
        # Calculate overall averages
        scores = [r.get("score", 0.0) for r in self.results]
        reasoning_scores = [r.get("reasoning_quality", 0.0) for r in self.results]
        
        avg_score = statistics.mean(scores) if scores else 0.0
        avg_reasoning = statistics.mean(reasoning_scores) if reasoning_scores else 0.0
        
        # Domain breakdown
        domain_stats = defaultdict(list)
        for result in self.results:
            domain = result.get("domain", "unknown")
            domain_stats[domain].append({
                "score": result.get("score", 0.0),
                "reasoning_quality": result.get("reasoning_quality", 0.0)
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
        logger.info("Cleared all collected results")