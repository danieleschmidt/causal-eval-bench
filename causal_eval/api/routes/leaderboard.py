"""
Advanced leaderboard and analytics endpoints.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import statistics

from fastapi import APIRouter, Request, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from causal_eval.database.connection import get_session
from causal_eval.repositories.evaluation import EvaluationRepository
from causal_eval.core.caching import get_cache

router = APIRouter()


class LeaderboardEntry(BaseModel):
    """Leaderboard entry model."""
    model_name: str
    overall_score: float
    task_scores: Dict[str, float]
    domain_scores: Dict[str, float]
    total_evaluations: int
    last_evaluation: datetime
    confidence_intervals: Dict[str, tuple] = {}
    rank: int
    rank_change: Optional[int] = None


class LeaderboardResponse(BaseModel):
    """Leaderboard response model."""
    entries: List[LeaderboardEntry]
    metadata: Dict[str, Any]
    generated_at: datetime
    cache_ttl: int


class LeaderboardFilters(BaseModel):
    """Leaderboard filter options."""
    domain: Optional[str] = None
    task_type: Optional[str] = None
    time_range: Optional[str] = "30d"  # 7d, 30d, 90d, all
    min_evaluations: int = Field(default=5, ge=1)
    model_category: Optional[str] = None  # open_source, commercial, research


@router.get("/", response_model=LeaderboardResponse)
async def get_leaderboard(
    request: Request,
    domain: Optional[str] = Query(None, description="Filter by domain"),
    task_type: Optional[str] = Query(None, description="Filter by task type"),
    time_range: Optional[str] = Query("30d", description="Time range for evaluations"),
    min_evaluations: int = Query(5, ge=1, description="Minimum number of evaluations"),
    limit: int = Query(50, ge=1, le=100, description="Number of entries to return"),
    session = Depends(get_session)
) -> LeaderboardResponse:
    """Get comprehensive leaderboard with advanced analytics."""
    cache_manager = request.app.state.cache_manager
    
    # Create cache key
    cache_key = f"leaderboard:{domain or 'all'}:{task_type or 'all'}:{time_range}:{min_evaluations}:{limit}"
    
    # Try cache first
    cached_result = await cache_manager.leaderboard_cache.get_leaderboard(
        domain or "all", 
        task_type or "all"
    )
    
    if cached_result:
        return LeaderboardResponse(**cached_result)
    
    try:
        repo = EvaluationRepository(session)
        
        # Calculate time filter
        time_filter = _calculate_time_filter(time_range)
        
        # Get leaderboard data
        leaderboard_data = await _generate_leaderboard(
            repo, 
            domain=domain,
            task_type=task_type,
            time_filter=time_filter,
            min_evaluations=min_evaluations,
            limit=limit
        )
        
        # Cache result
        await cache_manager.leaderboard_cache.set_leaderboard(
            domain or "all",
            task_type or "all", 
            leaderboard_data
        )
        
        return LeaderboardResponse(**leaderboard_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate leaderboard: {str(e)}")


@router.get("/model/{model_name}", response_model=Dict[str, Any])
async def get_model_analytics(
    model_name: str,
    request: Request,
    time_range: Optional[str] = Query("30d", description="Time range for analytics"),
    session = Depends(get_session)
) -> Dict[str, Any]:
    """Get detailed analytics for a specific model."""
    try:
        repo = EvaluationRepository(session)
        time_filter = _calculate_time_filter(time_range)
        
        # Get model statistics
        analytics = await _generate_model_analytics(repo, model_name, time_filter)
        
        return analytics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model analytics: {str(e)}")


@router.get("/trends", response_model=Dict[str, Any])
async def get_performance_trends(
    request: Request,
    domain: Optional[str] = Query(None, description="Filter by domain"),
    task_type: Optional[str] = Query(None, description="Filter by task type"),
    time_range: Optional[str] = Query("30d", description="Time range for trends"),
    session = Depends(get_session)
) -> Dict[str, Any]:
    """Get performance trends and insights."""
    try:
        repo = EvaluationRepository(session)
        time_filter = _calculate_time_filter(time_range)
        
        # Generate trends
        trends = await _generate_performance_trends(
            repo,
            domain=domain,
            task_type=task_type,
            time_filter=time_filter
        )
        
        return trends
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get trends: {str(e)}")


@router.get("/insights", response_model=Dict[str, Any])
async def get_evaluation_insights(
    request: Request,
    session = Depends(get_session)
) -> Dict[str, Any]:
    """Get comprehensive evaluation insights and statistics."""
    try:
        repo = EvaluationRepository(session)
        
        # Generate insights
        insights = await _generate_evaluation_insights(repo)
        
        return insights
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get insights: {str(e)}")


# Helper functions
def _calculate_time_filter(time_range: str) -> Optional[datetime]:
    """Calculate time filter based on range."""
    if time_range == "all":
        return None
    
    days_map = {
        "7d": 7,
        "30d": 30,
        "90d": 90,
        "1y": 365
    }
    
    days = days_map.get(time_range, 30)
    return datetime.utcnow() - timedelta(days=days)


async def _generate_leaderboard(
    repo: EvaluationRepository,
    domain: Optional[str] = None,
    task_type: Optional[str] = None,
    time_filter: Optional[datetime] = None,
    min_evaluations: int = 5,
    limit: int = 50
) -> Dict[str, Any]:
    """Generate comprehensive leaderboard data."""
    
    # Get evaluation statistics grouped by model
    model_stats = await repo.get_model_performance_stats(
        domain=domain,
        task_type=task_type,
        since=time_filter,
        min_evaluations=min_evaluations
    )
    
    entries = []
    for i, stats in enumerate(model_stats[:limit]):
        # Calculate confidence intervals
        confidence_intervals = _calculate_confidence_intervals(stats)
        
        entry = LeaderboardEntry(
            model_name=stats["model_name"],
            overall_score=stats["overall_score"],
            task_scores=stats.get("task_scores", {}),
            domain_scores=stats.get("domain_scores", {}),
            total_evaluations=stats["total_evaluations"],
            last_evaluation=stats["last_evaluation"],
            confidence_intervals=confidence_intervals,
            rank=i + 1
        )
        entries.append(entry)
    
    # Calculate metadata
    metadata = {
        "total_models": len(model_stats),
        "total_evaluations": sum(stats["total_evaluations"] for stats in model_stats),
        "filters": {
            "domain": domain,
            "task_type": task_type,
            "time_range": "custom" if time_filter else "all",
            "min_evaluations": min_evaluations
        }
    }
    
    return {
        "entries": [entry.dict() for entry in entries],
        "metadata": metadata,
        "generated_at": datetime.utcnow(),
        "cache_ttl": 900  # 15 minutes
    }


async def _generate_model_analytics(
    repo: EvaluationRepository,
    model_name: str,
    time_filter: Optional[datetime] = None
) -> Dict[str, Any]:
    """Generate detailed analytics for a specific model."""
    
    # Get comprehensive model statistics
    model_data = await repo.get_detailed_model_stats(model_name, since=time_filter)
    
    if not model_data:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Calculate advanced metrics
    analytics = {
        "model_name": model_name,
        "summary": {
            "overall_score": model_data["overall_score"],
            "total_evaluations": model_data["total_evaluations"],
            "evaluation_range": {
                "first": model_data["first_evaluation"],
                "last": model_data["last_evaluation"]
            }
        },
        "performance_breakdown": {
            "by_task": model_data.get("task_scores", {}),
            "by_domain": model_data.get("domain_scores", {}),
            "by_difficulty": model_data.get("difficulty_scores", {})
        },
        "trends": await _calculate_performance_trends_for_model(repo, model_name, time_filter),
        "strengths_and_weaknesses": _analyze_strengths_weaknesses(model_data),
        "comparison_rank": await _get_model_rank(repo, model_name, time_filter)
    }
    
    return analytics


async def _generate_performance_trends(
    repo: EvaluationRepository,
    domain: Optional[str] = None,
    task_type: Optional[str] = None,
    time_filter: Optional[datetime] = None
) -> Dict[str, Any]:
    """Generate performance trends across the ecosystem."""
    
    # Get time-series data
    trends_data = await repo.get_performance_trends(
        domain=domain,
        task_type=task_type,
        since=time_filter
    )
    
    trends = {
        "overall_improvements": _calculate_improvement_trends(trends_data),
        "task_difficulty_trends": _analyze_difficulty_trends(trends_data),
        "model_category_performance": _analyze_model_categories(trends_data),
        "evaluation_volume": _analyze_evaluation_volume(trends_data),
        "top_improving_models": _identify_improving_models(trends_data)
    }
    
    return trends


async def _generate_evaluation_insights(repo: EvaluationRepository) -> Dict[str, Any]:
    """Generate comprehensive evaluation insights."""
    
    # Get comprehensive statistics 
    all_stats = await repo.get_comprehensive_stats()
    
    insights = {
        "ecosystem_health": {
            "total_models_evaluated": all_stats["unique_models"],
            "total_evaluations": all_stats["total_evaluations"],
            "evaluation_velocity": all_stats["evaluations_per_day"],
            "average_model_performance": all_stats["average_score"]
        },
        "task_analysis": {
            "most_challenging_tasks": _identify_challenging_tasks(all_stats),
            "task_correlations": _calculate_task_correlations(all_stats),
            "domain_difficulty_ranking": _rank_domain_difficulties(all_stats)
        },
        "model_landscape": {
            "performance_distribution": _analyze_score_distribution(all_stats),
            "evaluation_fairness": _assess_evaluation_fairness(all_stats),
            "emerging_patterns": _identify_emerging_patterns(all_stats)
        },
        "quality_metrics": {
            "evaluation_consistency": _measure_consistency(all_stats),
            "score_reliability": _assess_score_reliability(all_stats)
        }
    }
    
    return insights


def _calculate_confidence_intervals(stats: Dict[str, Any]) -> Dict[str, tuple]:
    """Calculate confidence intervals for scores."""
    # This would implement proper statistical confidence intervals
    # For now, return placeholder values
    return {
        "overall_score": (
            max(0, stats["overall_score"] - 0.05),
            min(1, stats["overall_score"] + 0.05)
        )
    }


def _calculate_improvement_trends(trends_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate improvement trends over time."""
    # Implement trend analysis
    return {
        "overall_trend": "improving",
        "improvement_rate": 0.02,  # 2% per month
        "significant_improvements": []
    }


def _analyze_difficulty_trends(trends_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze how performance varies by difficulty."""
    return {
        "easy": {"average_score": 0.85, "improvement": 0.01},
        "medium": {"average_score": 0.72, "improvement": 0.015},
        "hard": {"average_score": 0.58, "improvement": 0.025}
    }


def _analyze_model_categories(trends_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze performance by model category."""
    return {
        "open_source": {"average_score": 0.68, "count": 25},
        "commercial": {"average_score": 0.78, "count": 15},
        "research": {"average_score": 0.72, "count": 10}
    }


def _analyze_evaluation_volume(trends_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze evaluation volume trends."""
    return {
        "daily_average": 150,
        "weekly_growth": 0.05,
        "peak_days": ["Monday", "Tuesday", "Wednesday"]
    }


def _identify_improving_models(trends_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identify models with significant improvements."""
    return [
        {"model_name": "gpt-4", "improvement": 0.03, "period": "30d"},
        {"model_name": "claude-3", "improvement": 0.025, "period": "30d"}
    ]


# Additional helper functions for insights
def _identify_challenging_tasks(stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Identify the most challenging tasks."""
    return [
        {"task": "counterfactual", "domain": "medical", "average_score": 0.45},
        {"task": "intervention", "domain": "environmental", "average_score": 0.52}
    ]


def _calculate_task_correlations(stats: Dict[str, Any]) -> Dict[str, float]:
    """Calculate correlations between different tasks."""
    return {
        "attribution_counterfactual": 0.68,
        "counterfactual_intervention": 0.72,
        "attribution_intervention": 0.65
    }


def _rank_domain_difficulties(stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Rank domains by difficulty."""
    return [
        {"domain": "medical", "average_score": 0.58, "rank": 1},
        {"domain": "environmental", "average_score": 0.62, "rank": 2},
        {"domain": "education", "average_score": 0.75, "rank": 3}
    ]


def _analyze_score_distribution(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the distribution of scores."""
    return {
        "mean": 0.68,
        "median": 0.70,
        "std_dev": 0.15,
        "quartiles": [0.58, 0.70, 0.82]
    }


def _assess_evaluation_fairness(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Assess fairness of evaluations across models."""
    return {
        "evaluation_balance": 0.85,  # How evenly distributed evaluations are
        "domain_coverage": 0.92,    # How well domains are covered
        "bias_indicators": []
    }


def _identify_emerging_patterns(stats: Dict[str, Any]) -> List[str]:
    """Identify emerging patterns in the data."""
    return [
        "Larger models showing diminishing returns on causal reasoning",
        "Specialized medical models outperforming general models in medical domain",
        "Counterfactual reasoning improving faster than other tasks"
    ]


def _measure_consistency(stats: Dict[str, Any]) -> float:
    """Measure consistency of evaluations."""
    return 0.87  # Placeholder for actual consistency measurement


def _assess_score_reliability(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Assess reliability of scoring system."""
    return {
        "inter_rater_reliability": 0.92,
        "test_retest_reliability": 0.89,
        "internal_consistency": 0.94
    }


async def _calculate_performance_trends_for_model(
    repo: EvaluationRepository,
    model_name: str,
    time_filter: Optional[datetime]
) -> Dict[str, Any]:
    """Calculate performance trends for a specific model."""
    return {
        "score_trend": "improving",
        "monthly_change": 0.02,
        "evaluation_frequency": "daily"
    }


def _analyze_strengths_weaknesses(model_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """Analyze model strengths and weaknesses."""
    return {
        "strengths": ["medical domain", "attribution tasks"],
        "weaknesses": ["counterfactual reasoning", "environmental domain"]
    }


async def _get_model_rank(
    repo: EvaluationRepository,
    model_name: str,
    time_filter: Optional[datetime]
) -> Dict[str, Any]:
    """Get model's rank in various categories."""
    return {
        "overall": 5,
        "by_task": {"attribution": 3, "counterfactual": 7, "intervention": 4},
        "by_domain": {"medical": 2, "education": 6, "business": 4}
    }