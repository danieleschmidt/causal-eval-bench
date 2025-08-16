"""
Research Discovery Module for Causal Reasoning Evaluation

This module implements advanced research discovery capabilities to identify
novel patterns, gaps, and opportunities in causal reasoning evaluation.
It provides data-driven insights for future research directions.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import logging
from collections import defaultdict, Counter
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class ResearchGap:
    """Represents an identified research gap or opportunity."""
    
    gap_type: str  # "performance", "methodology", "domain", "theoretical"
    title: str
    description: str
    evidence: Dict[str, Any]
    priority: str  # "high", "medium", "low"
    estimated_impact: float  # 0.0 to 1.0
    difficulty: str  # "easy", "medium", "hard"
    suggested_approaches: List[str]
    related_gaps: List[str] = field(default_factory=list)


@dataclass
class NovelPattern:
    """Represents a discovered novel pattern in the data."""
    
    pattern_type: str
    description: str
    statistical_significance: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    supporting_evidence: Dict[str, Any]
    implications: List[str]
    reproducibility_score: float


@dataclass
class ResearchOpportunity:
    """Represents a concrete research opportunity."""
    
    title: str
    research_question: str
    hypothesis: str
    methodology: str
    expected_outcomes: List[str]
    resource_requirements: Dict[str, Any]
    timeline: str
    potential_impact: str
    collaborators: List[str] = field(default_factory=list)


class PerformancePatternAnalyzer:
    """Analyzes performance patterns to identify research opportunities."""
    
    def __init__(self):
        self.patterns_cache = {}
        
    def analyze_performance_landscapes(
        self, 
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze performance landscapes across models and tasks."""
        
        logger.info("Analyzing performance landscapes")
        
        # Extract performance matrices
        model_task_matrix = self._build_performance_matrix(results)
        
        # Perform clustering analysis
        model_clusters = self._cluster_models(model_task_matrix)
        task_clusters = self._cluster_tasks(model_task_matrix.T)
        
        # Identify performance gaps
        performance_gaps = self._identify_performance_gaps(model_task_matrix)
        
        # Analyze scaling patterns
        scaling_patterns = self._analyze_scaling_patterns(results)
        
        # Detect ceiling effects
        ceiling_effects = self._detect_ceiling_effects(model_task_matrix)
        
        # Find unexpected correlations
        unexpected_correlations = self._find_unexpected_correlations(model_task_matrix)
        
        return {
            "model_task_matrix": model_task_matrix.tolist(),
            "model_clusters": model_clusters,
            "task_clusters": task_clusters,
            "performance_gaps": performance_gaps,
            "scaling_patterns": scaling_patterns,
            "ceiling_effects": ceiling_effects,
            "unexpected_correlations": unexpected_correlations
        }
    
    def _build_performance_matrix(self, results: List[Dict[str, Any]]) -> np.ndarray:
        """Build a performance matrix from experimental results."""
        
        # Extract unique models and tasks
        all_models = set()
        all_tasks = set()
        
        for result in results:
            if 'model_results' in result:
                all_models.update(result['model_results'].keys())
            if 'task_results' in result:
                all_tasks.update(result['task_results'].keys())
        
        models = sorted(list(all_models))
        tasks = sorted(list(all_tasks))
        
        # Build matrix
        matrix = np.zeros((len(models), len(tasks)))
        
        for result in results:
            model_results = result.get('model_results', {})
            for i, model in enumerate(models):
                if model in model_results:
                    for j, task in enumerate(tasks):
                        if task in model_results[model]:
                            score = model_results[model][task].get('score', 0)
                            matrix[i, j] = score
        
        return matrix
    
    def _cluster_models(self, performance_matrix: np.ndarray) -> Dict[str, Any]:
        """Cluster models based on performance patterns."""
        
        if performance_matrix.shape[0] < 3:
            return {"clusters": [], "analysis": "Insufficient models for clustering"}
        
        # Apply K-means clustering
        n_clusters = min(3, performance_matrix.shape[0])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(performance_matrix)
        
        # Analyze cluster characteristics
        clusters = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_performance = performance_matrix[cluster_indices]
            
            clusters.append({
                "cluster_id": cluster_id,
                "model_indices": cluster_indices.tolist(),
                "avg_performance": np.mean(cluster_performance),
                "std_performance": np.std(cluster_performance),
                "strengths": self._identify_cluster_strengths(cluster_performance),
                "weaknesses": self._identify_cluster_weaknesses(cluster_performance)
            })
        
        return {
            "clusters": clusters,
            "n_clusters": n_clusters,
            "silhouette_score": self._calculate_silhouette_score(performance_matrix, cluster_labels)
        }
    
    def _cluster_tasks(self, task_matrix: np.ndarray) -> Dict[str, Any]:
        """Cluster tasks based on difficulty patterns."""
        
        if task_matrix.shape[0] < 3:
            return {"clusters": [], "analysis": "Insufficient tasks for clustering"}
        
        # Apply K-means clustering
        n_clusters = min(3, task_matrix.shape[0])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(task_matrix)
        
        # Analyze task difficulty levels
        clusters = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_difficulty = np.mean(task_matrix[cluster_indices])
            
            difficulty_level = "easy" if cluster_difficulty > 0.8 else \
                              "medium" if cluster_difficulty > 0.6 else "hard"
            
            clusters.append({
                "cluster_id": cluster_id,
                "task_indices": cluster_indices.tolist(),
                "difficulty_level": difficulty_level,
                "avg_performance": cluster_difficulty,
                "variance": np.var(task_matrix[cluster_indices])
            })
        
        return {
            "clusters": clusters,
            "difficulty_distribution": {
                "easy_tasks": len([c for c in clusters if c["difficulty_level"] == "easy"]),
                "medium_tasks": len([c for c in clusters if c["difficulty_level"] == "medium"]),
                "hard_tasks": len([c for c in clusters if c["difficulty_level"] == "hard"])
            }
        }
    
    def _identify_performance_gaps(self, matrix: np.ndarray) -> List[Dict[str, Any]]:
        """Identify significant performance gaps."""
        
        gaps = []
        
        # Find tasks where all models perform poorly
        task_means = np.mean(matrix, axis=0)
        difficult_tasks = np.where(task_means < 0.5)[0]
        
        for task_idx in difficult_tasks:
            gaps.append({
                "type": "universal_difficulty",
                "task_index": int(task_idx),
                "avg_performance": float(task_means[task_idx]),
                "description": f"All models struggle with task {task_idx}",
                "opportunity": "Develop specialized approaches for this task type"
            })
        
        # Find models with inconsistent performance
        model_stds = np.std(matrix, axis=1)
        inconsistent_models = np.where(model_stds > 0.3)[0]
        
        for model_idx in inconsistent_models:
            gaps.append({
                "type": "inconsistent_performance",
                "model_index": int(model_idx),
                "performance_std": float(model_stds[model_idx]),
                "description": f"Model {model_idx} shows high performance variance",
                "opportunity": "Investigate and address performance inconsistencies"
            })
        
        # Find large gaps between best and worst performing models
        task_ranges = np.max(matrix, axis=0) - np.min(matrix, axis=0)
        high_variance_tasks = np.where(task_ranges > 0.4)[0]
        
        for task_idx in high_variance_tasks:
            gaps.append({
                "type": "high_model_variance",
                "task_index": int(task_idx),
                "performance_range": float(task_ranges[task_idx]),
                "description": f"Large performance gap between models on task {task_idx}",
                "opportunity": "Study what makes top models successful on this task"
            })
        
        return gaps
    
    def _analyze_scaling_patterns(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how performance scales with model size or complexity."""
        
        scaling_patterns = {
            "size_performance_correlation": None,
            "scaling_laws": {},
            "efficiency_analysis": {},
            "diminishing_returns": []
        }
        
        # Extract model size information if available
        model_sizes = {}
        model_performances = {}
        
        for result in results:
            model_results = result.get('model_results', {})
            for model, model_data in model_results.items():
                if 'model_size' in model_data:
                    model_sizes[model] = model_data['model_size']
                if 'overall_score' in model_data:
                    model_performances[model] = model_data['overall_score']
        
        if len(model_sizes) > 2 and len(model_performances) > 2:
            # Calculate correlation between size and performance
            common_models = set(model_sizes.keys()) & set(model_performances.keys())
            if len(common_models) > 2:
                sizes = [model_sizes[m] for m in common_models]
                perfs = [model_performances[m] for m in common_models]
                
                correlation, p_value = stats.pearsonr(sizes, perfs)
                scaling_patterns["size_performance_correlation"] = {
                    "correlation": correlation,
                    "p_value": p_value,
                    "significant": p_value < 0.05
                }
        
        return scaling_patterns
    
    def _detect_ceiling_effects(self, matrix: np.ndarray) -> Dict[str, Any]:
        """Detect ceiling effects in performance."""
        
        ceiling_effects = {
            "saturated_tasks": [],
            "ceiling_threshold": 0.95,
            "improvement_potential": {}
        }
        
        # Find tasks where top performance is very high
        max_performances = np.max(matrix, axis=0)
        saturated_tasks = np.where(max_performances > 0.95)[0]
        
        for task_idx in saturated_tasks:
            ceiling_effects["saturated_tasks"].append({
                "task_index": int(task_idx),
                "max_performance": float(max_performances[task_idx]),
                "description": f"Task {task_idx} may have reached performance ceiling"
            })
        
        # Calculate improvement potential for each task
        for task_idx in range(matrix.shape[1]):
            current_best = np.max(matrix[:, task_idx])
            avg_performance = np.mean(matrix[:, task_idx])
            improvement_potential = 1.0 - current_best
            
            ceiling_effects["improvement_potential"][str(task_idx)] = {
                "current_best": float(current_best),
                "average": float(avg_performance),
                "improvement_potential": float(improvement_potential),
                "difficulty_level": "easy" if current_best > 0.8 else 
                                  "medium" if current_best > 0.6 else "hard"
            }
        
        return ceiling_effects
    
    def _find_unexpected_correlations(self, matrix: np.ndarray) -> List[Dict[str, Any]]:
        """Find unexpected correlations in performance patterns."""
        
        correlations = []
        
        # Calculate task-task correlations
        task_corr_matrix = np.corrcoef(matrix.T)
        
        # Find surprisingly high or low correlations
        for i in range(task_corr_matrix.shape[0]):
            for j in range(i + 1, task_corr_matrix.shape[1]):
                corr = task_corr_matrix[i, j]
                
                # High unexpected correlation
                if corr > 0.8:
                    correlations.append({
                        "type": "high_task_correlation",
                        "task_indices": [i, j],
                        "correlation": float(corr),
                        "description": f"Tasks {i} and {j} show surprisingly high correlation",
                        "implication": "May indicate redundancy or shared underlying skills"
                    })
                
                # Low unexpected correlation (if tasks should be related)
                if corr < 0.1:
                    correlations.append({
                        "type": "low_task_correlation",
                        "task_indices": [i, j],
                        "correlation": float(corr),
                        "description": f"Tasks {i} and {j} show surprisingly low correlation",
                        "implication": "May indicate distinct skill requirements"
                    })
        
        return correlations
    
    def _identify_cluster_strengths(self, cluster_performance: np.ndarray) -> List[str]:
        """Identify strengths of a model cluster."""
        strengths = []
        
        task_means = np.mean(cluster_performance, axis=0)
        strong_tasks = np.where(task_means > 0.8)[0]
        
        if len(strong_tasks) > 0:
            strengths.append(f"Strong performance on {len(strong_tasks)} tasks")
        
        if np.std(task_means) < 0.1:
            strengths.append("Consistent performance across tasks")
        
        return strengths
    
    def _identify_cluster_weaknesses(self, cluster_performance: np.ndarray) -> List[str]:
        """Identify weaknesses of a model cluster."""
        weaknesses = []
        
        task_means = np.mean(cluster_performance, axis=0)
        weak_tasks = np.where(task_means < 0.5)[0]
        
        if len(weak_tasks) > 0:
            weaknesses.append(f"Poor performance on {len(weak_tasks)} tasks")
        
        if np.std(task_means) > 0.3:
            weaknesses.append("Inconsistent performance across tasks")
        
        return weaknesses
    
    def _calculate_silhouette_score(self, matrix: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score for clustering quality."""
        try:
            from sklearn.metrics import silhouette_score
            return float(silhouette_score(matrix, labels))
        except:
            return 0.0


class ResearchGapIdentifier:
    """Identifies gaps in current research and evaluation approaches."""
    
    def __init__(self):
        self.gap_patterns = self._load_gap_patterns()
    
    def identify_gaps(
        self, 
        performance_analysis: Dict[str, Any],
        literature_context: Optional[Dict[str, Any]] = None
    ) -> List[ResearchGap]:
        """Identify research gaps from performance analysis."""
        
        logger.info("Identifying research gaps")
        
        gaps = []
        
        # Performance gaps
        gaps.extend(self._identify_performance_gaps(performance_analysis))
        
        # Methodological gaps
        gaps.extend(self._identify_methodological_gaps(performance_analysis))
        
        # Domain coverage gaps
        gaps.extend(self._identify_domain_gaps(performance_analysis))
        
        # Theoretical gaps
        gaps.extend(self._identify_theoretical_gaps(performance_analysis))
        
        # Evaluation framework gaps
        gaps.extend(self._identify_evaluation_gaps(performance_analysis))
        
        # Prioritize gaps
        gaps = self._prioritize_gaps(gaps)
        
        return gaps
    
    def _identify_performance_gaps(self, analysis: Dict[str, Any]) -> List[ResearchGap]:
        """Identify gaps in model performance."""
        
        gaps = []
        
        # Check for universally difficult tasks
        if 'performance_gaps' in analysis:
            for gap in analysis['performance_gaps']:
                if gap['type'] == 'universal_difficulty':
                    research_gap = ResearchGap(
                        gap_type="performance",
                        title="Universal Task Difficulty",
                        description=f"All models struggle with specific task types (avg performance: {gap['avg_performance']:.3f})",
                        evidence={"performance_data": gap},
                        priority="high",
                        estimated_impact=0.8,
                        difficulty="medium",
                        suggested_approaches=[
                            "Develop specialized architectures for difficult tasks",
                            "Create targeted training datasets",
                            "Investigate task-specific reasoning patterns"
                        ]
                    )
                    gaps.append(research_gap)
        
        # Check for ceiling effects
        if 'ceiling_effects' in analysis:
            saturated_count = len(analysis['ceiling_effects']['saturated_tasks'])
            if saturated_count > 0:
                research_gap = ResearchGap(
                    gap_type="performance",
                    title="Performance Ceiling Effects",
                    description=f"Multiple tasks ({saturated_count}) show ceiling effects, limiting evaluation discrimination",
                    evidence={"ceiling_data": analysis['ceiling_effects']},
                    priority="medium",
                    estimated_impact=0.6,
                    difficulty="easy",
                    suggested_approaches=[
                        "Develop more challenging evaluation tasks",
                        "Create fine-grained performance metrics",
                        "Design adaptive difficulty systems"
                    ]
                )
                gaps.append(research_gap)
        
        return gaps
    
    def _identify_methodological_gaps(self, analysis: Dict[str, Any]) -> List[ResearchGap]:
        """Identify methodological gaps in evaluation approaches."""
        
        gaps = []
        
        # Limited evaluation metrics
        research_gap = ResearchGap(
            gap_type="methodology",
            title="Limited Evaluation Metrics",
            description="Current evaluation relies primarily on accuracy-based metrics, missing nuanced aspects of causal reasoning",
            evidence={"metric_analysis": "accuracy_focused"},
            priority="high",
            estimated_impact=0.7,
            difficulty="medium",
            suggested_approaches=[
                "Develop process-based evaluation metrics",
                "Create explanation quality assessments",
                "Design confidence calibration measures",
                "Implement reasoning path analysis"
            ]
        )
        gaps.append(research_gap)
        
        # Static evaluation framework
        research_gap = ResearchGap(
            gap_type="methodology",
            title="Static Evaluation Framework",
            description="Current framework uses fixed test sets, missing adaptive and interactive evaluation scenarios",
            evidence={"framework_analysis": "static_tests"},
            priority="medium",
            estimated_impact=0.6,
            difficulty="hard",
            suggested_approaches=[
                "Develop adaptive testing systems",
                "Create interactive evaluation environments",
                "Design personalized difficulty adjustment",
                "Implement real-time feedback mechanisms"
            ]
        )
        gaps.append(research_gap)
        
        return gaps
    
    def _identify_domain_gaps(self, analysis: Dict[str, Any]) -> List[ResearchGap]:
        """Identify gaps in domain coverage."""
        
        gaps = []
        
        # Limited domain diversity
        research_gap = ResearchGap(
            gap_type="domain",
            title="Limited Domain Diversity",
            description="Evaluation focuses on common domains, missing specialized fields and cross-domain reasoning",
            evidence={"domain_analysis": "limited_coverage"},
            priority="medium",
            estimated_impact=0.5,
            difficulty="medium",
            suggested_approaches=[
                "Expand to scientific domains (physics, chemistry, biology)",
                "Include social sciences (psychology, sociology)",
                "Add technical domains (engineering, computer science)",
                "Create cross-domain reasoning tasks"
            ]
        )
        gaps.append(research_gap)
        
        # Cultural and linguistic bias
        research_gap = ResearchGap(
            gap_type="domain",
            title="Cultural and Linguistic Bias",
            description="Evaluation primarily uses English and Western cultural contexts",
            evidence={"cultural_analysis": "western_bias"},
            priority="high",
            estimated_impact=0.8,
            difficulty="hard",
            suggested_approaches=[
                "Develop multilingual evaluation datasets",
                "Include diverse cultural contexts",
                "Create culture-specific causal reasoning tasks",
                "Collaborate with international research teams"
            ]
        )
        gaps.append(research_gap)
        
        return gaps
    
    def _identify_theoretical_gaps(self, analysis: Dict[str, Any]) -> List[ResearchGap]:
        """Identify theoretical gaps in understanding."""
        
        gaps = []
        
        # Mechanism understanding
        research_gap = ResearchGap(
            gap_type="theoretical",
            title="Causal Reasoning Mechanisms",
            description="Limited understanding of how language models actually perform causal reasoning",
            evidence={"mechanism_analysis": "black_box"},
            priority="high",
            estimated_impact=0.9,
            difficulty="hard",
            suggested_approaches=[
                "Develop interpretability methods for causal reasoning",
                "Create mechanistic analysis tools",
                "Design controlled intervention studies",
                "Build theoretical frameworks for LM causal reasoning"
            ]
        )
        gaps.append(research_gap)
        
        # Transfer learning
        research_gap = ResearchGap(
            gap_type="theoretical",
            title="Causal Knowledge Transfer",
            description="Unclear how causal knowledge transfers between domains and contexts",
            evidence={"transfer_analysis": "limited_study"},
            priority="medium",
            estimated_impact=0.7,
            difficulty="medium",
            suggested_approaches=[
                "Study cross-domain causal reasoning transfer",
                "Investigate few-shot causal learning",
                "Analyze causal abstraction capabilities",
                "Design transfer learning experiments"
            ]
        )
        gaps.append(research_gap)
        
        return gaps
    
    def _identify_evaluation_gaps(self, analysis: Dict[str, Any]) -> List[ResearchGap]:
        """Identify gaps in evaluation framework."""
        
        gaps = []
        
        # Human-AI comparison
        research_gap = ResearchGap(
            gap_type="methodology",
            title="Limited Human Baseline Comparison",
            description="Insufficient comparison with human causal reasoning performance",
            evidence={"human_comparison": "limited"},
            priority="high",
            estimated_impact=0.8,
            difficulty="medium",
            suggested_approaches=[
                "Conduct large-scale human evaluation studies",
                "Compare reasoning processes between humans and AI",
                "Analyze error patterns in both humans and models",
                "Design human-AI collaborative evaluation"
            ]
        )
        gaps.append(research_gap)
        
        return gaps
    
    def _prioritize_gaps(self, gaps: List[ResearchGap]) -> List[ResearchGap]:
        """Prioritize research gaps by impact and feasibility."""
        
        def priority_score(gap: ResearchGap) -> float:
            impact_weight = 0.6
            feasibility_weight = 0.4
            
            # Convert difficulty to feasibility score
            difficulty_to_feasibility = {"easy": 1.0, "medium": 0.7, "hard": 0.4}
            feasibility = difficulty_to_feasibility.get(gap.difficulty, 0.5)
            
            return gap.estimated_impact * impact_weight + feasibility * feasibility_weight
        
        return sorted(gaps, key=priority_score, reverse=True)
    
    def _load_gap_patterns(self) -> Dict[str, Any]:
        """Load predefined patterns for gap identification."""
        
        return {
            "performance_thresholds": {
                "poor": 0.5,
                "moderate": 0.7,
                "good": 0.85,
                "excellent": 0.95
            },
            "variance_thresholds": {
                "low": 0.1,
                "medium": 0.2,
                "high": 0.3
            },
            "correlation_thresholds": {
                "low": 0.3,
                "medium": 0.6,
                "high": 0.8
            }
        }


class FutureDirectionsSynthesizer:
    """Synthesizes research gaps into concrete future research directions."""
    
    def __init__(self):
        self.synthesizer_config = self._load_config()
    
    def synthesize_research_agenda(
        self,
        gaps: List[ResearchGap],
        performance_analysis: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Synthesize a comprehensive research agenda."""
        
        logger.info("Synthesizing research agenda")
        
        # Group gaps by type and priority
        gap_groups = self._group_gaps(gaps)
        
        # Generate research opportunities
        opportunities = self._generate_opportunities(gap_groups, performance_analysis)
        
        # Create roadmap
        roadmap = self._create_research_roadmap(opportunities, constraints)
        
        # Identify collaboration opportunities
        collaborations = self._identify_collaborations(opportunities)
        
        # Generate funding proposals
        funding_proposals = self._generate_funding_proposals(opportunities)
        
        return {
            "research_opportunities": opportunities,
            "roadmap": roadmap,
            "collaboration_opportunities": collaborations,
            "funding_proposals": funding_proposals,
            "priority_rankings": self._rank_opportunities(opportunities)
        }
    
    def _group_gaps(self, gaps: List[ResearchGap]) -> Dict[str, List[ResearchGap]]:
        """Group research gaps by type and priority."""
        
        groups = defaultdict(list)
        
        for gap in gaps:
            groups[gap.gap_type].append(gap)
        
        # Sort each group by priority and impact
        for gap_type in groups:
            groups[gap_type] = sorted(
                groups[gap_type],
                key=lambda g: (g.priority == "high", g.estimated_impact),
                reverse=True
            )
        
        return dict(groups)
    
    def _generate_opportunities(
        self,
        gap_groups: Dict[str, List[ResearchGap]],
        performance_analysis: Dict[str, Any]
    ) -> List[ResearchOpportunity]:
        """Generate concrete research opportunities from gaps."""
        
        opportunities = []
        
        # Performance improvement opportunities
        if "performance" in gap_groups:
            for gap in gap_groups["performance"][:3]:  # Top 3 performance gaps
                opportunity = ResearchOpportunity(
                    title=f"Advancing {gap.title}",
                    research_question=f"How can we address {gap.description.lower()}?",
                    hypothesis="Targeted architectural and training improvements can significantly enhance performance on difficult causal reasoning tasks",
                    methodology="Controlled experiments with specialized model architectures and training regimens",
                    expected_outcomes=[
                        "Improved performance on challenging tasks",
                        "Better understanding of task difficulty factors",
                        "Novel architectural insights"
                    ],
                    resource_requirements={
                        "computing": "High (GPU clusters for model training)",
                        "personnel": "3-4 researchers (ML, cognitive science)",
                        "timeframe": "12-18 months",
                        "funding": "$200K-500K"
                    },
                    timeline="18 months",
                    potential_impact="High - direct performance improvements"
                )
                opportunities.append(opportunity)
        
        # Methodological advancement opportunities
        if "methodology" in gap_groups:
            for gap in gap_groups["methodology"][:2]:  # Top 2 methodology gaps
                opportunity = ResearchOpportunity(
                    title=f"Developing Novel {gap.title}",
                    research_question=f"What new methodologies can address {gap.description.lower()}?",
                    hypothesis="Novel evaluation methodologies will reveal previously hidden aspects of causal reasoning capabilities",
                    methodology="Design and validate new evaluation frameworks with statistical validation",
                    expected_outcomes=[
                        "New evaluation metrics and frameworks",
                        "Better understanding of model capabilities",
                        "Improved evaluation standards"
                    ],
                    resource_requirements={
                        "computing": "Medium (evaluation infrastructure)",
                        "personnel": "2-3 researchers (methodology, statistics)",
                        "timeframe": "9-12 months",
                        "funding": "$100K-300K"
                    },
                    timeline="12 months",
                    potential_impact="Medium-High - improved evaluation quality"
                )
                opportunities.append(opportunity)
        
        # Domain expansion opportunities
        if "domain" in gap_groups:
            domain_gap = gap_groups["domain"][0] if gap_groups["domain"] else None
            if domain_gap:
                opportunity = ResearchOpportunity(
                    title="Expanding Causal Reasoning Evaluation Domains",
                    research_question="How does causal reasoning performance vary across diverse domains and cultures?",
                    hypothesis="Domain-specific and culturally-aware evaluation will reveal significant variations in causal reasoning capabilities",
                    methodology="Collaborative international study with domain experts",
                    expected_outcomes=[
                        "Comprehensive cross-domain evaluation framework",
                        "Cultural variation insights",
                        "Domain-specific improvement strategies"
                    ],
                    resource_requirements={
                        "computing": "Medium",
                        "personnel": "5-8 researchers (international collaboration)",
                        "timeframe": "18-24 months",
                        "funding": "$300K-800K"
                    },
                    timeline="24 months",
                    potential_impact="High - broader applicability and fairness"
                )
                opportunities.append(opportunity)
        
        # Theoretical understanding opportunities
        if "theoretical" in gap_groups:
            theoretical_gap = gap_groups["theoretical"][0] if gap_groups["theoretical"] else None
            if theoretical_gap:
                opportunity = ResearchOpportunity(
                    title="Understanding Causal Reasoning Mechanisms in LLMs",
                    research_question="What are the computational mechanisms underlying causal reasoning in language models?",
                    hypothesis="Mechanistic analysis will reveal specific neural circuits and computational patterns responsible for causal reasoning",
                    methodology="Interpretability research with controlled interventions and activation analysis",
                    expected_outcomes=[
                        "Mechanistic understanding of causal reasoning",
                        "Interpretability tools for causal cognition",
                        "Targeted improvement strategies"
                    ],
                    resource_requirements={
                        "computing": "High (large model analysis)",
                        "personnel": "3-4 researchers (interpretability, cognitive science)",
                        "timeframe": "15-20 months",
                        "funding": "$250K-600K"
                    },
                    timeline="20 months",
                    potential_impact="Very High - fundamental understanding"
                )
                opportunities.append(opportunity)
        
        return opportunities
    
    def _create_research_roadmap(
        self,
        opportunities: List[ResearchOpportunity],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a prioritized research roadmap."""
        
        constraints = constraints or {}
        max_budget = constraints.get("max_budget", 2000000)  # $2M default
        max_timeline = constraints.get("max_timeline", 36)  # 36 months default
        
        # Categorize by timeline and dependencies
        short_term = []  # 0-12 months
        medium_term = []  # 12-24 months
        long_term = []  # 24+ months
        
        for opp in opportunities:
            timeline_months = self._parse_timeline(opp.timeline)
            if timeline_months <= 12:
                short_term.append(opp)
            elif timeline_months <= 24:
                medium_term.append(opp)
            else:
                long_term.append(opp)
        
        # Create phased roadmap
        roadmap = {
            "phase_1_foundation": {
                "duration": "0-12 months",
                "focus": "Methodological improvements and immediate performance gains",
                "opportunities": short_term,
                "estimated_cost": sum(self._parse_budget(opp.resource_requirements.get("funding", "0")) for opp in short_term),
                "key_milestones": [
                    "New evaluation metrics developed",
                    "Performance improvements demonstrated",
                    "Initial theoretical insights gained"
                ]
            },
            "phase_2_expansion": {
                "duration": "12-24 months",
                "focus": "Domain expansion and deeper understanding",
                "opportunities": medium_term,
                "estimated_cost": sum(self._parse_budget(opp.resource_requirements.get("funding", "0")) for opp in medium_term),
                "key_milestones": [
                    "Cross-domain evaluation completed",
                    "Mechanistic insights developed",
                    "International collaborations established"
                ]
            },
            "phase_3_integration": {
                "duration": "24-36 months",
                "focus": "Integration and advanced applications",
                "opportunities": long_term,
                "estimated_cost": sum(self._parse_budget(opp.resource_requirements.get("funding", "0")) for opp in long_term),
                "key_milestones": [
                    "Comprehensive framework completed",
                    "Real-world applications demonstrated",
                    "Next-generation models developed"
                ]
            }
        }
        
        return roadmap
    
    def _identify_collaborations(self, opportunities: List[ResearchOpportunity]) -> List[Dict[str, Any]]:
        """Identify potential collaboration opportunities."""
        
        collaborations = []
        
        # Group opportunities by domain/expertise needed
        domain_groups = defaultdict(list)
        for opp in opportunities:
            if "domain" in opp.title.lower() or "cultural" in opp.title.lower():
                domain_groups["cross_cultural"].append(opp)
            if "mechanism" in opp.title.lower() or "interpretability" in opp.title.lower():
                domain_groups["interpretability"].append(opp)
            if "performance" in opp.title.lower():
                domain_groups["performance"].append(opp)
        
        # Suggest specific collaborations
        if "cross_cultural" in domain_groups:
            collaborations.append({
                "type": "International Research Consortium",
                "participants": [
                    "Leading universities in different regions",
                    "Cultural studies departments",
                    "Linguistics research groups"
                ],
                "opportunities": domain_groups["cross_cultural"],
                "benefits": [
                    "Diverse cultural perspectives",
                    "Access to multilingual datasets",
                    "Broader impact and applicability"
                ],
                "coordination_requirements": "Regular international meetings, shared data platforms"
            })
        
        if "interpretability" in domain_groups:
            collaborations.append({
                "type": "Cognitive Science Partnership",
                "participants": [
                    "Cognitive science labs",
                    "Neuroscience research groups",
                    "AI interpretability teams"
                ],
                "opportunities": domain_groups["interpretability"],
                "benefits": [
                    "Interdisciplinary insights",
                    "Human-AI comparison studies",
                    "Theoretical grounding"
                ],
                "coordination_requirements": "Joint lab meetings, shared experimental protocols"
            })
        
        return collaborations
    
    def _generate_funding_proposals(self, opportunities: List[ResearchOpportunity]) -> List[Dict[str, Any]]:
        """Generate funding proposal outlines."""
        
        proposals = []
        
        # Group opportunities for larger proposals
        high_impact_opps = [opp for opp in opportunities if "High" in opp.potential_impact]
        
        if len(high_impact_opps) >= 2:
            total_budget = sum(self._parse_budget(opp.resource_requirements.get("funding", "0")) for opp in high_impact_opps)
            
            proposal = {
                "title": "Advancing Causal Reasoning in Large Language Models: A Comprehensive Research Program",
                "type": "Large-scale research grant (NSF, NIH, or similar)",
                "duration": "36 months",
                "total_budget": f"${total_budget:,}",
                "opportunities_included": [opp.title for opp in high_impact_opps],
                "key_innovations": [
                    "Novel evaluation methodologies",
                    "Cross-cultural and domain expansion",
                    "Mechanistic understanding development",
                    "Performance breakthrough achievements"
                ],
                "broader_impacts": [
                    "Improved AI safety through better causal reasoning",
                    "Enhanced educational applications",
                    "Cross-cultural AI fairness",
                    "Scientific discovery acceleration"
                ],
                "team_composition": "Multi-institutional team with expertise in ML, cognitive science, and domain applications"
            }
            proposals.append(proposal)
        
        # Individual smaller proposals for quick wins
        quick_wins = [opp for opp in opportunities if self._parse_timeline(opp.timeline) <= 12]
        for opp in quick_wins:
            budget = self._parse_budget(opp.resource_requirements.get("funding", "0"))
            if budget <= 300000:  # Smaller grants
                proposal = {
                    "title": opp.title,
                    "type": "Small research grant or industry collaboration",
                    "duration": opp.timeline,
                    "total_budget": opp.resource_requirements.get("funding", "TBD"),
                    "research_question": opp.research_question,
                    "expected_outcomes": opp.expected_outcomes,
                    "potential_impact": opp.potential_impact,
                    "resource_requirements": opp.resource_requirements
                }
                proposals.append(proposal)
        
        return proposals
    
    def _rank_opportunities(self, opportunities: List[ResearchOpportunity]) -> List[Dict[str, Any]]:
        """Rank research opportunities by various criteria."""
        
        rankings = []
        
        for i, opp in enumerate(opportunities):
            # Calculate composite score
            impact_score = self._score_impact(opp.potential_impact)
            feasibility_score = self._score_feasibility(opp.resource_requirements)
            novelty_score = self._score_novelty(opp.title, opp.research_question)
            timeline_score = self._score_timeline(opp.timeline)
            
            composite_score = (
                impact_score * 0.35 +
                feasibility_score * 0.25 +
                novelty_score * 0.25 +
                timeline_score * 0.15
            )
            
            rankings.append({
                "opportunity": opp.title,
                "rank": i + 1,  # Will be reordered
                "composite_score": composite_score,
                "impact_score": impact_score,
                "feasibility_score": feasibility_score,
                "novelty_score": novelty_score,
                "timeline_score": timeline_score,
                "recommendation": self._generate_recommendation(opp, composite_score)
            })
        
        # Sort by composite score
        rankings.sort(key=lambda x: x["composite_score"], reverse=True)
        
        # Update ranks
        for i, ranking in enumerate(rankings):
            ranking["rank"] = i + 1
        
        return rankings
    
    def _parse_timeline(self, timeline: str) -> int:
        """Parse timeline string to months."""
        if "month" in timeline:
            return int(timeline.split()[0])
        return 12  # Default
    
    def _parse_budget(self, budget_str: str) -> int:
        """Parse budget string to integer."""
        if "$" in budget_str:
            # Extract numeric part
            import re
            numbers = re.findall(r'[\d,]+', budget_str.replace("$", "").replace("K", "000").replace("M", "000000"))
            if numbers:
                return int(numbers[0].replace(",", ""))
        return 0
    
    def _score_impact(self, impact_str: str) -> float:
        """Score impact level."""
        if "Very High" in impact_str:
            return 1.0
        elif "High" in impact_str:
            return 0.8
        elif "Medium" in impact_str:
            return 0.6
        else:
            return 0.4
    
    def _score_feasibility(self, resources: Dict[str, Any]) -> float:
        """Score feasibility based on resource requirements."""
        # Simpler timeline and budget = higher feasibility
        timeline = self._parse_timeline(resources.get("timeframe", "12 months"))
        budget = self._parse_budget(resources.get("funding", "$100K"))
        
        timeline_score = max(0, 1.0 - (timeline - 6) / 24)  # 6-30 months scale
        budget_score = max(0, 1.0 - (budget - 50000) / 1000000)  # $50K-$1M scale
        
        return (timeline_score + budget_score) / 2
    
    def _score_novelty(self, title: str, question: str) -> float:
        """Score novelty based on title and research question."""
        novelty_keywords = ["novel", "new", "innovative", "breakthrough", "first", "unexplored"]
        text = (title + " " + question).lower()
        
        novelty_count = sum(1 for keyword in novelty_keywords if keyword in text)
        return min(1.0, novelty_count / 3)  # Normalize to 0-1
    
    def _score_timeline(self, timeline: str) -> float:
        """Score timeline (shorter is better for quick impact)."""
        months = self._parse_timeline(timeline)
        return max(0, 1.0 - (months - 6) / 30)  # Prefer 6-36 month projects
    
    def _generate_recommendation(self, opp: ResearchOpportunity, score: float) -> str:
        """Generate recommendation for opportunity."""
        if score > 0.8:
            return "Highly recommended - pursue immediately"
        elif score > 0.6:
            return "Recommended - good opportunity for impact"
        elif score > 0.4:
            return "Consider - moderate potential"
        else:
            return "Lower priority - consider if resources available"
    
    def _load_config(self) -> Dict[str, Any]:
        """Load synthesizer configuration."""
        return {
            "impact_weights": {
                "performance": 0.35,
                "methodology": 0.25,
                "domain": 0.20,
                "theoretical": 0.20
            },
            "timeline_preferences": {
                "short_term": 0.4,
                "medium_term": 0.4,
                "long_term": 0.2
            }
        }