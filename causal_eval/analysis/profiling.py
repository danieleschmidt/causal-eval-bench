"""
Causal capability profiling for language models.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class CausalCapability(Enum):
    """Core causal reasoning capabilities."""
    CORRELATION_VS_CAUSATION = "correlation_vs_causation"
    CONFOUNDING_IDENTIFICATION = "confounding_identification"
    COUNTERFACTUAL_REASONING = "counterfactual_reasoning"
    INTERVENTION_PREDICTION = "intervention_prediction"
    CAUSAL_CHAIN_TRACING = "causal_chain_tracing"
    MECHANISM_UNDERSTANDING = "mechanism_understanding"
    TEMPORAL_REASONING = "temporal_reasoning"
    STATISTICAL_REASONING = "statistical_reasoning"
    DOMAIN_TRANSFER = "domain_transfer"
    COMPLEX_SCENARIOS = "complex_scenarios"


@dataclass
class CapabilityScore:
    """Score for a specific causal capability."""
    
    capability: CausalCapability
    overall_score: float
    consistency: float  # How consistent the performance is
    domain_scores: Dict[str, float] = field(default_factory=dict)
    difficulty_scores: Dict[str, float] = field(default_factory=dict)
    sample_size: int = 0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    
    def get_proficiency_level(self) -> str:
        """Get proficiency level description."""
        if self.overall_score >= 0.9:
            return "Expert"
        elif self.overall_score >= 0.8:
            return "Proficient"
        elif self.overall_score >= 0.7:
            return "Competent"
        elif self.overall_score >= 0.6:
            return "Developing"
        elif self.overall_score >= 0.5:
            return "Novice"
        else:
            return "Inadequate"


@dataclass
class CausalCapabilityProfile:
    """Complete causal reasoning capability profile."""
    
    model_name: str
    capability_scores: Dict[CausalCapability, CapabilityScore] = field(default_factory=dict)
    overall_causal_score: float = 0.0
    strengths: List[CausalCapability] = field(default_factory=list)
    weaknesses: List[CausalCapability] = field(default_factory=list)
    domain_performance: Dict[str, float] = field(default_factory=dict)
    improvement_recommendations: List[str] = field(default_factory=list)
    profile_timestamp: float = 0.0
    
    def __post_init__(self):
        """Calculate derived metrics."""
        if self.capability_scores:
            self._calculate_overall_score()
            self._identify_strengths_weaknesses()
            self._generate_recommendations()
    
    def _calculate_overall_score(self):
        """Calculate overall causal reasoning score."""
        if not self.capability_scores:
            return
        
        # Weighted average of capabilities
        weights = {
            CausalCapability.CORRELATION_VS_CAUSATION: 0.15,
            CausalCapability.CONFOUNDING_IDENTIFICATION: 0.15,
            CausalCapability.COUNTERFACTUAL_REASONING: 0.15,
            CausalCapability.INTERVENTION_PREDICTION: 0.15,
            CausalCapability.CAUSAL_CHAIN_TRACING: 0.10,
            CausalCapability.MECHANISM_UNDERSTANDING: 0.10,
            CausalCapability.TEMPORAL_REASONING: 0.08,
            CausalCapability.STATISTICAL_REASONING: 0.07,
            CausalCapability.DOMAIN_TRANSFER: 0.03,
            CausalCapability.COMPLEX_SCENARIOS: 0.02
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for capability, score_obj in self.capability_scores.items():
            weight = weights.get(capability, 0.05)
            weighted_sum += score_obj.overall_score * weight
            total_weight += weight
        
        self.overall_causal_score = weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _identify_strengths_weaknesses(self):
        """Identify model's strengths and weaknesses."""
        scores = [(cap, score.overall_score) for cap, score in self.capability_scores.items()]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Top 3 are strengths, bottom 3 are weaknesses
        self.strengths = [cap for cap, _ in scores[:3] if scores[0][1] - scores[2][1] > 0.1]
        self.weaknesses = [cap for cap, _ in scores[-3:] if scores[-1][1] < 0.7]
    
    def _generate_recommendations(self):
        """Generate improvement recommendations."""
        self.improvement_recommendations = []
        
        for capability in self.weaknesses:
            score_obj = self.capability_scores[capability]
            
            if capability == CausalCapability.CORRELATION_VS_CAUSATION:
                self.improvement_recommendations.append(
                    "Focus on distinguishing correlation from causation with explicit counter-examples"
                )
            elif capability == CausalCapability.CONFOUNDING_IDENTIFICATION:
                self.improvement_recommendations.append(
                    "Practice identifying third variables and confounding factors in observational studies"
                )
            elif capability == CausalCapability.COUNTERFACTUAL_REASONING:
                self.improvement_recommendations.append(
                    "Improve 'what-if' scenario analysis and alternative outcome prediction"
                )
            elif capability == CausalCapability.INTERVENTION_PREDICTION:
                self.improvement_recommendations.append(
                    "Strengthen understanding of intervention effects and experimental design"
                )
            elif capability == CausalCapability.CAUSAL_CHAIN_TRACING:
                self.improvement_recommendations.append(
                    "Practice multi-step causal reasoning and chain of causation analysis"
                )
        
        # Domain-specific recommendations
        if self.domain_performance:
            weakest_domain = min(self.domain_performance.items(), key=lambda x: x[1])
            if weakest_domain[1] < 0.6:
                self.improvement_recommendations.append(
                    f"Focus on domain-specific knowledge in {weakest_domain[0]}"
                )


class CausalProfiler:
    """Profiler for analyzing causal reasoning capabilities."""
    
    def __init__(self):
        """Initialize the profiler."""
        self.profiles = {}
        logger.info("Causal profiler initialized")
    
    def create_profile(
        self,
        model_name: str,
        evaluation_results: List[Dict[str, Any]]
    ) -> CausalCapabilityProfile:
        """Create a comprehensive capability profile."""
        
        logger.info(f"Creating causal capability profile for {model_name}")
        
        # Group results by capability
        capability_data = defaultdict(list)
        
        for result in evaluation_results:
            task_type = result.get("task_type", "unknown")
            domain = result.get("domain", "general")
            difficulty = result.get("difficulty", "medium")
            overall_score = result.get("overall_score", 0.0)
            
            # Map task types to capabilities
            capability = self._map_task_to_capability(task_type)
            if capability:
                capability_data[capability].append({
                    "score": overall_score,
                    "domain": domain,
                    "difficulty": difficulty,
                    "result": result
                })
        
        # Calculate capability scores
        capability_scores = {}
        domain_performance = defaultdict(list)
        
        for capability, data_points in capability_data.items():
            scores = [dp["score"] for dp in data_points]
            
            if not scores:
                continue
            
            # Overall statistics
            overall_score = np.mean(scores)
            consistency = 1.0 - np.std(scores)  # Higher consistency = lower variance
            
            # Domain-specific scores
            domain_scores = {}
            for domain in set(dp["domain"] for dp in data_points):
                domain_data = [dp["score"] for dp in data_points if dp["domain"] == domain]
                if domain_data:
                    domain_scores[domain] = np.mean(domain_data)
                    domain_performance[domain].extend(domain_data)
            
            # Difficulty-specific scores
            difficulty_scores = {}
            for difficulty in set(dp["difficulty"] for dp in data_points):
                difficulty_data = [dp["score"] for dp in data_points if dp["difficulty"] == difficulty]
                if difficulty_data:
                    difficulty_scores[difficulty] = np.mean(difficulty_data)
            
            # Confidence interval (bootstrap)
            ci_lower, ci_upper = self._calculate_bootstrap_ci(scores)
            
            capability_scores[capability] = CapabilityScore(
                capability=capability,
                overall_score=overall_score,
                consistency=max(0.0, consistency),  # Ensure non-negative
                domain_scores=domain_scores,
                difficulty_scores=difficulty_scores,
                sample_size=len(scores),
                confidence_interval=(ci_lower, ci_upper)
            )
        
        # Calculate domain performance averages
        domain_avg_performance = {
            domain: np.mean(scores) 
            for domain, scores in domain_performance.items()
        }
        
        # Create profile
        profile = CausalCapabilityProfile(
            model_name=model_name,
            capability_scores=capability_scores,
            domain_performance=domain_avg_performance,
            profile_timestamp=np.datetime64('now').astype(float)
        )
        
        # Store profile
        self.profiles[model_name] = profile
        
        logger.info(f"Profile created for {model_name} with overall score: {profile.overall_causal_score:.3f}")
        
        return profile
    
    def _map_task_to_capability(self, task_type: str) -> Optional[CausalCapability]:
        """Map task types to causal capabilities."""
        mapping = {
            "attribution": CausalCapability.CORRELATION_VS_CAUSATION,
            "causal_attribution": CausalCapability.CORRELATION_VS_CAUSATION,
            "counterfactual": CausalCapability.COUNTERFACTUAL_REASONING,
            "counterfactual_reasoning": CausalCapability.COUNTERFACTUAL_REASONING,
            "intervention": CausalCapability.INTERVENTION_PREDICTION,
            "causal_intervention": CausalCapability.INTERVENTION_PREDICTION,
            "chain": CausalCapability.CAUSAL_CHAIN_TRACING,
            "causal_chain": CausalCapability.CAUSAL_CHAIN_TRACING,
            "confounding": CausalCapability.CONFOUNDING_IDENTIFICATION,
            "confounding_analysis": CausalCapability.CONFOUNDING_IDENTIFICATION
        }
        
        return mapping.get(task_type)
    
    def _calculate_bootstrap_ci(
        self,
        data: List[float],
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval."""
        
        if len(data) < 2:
            return (0.0, 1.0)
        
        bootstrap_means = []
        n_samples = len(data)
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=n_samples, replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_means, lower_percentile)
        ci_upper = np.percentile(bootstrap_means, upper_percentile)
        
        return (ci_lower, ci_upper)
    
    def compare_profiles(
        self,
        model_a: str,
        model_b: str
    ) -> Dict[str, Any]:
        """Compare two model profiles."""
        
        if model_a not in self.profiles or model_b not in self.profiles:
            raise ValueError("Both models must have existing profiles")
        
        profile_a = self.profiles[model_a]
        profile_b = self.profiles[model_b]
        
        comparison = {
            "model_names": [model_a, model_b],
            "overall_scores": {
                model_a: profile_a.overall_causal_score,
                model_b: profile_b.overall_causal_score
            },
            "capability_comparison": {},
            "relative_strengths": {},
            "improvement_opportunities": {},
            "statistical_significance": {}
        }
        
        # Compare capabilities
        common_capabilities = set(profile_a.capability_scores.keys()) & set(profile_b.capability_scores.keys())
        
        for capability in common_capabilities:
            score_a = profile_a.capability_scores[capability]
            score_b = profile_b.capability_scores[capability]
            
            difference = score_a.overall_score - score_b.overall_score
            better_model = model_a if difference > 0 else model_b
            
            comparison["capability_comparison"][capability.value] = {
                f"{model_a}_score": score_a.overall_score,
                f"{model_b}_score": score_b.overall_score,
                "difference": abs(difference),
                "better_model": better_model,
                "significance": "high" if abs(difference) > 0.1 else "moderate" if abs(difference) > 0.05 else "low"
            }
        
        # Identify relative strengths
        comparison["relative_strengths"][model_a] = [
            cap.value for cap in profile_a.strengths 
            if cap not in profile_b.strengths
        ]
        comparison["relative_strengths"][model_b] = [
            cap.value for cap in profile_b.strengths 
            if cap not in profile_a.strengths
        ]
        
        # Improvement opportunities
        comparison["improvement_opportunities"][model_a] = [
            cap.value for cap in profile_a.weaknesses
        ]
        comparison["improvement_opportunities"][model_b] = [
            cap.value for cap in profile_b.weaknesses
        ]
        
        return comparison
    
    def generate_profile_report(
        self,
        model_name: str,
        include_detailed_analysis: bool = True
    ) -> str:
        """Generate a comprehensive profile report."""
        
        if model_name not in self.profiles:
            raise ValueError(f"No profile found for model: {model_name}")
        
        profile = self.profiles[model_name]
        
        report = f"# Causal Reasoning Capability Profile: {model_name}\n\n"
        
        # Executive Summary
        report += "## Executive Summary\n\n"
        report += f"- **Overall Causal Reasoning Score**: {profile.overall_causal_score:.3f} ({self._get_overall_proficiency(profile.overall_causal_score)})\n"
        report += f"- **Number of Capabilities Assessed**: {len(profile.capability_scores)}\n"
        report += f"- **Primary Strengths**: {', '.join([cap.value.replace('_', ' ').title() for cap in profile.strengths])}\n"
        report += f"- **Areas for Improvement**: {', '.join([cap.value.replace('_', ' ').title() for cap in profile.weaknesses])}\n\n"
        
        # Capability Breakdown
        report += "## Capability Assessment\n\n"
        
        sorted_capabilities = sorted(
            profile.capability_scores.items(),
            key=lambda x: x[1].overall_score,
            reverse=True
        )
        
        for capability, score_obj in sorted_capabilities:
            report += f"### {capability.value.replace('_', ' ').title()}\n"
            report += f"- **Score**: {score_obj.overall_score:.3f} ({score_obj.get_proficiency_level()})\n"
            report += f"- **Consistency**: {score_obj.consistency:.3f}\n"
            report += f"- **Sample Size**: {score_obj.sample_size}\n"
            report += f"- **Confidence Interval**: ({score_obj.confidence_interval[0]:.3f}, {score_obj.confidence_interval[1]:.3f})\n"
            
            if score_obj.domain_scores:
                report += f"- **Best Domain**: {max(score_obj.domain_scores.items(), key=lambda x: x[1])[0]} ({max(score_obj.domain_scores.values()):.3f})\n"
                report += f"- **Weakest Domain**: {min(score_obj.domain_scores.items(), key=lambda x: x[1])[0]} ({min(score_obj.domain_scores.values()):.3f})\n"
            
            report += "\n"
        
        # Domain Performance
        if profile.domain_performance:
            report += "## Domain-Specific Performance\n\n"
            sorted_domains = sorted(
                profile.domain_performance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for domain, score in sorted_domains:
                proficiency = self._get_overall_proficiency(score)
                report += f"- **{domain.title()}**: {score:.3f} ({proficiency})\n"
            report += "\n"
        
        # Recommendations
        if profile.improvement_recommendations:
            report += "## Improvement Recommendations\n\n"
            for i, recommendation in enumerate(profile.improvement_recommendations, 1):
                report += f"{i}. {recommendation}\n"
            report += "\n"
        
        if include_detailed_analysis:
            report += self._generate_detailed_analysis(profile)
        
        return report
    
    def _get_overall_proficiency(self, score: float) -> str:
        """Get proficiency level for overall scores."""
        if score >= 0.9:
            return "Expert Level"
        elif score >= 0.8:
            return "Proficient"
        elif score >= 0.7:
            return "Competent"
        elif score >= 0.6:
            return "Developing"
        elif score >= 0.5:
            return "Novice"
        else:
            return "Inadequate"
    
    def _generate_detailed_analysis(self, profile: CausalCapabilityProfile) -> str:
        """Generate detailed analysis section."""
        
        analysis = "## Detailed Analysis\n\n"
        
        # Consistency Analysis
        analysis += "### Consistency Analysis\n\n"
        consistency_scores = [score.consistency for score in profile.capability_scores.values()]
        avg_consistency = np.mean(consistency_scores)
        
        analysis += f"- **Average Consistency**: {avg_consistency:.3f}\n"
        
        most_consistent = max(profile.capability_scores.items(), key=lambda x: x[1].consistency)
        least_consistent = min(profile.capability_scores.items(), key=lambda x: x[1].consistency)
        
        analysis += f"- **Most Consistent**: {most_consistent[0].value.replace('_', ' ').title()} ({most_consistent[1].consistency:.3f})\n"
        analysis += f"- **Least Consistent**: {least_consistent[0].value.replace('_', ' ').title()} ({least_consistent[1].consistency:.3f})\n\n"
        
        # Performance Variability
        analysis += "### Performance Variability\n\n"
        scores = [score.overall_score for score in profile.capability_scores.values()]
        score_variance = np.var(scores)
        
        if score_variance < 0.01:
            variability = "Low - Performance is consistent across capabilities"
        elif score_variance < 0.05:
            variability = "Moderate - Some variation in capability strengths"
        else:
            variability = "High - Significant differences between capabilities"
        
        analysis += f"- **Performance Variability**: {variability}\n"
        analysis += f"- **Score Variance**: {score_variance:.4f}\n\n"
        
        return analysis
    
    def export_profile_data(
        self,
        model_name: str,
        format: str = "dict"
    ) -> Dict[str, Any]:
        """Export profile data in structured format."""
        
        if model_name not in self.profiles:
            raise ValueError(f"No profile found for model: {model_name}")
        
        profile = self.profiles[model_name]
        
        export_data = {
            "model_name": profile.model_name,
            "overall_score": profile.overall_causal_score,
            "profile_timestamp": profile.profile_timestamp,
            "capabilities": {}
        }
        
        for capability, score_obj in profile.capability_scores.items():
            export_data["capabilities"][capability.value] = {
                "overall_score": score_obj.overall_score,
                "consistency": score_obj.consistency,
                "proficiency_level": score_obj.get_proficiency_level(),
                "domain_scores": score_obj.domain_scores,
                "difficulty_scores": score_obj.difficulty_scores,
                "sample_size": score_obj.sample_size,
                "confidence_interval": score_obj.confidence_interval
            }
        
        export_data["domain_performance"] = profile.domain_performance
        export_data["strengths"] = [cap.value for cap in profile.strengths]
        export_data["weaknesses"] = [cap.value for cap in profile.weaknesses]
        export_data["recommendations"] = profile.improvement_recommendations
        
        return export_data