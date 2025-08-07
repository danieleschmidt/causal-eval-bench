"""
Error analysis tools for identifying patterns in causal reasoning failures.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter, defaultdict
import logging
import numpy as np

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of causal reasoning errors."""
    CORRELATION_CAUSATION = "correlation_causation_confusion"
    REVERSE_CAUSATION = "reverse_causation"
    CONFOUNDING_MISSED = "confounding_variables_missed"
    SPURIOUS_CORRELATION = "spurious_correlation_accepted"
    MECHANISM_MISUNDERSTANDING = "causal_mechanism_misunderstood"
    CHAIN_BROKEN = "causal_chain_broken"
    INTERVENTION_MISAPPLIED = "intervention_effects_misapplied"
    STATISTICAL_FALLACY = "statistical_reasoning_fallacy"
    DOMAIN_KNOWLEDGE_LACK = "domain_knowledge_insufficient"
    LOGICAL_INCONSISTENCY = "logical_inconsistency"


@dataclass
class ErrorPattern:
    """Pattern of errors identified in model responses."""
    
    error_type: ErrorType
    frequency: int
    confidence: float
    examples: List[str] = field(default_factory=list)
    contexts: List[str] = field(default_factory=list)
    severity: str = "medium"  # low, medium, high, critical
    suggested_improvements: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Post-process error pattern."""
        if not self.suggested_improvements:
            self.suggested_improvements = self._generate_improvement_suggestions()
    
    def _generate_improvement_suggestions(self) -> List[str]:
        """Generate suggestions based on error type."""
        suggestions_map = {
            ErrorType.CORRELATION_CAUSATION: [
                "Emphasize the difference between correlation and causation",
                "Provide examples of spurious correlations",
                "Teach Bradford Hill criteria for causation"
            ],
            ErrorType.REVERSE_CAUSATION: [
                "Practice identifying direction of causality",
                "Use temporal ordering as evidence",
                "Consider alternative causal directions"
            ],
            ErrorType.CONFOUNDING_MISSED: [
                "Train on identifying third variables",
                "Practice drawing causal diagrams",
                "Learn about Simpson's paradox"
            ],
            ErrorType.MECHANISM_MISUNDERSTANDING: [
                "Focus on underlying causal mechanisms",
                "Provide domain-specific training",
                "Emphasize biological plausibility"
            ]
        }
        return suggestions_map.get(self.error_type, ["Review causal reasoning fundamentals"])


class ErrorAnalyzer:
    """Advanced error analysis for causal reasoning evaluation."""
    
    def __init__(self):
        """Initialize the error analyzer."""
        self.error_patterns = []
        self.analysis_history = []
        logger.info("Error analyzer initialized")
    
    def analyze_response_errors(
        self,
        model_responses: List[str],
        ground_truth_labels: List[str],
        task_types: List[str],
        domains: List[str]
    ) -> Dict[str, Any]:
        """Comprehensive analysis of model response errors."""
        
        if len(set([len(model_responses), len(ground_truth_labels), len(task_types), len(domains)])) != 1:
            raise ValueError("All input lists must have the same length")
        
        error_analysis = {
            "total_responses": len(model_responses),
            "error_patterns": [],
            "error_frequency_by_type": {},
            "error_frequency_by_domain": {},
            "error_frequency_by_task": {},
            "severity_distribution": {},
            "improvement_priorities": []
        }
        
        # Identify error patterns
        for i, (response, truth, task, domain) in enumerate(zip(
            model_responses, ground_truth_labels, task_types, domains
        )):
            errors = self._identify_errors_in_response(response, truth, task, domain)
            
            for error_type, confidence in errors:
                error_analysis["error_frequency_by_type"][error_type.value] = \
                    error_analysis["error_frequency_by_type"].get(error_type.value, 0) + 1
                
                error_analysis["error_frequency_by_domain"][domain] = \
                    error_analysis["error_frequency_by_domain"].get(domain, 0) + 1
                
                error_analysis["error_frequency_by_task"][task] = \
                    error_analysis["error_frequency_by_task"].get(task, 0) + 1
                
                # Create or update error pattern
                self._update_error_pattern(error_type, response, domain, confidence)
        
        # Analyze patterns
        error_analysis["error_patterns"] = self._consolidate_error_patterns()
        error_analysis["severity_distribution"] = self._calculate_severity_distribution()
        error_analysis["improvement_priorities"] = self._prioritize_improvements()
        
        # Store analysis
        self.analysis_history.append(error_analysis)
        
        return error_analysis
    
    def _identify_errors_in_response(
        self,
        response: str,
        ground_truth: str,
        task_type: str,
        domain: str
    ) -> List[Tuple[ErrorType, float]]:
        """Identify specific error types in a model response."""
        
        errors = []
        response_lower = response.lower()
        truth_lower = ground_truth.lower()
        
        # Correlation-causation confusion
        if self._check_correlation_causation_error(response_lower, truth_lower):
            errors.append((ErrorType.CORRELATION_CAUSATION, 0.8))
        
        # Reverse causation error
        if self._check_reverse_causation_error(response_lower, truth_lower, task_type):
            errors.append((ErrorType.REVERSE_CAUSATION, 0.7))
        
        # Missing confounders
        if self._check_confounding_missed(response_lower, task_type):
            errors.append((ErrorType.CONFOUNDING_MISSED, 0.6))
        
        # Spurious correlation acceptance
        if self._check_spurious_correlation_acceptance(response_lower, truth_lower):
            errors.append((ErrorType.SPURIOUS_CORRELATION, 0.7))
        
        # Mechanism misunderstanding
        if self._check_mechanism_misunderstanding(response_lower, domain):
            errors.append((ErrorType.MECHANISM_MISUNDERSTANDING, 0.6))
        
        # Broken causal chain
        if task_type == "causal_chain" and self._check_chain_broken(response_lower):
            errors.append((ErrorType.CHAIN_BROKEN, 0.8))
        
        # Intervention misapplication
        if task_type == "intervention" and self._check_intervention_misapplied(response_lower):
            errors.append((ErrorType.INTERVENTION_MISAPPLIED, 0.7))
        
        # Statistical fallacies
        if self._check_statistical_fallacy(response_lower):
            errors.append((ErrorType.STATISTICAL_FALLACY, 0.6))
        
        # Domain knowledge insufficiency
        if self._check_domain_knowledge_lack(response_lower, domain):
            errors.append((ErrorType.DOMAIN_KNOWLEDGE_LACK, 0.5))
        
        # Logical inconsistency
        if self._check_logical_inconsistency(response_lower):
            errors.append((ErrorType.LOGICAL_INCONSISTENCY, 0.7))
        
        return errors
    
    def _check_correlation_causation_error(self, response: str, truth: str) -> bool:
        """Check for correlation-causation confusion."""
        causal_indicators = ["causes", "leads to", "results in", "brings about"]
        correlation_indicators = ["correlated", "associated", "related", "linked"]
        
        # Response claims causation when truth indicates correlation
        response_claims_causation = any(indicator in response for indicator in causal_indicators)
        truth_indicates_correlation = any(indicator in truth for indicator in correlation_indicators)
        
        return response_claims_causation and truth_indicates_correlation
    
    def _check_reverse_causation_error(self, response: str, truth: str, task_type: str) -> bool:
        """Check for reverse causation errors."""
        if "reverse" in truth and "reverse" not in response:
            return True
        
        # Look for direction indicators
        if "a causes b" in truth and "b causes a" in response:
            return True
        
        return False
    
    def _check_confounding_missed(self, response: str, task_type: str) -> bool:
        """Check if confounding variables were missed."""
        if task_type in ["attribution", "confounding"]:
            confounder_terms = ["confounder", "confounding", "third variable", "lurking variable"]
            return not any(term in response for term in confounder_terms)
        return False
    
    def _check_spurious_correlation_acceptance(self, response: str, truth: str) -> bool:
        """Check if spurious correlations were incorrectly accepted as causal."""
        if "spurious" in truth and "causal" in response:
            return "spurious" not in response
        return False
    
    def _check_mechanism_misunderstanding(self, response: str, domain: str) -> bool:
        """Check for domain-specific mechanism misunderstandings."""
        domain_mechanisms = {
            "medical": ["physiological", "biological", "pathophysiology", "pharmacological"],
            "education": ["cognitive", "learning", "memory", "pedagogical"],
            "economics": ["market", "supply", "demand", "behavioral"],
            "environmental": ["ecological", "climate", "atmospheric", "biological"]
        }
        
        if domain in domain_mechanisms:
            expected_terms = domain_mechanisms[domain]
            return not any(term in response for term in expected_terms)
        
        return False
    
    def _check_chain_broken(self, response: str) -> bool:
        """Check for broken causal chains."""
        chain_indicators = ["step", "then", "next", "leads to", "causes", "results in"]
        return len([indicator for indicator in chain_indicators if indicator in response]) < 2
    
    def _check_intervention_misapplied(self, response: str) -> bool:
        """Check for intervention effect misapplications."""
        intervention_terms = ["intervention", "manipulate", "control", "change"]
        effect_terms = ["effect", "impact", "result", "outcome"]
        
        has_intervention = any(term in response for term in intervention_terms)
        has_effect = any(term in response for term in effect_terms)
        
        return has_intervention and not has_effect
    
    def _check_statistical_fallacy(self, response: str) -> bool:
        """Check for statistical reasoning fallacies."""
        fallacy_indicators = [
            "correlation implies causation",
            "post hoc ergo propter hoc",
            "because it happened after",
            "must be caused by"
        ]
        return any(indicator in response for indicator in fallacy_indicators)
    
    def _check_domain_knowledge_lack(self, response: str, domain: str) -> bool:
        """Check for insufficient domain knowledge."""
        # This is a simplified check - in practice, would need more sophisticated analysis
        domain_terms = {
            "medical": ["diagnosis", "treatment", "symptom", "disease", "patient"],
            "education": ["learning", "student", "teacher", "curriculum", "assessment"],
            "business": ["revenue", "profit", "market", "customer", "competition"],
            "technology": ["algorithm", "system", "data", "process", "automation"]
        }
        
        if domain in domain_terms:
            expected_terms = domain_terms[domain]
            mentioned_terms = sum(1 for term in expected_terms if term in response)
            return mentioned_terms / len(expected_terms) < 0.3
        
        return False
    
    def _check_logical_inconsistency(self, response: str) -> bool:
        """Check for logical inconsistencies in reasoning."""
        # Look for contradictory statements
        contradictions = [
            ("increases", "decreases"),
            ("causes", "does not cause"),
            ("significant", "not significant"),
            ("positive", "negative")
        ]
        
        for pos, neg in contradictions:
            if pos in response and neg in response:
                # Check if they're in different contexts (more sophisticated analysis needed)
                return True
        
        return False
    
    def _update_error_pattern(
        self,
        error_type: ErrorType,
        response: str,
        domain: str,
        confidence: float
    ):
        """Update error pattern tracking."""
        # Find existing pattern or create new one
        existing_pattern = None
        for pattern in self.error_patterns:
            if pattern.error_type == error_type:
                existing_pattern = pattern
                break
        
        if existing_pattern:
            existing_pattern.frequency += 1
            existing_pattern.examples.append(response[:200] + "..." if len(response) > 200 else response)
            existing_pattern.contexts.append(domain)
            # Update confidence (weighted average)
            existing_pattern.confidence = (
                existing_pattern.confidence * (existing_pattern.frequency - 1) + confidence
            ) / existing_pattern.frequency
        else:
            new_pattern = ErrorPattern(
                error_type=error_type,
                frequency=1,
                confidence=confidence,
                examples=[response[:200] + "..." if len(response) > 200 else response],
                contexts=[domain]
            )
            self.error_patterns.append(new_pattern)
    
    def _consolidate_error_patterns(self) -> List[Dict[str, Any]]:
        """Consolidate error patterns into analysis format."""
        consolidated = []
        
        for pattern in self.error_patterns:
            # Determine severity based on frequency and confidence
            if pattern.frequency >= 5 and pattern.confidence >= 0.7:
                severity = "critical"
            elif pattern.frequency >= 3 and pattern.confidence >= 0.6:
                severity = "high"
            elif pattern.frequency >= 2:
                severity = "medium"
            else:
                severity = "low"
            
            pattern.severity = severity
            
            consolidated.append({
                "error_type": pattern.error_type.value,
                "frequency": pattern.frequency,
                "confidence": pattern.confidence,
                "severity": pattern.severity,
                "most_common_contexts": Counter(pattern.contexts).most_common(3),
                "example_responses": pattern.examples[:3],  # Show top 3 examples
                "improvement_suggestions": pattern.suggested_improvements
            })
        
        # Sort by severity and frequency
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        consolidated.sort(
            key=lambda x: (severity_order[x["severity"]], x["frequency"]),
            reverse=True
        )
        
        return consolidated
    
    def _calculate_severity_distribution(self) -> Dict[str, int]:
        """Calculate distribution of error severities."""
        severity_counts = Counter()
        for pattern in self.error_patterns:
            severity_counts[pattern.severity] += pattern.frequency
        
        return dict(severity_counts)
    
    def _prioritize_improvements(self) -> List[Dict[str, Any]]:
        """Prioritize improvement areas based on error analysis."""
        priorities = []
        
        # Sort patterns by impact (frequency * confidence * severity weight)
        severity_weights = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        
        for pattern in self.error_patterns:
            impact_score = (
                pattern.frequency * 
                pattern.confidence * 
                severity_weights[pattern.severity]
            )
            
            priorities.append({
                "error_type": pattern.error_type.value,
                "impact_score": impact_score,
                "improvement_suggestions": pattern.suggested_improvements,
                "affected_domains": Counter(pattern.contexts).most_common(3)
            })
        
        # Sort by impact score
        priorities.sort(key=lambda x: x["impact_score"], reverse=True)
        
        return priorities[:10]  # Top 10 priorities
    
    def generate_error_report(
        self,
        analysis_results: Dict[str, Any],
        title: str = "Causal Reasoning Error Analysis Report"
    ) -> str:
        """Generate a comprehensive error analysis report."""
        
        report = f"# {title}\n\n"
        
        # Summary
        report += "## Executive Summary\n\n"
        report += f"- **Total Responses Analyzed**: {analysis_results['total_responses']}\n"
        report += f"- **Error Patterns Identified**: {len(analysis_results['error_patterns'])}\n"
        report += f"- **Critical Issues**: {analysis_results['severity_distribution'].get('critical', 0)}\n"
        report += f"- **High Priority Issues**: {analysis_results['severity_distribution'].get('high', 0)}\n\n"
        
        # Top error patterns
        report += "## Most Critical Error Patterns\n\n"
        for i, pattern in enumerate(analysis_results['error_patterns'][:5], 1):
            report += f"### {i}. {pattern['error_type'].replace('_', ' ').title()}\n"
            report += f"- **Frequency**: {pattern['frequency']} occurrences\n"
            report += f"- **Confidence**: {pattern['confidence']:.2f}\n"
            report += f"- **Severity**: {pattern['severity']}\n"
            report += f"- **Common Contexts**: {', '.join([ctx[0] for ctx in pattern['most_common_contexts']])}\n"
            report += f"- **Key Improvements Needed**:\n"
            for suggestion in pattern['improvement_suggestions'][:3]:
                report += f"  - {suggestion}\n"
            report += "\n"
        
        # Domain-specific analysis
        report += "## Error Distribution by Domain\n\n"
        for domain, count in sorted(
            analysis_results['error_frequency_by_domain'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            report += f"- **{domain}**: {count} errors\n"
        
        report += "\n## Error Distribution by Task Type\n\n"
        for task, count in sorted(
            analysis_results['error_frequency_by_task'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            report += f"- **{task}**: {count} errors\n"
        
        # Improvement priorities
        report += "\n## Recommended Improvement Priorities\n\n"
        for i, priority in enumerate(analysis_results['improvement_priorities'][:5], 1):
            report += f"### Priority {i}: {priority['error_type'].replace('_', ' ').title()}\n"
            report += f"- **Impact Score**: {priority['impact_score']:.2f}\n"
            report += f"- **Recommended Actions**:\n"
            for action in priority['improvement_suggestions']:
                report += f"  - {action}\n"
            report += "\n"
        
        return report
    
    def compare_error_patterns(
        self,
        model_a_errors: Dict[str, Any],
        model_b_errors: Dict[str, Any],
        model_a_name: str = "Model A",
        model_b_name: str = "Model B"
    ) -> Dict[str, Any]:
        """Compare error patterns between two models."""
        
        comparison = {
            "model_names": [model_a_name, model_b_name],
            "error_pattern_differences": {},
            "strengths_and_weaknesses": {},
            "overall_assessment": {}
        }
        
        # Compare error frequencies
        a_patterns = {p["error_type"]: p for p in model_a_errors["error_patterns"]}
        b_patterns = {p["error_type"]: p for p in model_b_errors["error_patterns"]}
        
        all_error_types = set(a_patterns.keys()) | set(b_patterns.keys())
        
        for error_type in all_error_types:
            a_freq = a_patterns.get(error_type, {}).get("frequency", 0)
            b_freq = b_patterns.get(error_type, {}).get("frequency", 0)
            
            comparison["error_pattern_differences"][error_type] = {
                f"{model_a_name}_frequency": a_freq,
                f"{model_b_name}_frequency": b_freq,
                "difference": a_freq - b_freq,
                "better_model": model_b_name if a_freq > b_freq else model_a_name
            }
        
        # Identify strengths and weaknesses
        comparison["strengths_and_weaknesses"][model_a_name] = {
            "strengths": [
                error_type for error_type, diff in comparison["error_pattern_differences"].items()
                if diff["difference"] < -2  # Significantly fewer errors
            ],
            "weaknesses": [
                error_type for error_type, diff in comparison["error_pattern_differences"].items()
                if diff["difference"] > 2  # Significantly more errors
            ]
        }
        
        comparison["strengths_and_weaknesses"][model_b_name] = {
            "strengths": [
                error_type for error_type, diff in comparison["error_pattern_differences"].items()
                if diff["difference"] > 2
            ],
            "weaknesses": [
                error_type for error_type, diff in comparison["error_pattern_differences"].items()
                if diff["difference"] < -2
            ]
        }
        
        # Overall assessment
        a_total_errors = sum(p["frequency"] for p in model_a_errors["error_patterns"])
        b_total_errors = sum(p["frequency"] for p in model_b_errors["error_patterns"])
        
        comparison["overall_assessment"] = {
            "total_errors": {
                model_a_name: a_total_errors,
                model_b_name: b_total_errors
            },
            "better_overall": model_b_name if a_total_errors > b_total_errors else model_a_name,
            "improvement_gap": abs(a_total_errors - b_total_errors)
        }
        
        return comparison