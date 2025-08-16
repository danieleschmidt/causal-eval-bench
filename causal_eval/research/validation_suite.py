"""
Comprehensive Validation Suite for Causal Reasoning Research

This module provides rigorous validation tools for ensuring the reliability
and validity of causal reasoning evaluations and research findings.
"""

import numpy as np
import scipy.stats as stats
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod
import warnings
from sklearn.metrics import (
    cohen_kappa_score, 
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.model_selection import (
    cross_val_score, 
    StratifiedKFold,
    permutation_test_score
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation test."""
    
    test_name: str
    passed: bool
    score: float
    threshold: float
    confidence_interval: Optional[Tuple[float, float]] = None
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    interpretation: str = ""
    recommendations: List[str] = field(default_factory=list)


@dataclass 
class ReliabilityReport:
    """Comprehensive reliability assessment report."""
    
    overall_reliability_score: float
    test_results: List[ValidationResult]
    internal_consistency: Dict[str, float]
    test_retest_reliability: Dict[str, float]
    inter_rater_reliability: Dict[str, float]
    construct_validity: Dict[str, float]
    criterion_validity: Dict[str, float]
    recommendations: List[str]
    metadata: Dict[str, Any]


class ValidationTest(ABC):
    """Abstract base class for validation tests."""
    
    def __init__(self, name: str, threshold: float = 0.7):
        self.name = name
        self.threshold = threshold
    
    @abstractmethod
    def run_test(self, data: Any) -> ValidationResult:
        """Run the validation test on the provided data."""
        pass
    
    def get_description(self) -> str:
        """Get description of what this test validates."""
        return f"Validation test: {self.name}"


class InternalConsistencyTest(ValidationTest):
    """
    Test for internal consistency of evaluation metrics.
    
    Validates that different aspects of causal reasoning evaluation
    correlate appropriately with each other.
    """
    
    def __init__(self, threshold: float = 0.7):
        super().__init__("Internal Consistency", threshold)
    
    def run_test(self, data: List[Dict[str, Any]]) -> ValidationResult:
        """
        Test internal consistency using Cronbach's alpha and correlations.
        
        Args:
            data: List of evaluation results with multiple metric scores
        """
        try:
            # Extract metric scores
            metric_scores = self._extract_metric_matrix(data)
            
            if metric_scores.shape[1] < 2:
                return ValidationResult(
                    test_name=self.name,
                    passed=False,
                    score=0.0,
                    threshold=self.threshold,
                    interpretation="Insufficient metrics for consistency testing"
                )
            
            # Calculate Cronbach's alpha
            cronbach_alpha = self._calculate_cronbach_alpha(metric_scores)
            
            # Calculate inter-metric correlations
            correlation_matrix = np.corrcoef(metric_scores.T)
            mean_correlation = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
            
            # Overall consistency score
            consistency_score = (cronbach_alpha + mean_correlation) / 2
            
            passed = consistency_score >= self.threshold
            
            interpretation = self._interpret_consistency_score(consistency_score, cronbach_alpha, mean_correlation)
            
            recommendations = []
            if not passed:
                recommendations.extend([
                    "Consider revising metrics that show low correlation with others",
                    "Investigate potential redundancy in highly correlated metrics",
                    "Validate that metrics measure distinct aspects of causal reasoning"
                ])
            
            return ValidationResult(
                test_name=self.name,
                passed=passed,
                score=consistency_score,
                threshold=self.threshold,
                details={
                    "cronbach_alpha": cronbach_alpha,
                    "mean_inter_correlation": mean_correlation,
                    "correlation_matrix": correlation_matrix.tolist(),
                    "num_metrics": metric_scores.shape[1],
                    "num_observations": metric_scores.shape[0]
                },
                interpretation=interpretation,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error in internal consistency test: {e}")
            return ValidationResult(
                test_name=self.name,
                passed=False,
                score=0.0,
                threshold=self.threshold,
                interpretation=f"Test failed due to error: {str(e)}"
            )
    
    def _extract_metric_matrix(self, data: List[Dict[str, Any]]) -> np.ndarray:
        """Extract metric scores into a matrix."""
        if not data:
            return np.array([]).reshape(0, 0)
        
        # Get all metric names
        all_metrics = set()
        for item in data:
            if 'metric_scores' in item:
                all_metrics.update(item['metric_scores'].keys())
        
        metric_names = sorted(list(all_metrics))
        
        # Create matrix
        matrix = []
        for item in data:
            scores = item.get('metric_scores', {})
            row = []
            for metric in metric_names:
                if metric in scores:
                    score_data = scores[metric]
                    if isinstance(score_data, dict):
                        score = score_data.get('mean', score_data.get('score', 0))
                    else:
                        score = score_data
                    row.append(float(score))
                else:
                    row.append(0.0)
            matrix.append(row)
        
        return np.array(matrix)
    
    def _calculate_cronbach_alpha(self, metric_scores: np.ndarray) -> float:
        """Calculate Cronbach's alpha for internal consistency."""
        n_items = metric_scores.shape[1]
        
        if n_items < 2:
            return 0.0
        
        # Calculate item variances
        item_variances = np.var(metric_scores, axis=0, ddof=1)
        total_variance = np.var(np.sum(metric_scores, axis=1), ddof=1)
        
        # Cronbach's alpha formula
        alpha = (n_items / (n_items - 1)) * (1 - np.sum(item_variances) / total_variance)
        
        return max(0.0, min(1.0, alpha))  # Clamp to [0, 1]
    
    def _interpret_consistency_score(self, score: float, alpha: float, correlation: float) -> str:
        """Interpret the consistency score."""
        if score >= 0.9:
            return f"Excellent internal consistency (α={alpha:.3f}, r̄={correlation:.3f})"
        elif score >= 0.8:
            return f"Good internal consistency (α={alpha:.3f}, r̄={correlation:.3f})"
        elif score >= 0.7:
            return f"Acceptable internal consistency (α={alpha:.3f}, r̄={correlation:.3f})"
        elif score >= 0.6:
            return f"Questionable internal consistency (α={alpha:.3f}, r̄={correlation:.3f})"
        else:
            return f"Poor internal consistency (α={alpha:.3f}, r̄={correlation:.3f})"


class TestRetestReliabilityTest(ValidationTest):
    """
    Test for test-retest reliability of evaluations.
    
    Validates that evaluations produce consistent results when
    administered multiple times under similar conditions.
    """
    
    def __init__(self, threshold: float = 0.8):
        super().__init__("Test-Retest Reliability", threshold)
    
    def run_test(self, data: Dict[str, List[Any]]) -> ValidationResult:
        """
        Test test-retest reliability using correlation analysis.
        
        Args:
            data: Dictionary with 'test' and 'retest' evaluation results
        """
        try:
            test_scores = data.get('test', [])
            retest_scores = data.get('retest', [])
            
            if len(test_scores) != len(retest_scores) or len(test_scores) < 3:
                return ValidationResult(
                    test_name=self.name,
                    passed=False,
                    score=0.0,
                    threshold=self.threshold,
                    interpretation="Insufficient or mismatched test-retest data"
                )
            
            # Convert to numerical scores
            test_numeric = self._extract_scores(test_scores)
            retest_numeric = self._extract_scores(retest_scores)
            
            # Calculate test-retest correlation
            correlation, p_value = stats.pearsonr(test_numeric, retest_numeric)
            
            # Calculate intraclass correlation coefficient (ICC)
            icc = self._calculate_icc(test_numeric, retest_numeric)
            
            # Calculate mean absolute difference
            mean_abs_diff = np.mean(np.abs(np.array(test_numeric) - np.array(retest_numeric)))
            
            # Overall reliability score (weighted combination)
            reliability_score = (correlation + icc) / 2
            
            passed = reliability_score >= self.threshold
            
            # Confidence interval for correlation
            n = len(test_numeric)
            z_score = 0.5 * np.log((1 + correlation) / (1 - correlation))
            se = 1 / np.sqrt(n - 3)
            z_critical = stats.norm.ppf(0.975)  # 95% CI
            ci_lower = np.tanh(z_score - z_critical * se)
            ci_upper = np.tanh(z_score + z_critical * se)
            
            interpretation = self._interpret_reliability_score(reliability_score, correlation, icc)
            
            recommendations = []
            if not passed:
                recommendations.extend([
                    "Investigate sources of measurement error between test sessions",
                    "Consider increasing standardization of evaluation procedures",
                    "Examine whether time interval between tests is appropriate"
                ])
            
            return ValidationResult(
                test_name=self.name,
                passed=passed,
                score=reliability_score,
                threshold=self.threshold,
                confidence_interval=(ci_lower, ci_upper),
                p_value=p_value,
                details={
                    "pearson_correlation": correlation,
                    "intraclass_correlation": icc,
                    "mean_absolute_difference": mean_abs_diff,
                    "sample_size": n
                },
                interpretation=interpretation,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error in test-retest reliability test: {e}")
            return ValidationResult(
                test_name=self.name,
                passed=False,
                score=0.0,
                threshold=self.threshold,
                interpretation=f"Test failed due to error: {str(e)}"
            )
    
    def _extract_scores(self, results: List[Any]) -> List[float]:
        """Extract numerical scores from results."""
        scores = []
        for result in results:
            if isinstance(result, dict):
                # Try to extract overall score
                if 'overall_score' in result:
                    scores.append(float(result['overall_score']))
                elif 'score' in result:
                    scores.append(float(result['score']))
                elif 'metric_scores' in result:
                    # Average all metric scores
                    metric_values = []
                    for metric_data in result['metric_scores'].values():
                        if isinstance(metric_data, dict):
                            metric_values.append(metric_data.get('mean', 0))
                        else:
                            metric_values.append(metric_data)
                    scores.append(np.mean(metric_values) if metric_values else 0)
                else:
                    scores.append(0.0)
            elif isinstance(result, (int, float)):
                scores.append(float(result))
            else:
                scores.append(0.0)
        return scores
    
    def _calculate_icc(self, test_scores: List[float], retest_scores: List[float]) -> float:
        """Calculate intraclass correlation coefficient."""
        try:
            # Create data matrix for ICC calculation
            scores_matrix = np.column_stack([test_scores, retest_scores])
            
            # Calculate means
            grand_mean = np.mean(scores_matrix)
            subject_means = np.mean(scores_matrix, axis=1)
            occasion_means = np.mean(scores_matrix, axis=0)
            
            # Calculate sum of squares
            n_subjects = len(test_scores)
            n_occasions = 2
            
            ss_total = np.sum((scores_matrix - grand_mean) ** 2)
            ss_between_subjects = n_occasions * np.sum((subject_means - grand_mean) ** 2)
            ss_between_occasions = n_subjects * np.sum((occasion_means - grand_mean) ** 2)
            ss_error = ss_total - ss_between_subjects - ss_between_occasions
            
            # Calculate mean squares
            ms_between_subjects = ss_between_subjects / (n_subjects - 1)
            ms_error = ss_error / ((n_subjects - 1) * (n_occasions - 1))
            
            # ICC(2,1) - consistency
            if ms_error > 0:
                icc = (ms_between_subjects - ms_error) / (ms_between_subjects + ms_error)
            else:
                icc = 1.0
            
            return max(0.0, min(1.0, icc))
            
        except Exception:
            # Fallback to simple correlation if ICC calculation fails
            correlation, _ = stats.pearsonr(test_scores, retest_scores)
            return max(0.0, correlation)
    
    def _interpret_reliability_score(self, score: float, correlation: float, icc: float) -> str:
        """Interpret the reliability score."""
        if score >= 0.9:
            return f"Excellent test-retest reliability (r={correlation:.3f}, ICC={icc:.3f})"
        elif score >= 0.8:
            return f"Good test-retest reliability (r={correlation:.3f}, ICC={icc:.3f})"
        elif score >= 0.7:
            return f"Acceptable test-retest reliability (r={correlation:.3f}, ICC={icc:.3f})"
        elif score >= 0.6:
            return f"Questionable test-retest reliability (r={correlation:.3f}, ICC={icc:.3f})"
        else:
            return f"Poor test-retest reliability (r={correlation:.3f}, ICC={icc:.3f})"


class InterRaterReliabilityTest(ValidationTest):
    """
    Test for inter-rater reliability when multiple evaluators are involved.
    
    Validates that different human evaluators or evaluation systems
    produce consistent ratings for the same causal reasoning tasks.
    """
    
    def __init__(self, threshold: float = 0.8):
        super().__init__("Inter-Rater Reliability", threshold)
    
    def run_test(self, data: Dict[str, List[Any]]) -> ValidationResult:
        """
        Test inter-rater reliability using Cohen's kappa and correlations.
        
        Args:
            data: Dictionary with ratings from different raters
        """
        try:
            rater_names = list(data.keys())
            
            if len(rater_names) < 2:
                return ValidationResult(
                    test_name=self.name,
                    passed=False,
                    score=0.0,
                    threshold=self.threshold,
                    interpretation="Need at least 2 raters for inter-rater reliability"
                )
            
            # Extract ratings
            rater_scores = {}
            min_length = float('inf')
            
            for rater in rater_names:
                scores = self._extract_categorical_ratings(data[rater])
                rater_scores[rater] = scores
                min_length = min(min_length, len(scores))
            
            # Truncate to common length
            for rater in rater_names:
                rater_scores[rater] = rater_scores[rater][:min_length]
            
            if min_length < 3:
                return ValidationResult(
                    test_name=self.name,
                    passed=False,
                    score=0.0,
                    threshold=self.threshold,
                    interpretation="Insufficient ratings for reliability analysis"
                )
            
            # Calculate pairwise Cohen's kappa
            kappa_scores = []
            for i in range(len(rater_names)):
                for j in range(i + 1, len(rater_names)):
                    rater1 = rater_names[i]
                    rater2 = rater_names[j]
                    
                    try:
                        kappa = cohen_kappa_score(rater_scores[rater1], rater_scores[rater2])
                        kappa_scores.append(kappa)
                    except Exception as e:
                        logger.warning(f"Could not calculate kappa for {rater1} vs {rater2}: {e}")
            
            if not kappa_scores:
                return ValidationResult(
                    test_name=self.name,
                    passed=False,
                    score=0.0,
                    threshold=self.threshold,
                    interpretation="Could not calculate inter-rater agreement"
                )
            
            # Average Cohen's kappa
            mean_kappa = np.mean(kappa_scores)
            
            # Calculate Fleiss' kappa for multiple raters (if applicable)
            fleiss_kappa = self._calculate_fleiss_kappa(rater_scores)
            
            # Overall reliability score
            reliability_score = mean_kappa if fleiss_kappa is None else (mean_kappa + fleiss_kappa) / 2
            
            passed = reliability_score >= self.threshold
            
            interpretation = self._interpret_inter_rater_reliability(reliability_score, mean_kappa, fleiss_kappa)
            
            recommendations = []
            if not passed:
                recommendations.extend([
                    "Provide additional training for raters to improve agreement",
                    "Clarify evaluation criteria and scoring guidelines",
                    "Consider revising rating categories or scales"
                ])
            
            return ValidationResult(
                test_name=self.name,
                passed=passed,
                score=reliability_score,
                threshold=self.threshold,
                details={
                    "mean_cohens_kappa": mean_kappa,
                    "fleiss_kappa": fleiss_kappa,
                    "pairwise_kappas": kappa_scores,
                    "num_raters": len(rater_names),
                    "num_ratings": min_length,
                    "rater_names": rater_names
                },
                interpretation=interpretation,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error in inter-rater reliability test: {e}")
            return ValidationResult(
                test_name=self.name,
                passed=False,
                score=0.0,
                threshold=self.threshold,
                interpretation=f"Test failed due to error: {str(e)}"
            )
    
    def _extract_categorical_ratings(self, ratings: List[Any]) -> List[str]:
        """Extract categorical ratings from evaluation results."""
        categorical_ratings = []
        
        for rating in ratings:
            if isinstance(rating, dict):
                # Try to extract relationship type or prediction
                if 'predicted_relationship' in rating:
                    categorical_ratings.append(str(rating['predicted_relationship']))
                elif 'relationship_type' in rating:
                    categorical_ratings.append(str(rating['relationship_type']))
                elif 'prediction' in rating:
                    categorical_ratings.append(str(rating['prediction']))
                else:
                    # Use overall score binned into categories
                    score = rating.get('overall_score', rating.get('score', 0))
                    if score >= 0.8:
                        categorical_ratings.append('high')
                    elif score >= 0.6:
                        categorical_ratings.append('medium')
                    elif score >= 0.4:
                        categorical_ratings.append('low')
                    else:
                        categorical_ratings.append('very_low')
            elif isinstance(rating, str):
                categorical_ratings.append(rating)
            else:
                categorical_ratings.append('unknown')
        
        return categorical_ratings
    
    def _calculate_fleiss_kappa(self, rater_scores: Dict[str, List[str]]) -> Optional[float]:
        """Calculate Fleiss' kappa for multiple raters."""
        try:
            rater_names = list(rater_scores.keys())
            if len(rater_names) < 3:
                return None  # Fleiss' kappa needs at least 3 raters
            
            # Get all unique categories
            all_categories = set()
            for scores in rater_scores.values():
                all_categories.update(scores)
            categories = sorted(list(all_categories))
            
            n_subjects = len(rater_scores[rater_names[0]])
            n_raters = len(rater_names)
            n_categories = len(categories)
            
            # Create agreement matrix
            agreement_matrix = np.zeros((n_subjects, n_categories))
            
            for subject_idx in range(n_subjects):
                for rater in rater_names:
                    if subject_idx < len(rater_scores[rater]):
                        category = rater_scores[rater][subject_idx]
                        if category in categories:
                            cat_idx = categories.index(category)
                            agreement_matrix[subject_idx, cat_idx] += 1
            
            # Calculate Fleiss' kappa
            P_e = np.zeros(n_categories)
            for j in range(n_categories):
                P_e[j] = np.sum(agreement_matrix[:, j]) / (n_subjects * n_raters)
            
            P_bar = 0
            for i in range(n_subjects):
                for j in range(n_categories):
                    P_bar += agreement_matrix[i, j] * (agreement_matrix[i, j] - 1)
            P_bar = P_bar / (n_subjects * n_raters * (n_raters - 1))
            
            Pe_bar = np.sum(P_e ** 2)
            
            if 1 - Pe_bar == 0:
                return 0.0
            
            fleiss_kappa = (P_bar - Pe_bar) / (1 - Pe_bar)
            return max(-1.0, min(1.0, fleiss_kappa))
            
        except Exception as e:
            logger.warning(f"Could not calculate Fleiss' kappa: {e}")
            return None
    
    def _interpret_inter_rater_reliability(self, score: float, cohens_kappa: float, fleiss_kappa: Optional[float]) -> str:
        """Interpret the inter-rater reliability score."""
        fleiss_str = f", Fleiss' κ={fleiss_kappa:.3f}" if fleiss_kappa is not None else ""
        
        if score >= 0.9:
            return f"Excellent inter-rater reliability (κ̄={cohens_kappa:.3f}{fleiss_str})"
        elif score >= 0.8:
            return f"Good inter-rater reliability (κ̄={cohens_kappa:.3f}{fleiss_str})"
        elif score >= 0.7:
            return f"Acceptable inter-rater reliability (κ̄={cohens_kappa:.3f}{fleiss_str})"
        elif score >= 0.6:
            return f"Questionable inter-rater reliability (κ̄={cohens_kappa:.3f}{fleiss_str})"
        else:
            return f"Poor inter-rater reliability (κ̄={cohens_kappa:.3f}{fleiss_str})"


class ConstructValidityTest(ValidationTest):
    """
    Test for construct validity of causal reasoning evaluation.
    
    Validates that the evaluation actually measures causal reasoning
    capabilities rather than other cognitive abilities.
    """
    
    def __init__(self, threshold: float = 0.6):
        super().__init__("Construct Validity", threshold)
    
    def run_test(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Test construct validity using factor analysis and convergent/discriminant validity.
        
        Args:
            data: Dictionary with causal reasoning scores and related measures
        """
        try:
            causal_scores = data.get('causal_reasoning', [])
            related_measures = data.get('related_measures', {})
            unrelated_measures = data.get('unrelated_measures', {})
            
            if len(causal_scores) < 10:
                return ValidationResult(
                    test_name=self.name,
                    passed=False,
                    score=0.0,
                    threshold=self.threshold,
                    interpretation="Insufficient data for construct validity testing"
                )
            
            # Calculate convergent validity (correlation with related measures)
            convergent_validity = self._calculate_convergent_validity(causal_scores, related_measures)
            
            # Calculate discriminant validity (low correlation with unrelated measures)
            discriminant_validity = self._calculate_discriminant_validity(causal_scores, unrelated_measures)
            
            # Factor analysis (if enough variables)
            factor_loading = self._calculate_factor_loading(causal_scores, related_measures)
            
            # Overall construct validity score
            validity_score = (convergent_validity + discriminant_validity + factor_loading) / 3
            
            passed = validity_score >= self.threshold
            
            interpretation = self._interpret_construct_validity(
                validity_score, convergent_validity, discriminant_validity, factor_loading
            )
            
            recommendations = []
            if not passed:
                recommendations.extend([
                    "Investigate whether evaluation captures intended causal reasoning constructs",
                    "Consider adding measures of related cognitive abilities for validation",
                    "Examine potential confounding factors affecting construct validity"
                ])
            
            return ValidationResult(
                test_name=self.name,
                passed=passed,
                score=validity_score,
                threshold=self.threshold,
                details={
                    "convergent_validity": convergent_validity,
                    "discriminant_validity": discriminant_validity,
                    "factor_loading": factor_loading,
                    "num_observations": len(causal_scores)
                },
                interpretation=interpretation,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error in construct validity test: {e}")
            return ValidationResult(
                test_name=self.name,
                passed=False,
                score=0.0,
                threshold=self.threshold,
                interpretation=f"Test failed due to error: {str(e)}"
            )
    
    def _calculate_convergent_validity(self, causal_scores: List[float], related_measures: Dict[str, List[float]]) -> float:
        """Calculate convergent validity with related measures."""
        if not related_measures:
            return 0.5  # Neutral score if no related measures
        
        correlations = []
        for measure_name, measure_scores in related_measures.items():
            if len(measure_scores) == len(causal_scores):
                try:
                    corr, _ = stats.pearsonr(causal_scores, measure_scores)
                    correlations.append(abs(corr))  # Absolute value for convergent validity
                except Exception:
                    continue
        
        return np.mean(correlations) if correlations else 0.5
    
    def _calculate_discriminant_validity(self, causal_scores: List[float], unrelated_measures: Dict[str, List[float]]) -> float:
        """Calculate discriminant validity with unrelated measures."""
        if not unrelated_measures:
            return 0.5  # Neutral score if no unrelated measures
        
        correlations = []
        for measure_name, measure_scores in unrelated_measures.items():
            if len(measure_scores) == len(causal_scores):
                try:
                    corr, _ = stats.pearsonr(causal_scores, measure_scores)
                    correlations.append(abs(corr))  # Low correlation is good for discriminant validity
                except Exception:
                    continue
        
        # Discriminant validity: lower correlation is better
        if correlations:
            return 1.0 - np.mean(correlations)
        else:
            return 0.5
    
    def _calculate_factor_loading(self, causal_scores: List[float], related_measures: Dict[str, List[float]]) -> float:
        """Calculate factor loading for causal reasoning construct."""
        try:
            # Simple proxy: correlation with first principal component
            if not related_measures:
                return 0.5
            
            # Combine all measures into a matrix
            all_measures = [causal_scores]
            for measure_scores in related_measures.values():
                if len(measure_scores) == len(causal_scores):
                    all_measures.append(measure_scores)
            
            if len(all_measures) < 2:
                return 0.5
            
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(all_measures)
            
            # Use first row (causal reasoning correlations) as proxy for factor loading
            factor_loading = np.mean(np.abs(corr_matrix[0, 1:]))  # Exclude self-correlation
            
            return factor_loading
            
        except Exception:
            return 0.5
    
    def _interpret_construct_validity(self, score: float, convergent: float, discriminant: float, factor: float) -> str:
        """Interpret the construct validity score."""
        if score >= 0.8:
            return f"Strong construct validity (convergent: {convergent:.3f}, discriminant: {discriminant:.3f}, factor: {factor:.3f})"
        elif score >= 0.6:
            return f"Adequate construct validity (convergent: {convergent:.3f}, discriminant: {discriminant:.3f}, factor: {factor:.3f})"
        elif score >= 0.4:
            return f"Questionable construct validity (convergent: {convergent:.3f}, discriminant: {discriminant:.3f}, factor: {factor:.3f})"
        else:
            return f"Poor construct validity (convergent: {convergent:.3f}, discriminant: {discriminant:.3f}, factor: {factor:.3f})"


class ValidationSuite:
    """
    Comprehensive validation suite for causal reasoning research.
    
    This suite orchestrates multiple validation tests to provide
    a thorough assessment of evaluation reliability and validity.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize validation tests
        self.tests = {
            'internal_consistency': InternalConsistencyTest(
                threshold=self.config.get('internal_consistency_threshold', 0.7)
            ),
            'test_retest_reliability': TestRetestReliabilityTest(
                threshold=self.config.get('test_retest_threshold', 0.8)
            ),
            'inter_rater_reliability': InterRaterReliabilityTest(
                threshold=self.config.get('inter_rater_threshold', 0.8)
            ),
            'construct_validity': ConstructValidityTest(
                threshold=self.config.get('construct_validity_threshold', 0.6)
            )
        }
        
        self.validation_history = []
    
    def run_comprehensive_validation(self, validation_data: Dict[str, Any]) -> ReliabilityReport:
        """
        Run comprehensive validation across all available tests.
        
        Args:
            validation_data: Dictionary containing data for different validation tests
        
        Returns:
            Comprehensive reliability report
        """
        logger.info("Starting comprehensive validation suite")
        
        test_results = []
        
        # Run each validation test
        for test_name, test in self.tests.items():
            if test_name in validation_data:
                logger.info(f"Running {test_name} validation")
                try:
                    result = test.run_test(validation_data[test_name])
                    test_results.append(result)
                except Exception as e:
                    logger.error(f"Error in {test_name} validation: {e}")
                    test_results.append(ValidationResult(
                        test_name=test_name,
                        passed=False,
                        score=0.0,
                        threshold=test.threshold,
                        interpretation=f"Test failed: {str(e)}"
                    ))
            else:
                logger.warning(f"No data provided for {test_name} validation")
        
        # Calculate overall reliability score
        overall_score = self._calculate_overall_reliability(test_results)
        
        # Extract specific reliability measures
        internal_consistency = self._extract_internal_consistency(test_results)
        test_retest = self._extract_test_retest_reliability(test_results)
        inter_rater = self._extract_inter_rater_reliability(test_results)
        construct_validity = self._extract_construct_validity(test_results)
        criterion_validity = {}  # Placeholder for future implementation
        
        # Generate recommendations
        recommendations = self._generate_validation_recommendations(test_results)
        
        # Create comprehensive report
        report = ReliabilityReport(
            overall_reliability_score=overall_score,
            test_results=test_results,
            internal_consistency=internal_consistency,
            test_retest_reliability=test_retest,
            inter_rater_reliability=inter_rater,
            construct_validity=construct_validity,
            criterion_validity=criterion_validity,
            recommendations=recommendations,
            metadata={
                'validation_timestamp': pd.Timestamp.now().isoformat(),
                'tests_run': len(test_results),
                'tests_passed': sum(1 for r in test_results if r.passed),
                'config': self.config
            }
        )
        
        # Store in history
        self.validation_history.append(report)
        
        logger.info(f"Validation complete. Overall reliability: {overall_score:.3f}")
        
        return report
    
    def _calculate_overall_reliability(self, test_results: List[ValidationResult]) -> float:
        """Calculate overall reliability score from test results."""
        if not test_results:
            return 0.0
        
        # Weight different aspects of reliability
        weights = {
            'Internal Consistency': 0.25,
            'Test-Retest Reliability': 0.35,
            'Inter-Rater Reliability': 0.25,
            'Construct Validity': 0.15
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for result in test_results:
            weight = weights.get(result.test_name, 0.1)
            weighted_score += result.score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _extract_internal_consistency(self, test_results: List[ValidationResult]) -> Dict[str, float]:
        """Extract internal consistency measures."""
        for result in test_results:
            if result.test_name == 'Internal Consistency':
                return {
                    'cronbach_alpha': result.details.get('cronbach_alpha', 0.0),
                    'mean_inter_correlation': result.details.get('mean_inter_correlation', 0.0),
                    'overall_score': result.score
                }
        return {}
    
    def _extract_test_retest_reliability(self, test_results: List[ValidationResult]) -> Dict[str, float]:
        """Extract test-retest reliability measures."""
        for result in test_results:
            if result.test_name == 'Test-Retest Reliability':
                return {
                    'pearson_correlation': result.details.get('pearson_correlation', 0.0),
                    'intraclass_correlation': result.details.get('intraclass_correlation', 0.0),
                    'overall_score': result.score
                }
        return {}
    
    def _extract_inter_rater_reliability(self, test_results: List[ValidationResult]) -> Dict[str, float]:
        """Extract inter-rater reliability measures."""
        for result in test_results:
            if result.test_name == 'Inter-Rater Reliability':
                return {
                    'mean_cohens_kappa': result.details.get('mean_cohens_kappa', 0.0),
                    'fleiss_kappa': result.details.get('fleiss_kappa', 0.0),
                    'overall_score': result.score
                }
        return {}
    
    def _extract_construct_validity(self, test_results: List[ValidationResult]) -> Dict[str, float]:
        """Extract construct validity measures."""
        for result in test_results:
            if result.test_name == 'Construct Validity':
                return {
                    'convergent_validity': result.details.get('convergent_validity', 0.0),
                    'discriminant_validity': result.details.get('discriminant_validity', 0.0),
                    'factor_loading': result.details.get('factor_loading', 0.0),
                    'overall_score': result.score
                }
        return {}
    
    def _generate_validation_recommendations(self, test_results: List[ValidationResult]) -> List[str]:
        """Generate comprehensive validation recommendations."""
        recommendations = []
        
        # Aggregate recommendations from individual tests
        for result in test_results:
            recommendations.extend(result.recommendations)
        
        # Add overall recommendations based on patterns
        failed_tests = [r for r in test_results if not r.passed]
        if len(failed_tests) > len(test_results) // 2:
            recommendations.append(
                "Multiple validation tests failed - consider fundamental revision of evaluation approach"
            )
        
        # Check for specific patterns
        low_scores = [r for r in test_results if r.score < 0.5]
        if low_scores:
            recommendations.append(
                f"Very low scores in {len(low_scores)} validation test(s) - investigate systematic issues"
            )
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def save_validation_report(self, report: ReliabilityReport, filepath: str) -> None:
        """Save validation report to file."""
        # Convert report to serializable format
        report_dict = {
            'overall_reliability_score': report.overall_reliability_score,
            'test_results': [
                {
                    'test_name': r.test_name,
                    'passed': r.passed,
                    'score': r.score,
                    'threshold': r.threshold,
                    'confidence_interval': r.confidence_interval,
                    'p_value': r.p_value,
                    'effect_size': r.effect_size,
                    'details': r.details,
                    'interpretation': r.interpretation,
                    'recommendations': r.recommendations
                }
                for r in report.test_results
            ],
            'internal_consistency': report.internal_consistency,
            'test_retest_reliability': report.test_retest_reliability,
            'inter_rater_reliability': report.inter_rater_reliability,
            'construct_validity': report.construct_validity,
            'criterion_validity': report.criterion_validity,
            'recommendations': report.recommendations,
            'metadata': report.metadata
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to {filepath}")
    
    def generate_validation_plots(self, report: ReliabilityReport, output_dir: str) -> Dict[str, str]:
        """Generate visualization plots for validation results."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        plots = {}
        
        # Plot 1: Overall validation scores
        fig, ax = plt.subplots(figsize=(10, 6))
        
        test_names = [r.test_name for r in report.test_results]
        test_scores = [r.score for r in report.test_results]
        test_thresholds = [r.threshold for r in report.test_results]
        
        colors = ['green' if r.passed else 'red' for r in report.test_results]
        
        bars = ax.bar(test_names, test_scores, color=colors, alpha=0.7)
        ax.axhline(y=np.mean(test_thresholds), color='black', linestyle='--', 
                  label=f'Mean Threshold ({np.mean(test_thresholds):.2f})')
        
        ax.set_ylabel('Validation Score')
        ax.set_title('Validation Test Results')
        ax.set_ylim(0, 1)
        ax.legend()
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plot_file = output_path / 'validation_scores.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        plots['validation_scores'] = str(plot_file)
        
        # Plot 2: Reliability components radar chart
        if len(report.test_results) >= 3:
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            
            angles = np.linspace(0, 2 * np.pi, len(test_names), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            scores = test_scores + test_scores[:1]  # Complete the circle
            thresholds = test_thresholds + test_thresholds[:1]
            
            ax.plot(angles, scores, 'o-', linewidth=2, label='Actual Scores')
            ax.fill(angles, scores, alpha=0.25)
            ax.plot(angles, thresholds, 's--', linewidth=1, label='Thresholds')
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(test_names)
            ax.set_ylim(0, 1)
            ax.set_title('Validation Profile', y=1.08)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
            
            plt.tight_layout()
            
            plot_file = output_path / 'validation_radar.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            plots['validation_radar'] = str(plot_file)
        
        return plots