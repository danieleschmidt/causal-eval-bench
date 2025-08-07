"""
Statistical analysis tools for causal reasoning evaluation.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
from scipy.stats import mannwhitneyu, wilcoxon, chi2_contingency, fisher_exact
import pandas as pd

logger = logging.getLogger(__name__)


class TestType(Enum):
    """Types of statistical tests."""
    T_TEST = "t_test"
    MANN_WHITNEY = "mann_whitney"  
    WILCOXON = "wilcoxon"
    CHI_SQUARE = "chi_square"
    FISHER_EXACT = "fisher_exact"
    MCNEMAR = "mcnemar"
    PAIRED_T_TEST = "paired_t_test"


@dataclass
class SignificanceTest:
    """Results of a statistical significance test."""
    
    test_type: TestType
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    alpha: float = 0.05
    interpretation: str = ""
    assumptions_met: bool = True
    sample_sizes: Dict[str, int] = None
    
    def __post_init__(self):
        """Post-process results."""
        self.is_significant = self.p_value < self.alpha
        if not self.interpretation:
            self.interpretation = self._generate_interpretation()
    
    def _generate_interpretation(self) -> str:
        """Generate human-readable interpretation."""
        significance = "significant" if self.is_significant else "not significant"
        effect_magnitude = self._interpret_effect_size()
        
        return (
            f"The test shows a {significance} difference (p={self.p_value:.4f}) "
            f"with a {effect_magnitude} effect size ({self.effect_size:.3f})"
        )
    
    def _interpret_effect_size(self) -> str:
        """Interpret effect size magnitude."""
        abs_effect = abs(self.effect_size)
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"


class StatisticalAnalyzer:
    """Advanced statistical analysis for causal reasoning evaluations."""
    
    def __init__(self, alpha: float = 0.05):
        """Initialize the analyzer."""
        self.alpha = alpha
        logger.info(f"Statistical analyzer initialized with alpha={alpha}")
    
    def compare_model_performance(
        self,
        model_a_scores: List[float],
        model_b_scores: List[float],
        model_a_name: str = "Model A",
        model_b_name: str = "Model B",
        test_type: Optional[TestType] = None,
        paired: bool = False
    ) -> SignificanceTest:
        """Compare performance between two models."""
        
        if len(model_a_scores) != len(model_b_scores) and paired:
            raise ValueError("Paired test requires equal sample sizes")
        
        # Auto-select test if not specified
        if test_type is None:
            test_type = self._select_appropriate_test(
                model_a_scores, model_b_scores, paired
            )
        
        logger.info(f"Comparing {model_a_name} vs {model_b_name} using {test_type.value}")
        
        if test_type == TestType.T_TEST:
            return self._independent_t_test(model_a_scores, model_b_scores)
        elif test_type == TestType.PAIRED_T_TEST:
            return self._paired_t_test(model_a_scores, model_b_scores)
        elif test_type == TestType.MANN_WHITNEY:
            return self._mann_whitney_test(model_a_scores, model_b_scores)
        elif test_type == TestType.WILCOXON:
            return self._wilcoxon_test(model_a_scores, model_b_scores)
        else:
            raise ValueError(f"Unsupported test type: {test_type}")
    
    def mcnemar_test(
        self,
        model_a_correct: List[bool],
        model_b_correct: List[bool]
    ) -> SignificanceTest:
        """McNemar's test for comparing paired binary outcomes."""
        
        if len(model_a_correct) != len(model_b_correct):
            raise ValueError("McNemar test requires equal sample sizes")
        
        # Create contingency table
        both_correct = sum(1 for a, b in zip(model_a_correct, model_b_correct) if a and b)
        a_correct_b_wrong = sum(1 for a, b in zip(model_a_correct, model_b_correct) if a and not b)
        a_wrong_b_correct = sum(1 for a, b in zip(model_a_correct, model_b_correct) if not a and b)
        both_wrong = sum(1 for a, b in zip(model_a_correct, model_b_correct) if not a and not b)
        
        # McNemar's test statistic
        n_discordant = a_correct_b_wrong + a_wrong_b_correct
        
        if n_discordant == 0:
            # No discordant pairs
            p_value = 1.0
            statistic = 0.0
            effect_size = 0.0
        else:
            # Chi-square test statistic
            statistic = ((abs(a_correct_b_wrong - a_wrong_b_correct) - 1) ** 2) / n_discordant
            p_value = 1 - stats.chi2.cdf(statistic, df=1)
            
            # Effect size (odds ratio)
            if a_wrong_b_correct == 0:
                effect_size = float('inf')
            else:
                effect_size = a_correct_b_wrong / a_wrong_b_correct
        
        # Confidence interval for difference in proportions
        n = len(model_a_correct)
        p_a = sum(model_a_correct) / n
        p_b = sum(model_b_correct) / n
        diff = p_a - p_b
        
        se_diff = np.sqrt((a_correct_b_wrong + a_wrong_b_correct) / (n ** 2))
        ci_lower = diff - 1.96 * se_diff
        ci_upper = diff + 1.96 * se_diff
        
        return SignificanceTest(
            test_type=TestType.MCNEMAR,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=p_value < self.alpha,
            alpha=self.alpha,
            sample_sizes={"total": n, "discordant": n_discordant}
        )
    
    def analyze_task_difficulty(
        self,
        scores_by_difficulty: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """Analyze performance across different difficulty levels."""
        
        difficulties = list(scores_by_difficulty.keys())
        if len(difficulties) < 2:
            return {"error": "Need at least 2 difficulty levels for analysis"}
        
        # ANOVA test for overall difference
        scores_groups = [scores_by_difficulty[diff] for diff in difficulties]
        f_statistic, p_value = stats.f_oneway(*scores_groups)
        
        # Post-hoc pairwise comparisons
        pairwise_comparisons = {}
        for i, diff_a in enumerate(difficulties):
            for diff_b in difficulties[i+1:]:
                comparison = self.compare_model_performance(
                    scores_by_difficulty[diff_a],
                    scores_by_difficulty[diff_b],
                    model_a_name=diff_a,
                    model_b_name=diff_b
                )
                pairwise_comparisons[f"{diff_a}_vs_{diff_b}"] = comparison
        
        # Effect sizes between consecutive difficulties
        effect_sizes = {}
        if len(difficulties) > 2:
            sorted_difficulties = sorted(difficulties)
            for i in range(len(sorted_difficulties) - 1):
                current = sorted_difficulties[i]
                next_diff = sorted_difficulties[i + 1]
                
                mean_current = np.mean(scores_by_difficulty[current])
                mean_next = np.mean(scores_by_difficulty[next_diff])
                pooled_std = np.sqrt((
                    np.var(scores_by_difficulty[current]) + 
                    np.var(scores_by_difficulty[next_diff])
                ) / 2)
                
                effect_sizes[f"{current}_to_{next_diff}"] = (mean_next - mean_current) / pooled_std
        
        return {
            "anova_f_statistic": f_statistic,
            "anova_p_value": p_value,
            "overall_significant": p_value < self.alpha,
            "pairwise_comparisons": pairwise_comparisons,
            "effect_sizes": effect_sizes,
            "means_by_difficulty": {
                diff: np.mean(scores) for diff, scores in scores_by_difficulty.items()
            },
            "std_by_difficulty": {
                diff: np.std(scores) for diff, scores in scores_by_difficulty.items()
            }
        }
    
    def bootstrap_confidence_interval(
        self,
        data: List[float],
        statistic_func=np.mean,
        n_bootstrap: int = 10000,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for a statistic."""
        
        bootstrap_stats = []
        n_samples = len(data)
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            bootstrap_sample = np.random.choice(data, size=n_samples, replace=True)
            bootstrap_stats.append(statistic_func(bootstrap_sample))
        
        # Calculate percentiles for confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)
        
        return (ci_lower, ci_upper)
    
    def power_analysis(
        self,
        effect_size: float,
        sample_size: int,
        alpha: float = None,
        test_type: TestType = TestType.T_TEST
    ) -> Dict[str, float]:
        """Calculate statistical power for a given effect size and sample size."""
        
        if alpha is None:
            alpha = self.alpha
        
        # This is a simplified power calculation
        # For more complex analyses, consider using specialized libraries
        
        if test_type in [TestType.T_TEST, TestType.PAIRED_T_TEST]:
            # Cohen's d to power calculation
            delta = effect_size * np.sqrt(sample_size / 2)
            critical_t = stats.t.ppf(1 - alpha/2, df=sample_size-1)
            power = 1 - stats.t.cdf(critical_t - delta, df=sample_size-1) + stats.t.cdf(-critical_t - delta, df=sample_size-1)
        else:
            # Approximate power for non-parametric tests
            power = 0.8  # Placeholder - actual calculation depends on specific test
        
        return {
            "power": power,
            "effect_size": effect_size,
            "sample_size": sample_size,
            "alpha": alpha,
            "test_type": test_type.value
        }
    
    def sample_size_calculation(
        self,
        effect_size: float,
        power: float = 0.8,
        alpha: float = None,
        test_type: TestType = TestType.T_TEST
    ) -> int:
        """Calculate required sample size for desired power."""
        
        if alpha is None:
            alpha = self.alpha
        
        if test_type in [TestType.T_TEST, TestType.PAIRED_T_TEST]:
            # Simplified calculation for t-tests
            z_alpha = stats.norm.ppf(1 - alpha/2)
            z_beta = stats.norm.ppf(power)
            
            n = ((z_alpha + z_beta) / effect_size) ** 2 * 2
            return int(np.ceil(n))
        else:
            # Conservative estimate for non-parametric tests (15% larger sample)
            parametric_n = self.sample_size_calculation(effect_size, power, alpha, TestType.T_TEST)
            return int(np.ceil(parametric_n * 1.15))
    
    def _select_appropriate_test(
        self,
        group_a: List[float],
        group_b: List[float],
        paired: bool = False
    ) -> TestType:
        """Automatically select appropriate statistical test."""
        
        # Check for normality using Shapiro-Wilk test
        _, p_a = stats.shapiro(group_a)
        _, p_b = stats.shapiro(group_b)
        
        both_normal = p_a > 0.05 and p_b > 0.05
        
        if paired:
            return TestType.PAIRED_T_TEST if both_normal else TestType.WILCOXON
        else:
            return TestType.T_TEST if both_normal else TestType.MANN_WHITNEY
    
    def _independent_t_test(
        self,
        group_a: List[float],
        group_b: List[float]
    ) -> SignificanceTest:
        """Perform independent samples t-test."""
        
        statistic, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
        
        # Cohen's d effect size
        mean_a, mean_b = np.mean(group_a), np.mean(group_b)
        pooled_std = np.sqrt((np.var(group_a) + np.var(group_b)) / 2)
        cohens_d = (mean_a - mean_b) / pooled_std
        
        # Confidence interval for difference in means
        se_diff = np.sqrt(np.var(group_a)/len(group_a) + np.var(group_b)/len(group_b))
        df = len(group_a) + len(group_b) - 2
        t_critical = stats.t.ppf(1 - self.alpha/2, df)
        
        diff = mean_a - mean_b
        ci_lower = diff - t_critical * se_diff
        ci_upper = diff + t_critical * se_diff
        
        return SignificanceTest(
            test_type=TestType.T_TEST,
            statistic=statistic,
            p_value=p_value,
            effect_size=cohens_d,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=p_value < self.alpha,
            alpha=self.alpha,
            sample_sizes={"group_a": len(group_a), "group_b": len(group_b)}
        )
    
    def _paired_t_test(
        self,
        group_a: List[float],
        group_b: List[float]
    ) -> SignificanceTest:
        """Perform paired samples t-test."""
        
        statistic, p_value = stats.ttest_rel(group_a, group_b)
        
        # Effect size for paired samples
        differences = np.array(group_a) - np.array(group_b)
        effect_size = np.mean(differences) / np.std(differences)
        
        # Confidence interval for mean difference
        mean_diff = np.mean(differences)
        se_diff = np.std(differences) / np.sqrt(len(differences))
        t_critical = stats.t.ppf(1 - self.alpha/2, len(differences) - 1)
        
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff
        
        return SignificanceTest(
            test_type=TestType.PAIRED_T_TEST,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=p_value < self.alpha,
            alpha=self.alpha,
            sample_sizes={"pairs": len(group_a)}
        )
    
    def _mann_whitney_test(
        self,
        group_a: List[float],
        group_b: List[float]
    ) -> SignificanceTest:
        """Perform Mann-Whitney U test."""
        
        statistic, p_value = mannwhitneyu(group_a, group_b, alternative='two-sided')
        
        # Effect size (rank-biserial correlation)
        n_a, n_b = len(group_a), len(group_b)
        u_statistic = statistic
        effect_size = 1 - (2 * u_statistic) / (n_a * n_b)
        
        # Confidence interval (approximate)
        ci_lower, ci_upper = (-1, 1)  # Placeholder - exact CI calculation is complex
        
        return SignificanceTest(
            test_type=TestType.MANN_WHITNEY,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=p_value < self.alpha,
            alpha=self.alpha,
            assumptions_met=True,  # Non-parametric test
            sample_sizes={"group_a": n_a, "group_b": n_b}
        )
    
    def _wilcoxon_test(
        self,
        group_a: List[float],
        group_b: List[float]
    ) -> SignificanceTest:
        """Perform Wilcoxon signed-rank test."""
        
        statistic, p_value = wilcoxon(group_a, group_b)
        
        # Effect size (matched pairs rank-biserial correlation)
        differences = np.array(group_a) - np.array(group_b)
        effect_size = statistic / (len(differences) * (len(differences) + 1) / 4)
        
        # Confidence interval (approximate)
        ci_lower, ci_upper = (-1, 1)  # Placeholder
        
        return SignificanceTest(
            test_type=TestType.WILCOXON,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=p_value < self.alpha,
            alpha=self.alpha,
            assumptions_met=True,  # Non-parametric test
            sample_sizes={"pairs": len(group_a)}
        )
    
    def multiple_comparisons_correction(
        self,
        p_values: List[float],
        method: str = "bonferroni"
    ) -> Tuple[List[float], List[bool]]:
        """Apply multiple comparisons correction."""
        
        n_tests = len(p_values)
        
        if method == "bonferroni":
            corrected_alpha = self.alpha / n_tests
            significant = [p < corrected_alpha for p in p_values]
            adjusted_p = [min(p * n_tests, 1.0) for p in p_values]
        
        elif method == "holm":
            # Holm-Bonferroni method
            indexed_p = list(enumerate(p_values))
            indexed_p.sort(key=lambda x: x[1])  # Sort by p-value
            
            adjusted_p = [0.0] * n_tests
            significant = [False] * n_tests
            
            for i, (original_idx, p) in enumerate(indexed_p):
                adjusted_alpha = self.alpha / (n_tests - i)
                adjusted_p[original_idx] = min(p * (n_tests - i), 1.0)
                significant[original_idx] = p < adjusted_alpha
                
                # Stop if this test is not significant (Holm's step-down procedure)
                if not significant[original_idx]:
                    break
        
        else:
            raise ValueError(f"Unknown correction method: {method}")
        
        return adjusted_p, significant
    
    def generate_statistical_report(
        self,
        comparisons: List[SignificanceTest],
        title: str = "Statistical Analysis Report"
    ) -> str:
        """Generate a comprehensive statistical report."""
        
        report = f"# {title}\n\n"
        report += f"Analysis conducted with α = {self.alpha}\n\n"
        
        significant_count = sum(1 for comp in comparisons if comp.is_significant)
        report += f"## Summary\n"
        report += f"- Total comparisons: {len(comparisons)}\n"
        report += f"- Significant results: {significant_count}\n"
        report += f"- Non-significant results: {len(comparisons) - significant_count}\n\n"
        
        report += "## Individual Tests\n\n"
        for i, comparison in enumerate(comparisons, 1):
            report += f"### Test {i}: {comparison.test_type.value.replace('_', ' ').title()}\n"
            report += f"- **Statistic**: {comparison.statistic:.4f}\n"
            report += f"- **P-value**: {comparison.p_value:.4f}\n"
            report += f"- **Effect Size**: {comparison.effect_size:.4f}\n"
            report += f"- **Significance**: {'Yes' if comparison.is_significant else 'No'}\n"
            report += f"- **Interpretation**: {comparison.interpretation}\n\n"
        
        return report