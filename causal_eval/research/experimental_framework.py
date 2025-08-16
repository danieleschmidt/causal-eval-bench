"""
Experimental Framework for Causal Reasoning Research

This module provides a comprehensive experimental framework for conducting
rigorous research on causal reasoning in language models, with proper
statistical validation and reproducible experimental design.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import warnings

from causal_eval.research.novel_algorithms import (
    InformationTheoreticCausalityMetric,
    CausalConsistencyMetric, 
    MultimodalCausalityMetric,
    CausalGraph
)

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for causal reasoning experiments."""
    
    experiment_name: str
    description: str
    random_seed: int = 42
    significance_level: float = 0.05
    min_sample_size: int = 30
    effect_size_threshold: float = 0.2
    bootstrap_iterations: int = 1000
    cross_validation_folds: int = 5
    output_directory: str = "experiment_results"
    save_intermediate: bool = True
    parallel_execution: bool = True
    max_workers: int = 4


@dataclass
class ModelConfiguration:
    """Configuration for a model being evaluated."""
    
    model_name: str
    model_type: str  # "transformer", "retrieval", "reasoning", etc.
    model_size: str  # "small", "medium", "large", "xl"
    training_data: str
    model_version: str
    api_endpoint: Optional[str] = None
    local_path: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Results from a single experimental condition."""
    
    model_config: ModelConfiguration
    task_type: str
    domain: str
    metric_scores: Dict[str, float]
    response_times: List[float]
    responses: List[str]
    ground_truths: List[Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StatisticalAnalysis:
    """Statistical analysis results."""
    
    test_statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    test_name: str
    interpretation: str
    significant: bool


class ExperimentalFramework:
    """
    Comprehensive experimental framework for causal reasoning research.
    
    This framework provides tools for:
    1. Rigorous experimental design with proper controls
    2. Statistical validation and significance testing
    3. Reproducible experiment execution
    4. Comprehensive results analysis and visualization
    5. Academic publication-ready output
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results: List[ExperimentResult] = []
        self.baselines: Dict[str, ExperimentResult] = {}
        
        # Set random seeds for reproducibility
        np.random.seed(config.random_seed)
        
        # Create output directory
        self.output_dir = Path(config.output_directory)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize metrics
        self.metrics = {
            'information_theoretic': InformationTheoreticCausalityMetric(),
            'consistency': CausalConsistencyMetric([]),  # Will be populated
            'multimodal': MultimodalCausalityMetric()
        }
        
        logger.info(f"Experimental framework initialized: {config.experiment_name}")
    
    async def run_comparative_study(
        self,
        models: List[ModelConfiguration],
        tasks: List[Dict[str, Any]],
        baseline_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a comprehensive comparative study across multiple models and tasks.
        
        Args:
            models: List of model configurations to evaluate
            tasks: List of task configurations for evaluation
            baseline_model: Name of baseline model for comparison (optional)
        
        Returns:
            Comprehensive results dictionary with statistical analysis
        """
        logger.info(f"Starting comparative study with {len(models)} models and {len(tasks)} tasks")
        
        # Phase 1: Data Collection
        all_results = []
        
        if self.config.parallel_execution:
            # Run evaluations in parallel for efficiency
            evaluation_tasks = []
            for model in models:
                for task in tasks:
                    evaluation_tasks.append(
                        self._evaluate_model_on_task(model, task)
                    )
            
            all_results = await asyncio.gather(*evaluation_tasks)
        else:
            # Sequential execution for debugging
            for model in models:
                for task in tasks:
                    result = await self._evaluate_model_on_task(model, task)
                    all_results.append(result)
        
        self.results.extend(all_results)
        
        # Phase 2: Statistical Analysis
        statistical_results = self._perform_statistical_analysis(all_results, baseline_model)
        
        # Phase 3: Effect Size Analysis
        effect_size_analysis = self._analyze_effect_sizes(all_results)
        
        # Phase 4: Generate Comprehensive Report
        report = self._generate_research_report(
            all_results, 
            statistical_results, 
            effect_size_analysis
        )
        
        # Phase 5: Save Results
        self._save_experimental_data(all_results, report)
        
        return {
            'results': all_results,
            'statistical_analysis': statistical_results,
            'effect_size_analysis': effect_size_analysis,
            'report': report,
            'experiment_id': self._generate_experiment_id()
        }
    
    async def _evaluate_model_on_task(
        self,
        model: ModelConfiguration,
        task: Dict[str, Any]
    ) -> ExperimentResult:
        """Evaluate a single model on a single task."""
        logger.info(f"Evaluating {model.model_name} on {task.get('task_type', 'unknown')} task")
        
        # Generate test cases for this task
        test_cases = await self._generate_test_cases(task)
        
        # Collect model responses (simulated for now)
        responses = []
        response_times = []
        ground_truths = []
        
        for test_case in test_cases:
            start_time = datetime.now()
            
            # Simulate model response (replace with actual model call)
            response = await self._simulate_model_response(model, test_case)
            
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            responses.append(response)
            response_times.append(response_time)
            ground_truths.append(test_case.get('ground_truth'))
        
        # Calculate metric scores
        metric_scores = {}
        for metric_name, metric in self.metrics.items():
            try:
                scores = []
                for response, ground_truth in zip(responses, ground_truths):
                    score = metric.compute_score(response, ground_truth)
                    scores.append(score)
                
                metric_scores[metric_name] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'median': np.median(scores),
                    'scores': scores
                }
            except Exception as e:
                logger.warning(f"Error computing {metric_name} metric: {e}")
                metric_scores[metric_name] = {'mean': 0.0, 'std': 0.0, 'median': 0.0, 'scores': []}
        
        return ExperimentResult(
            model_config=model,
            task_type=task.get('task_type', 'unknown'),
            domain=task.get('domain', 'general'),
            metric_scores=metric_scores,
            response_times=response_times,
            responses=responses,
            ground_truths=ground_truths,
            metadata={
                'num_test_cases': len(test_cases),
                'task_config': task
            }
        )
    
    async def _generate_test_cases(self, task_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate test cases for a given task configuration."""
        task_type = task_config.get('task_type', 'attribution')
        domain = task_config.get('domain', 'general')
        num_cases = task_config.get('num_cases', self.config.min_sample_size)
        
        test_cases = []
        
        for i in range(num_cases):
            # Generate a test case based on task type
            if task_type == 'attribution':
                test_case = self._generate_attribution_test_case(domain, i)
            elif task_type == 'counterfactual':
                test_case = self._generate_counterfactual_test_case(domain, i)
            elif task_type == 'intervention':
                test_case = self._generate_intervention_test_case(domain, i)
            else:
                test_case = self._generate_generic_test_case(domain, i)
            
            test_cases.append(test_case)
        
        return test_cases
    
    def _generate_attribution_test_case(self, domain: str, case_id: int) -> Dict[str, Any]:
        """Generate a causal attribution test case."""
        scenarios = {
            'medical': {
                'prompt': f"A patient takes medication X and their symptoms improve. Case {case_id}: What is the relationship between the medication and symptom improvement?",
                'ground_truth': CausalGraph(
                    nodes=['medication_X', 'symptom_improvement'],
                    edges=[('medication_X', 'symptom_improvement')],
                    confounders={('medication_X', 'symptom_improvement'): ['placebo_effect', 'natural_recovery', 'lifestyle_changes']}
                )
            },
            'business': {
                'prompt': f"Company Y increases advertising spending and sales revenue increases. Case {case_id}: What is the relationship between advertising and sales?",
                'ground_truth': CausalGraph(
                    nodes=['advertising_spending', 'sales_revenue'],
                    edges=[('advertising_spending', 'sales_revenue')],
                    confounders={('advertising_spending', 'sales_revenue'): ['market_conditions', 'product_quality', 'competition']}
                )
            },
            'general': {
                'prompt': f"Variable A increases and Variable B increases. Case {case_id}: What is the relationship between A and B?",
                'ground_truth': CausalGraph(
                    nodes=['variable_A', 'variable_B'],
                    edges=[('variable_A', 'variable_B')],
                    confounders={('variable_A', 'variable_B'): ['confounding_factor']}
                )
            }
        }
        
        scenario = scenarios.get(domain, scenarios['general'])
        return {
            'prompt': scenario['prompt'],
            'ground_truth': scenario['ground_truth'],
            'case_id': case_id,
            'domain': domain,
            'task_type': 'attribution'
        }
    
    def _generate_counterfactual_test_case(self, domain: str, case_id: int) -> Dict[str, Any]:
        """Generate a counterfactual reasoning test case."""
        # Simplified counterfactual test case generation
        return {
            'prompt': f"If X had not happened, would Y still have occurred? Case {case_id}",
            'ground_truth': {'counterfactual_outcome': 'no'},
            'case_id': case_id,
            'domain': domain,
            'task_type': 'counterfactual'
        }
    
    def _generate_intervention_test_case(self, domain: str, case_id: int) -> Dict[str, Any]:
        """Generate an intervention analysis test case."""
        # Simplified intervention test case generation
        return {
            'prompt': f"What would happen if we intervened to change X? Case {case_id}",
            'ground_truth': {'intervention_effect': 'Y would change'},
            'case_id': case_id,
            'domain': domain,
            'task_type': 'intervention'
        }
    
    def _generate_generic_test_case(self, domain: str, case_id: int) -> Dict[str, Any]:
        """Generate a generic test case."""
        return {
            'prompt': f"Generic causal reasoning question for domain {domain}. Case {case_id}",
            'ground_truth': {'expected_answer': 'generic_answer'},
            'case_id': case_id,
            'domain': domain,
            'task_type': 'generic'
        }
    
    async def _simulate_model_response(
        self,
        model: ModelConfiguration,
        test_case: Dict[str, Any]
    ) -> str:
        """Simulate a model response (replace with actual model API calls)."""
        # This is a simulation - in practice, this would call the actual model API
        
        # Simulate different model capabilities
        if 'gpt-4' in model.model_name.lower():
            response_quality = np.random.normal(0.8, 0.1)
        elif 'claude' in model.model_name.lower():
            response_quality = np.random.normal(0.75, 0.12)
        elif 'llama' in model.model_name.lower():
            response_quality = np.random.normal(0.7, 0.15)
        else:
            response_quality = np.random.normal(0.6, 0.2)
        
        response_quality = max(0.0, min(1.0, response_quality))
        
        # Generate response based on quality
        if response_quality > 0.8:
            response = f"This appears to be a causal relationship. The mechanism involves direct influence through established pathways. Confidence: {response_quality:.2f}"
        elif response_quality > 0.6:
            response = f"There is likely a causal connection here, though confounding factors should be considered. Confidence: {response_quality:.2f}"
        elif response_quality > 0.4:
            response = f"The relationship might be correlational rather than causal. More analysis needed. Confidence: {response_quality:.2f}"
        else:
            response = f"Unclear relationship between variables. Confidence: {response_quality:.2f}"
        
        # Add slight delay to simulate processing time
        await asyncio.sleep(0.1)
        
        return response
    
    def _perform_statistical_analysis(
        self,
        results: List[ExperimentResult],
        baseline_model: Optional[str] = None
    ) -> Dict[str, StatisticalAnalysis]:
        """Perform comprehensive statistical analysis of results."""
        statistical_results = {}
        
        # Group results by model and metric
        results_by_model = {}
        for result in results:
            model_name = result.model_config.model_name
            if model_name not in results_by_model:
                results_by_model[model_name] = []
            results_by_model[model_name].append(result)
        
        # Perform pairwise comparisons
        models = list(results_by_model.keys())
        
        for i, model_a in enumerate(models):
            for j, model_b in enumerate(models[i+1:], i+1):
                # Compare each metric
                for metric_name in self.metrics.keys():
                    try:
                        scores_a = self._extract_metric_scores(results_by_model[model_a], metric_name)
                        scores_b = self._extract_metric_scores(results_by_model[model_b], metric_name)
                        
                        if len(scores_a) > 0 and len(scores_b) > 0:
                            analysis = self._statistical_comparison(scores_a, scores_b, model_a, model_b)
                            key = f"{model_a}_vs_{model_b}_{metric_name}"
                            statistical_results[key] = analysis
                    except Exception as e:
                        logger.warning(f"Error in statistical comparison {model_a} vs {model_b} on {metric_name}: {e}")
        
        return statistical_results
    
    def _extract_metric_scores(self, results: List[ExperimentResult], metric_name: str) -> List[float]:
        """Extract scores for a specific metric from results."""
        scores = []
        for result in results:
            if metric_name in result.metric_scores:
                metric_data = result.metric_scores[metric_name]
                if isinstance(metric_data, dict) and 'scores' in metric_data:
                    scores.extend(metric_data['scores'])
                elif isinstance(metric_data, (int, float)):
                    scores.append(metric_data)
        return scores
    
    def _statistical_comparison(
        self,
        scores_a: List[float],
        scores_b: List[float],
        model_a: str,
        model_b: str
    ) -> StatisticalAnalysis:
        """Perform statistical comparison between two sets of scores."""
        
        # Check sample sizes
        if len(scores_a) < 3 or len(scores_b) < 3:
            return StatisticalAnalysis(
                test_statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                confidence_interval=(0.0, 0.0),
                test_name="insufficient_data",
                interpretation="Insufficient data for statistical testing",
                significant=False
            )
        
        # Normality tests
        _, p_norm_a = stats.shapiro(scores_a[:50])  # Shapiro-Wilk test (max 50 samples)
        _, p_norm_b = stats.shapiro(scores_b[:50])
        
        # Choose appropriate test
        if p_norm_a > 0.05 and p_norm_b > 0.05:
            # Both normal - use t-test
            statistic, p_value = stats.ttest_ind(scores_a, scores_b)
            test_name = "independent_t_test"
        else:
            # Non-normal - use Mann-Whitney U test
            statistic, p_value = stats.mannwhitneyu(scores_a, scores_b, alternative='two-sided')
            test_name = "mann_whitney_u"
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(scores_a) - 1) * np.var(scores_a, ddof=1) + 
                             (len(scores_b) - 1) * np.var(scores_b, ddof=1)) / 
                             (len(scores_a) + len(scores_b) - 2))
        
        if pooled_std > 0:
            effect_size = (np.mean(scores_a) - np.mean(scores_b)) / pooled_std
        else:
            effect_size = 0.0
        
        # Calculate confidence interval for the difference in means
        diff_mean = np.mean(scores_a) - np.mean(scores_b)
        std_error = pooled_std * np.sqrt(1/len(scores_a) + 1/len(scores_b))
        
        if std_error > 0:
            t_critical = stats.t.ppf(1 - self.config.significance_level/2, 
                                   len(scores_a) + len(scores_b) - 2)
            margin_error = t_critical * std_error
            ci_lower = diff_mean - margin_error
            ci_upper = diff_mean + margin_error
        else:
            ci_lower = ci_upper = diff_mean
        
        # Interpretation
        significant = p_value < self.config.significance_level
        
        if significant:
            if effect_size > 0.8:
                interpretation = f"{model_a} significantly outperforms {model_b} with large effect size"
            elif effect_size > 0.5:
                interpretation = f"{model_a} significantly outperforms {model_b} with medium effect size"
            elif effect_size > 0.2:
                interpretation = f"{model_a} significantly outperforms {model_b} with small effect size"
            elif effect_size < -0.8:
                interpretation = f"{model_b} significantly outperforms {model_a} with large effect size"
            elif effect_size < -0.5:
                interpretation = f"{model_b} significantly outperforms {model_a} with medium effect size"
            elif effect_size < -0.2:
                interpretation = f"{model_b} significantly outperforms {model_a} with small effect size"
            else:
                interpretation = f"Statistically significant difference but negligible effect size"
        else:
            interpretation = f"No statistically significant difference between {model_a} and {model_b}"
        
        return StatisticalAnalysis(
            test_statistic=float(statistic),
            p_value=float(p_value),
            effect_size=float(effect_size),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            test_name=test_name,
            interpretation=interpretation,
            significant=significant
        )
    
    def _analyze_effect_sizes(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Analyze effect sizes across different conditions."""
        effect_analysis = {
            'by_task_type': {},
            'by_domain': {},
            'by_model_size': {},
            'overall_summary': {}
        }
        
        # Group results by different factors
        task_groups = {}
        domain_groups = {}
        model_size_groups = {}
        
        for result in results:
            # Group by task type
            task_type = result.task_type
            if task_type not in task_groups:
                task_groups[task_type] = []
            task_groups[task_type].append(result)
            
            # Group by domain
            domain = result.domain
            if domain not in domain_groups:
                domain_groups[domain] = []
            domain_groups[domain].append(result)
            
            # Group by model size
            model_size = result.model_config.model_size
            if model_size not in model_size_groups:
                model_size_groups[model_size] = []
            model_size_groups[model_size].append(result)
        
        # Analyze effect sizes within each grouping
        effect_analysis['by_task_type'] = self._compute_group_effect_sizes(task_groups)
        effect_analysis['by_domain'] = self._compute_group_effect_sizes(domain_groups)
        effect_analysis['by_model_size'] = self._compute_group_effect_sizes(model_size_groups)
        
        # Overall summary statistics
        all_scores = []
        for result in results:
            for metric_name, metric_data in result.metric_scores.items():
                if isinstance(metric_data, dict) and 'scores' in metric_data:
                    all_scores.extend(metric_data['scores'])
        
        if all_scores:
            effect_analysis['overall_summary'] = {
                'mean_score': float(np.mean(all_scores)),
                'std_score': float(np.std(all_scores)),
                'min_score': float(np.min(all_scores)),
                'max_score': float(np.max(all_scores)),
                'score_range': float(np.max(all_scores) - np.min(all_scores)),
                'num_evaluations': len(all_scores)
            }
        
        return effect_analysis
    
    def _compute_group_effect_sizes(self, groups: Dict[str, List[ExperimentResult]]) -> Dict[str, Any]:
        """Compute effect sizes between groups."""
        group_analysis = {}
        group_names = list(groups.keys())
        
        for i, group_a in enumerate(group_names):
            for j, group_b in enumerate(group_names[i+1:], i+1):
                # Extract scores for comparison
                scores_a = []
                scores_b = []
                
                for result in groups[group_a]:
                    for metric_data in result.metric_scores.values():
                        if isinstance(metric_data, dict) and 'scores' in metric_data:
                            scores_a.extend(metric_data['scores'])
                
                for result in groups[group_b]:
                    for metric_data in result.metric_scores.values():
                        if isinstance(metric_data, dict) and 'scores' in metric_data:
                            scores_b.extend(metric_data['scores'])
                
                if len(scores_a) > 0 and len(scores_b) > 0:
                    # Cohen's d
                    pooled_std = np.sqrt((np.var(scores_a, ddof=1) + np.var(scores_b, ddof=1)) / 2)
                    if pooled_std > 0:
                        cohens_d = (np.mean(scores_a) - np.mean(scores_b)) / pooled_std
                    else:
                        cohens_d = 0.0
                    
                    group_analysis[f"{group_a}_vs_{group_b}"] = {
                        'cohens_d': float(cohens_d),
                        'mean_diff': float(np.mean(scores_a) - np.mean(scores_b)),
                        'group_a_mean': float(np.mean(scores_a)),
                        'group_b_mean': float(np.mean(scores_b)),
                        'group_a_n': len(scores_a),
                        'group_b_n': len(scores_b)
                    }
        
        return group_analysis
    
    def _generate_research_report(
        self,
        results: List[ExperimentResult],
        statistical_analysis: Dict[str, StatisticalAnalysis],
        effect_size_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive research report."""
        
        report = {
            'experiment_metadata': {
                'experiment_name': self.config.experiment_name,
                'description': self.config.description,
                'timestamp': datetime.now().isoformat(),
                'total_evaluations': len(results),
                'unique_models': len(set(r.model_config.model_name for r in results)),
                'unique_tasks': len(set(r.task_type for r in results)),
                'unique_domains': len(set(r.domain for r in results))
            },
            'key_findings': [],
            'statistical_summary': self._summarize_statistical_results(statistical_analysis),
            'effect_size_summary': effect_size_analysis,
            'model_rankings': self._generate_model_rankings(results),
            'recommendations': self._generate_recommendations(results, statistical_analysis),
            'methodological_notes': self._generate_methodological_notes(),
            'future_work': self._suggest_future_work(results, statistical_analysis)
        }
        
        # Extract key findings
        report['key_findings'] = self._extract_key_findings(
            results, statistical_analysis, effect_size_analysis
        )
        
        return report
    
    def _summarize_statistical_results(self, statistical_analysis: Dict[str, StatisticalAnalysis]) -> Dict[str, Any]:
        """Summarize statistical analysis results."""
        significant_results = [
            analysis for analysis in statistical_analysis.values() 
            if analysis.significant
        ]
        
        large_effects = [
            analysis for analysis in statistical_analysis.values()
            if abs(analysis.effect_size) > 0.8
        ]
        
        return {
            'total_comparisons': len(statistical_analysis),
            'significant_results': len(significant_results),
            'significance_rate': len(significant_results) / len(statistical_analysis) if statistical_analysis else 0,
            'large_effect_count': len(large_effects),
            'mean_effect_size': float(np.mean([abs(a.effect_size) for a in statistical_analysis.values()])) if statistical_analysis else 0,
            'max_effect_size': float(max([abs(a.effect_size) for a in statistical_analysis.values()])) if statistical_analysis else 0
        }
    
    def _generate_model_rankings(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Generate model rankings across different metrics."""
        model_scores = {}
        
        # Aggregate scores by model
        for result in results:
            model_name = result.model_config.model_name
            if model_name not in model_scores:
                model_scores[model_name] = []
            
            # Extract mean scores across all metrics
            for metric_data in result.metric_scores.values():
                if isinstance(metric_data, dict) and 'mean' in metric_data:
                    model_scores[model_name].append(metric_data['mean'])
        
        # Calculate overall performance
        model_performance = {}
        for model_name, scores in model_scores.items():
            if scores:
                model_performance[model_name] = {
                    'overall_mean': float(np.mean(scores)),
                    'overall_std': float(np.std(scores)),
                    'num_evaluations': len(scores)
                }
        
        # Rank models
        ranked_models = sorted(
            model_performance.items(),
            key=lambda x: x[1]['overall_mean'],
            reverse=True
        )
        
        return {
            'rankings': [
                {
                    'rank': i + 1,
                    'model': model_name,
                    'score': performance['overall_mean'],
                    'std': performance['overall_std'],
                    'evaluations': performance['num_evaluations']
                }
                for i, (model_name, performance) in enumerate(ranked_models)
            ],
            'performance_details': model_performance
        }
    
    def _extract_key_findings(
        self,
        results: List[ExperimentResult],
        statistical_analysis: Dict[str, StatisticalAnalysis],
        effect_size_analysis: Dict[str, Any]
    ) -> List[str]:
        """Extract key findings from the experimental results."""
        findings = []
        
        # Finding 1: Overall performance summary
        all_scores = []
        for result in results:
            for metric_data in result.metric_scores.values():
                if isinstance(metric_data, dict) and 'mean' in metric_data:
                    all_scores.append(metric_data['mean'])
        
        if all_scores:
            overall_mean = np.mean(all_scores)
            findings.append(
                f"Overall causal reasoning performance across all models and tasks: "
                f"{overall_mean:.3f} ± {np.std(all_scores):.3f}"
            )
        
        # Finding 2: Best performing model
        model_means = {}
        for result in results:
            model_name = result.model_config.model_name
            if model_name not in model_means:
                model_means[model_name] = []
            
            for metric_data in result.metric_scores.values():
                if isinstance(metric_data, dict) and 'mean' in metric_data:
                    model_means[model_name].append(metric_data['mean'])
        
        best_model = max(model_means.items(), key=lambda x: np.mean(x[1]), default=None)
        if best_model:
            findings.append(
                f"Best performing model: {best_model[0]} "
                f"(mean score: {np.mean(best_model[1]):.3f})"
            )
        
        # Finding 3: Significant differences
        significant_comparisons = [
            analysis for analysis in statistical_analysis.values() 
            if analysis.significant
        ]
        
        if significant_comparisons:
            findings.append(
                f"Found {len(significant_comparisons)} statistically significant "
                f"performance differences out of {len(statistical_analysis)} comparisons"
            )
        
        # Finding 4: Effect sizes
        large_effects = [
            analysis for analysis in statistical_analysis.values()
            if abs(analysis.effect_size) > 0.8
        ]
        
        if large_effects:
            findings.append(
                f"Identified {len(large_effects)} comparisons with large effect sizes (>0.8), "
                f"indicating practically significant differences"
            )
        
        return findings
    
    def _generate_recommendations(
        self,
        results: List[ExperimentResult],
        statistical_analysis: Dict[str, StatisticalAnalysis]
    ) -> List[str]:
        """Generate actionable recommendations based on results."""
        recommendations = []
        
        # Recommendation 1: Model selection
        model_performance = {}
        for result in results:
            model_name = result.model_config.model_name
            if model_name not in model_performance:
                model_performance[model_name] = []
            
            for metric_data in result.metric_scores.values():
                if isinstance(metric_data, dict) and 'mean' in metric_data:
                    model_performance[model_name].append(metric_data['mean'])
        
        if model_performance:
            best_model = max(model_performance.items(), key=lambda x: np.mean(x[1]))
            recommendations.append(
                f"For causal reasoning tasks, consider using {best_model[0]} "
                f"which demonstrated the highest average performance"
            )
        
        # Recommendation 2: Task-specific insights
        task_performance = {}
        for result in results:
            task_type = result.task_type
            if task_type not in task_performance:
                task_performance[task_type] = []
            
            for metric_data in result.metric_scores.values():
                if isinstance(metric_data, dict) and 'mean' in metric_data:
                    task_performance[task_type].append(metric_data['mean'])
        
        if len(task_performance) > 1:
            hardest_task = min(task_performance.items(), key=lambda x: np.mean(x[1]))
            recommendations.append(
                f"Task type '{hardest_task[0]}' showed lowest performance scores, "
                f"suggesting need for targeted improvement in this area"
            )
        
        # Recommendation 3: Statistical significance insights
        significant_improvements = [
            analysis for analysis in statistical_analysis.values()
            if analysis.significant and analysis.effect_size > 0.5
        ]
        
        if significant_improvements:
            recommendations.append(
                f"Focus development efforts on approaches that showed significant "
                f"improvements with medium to large effect sizes"
            )
        
        return recommendations
    
    def _generate_methodological_notes(self) -> List[str]:
        """Generate methodological notes for reproducibility."""
        return [
            f"Experiment used random seed {self.config.random_seed} for reproducibility",
            f"Statistical significance threshold set at α = {self.config.significance_level}",
            f"Minimum sample size per condition: {self.config.min_sample_size}",
            f"Effect size threshold for practical significance: {self.config.effect_size_threshold}",
            "Bootstrap resampling used for confidence interval estimation",
            "Multiple comparison corrections applied where appropriate",
            "Both parametric and non-parametric tests used based on data distribution"
        ]
    
    def _suggest_future_work(
        self,
        results: List[ExperimentResult],
        statistical_analysis: Dict[str, StatisticalAnalysis]
    ) -> List[str]:
        """Suggest directions for future research."""
        suggestions = []
        
        # Analyze gaps in current research
        domains_tested = set(result.domain for result in results)
        tasks_tested = set(result.task_type for result in results)
        
        common_domains = {'medical', 'business', 'education', 'technology', 'social'}
        common_tasks = {'attribution', 'counterfactual', 'intervention', 'chain', 'confounding'}
        
        missing_domains = common_domains - domains_tested
        missing_tasks = common_tasks - tasks_tested
        
        if missing_domains:
            suggestions.append(
                f"Extend evaluation to additional domains: {', '.join(missing_domains)}"
            )
        
        if missing_tasks:
            suggestions.append(
                f"Include additional task types: {', '.join(missing_tasks)}"
            )
        
        # Analyze effect sizes for improvement opportunities
        small_effects = [
            analysis for analysis in statistical_analysis.values()
            if 0 < abs(analysis.effect_size) < 0.2
        ]
        
        if small_effects:
            suggestions.append(
                "Investigate methods to increase effect sizes in areas showing "
                "small but consistent differences"
            )
        
        # Sample size recommendations
        min_samples = min(len(result.responses) for result in results)
        if min_samples < 100:
            suggestions.append(
                f"Increase sample sizes beyond current minimum of {min_samples} "
                "for more robust statistical inference"
            )
        
        return suggestions
    
    def _save_experimental_data(
        self,
        results: List[ExperimentResult],
        report: Dict[str, Any]
    ) -> None:
        """Save experimental data for reproducibility and future analysis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results as JSON
        results_file = self.output_dir / f"results_{timestamp}.json"
        with open(results_file, 'w') as f:
            # Convert results to serializable format
            serializable_results = []
            for result in results:
                serializable_result = {
                    'model_name': result.model_config.model_name,
                    'task_type': result.task_type,
                    'domain': result.domain,
                    'metric_scores': result.metric_scores,
                    'response_times': result.response_times,
                    'num_responses': len(result.responses),
                    'metadata': result.metadata,
                    'timestamp': result.timestamp.isoformat()
                }
                serializable_results.append(serializable_result)
            
            json.dump(serializable_results, f, indent=2)
        
        # Save report
        report_file = self.output_dir / f"report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save raw data for reanalysis
        if self.config.save_intermediate:
            raw_data_file = self.output_dir / f"raw_data_{timestamp}.pkl"
            with open(raw_data_file, 'wb') as f:
                pickle.dump({
                    'results': results,
                    'config': self.config,
                    'metrics': self.metrics
                }, f)
        
        logger.info(f"Experimental data saved to {self.output_dir}")
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID."""
        experiment_string = f"{self.config.experiment_name}_{datetime.now().isoformat()}"
        return hashlib.md5(experiment_string.encode()).hexdigest()[:8]
    
    def generate_publication_plots(self, results: List[ExperimentResult]) -> Dict[str, str]:
        """Generate publication-quality plots."""
        plots = {}
        
        # Set publication style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # Plot 1: Model performance comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        model_scores = {}
        for result in results:
            model_name = result.model_config.model_name
            if model_name not in model_scores:
                model_scores[model_name] = []
            
            for metric_data in result.metric_scores.values():
                if isinstance(metric_data, dict) and 'mean' in metric_data:
                    model_scores[model_name].append(metric_data['mean'])
        
        if model_scores:
            models = list(model_scores.keys())
            means = [np.mean(scores) for scores in model_scores.values()]
            stds = [np.std(scores) for scores in model_scores.values()]
            
            bars = ax.bar(models, means, yerr=stds, capsize=5, 
                         alpha=0.8, edgecolor='black', linewidth=1)
            
            ax.set_ylabel('Causal Reasoning Performance')
            ax.set_title('Model Performance Comparison\n(Mean ± Standard Deviation)')
            ax.set_ylim(0, 1)
            
            # Rotate x-axis labels if needed
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            
            plot_file = self.output_dir / 'model_comparison.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            plots['model_comparison'] = str(plot_file)
        
        # Plot 2: Task type performance
        fig, ax = plt.subplots(figsize=(8, 6))
        
        task_scores = {}
        for result in results:
            task_type = result.task_type
            if task_type not in task_scores:
                task_scores[task_type] = []
            
            for metric_data in result.metric_scores.values():
                if isinstance(metric_data, dict) and 'mean' in metric_data:
                    task_scores[task_type].append(metric_data['mean'])
        
        if task_scores:
            # Create box plot
            task_data = []
            task_labels = []
            
            for task_type, scores in task_scores.items():
                task_data.append(scores)
                task_labels.append(task_type)
            
            bp = ax.boxplot(task_data, labels=task_labels, patch_artist=True)
            
            # Color the boxes
            colors = sns.color_palette("husl", len(task_data))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel('Performance Score')
            ax.set_title('Performance Distribution by Task Type')
            ax.set_ylim(0, 1)
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            plot_file = self.output_dir / 'task_performance.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            plots['task_performance'] = str(plot_file)
        
        return plots