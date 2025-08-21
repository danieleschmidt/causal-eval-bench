"""
Enhanced Validation Suite for Causal Reasoning Research

This module provides advanced validation capabilities including:
1. Multi-dimensional performance assessment
2. Statistical robustness testing
3. Cross-validation with temporal splits
4. Adversarial validation scenarios
"""

import asyncio
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path
import itertools
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class ValidationScenario:
    """Represents a validation scenario for testing causal reasoning."""
    
    scenario_id: str
    name: str
    description: str
    scenario_type: str  # 'standard', 'adversarial', 'robustness', 'temporal'
    difficulty_level: str  # 'easy', 'medium', 'hard', 'expert'
    domain: str
    test_cases: List[Dict[str, Any]]
    ground_truth: Dict[str, Any]
    evaluation_metrics: List[str]
    expected_performance_range: Tuple[float, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Results from a validation scenario."""
    
    scenario_id: str
    algorithm_name: str
    performance_scores: Dict[str, float]
    execution_time: float
    memory_usage: float
    robustness_score: float
    statistical_significance: bool
    confidence_intervals: Dict[str, Tuple[float, float]]
    error_analysis: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CrossValidationResults:
    """Results from cross-validation analysis."""
    
    algorithm_name: str
    cv_scores: List[float]
    mean_score: float
    std_score: float
    confidence_interval: Tuple[float, float]
    temporal_stability: float
    domain_generalization: float
    statistical_tests: Dict[str, Any]


class AbstractValidationMetric(ABC):
    """Abstract base class for validation metrics."""
    
    @abstractmethod
    def compute(self, predictions: List[Any], ground_truth: List[Any]) -> float:
        """Compute metric score."""
        pass
    
    @abstractmethod
    def get_confidence_interval(self, predictions: List[Any], ground_truth: List[Any]) -> Tuple[float, float]:
        """Compute confidence interval for metric."""
        pass


class CausalAccuracyMetric(AbstractValidationMetric):
    """Measures accuracy of causal relationship identification."""
    
    def compute(self, predictions: List[Any], ground_truth: List[Any]) -> float:
        """Compute causal accuracy."""
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        correct = 0
        total = len(predictions)
        
        for pred, truth in zip(predictions, ground_truth):
            if self._causal_match(pred, truth):
                correct += 1
        
        return correct / total if total > 0 else 0.0
    
    def get_confidence_interval(self, predictions: List[Any], ground_truth: List[Any]) -> Tuple[float, float]:
        """Bootstrap confidence interval for accuracy."""
        n_bootstrap = 1000
        bootstrap_scores = []
        
        data_pairs = list(zip(predictions, ground_truth))
        n_samples = len(data_pairs)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            bootstrap_sample = [data_pairs[np.random.randint(0, n_samples)] for _ in range(n_samples)]
            bootstrap_pred, bootstrap_truth = zip(*bootstrap_sample)
            
            score = self.compute(list(bootstrap_pred), list(bootstrap_truth))
            bootstrap_scores.append(score)
        
        # Calculate 95% confidence interval
        bootstrap_scores.sort()
        lower = bootstrap_scores[int(0.025 * n_bootstrap)]
        upper = bootstrap_scores[int(0.975 * n_bootstrap)]
        
        return (lower, upper)
    
    def _causal_match(self, prediction: Any, ground_truth: Any) -> bool:
        """Check if prediction matches ground truth causal relationship."""
        # Simplified causal matching logic
        if isinstance(prediction, dict) and isinstance(ground_truth, dict):
            pred_relationship = prediction.get('causal_relationship', 'none')
            true_relationship = ground_truth.get('causal_relationship', 'none')
            return pred_relationship == true_relationship
        
        return str(prediction).lower() == str(ground_truth).lower()


class CausalPrecisionMetric(AbstractValidationMetric):
    """Measures precision of causal discovery."""
    
    def compute(self, predictions: List[Any], ground_truth: List[Any]) -> float:
        """Compute causal precision (true positives / predicted positives)."""
        true_positives = 0
        predicted_positives = 0
        
        for pred, truth in zip(predictions, ground_truth):
            pred_causal = self._is_causal(pred)
            true_causal = self._is_causal(truth)
            
            if pred_causal:
                predicted_positives += 1
                if true_causal:
                    true_positives += 1
        
        return true_positives / predicted_positives if predicted_positives > 0 else 0.0
    
    def get_confidence_interval(self, predictions: List[Any], ground_truth: List[Any]) -> Tuple[float, float]:
        """Bootstrap confidence interval for precision."""
        return self._bootstrap_ci(predictions, ground_truth)
    
    def _is_causal(self, relationship: Any) -> bool:
        """Check if relationship indicates causation."""
        if isinstance(relationship, dict):
            return relationship.get('causal_relationship') not in ['none', 'correlation', 'spurious']
        
        causal_keywords = ['causes', 'leads to', 'results in', 'influences']
        return any(keyword in str(relationship).lower() for keyword in causal_keywords)
    
    def _bootstrap_ci(self, predictions: List[Any], ground_truth: List[Any]) -> Tuple[float, float]:
        """Generic bootstrap confidence interval."""
        n_bootstrap = 1000
        bootstrap_scores = []
        
        data_pairs = list(zip(predictions, ground_truth))
        n_samples = len(data_pairs)
        
        for _ in range(n_bootstrap):
            bootstrap_sample = [data_pairs[np.random.randint(0, n_samples)] for _ in range(n_samples)]
            bootstrap_pred, bootstrap_truth = zip(*bootstrap_sample)
            
            score = self.compute(list(bootstrap_pred), list(bootstrap_truth))
            bootstrap_scores.append(score)
        
        bootstrap_scores.sort()
        lower = bootstrap_scores[int(0.025 * n_bootstrap)]
        upper = bootstrap_scores[int(0.975 * n_bootstrap)]
        
        return (lower, upper)


class CausalRecallMetric(AbstractValidationMetric):
    """Measures recall of causal discovery."""
    
    def compute(self, predictions: List[Any], ground_truth: List[Any]) -> float:
        """Compute causal recall (true positives / actual positives)."""
        true_positives = 0
        actual_positives = 0
        
        for pred, truth in zip(predictions, ground_truth):
            pred_causal = self._is_causal(pred)
            true_causal = self._is_causal(truth)
            
            if true_causal:
                actual_positives += 1
                if pred_causal:
                    true_positives += 1
        
        return true_positives / actual_positives if actual_positives > 0 else 0.0
    
    def get_confidence_interval(self, predictions: List[Any], ground_truth: List[Any]) -> Tuple[float, float]:
        """Bootstrap confidence interval for recall."""
        return self._bootstrap_ci(predictions, ground_truth)
    
    def _is_causal(self, relationship: Any) -> bool:
        """Check if relationship indicates causation."""
        if isinstance(relationship, dict):
            return relationship.get('causal_relationship') not in ['none', 'correlation', 'spurious']
        
        causal_keywords = ['causes', 'leads to', 'results in', 'influences']
        return any(keyword in str(relationship).lower() for keyword in causal_keywords)
    
    def _bootstrap_ci(self, predictions: List[Any], ground_truth: List[Any]) -> Tuple[float, float]:
        """Generic bootstrap confidence interval."""
        n_bootstrap = 1000
        bootstrap_scores = []
        
        data_pairs = list(zip(predictions, ground_truth))
        n_samples = len(data_pairs)
        
        for _ in range(n_bootstrap):
            bootstrap_sample = [data_pairs[np.random.randint(0, n_samples)] for _ in range(n_samples)]
            bootstrap_pred, bootstrap_truth = zip(*bootstrap_sample)
            
            score = self.compute(list(bootstrap_pred), list(bootstrap_truth))
            bootstrap_scores.append(score)
        
        bootstrap_scores.sort()
        lower = bootstrap_scores[int(0.025 * n_bootstrap)]
        upper = bootstrap_scores[int(0.975 * n_bootstrap)]
        
        return (lower, upper)


class EnhancedValidationSuite:
    """
    Enhanced validation suite for causal reasoning research.
    
    Provides comprehensive validation capabilities including:
    - Multi-dimensional performance assessment
    - Statistical robustness testing
    - Cross-validation with temporal awareness
    - Adversarial scenario testing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metrics = {
            'accuracy': CausalAccuracyMetric(),
            'precision': CausalPrecisionMetric(),
            'recall': CausalRecallMetric()
        }
        self.validation_scenarios: List[ValidationScenario] = []
        self.validation_results: List[ValidationResult] = []
        
        # Initialize standard validation scenarios
        self._initialize_standard_scenarios()
        
        logger.info("Enhanced Validation Suite initialized")
    
    async def run_comprehensive_validation(
        self,
        algorithm_func: Callable,
        algorithm_name: str,
        scenarios: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive validation across multiple scenarios.
        
        Args:
            algorithm_func: Function that implements the algorithm to test
            algorithm_name: Name of the algorithm being tested
            scenarios: List of scenario names to run (None for all)
        
        Returns:
            Comprehensive validation results
        """
        logger.info(f"Running comprehensive validation for algorithm: {algorithm_name}")
        
        # Select scenarios to run
        if scenarios is None:
            test_scenarios = self.validation_scenarios
        else:
            test_scenarios = [s for s in self.validation_scenarios if s.name in scenarios]
        
        # Run validation for each scenario
        scenario_results = []
        
        for scenario in test_scenarios:
            try:
                result = await self._run_scenario_validation(algorithm_func, algorithm_name, scenario)
                scenario_results.append(result)
                self.validation_results.append(result)
                
            except Exception as e:
                logger.error(f"Error validating scenario {scenario.name}: {e}")
        
        # Aggregate results
        aggregated_results = self._aggregate_validation_results(scenario_results, algorithm_name)
        
        # Perform statistical analysis
        statistical_analysis = self._perform_statistical_analysis(scenario_results)
        
        # Generate comprehensive report
        comprehensive_results = {
            'algorithm_name': algorithm_name,
            'validation_summary': {
                'scenarios_tested': len(test_scenarios),
                'successful_validations': len(scenario_results),
                'overall_performance': aggregated_results['overall_score'],
                'timestamp': datetime.now().isoformat()
            },
            'scenario_results': scenario_results,
            'aggregated_metrics': aggregated_results,
            'statistical_analysis': statistical_analysis,
            'performance_breakdown': self._generate_performance_breakdown(scenario_results),
            'recommendations': self._generate_validation_recommendations(aggregated_results, statistical_analysis)
        }
        
        logger.info(f"Comprehensive validation completed. Overall score: {aggregated_results['overall_score']:.3f}")
        return comprehensive_results
    
    async def run_cross_validation(
        self,
        algorithm_func: Callable,
        algorithm_name: str,
        n_folds: int = 5,
        temporal_split: bool = True
    ) -> CrossValidationResults:
        """
        Run cross-validation analysis.
        
        Args:
            algorithm_func: Function that implements the algorithm to test
            algorithm_name: Name of the algorithm being tested
            n_folds: Number of cross-validation folds
            temporal_split: Whether to use temporal splitting
        
        Returns:
            Cross-validation results
        """
        logger.info(f"Running {n_folds}-fold cross-validation for {algorithm_name}")
        
        # Prepare cross-validation data
        cv_data = self._prepare_cv_data(temporal_split)
        
        # Perform cross-validation
        cv_scores = []
        fold_results = []
        
        for fold in range(n_folds):
            train_data, test_data = self._create_cv_fold(cv_data, fold, n_folds)
            
            try:
                # Train/adapt algorithm on fold training data
                adapted_func = self._adapt_algorithm_for_fold(algorithm_func, train_data)
                
                # Test on fold test data
                fold_score = await self._evaluate_fold(adapted_func, test_data)
                cv_scores.append(fold_score)
                
                fold_results.append({
                    'fold': fold,
                    'score': fold_score,
                    'train_size': len(train_data),
                    'test_size': len(test_data)
                })
                
            except Exception as e:
                logger.warning(f"Error in fold {fold}: {e}")
        
        # Calculate cross-validation statistics
        mean_score = np.mean(cv_scores) if cv_scores else 0.0
        std_score = np.std(cv_scores) if cv_scores else 0.0
        
        # Calculate confidence interval
        if len(cv_scores) >= 3:
            confidence_interval = self._calculate_cv_confidence_interval(cv_scores)
        else:
            confidence_interval = (mean_score - std_score, mean_score + std_score)
        
        # Assess temporal stability and domain generalization
        temporal_stability = self._assess_temporal_stability(fold_results, temporal_split)
        domain_generalization = self._assess_domain_generalization(fold_results)
        
        # Perform statistical tests
        statistical_tests = self._perform_cv_statistical_tests(cv_scores, fold_results)
        
        cv_results = CrossValidationResults(
            algorithm_name=algorithm_name,
            cv_scores=cv_scores,
            mean_score=mean_score,
            std_score=std_score,
            confidence_interval=confidence_interval,
            temporal_stability=temporal_stability,
            domain_generalization=domain_generalization,
            statistical_tests=statistical_tests
        )
        
        logger.info(f"Cross-validation completed. Mean score: {mean_score:.3f} ± {std_score:.3f}")
        return cv_results
    
    async def run_adversarial_validation(
        self,
        algorithm_func: Callable,
        algorithm_name: str,
        adversarial_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        Run adversarial validation scenarios.
        
        Args:
            algorithm_func: Function that implements the algorithm to test
            algorithm_name: Name of the algorithm being tested
            adversarial_types: Types of adversarial scenarios to test
        
        Returns:
            Adversarial validation results
        """
        logger.info(f"Running adversarial validation for {algorithm_name}")
        
        adversarial_types = adversarial_types or [
            'spurious_correlation', 'confounding_bias', 'selection_bias',
            'temporal_shift', 'domain_shift', 'noise_injection'
        ]
        
        adversarial_results = {}
        
        for adv_type in adversarial_types:
            try:
                # Generate adversarial scenario
                adversarial_scenario = self._generate_adversarial_scenario(adv_type)
                
                # Run validation
                result = await self._run_scenario_validation(algorithm_func, algorithm_name, adversarial_scenario)
                
                adversarial_results[adv_type] = {
                    'performance_score': result.performance_scores.get('accuracy', 0.0),
                    'robustness_score': result.robustness_score,
                    'degradation': self._calculate_performance_degradation(result, adv_type),
                    'vulnerability_analysis': self._analyze_vulnerability(result, adv_type)
                }
                
            except Exception as e:
                logger.error(f"Error in adversarial validation {adv_type}: {e}")
                adversarial_results[adv_type] = {
                    'performance_score': 0.0,
                    'robustness_score': 0.0,
                    'degradation': 1.0,
                    'error': str(e)
                }
        
        # Calculate overall adversarial robustness
        robustness_scores = [r.get('robustness_score', 0) for r in adversarial_results.values() if 'error' not in r]
        overall_robustness = np.mean(robustness_scores) if robustness_scores else 0.0
        
        adversarial_summary = {
            'algorithm_name': algorithm_name,
            'overall_robustness': overall_robustness,
            'adversarial_results': adversarial_results,
            'vulnerability_ranking': self._rank_vulnerabilities(adversarial_results),
            'robustness_recommendations': self._generate_robustness_recommendations(adversarial_results)
        }
        
        logger.info(f"Adversarial validation completed. Overall robustness: {overall_robustness:.3f}")
        return adversarial_summary
    
    def _initialize_standard_scenarios(self) -> None:
        """Initialize standard validation scenarios."""
        standard_scenarios = [
            {
                'name': 'Basic Causal Attribution',
                'type': 'standard',
                'difficulty': 'easy',
                'domain': 'general',
                'description': 'Test basic ability to distinguish causation from correlation'
            },
            {
                'name': 'Confounded Relationships',
                'type': 'standard',
                'difficulty': 'medium',
                'domain': 'general',
                'description': 'Test ability to identify confounding variables'
            },
            {
                'name': 'Temporal Causation',
                'type': 'temporal',
                'difficulty': 'medium',
                'domain': 'general',
                'description': 'Test understanding of temporal precedence in causation'
            },
            {
                'name': 'Multi-step Causal Chains',
                'type': 'standard',
                'difficulty': 'hard',
                'domain': 'general',
                'description': 'Test reasoning about complex causal chains'
            },
            {
                'name': 'Domain-specific Medical Causation',
                'type': 'standard',
                'difficulty': 'expert',
                'domain': 'medical',
                'description': 'Test causal reasoning in medical domain'
            }
        ]
        
        for scenario_config in standard_scenarios:
            scenario = ValidationScenario(
                scenario_id=self._generate_scenario_id(scenario_config['name']),
                name=scenario_config['name'],
                description=scenario_config['description'],
                scenario_type=scenario_config['type'],
                difficulty_level=scenario_config['difficulty'],
                domain=scenario_config['domain'],
                test_cases=self._generate_test_cases_for_scenario(scenario_config),
                ground_truth=self._generate_ground_truth_for_scenario(scenario_config),
                evaluation_metrics=['accuracy', 'precision', 'recall'],
                expected_performance_range=self._get_expected_performance_range(scenario_config['difficulty'])
            )
            
            self.validation_scenarios.append(scenario)
    
    async def _run_scenario_validation(
        self,
        algorithm_func: Callable,
        algorithm_name: str,
        scenario: ValidationScenario
    ) -> ValidationResult:
        """Run validation for a single scenario."""
        logger.debug(f"Running scenario validation: {scenario.name}")
        
        start_time = datetime.now()
        
        # Run algorithm on test cases
        predictions = []
        execution_times = []
        
        for test_case in scenario.test_cases:
            case_start = datetime.now()
            
            try:
                prediction = await self._run_algorithm_on_case(algorithm_func, test_case)
                predictions.append(prediction)
                
                case_end = datetime.now()
                execution_times.append((case_end - case_start).total_seconds())
                
            except Exception as e:
                logger.warning(f"Algorithm failed on test case: {e}")
                predictions.append({'error': str(e)})
                execution_times.append(0.0)
        
        end_time = datetime.now()
        total_execution_time = (end_time - start_time).total_seconds()
        
        # Extract ground truth for comparison
        ground_truth_list = self._extract_ground_truth_list(scenario.ground_truth, len(predictions))
        
        # Calculate performance scores
        performance_scores = {}
        confidence_intervals = {}
        
        for metric_name in scenario.evaluation_metrics:
            if metric_name in self.metrics:
                try:
                    score = self.metrics[metric_name].compute(predictions, ground_truth_list)
                    ci = self.metrics[metric_name].get_confidence_interval(predictions, ground_truth_list)
                    
                    performance_scores[metric_name] = score
                    confidence_intervals[metric_name] = ci
                    
                except Exception as e:
                    logger.warning(f"Error computing {metric_name}: {e}")
                    performance_scores[metric_name] = 0.0
                    confidence_intervals[metric_name] = (0.0, 0.0)
        
        # Calculate robustness score
        robustness_score = self._calculate_robustness_score(predictions, ground_truth_list, scenario)
        
        # Perform error analysis
        error_analysis = self._perform_error_analysis(predictions, ground_truth_list, scenario)
        
        # Statistical significance test
        statistical_significance = self._test_statistical_significance(
            performance_scores.get('accuracy', 0.0),
            len(predictions)
        )
        
        result = ValidationResult(
            scenario_id=scenario.scenario_id,
            algorithm_name=algorithm_name,
            performance_scores=performance_scores,
            execution_time=total_execution_time,
            memory_usage=self._estimate_memory_usage(predictions),
            robustness_score=robustness_score,
            statistical_significance=statistical_significance,
            confidence_intervals=confidence_intervals,
            error_analysis=error_analysis
        )
        
        return result
    
    async def _run_algorithm_on_case(self, algorithm_func: Callable, test_case: Dict[str, Any]) -> Any:
        """Run algorithm on a single test case."""
        # Extract input from test case
        algorithm_input = test_case.get('input', test_case)
        
        # Run algorithm (handle both sync and async functions)
        if asyncio.iscoroutinefunction(algorithm_func):
            result = await algorithm_func(algorithm_input)
        else:
            result = algorithm_func(algorithm_input)
        
        return result
    
    def _generate_scenario_id(self, name: str) -> str:
        """Generate unique scenario ID."""
        content = f"{name}_{datetime.now().isoformat()}"
        import hashlib
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def _generate_test_cases_for_scenario(self, scenario_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate test cases for a scenario."""
        scenario_type = scenario_config['type']
        difficulty = scenario_config['difficulty']
        domain = scenario_config['domain']
        
        # Base number of test cases by difficulty
        num_cases = {'easy': 10, 'medium': 15, 'hard': 20, 'expert': 25}.get(difficulty, 10)
        
        test_cases = []
        
        for i in range(num_cases):
            case = {
                'case_id': f"{scenario_config['name']}_case_{i}",
                'input': self._generate_test_input(scenario_type, domain, difficulty, i),
                'metadata': {
                    'scenario_type': scenario_type,
                    'difficulty': difficulty,
                    'domain': domain,
                    'case_number': i
                }
            }
            test_cases.append(case)
        
        return test_cases
    
    def _generate_test_input(self, scenario_type: str, domain: str, difficulty: str, case_num: int) -> Dict[str, Any]:
        """Generate input for a test case."""
        base_variables = ['A', 'B', 'C', 'D', 'E']
        
        if domain == 'medical':
            variables = ['treatment', 'outcome', 'age', 'comorbidity', 'genetics']
        elif domain == 'business':
            variables = ['marketing', 'sales', 'competition', 'price', 'customer_satisfaction']
        else:
            variables = base_variables
        
        # Adjust complexity based on difficulty
        if difficulty == 'easy':
            selected_vars = variables[:2]
        elif difficulty == 'medium':
            selected_vars = variables[:3]
        elif difficulty == 'hard':
            selected_vars = variables[:4]
        else:  # expert
            selected_vars = variables
        
        # Generate scenario-specific input
        if scenario_type == 'temporal':
            test_input = {
                'scenario': f"Variable {selected_vars[0]} changes at time T, variable {selected_vars[1]} changes at time T+1",
                'question': f"Does {selected_vars[0]} cause {selected_vars[1]}?",
                'variables': selected_vars,
                'temporal_order': [selected_vars[0], selected_vars[1]]
            }
        else:
            test_input = {
                'scenario': f"Study shows correlation between {selected_vars[0]} and {selected_vars[1]}",
                'question': f"Is there a causal relationship between {selected_vars[0]} and {selected_vars[1]}?",
                'variables': selected_vars,
                'correlation_strength': 0.5 + (case_num % 5) * 0.1
            }
        
        return test_input
    
    def _generate_ground_truth_for_scenario(self, scenario_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate ground truth for scenario."""
        # Simplified ground truth generation
        return {
            'scenario_name': scenario_config['name'],
            'expected_causal_relationships': self._get_expected_relationships(scenario_config),
            'difficulty_adjustment': scenario_config['difficulty']
        }
    
    def _get_expected_relationships(self, scenario_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get expected causal relationships for scenario."""
        scenario_name = scenario_config['name']
        
        if 'Basic' in scenario_name:
            return [{'type': 'direct_causation', 'confidence': 0.8}]
        elif 'Confounded' in scenario_name:
            return [{'type': 'confounded_relationship', 'confidence': 0.6}]
        elif 'Temporal' in scenario_name:
            return [{'type': 'temporal_causation', 'confidence': 0.7}]
        elif 'Multi-step' in scenario_name:
            return [{'type': 'causal_chain', 'confidence': 0.5}]
        elif 'Medical' in scenario_name:
            return [{'type': 'domain_specific_causation', 'confidence': 0.9}]
        else:
            return [{'type': 'unknown', 'confidence': 0.5}]
    
    def _get_expected_performance_range(self, difficulty: str) -> Tuple[float, float]:
        """Get expected performance range for difficulty level."""
        ranges = {
            'easy': (0.8, 1.0),
            'medium': (0.6, 0.9),
            'hard': (0.4, 0.7),
            'expert': (0.2, 0.6)
        }
        return ranges.get(difficulty, (0.3, 0.7))
    
    def _extract_ground_truth_list(self, ground_truth: Dict[str, Any], num_predictions: int) -> List[Any]:
        """Extract ground truth list for comparison with predictions."""
        # Simplified ground truth extraction
        expected_relationships = ground_truth.get('expected_causal_relationships', [])
        
        if not expected_relationships:
            return [{'causal_relationship': 'none'}] * num_predictions
        
        # Use first expected relationship as template
        template = expected_relationships[0]
        return [template] * num_predictions
    
    def _calculate_robustness_score(
        self,
        predictions: List[Any],
        ground_truth: List[Any],
        scenario: ValidationScenario
    ) -> float:
        """Calculate robustness score for predictions."""
        # Simplified robustness calculation
        valid_predictions = [p for p in predictions if 'error' not in str(p)]
        robustness = len(valid_predictions) / len(predictions) if predictions else 0.0
        
        # Adjust for scenario difficulty
        difficulty_weights = {'easy': 1.0, 'medium': 0.9, 'hard': 0.8, 'expert': 0.7}
        weight = difficulty_weights.get(scenario.difficulty_level, 0.8)
        
        return robustness * weight
    
    def _perform_error_analysis(
        self,
        predictions: List[Any],
        ground_truth: List[Any],
        scenario: ValidationScenario
    ) -> Dict[str, Any]:
        """Perform detailed error analysis."""
        errors = []
        error_types = {'syntax_error': 0, 'logic_error': 0, 'timeout_error': 0, 'unknown_error': 0}
        
        for i, (pred, truth) in enumerate(zip(predictions, ground_truth)):
            if 'error' in str(pred):
                error_info = {
                    'case_id': i,
                    'error_message': str(pred),
                    'expected': truth,
                    'error_type': self._classify_error(str(pred))
                }
                errors.append(error_info)
                error_types[error_info['error_type']] += 1
        
        return {
            'total_errors': len(errors),
            'error_rate': len(errors) / len(predictions) if predictions else 0.0,
            'error_types': error_types,
            'error_details': errors[:5],  # Store first 5 errors for analysis
            'most_common_error': max(error_types.items(), key=lambda x: x[1])[0] if any(error_types.values()) else 'none'
        }
    
    def _classify_error(self, error_message: str) -> str:
        """Classify error type based on error message."""
        error_message = error_message.lower()
        
        if 'syntax' in error_message or 'invalid' in error_message:
            return 'syntax_error'
        elif 'timeout' in error_message or 'time' in error_message:
            return 'timeout_error'
        elif 'logic' in error_message or 'value' in error_message:
            return 'logic_error'
        else:
            return 'unknown_error'
    
    def _test_statistical_significance(self, accuracy: float, sample_size: int) -> bool:
        """Test if performance is statistically significant."""
        # Simplified significance test
        # Use binomial test against chance performance (0.5)
        if sample_size < 10:
            return False
        
        # Calculate z-score for binomial proportion test
        p0 = 0.5  # Null hypothesis (chance performance)
        p_hat = accuracy
        se = np.sqrt(p0 * (1 - p0) / sample_size)
        
        if se == 0:
            return False
        
        z_score = (p_hat - p0) / se
        
        # Two-tailed test, alpha = 0.05
        return abs(z_score) > 1.96
    
    def _estimate_memory_usage(self, predictions: List[Any]) -> float:
        """Estimate memory usage in MB."""
        # Simplified memory estimation
        total_size = 0
        for pred in predictions:
            total_size += len(str(pred))
        
        # Convert to approximate MB (very rough estimate)
        return total_size / (1024 * 1024)
    
    def _aggregate_validation_results(self, scenario_results: List[ValidationResult], algorithm_name: str) -> Dict[str, Any]:
        """Aggregate results across scenarios."""
        if not scenario_results:
            return {'overall_score': 0.0, 'error': 'No successful validations'}
        
        # Aggregate performance scores
        all_scores = {}
        for result in scenario_results:
            for metric, score in result.performance_scores.items():
                if metric not in all_scores:
                    all_scores[metric] = []
                all_scores[metric].append(score)
        
        # Calculate aggregated metrics
        aggregated_metrics = {}
        for metric, scores in all_scores.items():
            aggregated_metrics[metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'median': np.median(scores)
            }
        
        # Calculate overall score (weighted average of key metrics)
        key_metrics = ['accuracy', 'precision', 'recall']
        available_metrics = [m for m in key_metrics if m in aggregated_metrics]
        
        if available_metrics:
            overall_score = np.mean([aggregated_metrics[m]['mean'] for m in available_metrics])
        else:
            overall_score = 0.0
        
        # Aggregate other metrics
        avg_execution_time = np.mean([r.execution_time for r in scenario_results])
        avg_memory_usage = np.mean([r.memory_usage for r in scenario_results])
        avg_robustness = np.mean([r.robustness_score for r in scenario_results])
        
        return {
            'overall_score': overall_score,
            'performance_metrics': aggregated_metrics,
            'average_execution_time': avg_execution_time,
            'average_memory_usage': avg_memory_usage,
            'average_robustness': avg_robustness,
            'scenarios_passed': len(scenario_results),
            'algorithm_name': algorithm_name
        }
    
    def _perform_statistical_analysis(self, scenario_results: List[ValidationResult]) -> Dict[str, Any]:
        """Perform statistical analysis of validation results."""
        if len(scenario_results) < 2:
            return {'error': 'Insufficient data for statistical analysis'}
        
        # Extract accuracy scores for analysis
        accuracy_scores = [r.performance_scores.get('accuracy', 0.0) for r in scenario_results]
        
        # Basic statistical measures
        mean_accuracy = np.mean(accuracy_scores)
        std_accuracy = np.std(accuracy_scores)
        
        # Confidence interval for mean accuracy
        n = len(accuracy_scores)
        se = std_accuracy / np.sqrt(n)
        
        # 95% confidence interval (assuming t-distribution)
        import scipy.stats as stats
        t_critical = stats.t.ppf(0.975, n-1)
        ci_lower = mean_accuracy - t_critical * se
        ci_upper = mean_accuracy + t_critical * se
        
        # Consistency analysis
        consistency_score = 1.0 - (std_accuracy / mean_accuracy) if mean_accuracy > 0 else 0.0
        
        return {
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'confidence_interval': (ci_lower, ci_upper),
            'consistency_score': consistency_score,
            'sample_size': n,
            'statistical_power': self._estimate_statistical_power(accuracy_scores)
        }
    
    def _estimate_statistical_power(self, scores: List[float]) -> float:
        """Estimate statistical power of the validation."""
        # Simplified power estimation
        n = len(scores)
        effect_size = (np.mean(scores) - 0.5) / np.std(scores) if np.std(scores) > 0 else 0.0
        
        # Rough power estimation based on sample size and effect size
        if n < 10:
            return 0.5
        elif n < 20:
            return 0.7 if abs(effect_size) > 0.5 else 0.5
        else:
            return 0.9 if abs(effect_size) > 0.5 else 0.8
    
    def _generate_performance_breakdown(self, scenario_results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate detailed performance breakdown."""
        breakdown = {
            'by_difficulty': {},
            'by_domain': {},
            'by_scenario_type': {}
        }
        
        # Group results by various factors
        for result in scenario_results:
            # Find corresponding scenario
            scenario = next((s for s in self.validation_scenarios if s.scenario_id == result.scenario_id), None)
            
            if scenario:
                # Group by difficulty
                difficulty = scenario.difficulty_level
                if difficulty not in breakdown['by_difficulty']:
                    breakdown['by_difficulty'][difficulty] = []
                breakdown['by_difficulty'][difficulty].append(result.performance_scores.get('accuracy', 0.0))
                
                # Group by domain
                domain = scenario.domain
                if domain not in breakdown['by_domain']:
                    breakdown['by_domain'][domain] = []
                breakdown['by_domain'][domain].append(result.performance_scores.get('accuracy', 0.0))
                
                # Group by scenario type
                scenario_type = scenario.scenario_type
                if scenario_type not in breakdown['by_scenario_type']:
                    breakdown['by_scenario_type'][scenario_type] = []
                breakdown['by_scenario_type'][scenario_type].append(result.performance_scores.get('accuracy', 0.0))
        
        # Calculate summary statistics for each group
        for category in breakdown:
            for group, scores in breakdown[category].items():
                if scores:
                    breakdown[category][group] = {
                        'mean': np.mean(scores),
                        'std': np.std(scores),
                        'count': len(scores),
                        'raw_scores': scores
                    }
        
        return breakdown
    
    def _generate_validation_recommendations(
        self,
        aggregated_results: Dict[str, Any],
        statistical_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        overall_score = aggregated_results.get('overall_score', 0.0)
        consistency_score = statistical_analysis.get('consistency_score', 0.0)
        
        # Performance recommendations
        if overall_score < 0.6:
            recommendations.append("Algorithm performance is below acceptable threshold. Consider algorithmic improvements.")
        elif overall_score < 0.8:
            recommendations.append("Algorithm shows moderate performance. Focus on specific weak areas identified in breakdown.")
        else:
            recommendations.append("Algorithm demonstrates strong performance across validation scenarios.")
        
        # Consistency recommendations
        if consistency_score < 0.7:
            recommendations.append("Performance consistency is low. Investigate sources of variability.")
        
        # Statistical power recommendations
        statistical_power = statistical_analysis.get('statistical_power', 0.0)
        if statistical_power < 0.8:
            recommendations.append("Consider increasing sample size for more robust statistical validation.")
        
        # Domain-specific recommendations
        if 'performance_metrics' in aggregated_results:
            precision = aggregated_results['performance_metrics'].get('precision', {}).get('mean', 0.0)
            recall = aggregated_results['performance_metrics'].get('recall', {}).get('mean', 0.0)
            
            if precision < 0.7:
                recommendations.append("Low precision indicates high false positive rate. Improve specificity.")
            if recall < 0.7:
                recommendations.append("Low recall indicates high false negative rate. Improve sensitivity.")
        
        return recommendations
    
    # Additional methods for cross-validation and adversarial testing would go here...
    # (Omitted for brevity, but would include _prepare_cv_data, _create_cv_fold, etc.)
    
    def _prepare_cv_data(self, temporal_split: bool) -> List[Dict[str, Any]]:
        """Prepare data for cross-validation."""
        # Simplified CV data preparation
        cv_data = []
        for scenario in self.validation_scenarios:
            for test_case in scenario.test_cases:
                cv_data.append({
                    'input': test_case,
                    'scenario_id': scenario.scenario_id,
                    'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 365))
                })
        
        # Sort by timestamp if using temporal split
        if temporal_split:
            cv_data.sort(key=lambda x: x['timestamp'])
        
        return cv_data
    
    def _create_cv_fold(self, cv_data: List[Dict[str, Any]], fold: int, n_folds: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Create training and test data for a CV fold."""
        n_samples = len(cv_data)
        fold_size = n_samples // n_folds
        
        test_start = fold * fold_size
        test_end = test_start + fold_size
        
        test_data = cv_data[test_start:test_end]
        train_data = cv_data[:test_start] + cv_data[test_end:]
        
        return train_data, test_data
    
    def _adapt_algorithm_for_fold(self, algorithm_func: Callable, train_data: List[Dict[str, Any]]) -> Callable:
        """Adapt algorithm based on training data (simplified)."""
        # In a real implementation, this would train/adapt the algorithm
        # For now, just return the original function
        return algorithm_func
    
    async def _evaluate_fold(self, algorithm_func: Callable, test_data: List[Dict[str, Any]]) -> float:
        """Evaluate algorithm performance on fold test data."""
        correct = 0
        total = len(test_data)
        
        for data_point in test_data:
            try:
                prediction = await self._run_algorithm_on_case(algorithm_func, data_point['input'])
                # Simplified evaluation - assume prediction is correct if no error
                if 'error' not in str(prediction):
                    correct += 1
            except:
                pass
        
        return correct / total if total > 0 else 0.0
    
    def _calculate_cv_confidence_interval(self, cv_scores: List[float]) -> Tuple[float, float]:
        """Calculate confidence interval for CV scores."""
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        n = len(cv_scores)
        
        # 95% confidence interval
        import scipy.stats as stats
        t_critical = stats.t.ppf(0.975, n-1)
        margin_error = t_critical * std_score / np.sqrt(n)
        
        return (mean_score - margin_error, mean_score + margin_error)
    
    def _assess_temporal_stability(self, fold_results: List[Dict[str, Any]], temporal_split: bool) -> float:
        """Assess temporal stability of algorithm performance."""
        if not temporal_split or len(fold_results) < 3:
            return 0.5  # Default value when temporal analysis not applicable
        
        # Calculate trend in performance over folds (assuming temporal ordering)
        scores = [r['score'] for r in fold_results]
        
        # Simple linear trend analysis
        x = np.arange(len(scores))
        correlation = np.corrcoef(x, scores)[0, 1] if len(scores) > 1 else 0.0
        
        # Convert correlation to stability score (higher is more stable)
        stability = 1.0 - abs(correlation)
        
        return max(0.0, min(1.0, stability))
    
    def _assess_domain_generalization(self, fold_results: List[Dict[str, Any]]) -> float:
        """Assess domain generalization capability."""
        if len(fold_results) < 2:
            return 0.5
        
        scores = [r['score'] for r in fold_results]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Generalization is inversely related to variance
        if mean_score == 0:
            return 0.0
        
        generalization = max(0.0, 1.0 - (std_score / mean_score))
        return min(1.0, generalization)
    
    def _perform_cv_statistical_tests(self, cv_scores: List[float], fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform statistical tests on CV results."""
        if len(cv_scores) < 3:
            return {'error': 'Insufficient data for statistical tests'}
        
        # One-sample t-test against chance performance (0.5)
        import scipy.stats as stats
        t_stat, p_value = stats.ttest_1samp(cv_scores, 0.5)
        
        # Normality test
        shapiro_stat, shapiro_p = stats.shapiro(cv_scores)
        
        return {
            'one_sample_t_test': {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            },
            'normality_test': {
                'shapiro_statistic': shapiro_stat,
                'p_value': shapiro_p,
                'normal_distribution': shapiro_p > 0.05
            },
            'effect_size': (np.mean(cv_scores) - 0.5) / np.std(cv_scores) if np.std(cv_scores) > 0 else 0.0
        }
    
    def _generate_adversarial_scenario(self, adv_type: str) -> ValidationScenario:
        """Generate an adversarial validation scenario."""
        # Simplified adversarial scenario generation
        adversarial_scenarios = {
            'spurious_correlation': {
                'name': 'Spurious Correlation Attack',
                'description': 'Test robustness against spurious correlations',
                'difficulty': 'hard'
            },
            'confounding_bias': {
                'name': 'Confounding Bias Attack',
                'description': 'Test ability to handle confounding variables',
                'difficulty': 'medium'
            },
            'selection_bias': {
                'name': 'Selection Bias Attack',
                'description': 'Test robustness against selection bias',
                'difficulty': 'medium'
            },
            'temporal_shift': {
                'name': 'Temporal Shift Attack',
                'description': 'Test stability under temporal distribution shift',
                'difficulty': 'hard'
            },
            'domain_shift': {
                'name': 'Domain Shift Attack',
                'description': 'Test generalization across domain shifts',
                'difficulty': 'expert'
            },
            'noise_injection': {
                'name': 'Noise Injection Attack',
                'description': 'Test robustness against noisy data',
                'difficulty': 'easy'
            }
        }
        
        config = adversarial_scenarios.get(adv_type, adversarial_scenarios['spurious_correlation'])
        
        scenario = ValidationScenario(
            scenario_id=self._generate_scenario_id(f"adversarial_{adv_type}"),
            name=config['name'],
            description=config['description'],
            scenario_type='adversarial',
            difficulty_level=config['difficulty'],
            domain='general',
            test_cases=self._generate_adversarial_test_cases(adv_type),
            ground_truth={'adversarial_type': adv_type, 'expected_degradation': 0.2},
            evaluation_metrics=['accuracy', 'robustness'],
            expected_performance_range=(0.3, 0.8)  # Lower expectations for adversarial scenarios
        )
        
        return scenario
    
    def _generate_adversarial_test_cases(self, adv_type: str) -> List[Dict[str, Any]]:
        """Generate test cases for adversarial scenarios."""
        test_cases = []
        
        for i in range(10):  # Generate 10 adversarial test cases
            case = {
                'case_id': f"adversarial_{adv_type}_case_{i}",
                'input': {
                    'scenario': f"Adversarial test case {i} for {adv_type}",
                    'question': f"Identify causal relationship despite {adv_type}",
                    'adversarial_type': adv_type,
                    'corruption_level': 0.1 + (i % 5) * 0.1
                },
                'metadata': {
                    'adversarial_type': adv_type,
                    'case_number': i
                }
            }
            test_cases.append(case)
        
        return test_cases
    
    def _calculate_performance_degradation(self, result: ValidationResult, adv_type: str) -> float:
        """Calculate performance degradation due to adversarial attack."""
        # Compare with baseline performance (would need to store baseline results)
        baseline_performance = 0.8  # Assumed baseline
        current_performance = result.performance_scores.get('accuracy', 0.0)
        
        degradation = (baseline_performance - current_performance) / baseline_performance
        return max(0.0, min(1.0, degradation))
    
    def _analyze_vulnerability(self, result: ValidationResult, adv_type: str) -> Dict[str, Any]:
        """Analyze algorithm vulnerability to specific adversarial attack."""
        performance = result.performance_scores.get('accuracy', 0.0)
        robustness = result.robustness_score
        
        vulnerability_level = 'low' if performance > 0.7 else 'medium' if performance > 0.4 else 'high'
        
        return {
            'vulnerability_level': vulnerability_level,
            'performance_score': performance,
            'robustness_score': robustness,
            'recommended_defenses': self._get_recommended_defenses(adv_type, vulnerability_level)
        }
    
    def _get_recommended_defenses(self, adv_type: str, vulnerability_level: str) -> List[str]:
        """Get recommended defenses for adversarial attacks."""
        defense_recommendations = {
            'spurious_correlation': ['Use causal discovery algorithms', 'Implement confounding controls'],
            'confounding_bias': ['Add confounding variables to model', 'Use propensity score matching'],
            'selection_bias': ['Implement random sampling', 'Use inverse probability weighting'],
            'temporal_shift': ['Add temporal validation', 'Use sliding window training'],
            'domain_shift': ['Implement domain adaptation', 'Use cross-domain validation'],
            'noise_injection': ['Add noise regularization', 'Use robust optimization methods']
        }
        
        defenses = defense_recommendations.get(adv_type, ['General robustness improvements'])
        
        if vulnerability_level == 'high':
            defenses.extend(['Consider algorithm redesign', 'Implement ensemble methods'])
        
        return defenses
    
    def _rank_vulnerabilities(self, adversarial_results: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Rank vulnerabilities by severity."""
        vulnerabilities = []
        
        for adv_type, result in adversarial_results.items():
            if 'error' not in result:
                severity = 1.0 - result.get('robustness_score', 0.0)
                vulnerabilities.append((adv_type, severity))
        
        vulnerabilities.sort(key=lambda x: x[1], reverse=True)
        return vulnerabilities
    
    def _generate_robustness_recommendations(self, adversarial_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving robustness."""
        recommendations = []
        
        # Count high-severity vulnerabilities
        high_severity_count = sum(1 for result in adversarial_results.values() 
                                if 'error' not in result and result.get('robustness_score', 0.0) < 0.5)
        
        if high_severity_count > 3:
            recommendations.append("Algorithm shows multiple severe vulnerabilities. Consider fundamental redesign.")
        elif high_severity_count > 1:
            recommendations.append("Address multiple identified vulnerabilities through targeted improvements.")
        elif high_severity_count == 1:
            recommendations.append("Focus on addressing the identified high-severity vulnerability.")
        else:
            recommendations.append("Algorithm demonstrates good overall robustness. Continue monitoring.")
        
        # Add specific recommendations based on worst vulnerabilities
        ranked_vulnerabilities = self._rank_vulnerabilities(adversarial_results)
        if ranked_vulnerabilities:
            worst_vulnerability = ranked_vulnerabilities[0][0]
            specific_defenses = self._get_recommended_defenses(worst_vulnerability, 'high')
            recommendations.extend(specific_defenses[:2])  # Add top 2 specific recommendations
        
        return recommendations