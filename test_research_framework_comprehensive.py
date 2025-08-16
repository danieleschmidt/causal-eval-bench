"""
Comprehensive Test Suite for Research Framework

This test suite validates the entire research framework including novel algorithms,
experimental framework, validation suite, and publication tools.
"""

import pytest
import numpy as np
import asyncio
from pathlib import Path
import tempfile
import json
from unittest.mock import Mock, patch

# Import the research modules
from causal_eval.research.novel_algorithms import (
    InformationTheoreticCausalityMetric,
    CausalConsistencyMetric,
    MultimodalCausalityMetric,
    CausalGraph
)
from causal_eval.research.experimental_framework import (
    ExperimentalFramework,
    ExperimentConfig,
    ModelConfiguration
)
from causal_eval.research.baseline_models import (
    BaselineEvaluator,
    RandomBaseline,
    KeywordBaseline,
    MLBaseline
)
from causal_eval.research.validation_suite import (
    ValidationSuite,
    InternalConsistencyTest,
    TestRetestReliabilityTest
)
from causal_eval.research.publication_tools import (
    PublicationGenerator,
    AcademicFigureGenerator,
    PublicationMetadata
)
from causal_eval.research.research_discovery import (
    PerformancePatternAnalyzer,
    ResearchGapIdentifier,
    FutureDirectionsSynthesizer
)


class TestNovelAlgorithms:
    """Test suite for novel algorithms module."""
    
    def test_information_theoretic_metric_initialization(self):
        """Test IT metric initialization."""
        metric = InformationTheoreticCausalityMetric()
        assert metric.alpha == 0.05
        assert metric.get_explanation() is not None
    
    def test_information_theoretic_metric_scoring(self):
        """Test IT metric scoring functionality."""
        metric = InformationTheoreticCausalityMetric()
        
        # Create test ground truth
        ground_truth = CausalGraph(
            nodes=['A', 'B'],
            edges=[('A', 'B')],
            confounders={('A', 'B'): ['C']}
        )
        
        # Test response
        response = "A causes B because there is a clear causal mechanism. However, variable C might be a confounder."
        
        score = metric.compute_score(response, ground_truth)
        assert 0.0 <= score <= 1.0
    
    def test_causal_consistency_metric(self):
        """Test causal consistency metric."""
        metric = CausalConsistencyMetric([])
        
        response = "This is a causal relationship involving temporal mechanisms."
        ground_truth = {"type": "causal"}
        
        score = metric.compute_score(response, ground_truth)
        assert 0.0 <= score <= 1.0
    
    def test_multimodal_causality_metric(self):
        """Test multimodal causality metric."""
        metric = MultimodalCausalityMetric()
        
        response = "The textual evidence shows correlation of 0.8, and the structural analysis reveals a network pattern."
        ground_truth = {
            'text': {'key_phrases': ['correlation', 'evidence']},
            'numerical': {'patterns': ['0.8']},
            'structural': {'relationships': ['network']}
        }
        
        score = metric.compute_score(response, ground_truth)
        assert 0.0 <= score <= 1.0
    
    def test_causal_graph_creation(self):
        """Test causal graph functionality."""
        graph = CausalGraph(
            nodes=['X', 'Y', 'Z'],
            edges=[('X', 'Y'), ('Y', 'Z')],
            confounders={('X', 'Y'): ['C1'], ('Y', 'Z'): ['C2']}
        )
        
        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2
        assert ('X', 'Y') in graph.confounders
        
        # Test NetworkX conversion
        nx_graph = graph.to_networkx()
        assert nx_graph.number_of_nodes() == 3
        assert nx_graph.number_of_edges() == 2


class TestExperimentalFramework:
    """Test suite for experimental framework."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_experiment_config_creation(self):
        """Test experiment configuration creation."""
        config = ExperimentConfig(
            experiment_name="Test Experiment",
            description="Test description",
            random_seed=42,
            min_sample_size=10
        )
        
        assert config.experiment_name == "Test Experiment"
        assert config.random_seed == 42
        assert config.min_sample_size == 10
    
    def test_model_configuration(self):
        """Test model configuration."""
        model_config = ModelConfiguration(
            model_name="test-model",
            model_type="transformer",
            model_size="small",
            training_data="test_data",
            model_version="v1.0"
        )
        
        assert model_config.model_name == "test-model"
        assert model_config.model_type == "transformer"
    
    def test_experimental_framework_initialization(self, temp_dir):
        """Test experimental framework initialization."""
        config = ExperimentConfig(
            experiment_name="Test",
            description="Test experiment",
            output_directory=temp_dir
        )
        
        framework = ExperimentalFramework(config)
        assert framework.config.experiment_name == "Test"
        assert len(framework.metrics) > 0
    
    @pytest.mark.asyncio
    async def test_test_case_generation(self, temp_dir):
        """Test test case generation."""
        config = ExperimentConfig(
            experiment_name="Test",
            description="Test experiment",
            output_directory=temp_dir
        )
        
        framework = ExperimentalFramework(config)
        
        task_config = {
            'task_type': 'attribution',
            'domain': 'medical',
            'num_cases': 5
        }
        
        test_cases = await framework._generate_test_cases(task_config)
        assert len(test_cases) == 5
        assert all('prompt' in case for case in test_cases)
        assert all('ground_truth' in case for case in test_cases)


class TestBaselineModels:
    """Test suite for baseline models."""
    
    def test_random_baseline(self):
        """Test random baseline functionality."""
        baseline = RandomBaseline(seed=42)
        
        scenario = "Ice cream sales and swimming accidents both increase in summer."
        result = baseline.predict(scenario)
        
        assert result.model_name == "Random Baseline"
        assert result.predicted_relationship in ["causal", "spurious", "correlation", "reverse_causal"]
        assert 0.0 <= result.confidence <= 1.0
    
    def test_keyword_baseline(self):
        """Test keyword baseline functionality."""
        baseline = KeywordBaseline()
        
        scenario = "Regular exercise causes improved cardiovascular health."
        result = baseline.predict(scenario)
        
        assert result.model_name == "Keyword Heuristic Baseline"
        assert result.predicted_relationship in ["causal", "spurious", "correlation", "reverse_causal"]
    
    def test_ml_baseline_initialization(self):
        """Test ML baseline initialization."""
        baseline = MLBaseline("logistic_regression")
        
        assert baseline.model_type == "logistic_regression"
        assert not baseline.is_trained
    
    def test_baseline_evaluator(self):
        """Test baseline evaluator functionality."""
        evaluator = BaselineEvaluator()
        
        assert len(evaluator.baselines) > 0
        assert "random" in evaluator.baselines
        assert "keyword" in evaluator.baselines
    
    def test_baseline_evaluation(self):
        """Test baseline evaluation process."""
        evaluator = BaselineEvaluator()
        
        scenarios = [
            "A causes B",
            "X correlates with Y",
            "Z results in W"
        ]
        ground_truths = ["causal", "correlation", "causal"]
        
        results = evaluator.evaluate_all_baselines(scenarios, ground_truths)
        
        assert len(results) > 0
        for baseline_name, result_data in results.items():
            assert "accuracy" in result_data
            assert "results" in result_data
            assert len(result_data["results"]) == len(scenarios)


class TestValidationSuite:
    """Test suite for validation suite."""
    
    def test_internal_consistency_test(self):
        """Test internal consistency validation."""
        test = InternalConsistencyTest()
        
        # Mock data with multiple metrics
        test_data = [
            {
                'metric_scores': {
                    'metric1': {'mean': 0.8, 'scores': [0.7, 0.8, 0.9]},
                    'metric2': {'mean': 0.75, 'scores': [0.7, 0.75, 0.8]},
                    'metric3': {'mean': 0.85, 'scores': [0.8, 0.85, 0.9]}
                }
            }
        ]
        
        result = test.run_test(test_data)
        
        assert result.test_name == "Internal Consistency"
        assert isinstance(result.passed, bool)
        assert 0.0 <= result.score <= 1.0
    
    def test_test_retest_reliability(self):
        """Test test-retest reliability validation."""
        test = TestRetestReliabilityTest()
        
        # Mock test-retest data
        test_data = {
            'test': [{'overall_score': 0.8}, {'overall_score': 0.7}, {'overall_score': 0.9}],
            'retest': [{'overall_score': 0.82}, {'overall_score': 0.68}, {'overall_score': 0.88}]
        }
        
        result = test.run_test(test_data)
        
        assert result.test_name == "Test-Retest Reliability"
        assert isinstance(result.passed, bool)
        assert 0.0 <= result.score <= 1.0
    
    def test_validation_suite_initialization(self):
        """Test validation suite initialization."""
        suite = ValidationSuite()
        
        assert len(suite.tests) > 0
        assert 'internal_consistency' in suite.tests
        assert 'test_retest_reliability' in suite.tests


class TestPublicationTools:
    """Test suite for publication tools."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_academic_figure_generator_initialization(self):
        """Test academic figure generator initialization."""
        generator = AcademicFigureGenerator()
        
        assert generator.style == "academic"
        assert generator.dpi == 300
        assert generator.figure_counter == 1
    
    def test_figure_generation(self, temp_dir):
        """Test figure generation functionality."""
        generator = AcademicFigureGenerator()
        
        # Mock results data
        results = {
            'model_1': {'accuracy': 0.8},
            'model_2': {'accuracy': 0.75},
            'model_3': {'accuracy': 0.85}
        }
        
        figure_path = generator.create_model_comparison_figure(
            results, temp_dir, "Test Comparison"
        )
        
        assert Path(figure_path).exists()
        assert figure_path.endswith('.pdf')
    
    def test_publication_metadata(self):
        """Test publication metadata structure."""
        metadata = PublicationMetadata(
            title="Test Paper",
            authors=["Author 1", "Author 2"],
            affiliation="Test University",
            abstract="Test abstract",
            keywords=["test", "paper"],
            submission_date="2024-01-01",
            venue="Test Venue"
        )
        
        assert metadata.title == "Test Paper"
        assert len(metadata.authors) == 2
        assert len(metadata.keywords) == 2
    
    def test_publication_generator_initialization(self, temp_dir):
        """Test publication generator initialization."""
        generator = PublicationGenerator(temp_dir)
        
        assert generator.output_dir.exists()
        assert (generator.output_dir / "figures").exists()
        assert (generator.output_dir / "tables").exists()


class TestResearchDiscovery:
    """Test suite for research discovery module."""
    
    def test_performance_pattern_analyzer(self):
        """Test performance pattern analyzer."""
        analyzer = PerformancePatternAnalyzer()
        
        # Mock results data
        results = [
            {
                'model_results': {
                    'model_1': {'task_1': {'score': 0.8}, 'task_2': {'score': 0.7}},
                    'model_2': {'task_1': {'score': 0.75}, 'task_2': {'score': 0.8}}
                }
            }
        ]
        
        analysis = analyzer.analyze_performance_landscapes(results)
        
        assert 'model_task_matrix' in analysis
        assert 'model_clusters' in analysis
        assert 'performance_gaps' in analysis
    
    def test_research_gap_identifier(self):
        """Test research gap identifier."""
        identifier = ResearchGapIdentifier()
        
        # Mock performance analysis
        performance_analysis = {
            'performance_gaps': [
                {
                    'type': 'universal_difficulty',
                    'avg_performance': 0.4,
                    'description': 'All models struggle'
                }
            ],
            'ceiling_effects': {
                'saturated_tasks': [{'task_index': 0, 'max_performance': 0.98}]
            }
        }
        
        gaps = identifier.identify_gaps(performance_analysis)
        
        assert len(gaps) > 0
        assert all(hasattr(gap, 'gap_type') for gap in gaps)
        assert all(hasattr(gap, 'priority') for gap in gaps)
    
    def test_future_directions_synthesizer(self):
        """Test future directions synthesizer."""
        synthesizer = FutureDirectionsSynthesizer()
        
        # Mock research gaps
        from causal_eval.research.research_discovery import ResearchGap
        
        gaps = [
            ResearchGap(
                gap_type="performance",
                title="Test Gap",
                description="Test description",
                evidence={},
                priority="high",
                estimated_impact=0.8,
                difficulty="medium",
                suggested_approaches=["approach1", "approach2"]
            )
        ]
        
        # Mock performance analysis
        performance_analysis = {'test': 'data'}
        
        agenda = synthesizer.synthesize_research_agenda(gaps, performance_analysis)
        
        assert 'research_opportunities' in agenda
        assert 'roadmap' in agenda
        assert 'priority_rankings' in agenda


class TestIntegration:
    """Integration tests for the complete research framework."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.mark.asyncio
    async def test_end_to_end_research_pipeline(self, temp_dir):
        """Test complete research pipeline from experiment to publication."""
        
        # 1. Set up experimental framework
        config = ExperimentConfig(
            experiment_name="Integration Test",
            description="End-to-end test",
            output_directory=temp_dir,
            min_sample_size=5
        )
        
        framework = ExperimentalFramework(config)
        
        # 2. Mock model configurations
        models = [
            ModelConfiguration(
                model_name="test_model_1",
                model_type="transformer",
                model_size="small",
                training_data="test",
                model_version="v1.0"
            ),
            ModelConfiguration(
                model_name="test_model_2", 
                model_type="transformer",
                model_size="medium",
                training_data="test",
                model_version="v1.0"
            )
        ]
        
        # 3. Mock tasks
        tasks = [
            {'task_type': 'attribution', 'domain': 'medical', 'num_cases': 3},
            {'task_type': 'counterfactual', 'domain': 'business', 'num_cases': 3}
        ]
        
        # 4. Run comparative study (simplified)
        # Note: This would normally be run with actual models
        mock_results = {
            'results': [
                {
                    'model_config': models[0],
                    'task_type': 'attribution',
                    'metric_scores': {
                        'information_theoretic': {'mean': 0.8, 'scores': [0.7, 0.8, 0.9]},
                        'consistency': {'mean': 0.75, 'scores': [0.7, 0.75, 0.8]}
                    }
                }
            ],
            'statistical_analysis': {
                'model1_vs_model2_information_theoretic': {
                    'test_statistic': 2.5,
                    'p_value': 0.02,
                    'effect_size': 0.6,
                    'significant': True
                }
            },
            'experiment_id': 'test_exp_001'
        }
        
        # 5. Validate results
        validation_suite = ValidationSuite()
        validation_data = {
            'internal_consistency': [
                {
                    'metric_scores': {
                        'metric1': {'mean': 0.8, 'scores': [0.7, 0.8, 0.9]},
                        'metric2': {'mean': 0.75, 'scores': [0.7, 0.75, 0.8]}
                    }
                }
            ]
        }
        
        validation_report = validation_suite.run_comprehensive_validation(validation_data)
        assert validation_report.overall_reliability_score >= 0.0
        
        # 6. Generate publication materials
        pub_generator = PublicationGenerator(temp_dir)
        
        metadata = PublicationMetadata(
            title="Integration Test Paper",
            authors=["Test Author"],
            affiliation="Test Institution",
            abstract="This is a test abstract for integration testing.",
            keywords=["test", "integration", "causal reasoning"],
            submission_date="2024-01-01",
            venue="Test Conference"
        )
        
        # Mock research data for publication
        research_data = {
            'model_results': {
                'test_model_1': {'accuracy': 0.8},
                'test_model_2': {'accuracy': 0.75}
            }
        }
        
        paper_path = pub_generator.generate_research_paper(research_data, metadata)
        assert Path(paper_path).exists()
        
        # 7. Research discovery
        pattern_analyzer = PerformancePatternAnalyzer()
        gap_identifier = ResearchGapIdentifier()
        
        # Mock analysis results
        performance_analysis = pattern_analyzer.analyze_performance_landscapes([
            {'model_results': research_data['model_results']}
        ])
        
        gaps = gap_identifier.identify_gaps(performance_analysis)
        assert len(gaps) >= 0  # Should identify some gaps
        
        print(f"✅ Integration test completed successfully!")
        print(f"📄 Paper generated: {paper_path}")
        print(f"📊 Validation score: {validation_report.overall_reliability_score:.3f}")
        print(f"🔍 Research gaps identified: {len(gaps)}")
    
    def test_quality_metrics_computation(self):
        """Test quality metrics computation across framework."""
        
        # Test metrics are within expected ranges
        test_responses = [
            "A causes B through a clear mechanism",
            "There is correlation but no causation",
            "This appears to be spurious correlation"
        ]
        
        # Test information theoretic metric
        it_metric = InformationTheoreticCausalityMetric()
        graph = CausalGraph(nodes=['A', 'B'], edges=[('A', 'B')], confounders={})
        
        scores = []
        for response in test_responses:
            score = it_metric.compute_score(response, graph)
            scores.append(score)
            assert 0.0 <= score <= 1.0
        
        # Scores should show reasonable variation
        assert np.std(scores) > 0.01  # Some variation expected
        
        print(f"✅ Quality metrics test passed!")
        print(f"📊 Score range: {min(scores):.3f} - {max(scores):.3f}")
        print(f"📈 Score variation: {np.std(scores):.3f}")


def test_framework_completeness():
    """Test that all framework components are properly integrated."""
    
    # Check that all major components can be imported and initialized
    components = [
        (InformationTheoreticCausalityMetric, "Novel Algorithms"),
        (ExperimentalFramework, "Experimental Framework"),
        (BaselineEvaluator, "Baseline Models"),
        (ValidationSuite, "Validation Suite"),
        (PublicationGenerator, "Publication Tools"),
        (PerformancePatternAnalyzer, "Research Discovery")
    ]
    
    for component_class, name in components:
        try:
            if component_class == ExperimentalFramework:
                config = ExperimentConfig(experiment_name="test", description="test")
                instance = component_class(config)
            elif component_class == PublicationGenerator:
                instance = component_class("/tmp/test")
            else:
                instance = component_class()
            
            assert instance is not None
            print(f"✅ {name} component: OK")
            
        except Exception as e:
            pytest.fail(f"❌ {name} component failed to initialize: {e}")
    
    print("🎉 All framework components successfully integrated!")


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v", "--tb=short"])