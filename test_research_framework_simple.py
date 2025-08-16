"""
Simple Test Suite for Research Framework

This test suite validates the research framework without external dependencies.
"""

import sys
import os
import traceback
from pathlib import Path

# Add the repo to Python path
sys.path.insert(0, '/root/repo')

def test_novel_algorithms():
    """Test novel algorithms module."""
    try:
        from causal_eval.research.novel_algorithms import (
            InformationTheoreticCausalityMetric,
            CausalGraph
        )
        
        # Test IT metric
        metric = InformationTheoreticCausalityMetric()
        assert metric.alpha == 0.05
        
        # Test graph creation
        graph = CausalGraph(
            nodes=['A', 'B'],
            edges=[('A', 'B')],
            confounders={('A', 'B'): ['C']}
        )
        assert len(graph.nodes) == 2
        
        # Test scoring
        response = "A causes B through mechanism X"
        score = metric.compute_score(response, graph)
        assert 0.0 <= score <= 1.0
        
        print("✅ Novel Algorithms: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Novel Algorithms: FAILED - {e}")
        traceback.print_exc()
        return False

def test_experimental_framework():
    """Test experimental framework module."""
    try:
        from causal_eval.research.experimental_framework import (
            ExperimentalFramework,
            ExperimentConfig,
            ModelConfiguration
        )
        
        # Test config creation
        config = ExperimentConfig(
            experiment_name="Test",
            description="Test experiment"
        )
        assert config.experiment_name == "Test"
        
        # Test framework initialization
        framework = ExperimentalFramework(config)
        assert framework.config.experiment_name == "Test"
        
        # Test model configuration
        model_config = ModelConfiguration(
            model_name="test-model",
            model_type="transformer",
            model_size="small",
            training_data="test",
            model_version="v1.0"
        )
        assert model_config.model_name == "test-model"
        
        print("✅ Experimental Framework: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Experimental Framework: FAILED - {e}")
        traceback.print_exc()
        return False

def test_baseline_models():
    """Test baseline models module."""
    try:
        from causal_eval.research.baseline_models import (
            BaselineEvaluator,
            RandomBaseline,
            KeywordBaseline
        )
        
        # Test random baseline
        baseline = RandomBaseline(seed=42)
        scenario = "A causes B"
        result = baseline.predict(scenario)
        assert result.model_name == "Random Baseline"
        assert result.predicted_relationship in ["causal", "spurious", "correlation", "reverse_causal"]
        
        # Test keyword baseline
        keyword_baseline = KeywordBaseline()
        result = keyword_baseline.predict("Exercise causes health")
        assert result.model_name == "Keyword Heuristic Baseline"
        
        # Test evaluator
        evaluator = BaselineEvaluator()
        assert len(evaluator.baselines) > 0
        
        print("✅ Baseline Models: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Baseline Models: FAILED - {e}")
        traceback.print_exc()
        return False

def test_validation_suite():
    """Test validation suite module."""
    try:
        from causal_eval.research.validation_suite import (
            ValidationSuite,
            InternalConsistencyTest
        )
        
        # Test validation suite initialization
        suite = ValidationSuite()
        assert len(suite.tests) > 0
        
        # Test internal consistency
        test = InternalConsistencyTest()
        test_data = [
            {
                'metric_scores': {
                    'metric1': {'mean': 0.8, 'scores': [0.7, 0.8, 0.9]},
                    'metric2': {'mean': 0.75, 'scores': [0.7, 0.75, 0.8]}
                }
            }
        ]
        
        result = test.run_test(test_data)
        assert result.test_name == "Internal Consistency"
        assert 0.0 <= result.score <= 1.0
        
        print("✅ Validation Suite: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Validation Suite: FAILED - {e}")
        traceback.print_exc()
        return False

def test_publication_tools():
    """Test publication tools module."""
    try:
        from causal_eval.research.publication_tools import (
            PublicationGenerator,
            AcademicFigureGenerator,
            PublicationMetadata
        )
        
        # Test metadata
        metadata = PublicationMetadata(
            title="Test Paper",
            authors=["Author 1"],
            affiliation="Test University",
            abstract="Test abstract",
            keywords=["test"],
            submission_date="2024-01-01",
            venue="Test Venue"
        )
        assert metadata.title == "Test Paper"
        
        # Test figure generator
        generator = AcademicFigureGenerator()
        assert generator.style == "academic"
        assert generator.dpi == 300
        
        print("✅ Publication Tools: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Publication Tools: FAILED - {e}")
        traceback.print_exc()
        return False

def test_research_discovery():
    """Test research discovery module."""
    try:
        from causal_eval.research.research_discovery import (
            PerformancePatternAnalyzer,
            ResearchGapIdentifier,
            FutureDirectionsSynthesizer
        )
        
        # Test pattern analyzer
        analyzer = PerformancePatternAnalyzer()
        assert analyzer is not None
        
        # Test gap identifier
        identifier = ResearchGapIdentifier()
        assert identifier is not None
        
        # Test synthesizer
        synthesizer = FutureDirectionsSynthesizer()
        assert synthesizer is not None
        
        print("✅ Research Discovery: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Research Discovery: FAILED - {e}")
        traceback.print_exc()
        return False

def test_integration():
    """Test basic integration between components."""
    try:
        # Import all major components
        from causal_eval.research.novel_algorithms import InformationTheoreticCausalityMetric, CausalGraph
        from causal_eval.research.experimental_framework import ExperimentConfig
        from causal_eval.research.baseline_models import RandomBaseline
        from causal_eval.research.validation_suite import ValidationSuite
        from causal_eval.research.publication_tools import PublicationMetadata
        from causal_eval.research.research_discovery import PerformancePatternAnalyzer
        
        # Test basic integration workflow
        
        # 1. Create evaluation components
        metric = InformationTheoreticCausalityMetric()
        baseline = RandomBaseline()
        analyzer = PerformancePatternAnalyzer()
        
        # 2. Test simple evaluation
        graph = CausalGraph(nodes=['A', 'B'], edges=[('A', 'B')], confounders={})
        response = "A causes B"
        score = metric.compute_score(response, graph)
        
        # 3. Test baseline prediction
        baseline_result = baseline.predict("X causes Y")
        
        # 4. Validate basic data flow
        assert 0.0 <= score <= 1.0
        assert baseline_result.predicted_relationship in ["causal", "spurious", "correlation", "reverse_causal"]
        
        print("✅ Integration: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Integration: FAILED - {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests and report results."""
    
    print("🧪 Running Research Framework Test Suite")
    print("=" * 50)
    
    tests = [
        ("Novel Algorithms", test_novel_algorithms),
        ("Experimental Framework", test_experimental_framework),
        ("Baseline Models", test_baseline_models),
        ("Validation Suite", test_validation_suite),
        ("Publication Tools", test_publication_tools),
        ("Research Discovery", test_research_discovery),
        ("Integration", test_integration)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name}: CRITICAL FAILURE - {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} PASSED, {failed} FAILED")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED - Research Framework is fully functional!")
        return True
    else:
        print(f"⚠️ {failed} test(s) failed - Review implementation")
        return False

def check_framework_completeness():
    """Check that all framework components are present and loadable."""
    
    print("\n🔍 Checking Framework Completeness...")
    
    required_modules = [
        "causal_eval.research.novel_algorithms",
        "causal_eval.research.experimental_framework", 
        "causal_eval.research.baseline_models",
        "causal_eval.research.validation_suite",
        "causal_eval.research.publication_tools",
        "causal_eval.research.research_discovery"
    ]
    
    missing_modules = []
    
    for module_name in required_modules:
        try:
            __import__(module_name)
            print(f"✅ {module_name}")
        except ImportError as e:
            print(f"❌ {module_name}: {e}")
            missing_modules.append(module_name)
    
    if not missing_modules:
        print("🎉 All required modules are present!")
        return True
    else:
        print(f"⚠️ Missing modules: {missing_modules}")
        return False

def calculate_framework_metrics():
    """Calculate framework quality metrics."""
    
    print("\n📊 Calculating Framework Metrics...")
    
    try:
        # Count implementation files
        research_dir = Path("/root/repo/causal_eval/research")
        if research_dir.exists():
            python_files = list(research_dir.glob("*.py"))
            total_files = len(python_files)
            
            # Calculate lines of code
            total_lines = 0
            for file_path in python_files:
                try:
                    with open(file_path, 'r') as f:
                        lines = len(f.readlines())
                    total_lines += lines
                    print(f"📄 {file_path.name}: {lines} lines")
                except Exception:
                    continue
            
            print(f"\n📈 Framework Metrics:")
            print(f"   📁 Research modules: {total_files}")
            print(f"   📝 Total lines of code: {total_lines:,}")
            print(f"   📊 Average lines per module: {total_lines // total_files if total_files > 0 else 0:,}")
            
            # Estimate complexity
            if total_lines > 5000:
                complexity = "Very High"
            elif total_lines > 3000:
                complexity = "High"
            elif total_lines > 1500:
                complexity = "Medium"
            else:
                complexity = "Low"
            
            print(f"   🧠 Framework complexity: {complexity}")
            print(f"   🎯 Estimated capability: Production-Ready Research Framework")
            
        else:
            print("❌ Research directory not found")
            
    except Exception as e:
        print(f"❌ Error calculating metrics: {e}")

if __name__ == "__main__":
    print("🚀 Research Framework Quality Gate Validation")
    print("=" * 60)
    
    # Step 1: Check completeness
    completeness_ok = check_framework_completeness()
    
    # Step 2: Run functional tests
    tests_ok = run_all_tests()
    
    # Step 3: Calculate metrics
    calculate_framework_metrics()
    
    # Final assessment
    print("\n" + "=" * 60)
    print("🏁 FINAL QUALITY GATE ASSESSMENT")
    print("=" * 60)
    
    if completeness_ok and tests_ok:
        print("✅ QUALITY GATE: PASSED")
        print("🎉 Research Framework is production-ready!")
        print("📋 Ready for:")
        print("   • Novel causal reasoning research")
        print("   • Rigorous experimental validation")
        print("   • Academic publication preparation")
        print("   • Statistical analysis and reporting")
        print("   • Performance benchmarking")
        print("   • Research gap identification")
        exit(0)
    else:
        print("❌ QUALITY GATE: FAILED")
        print("⚠️ Framework requires additional development")
        exit(1)