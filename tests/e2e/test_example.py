"""Example end-to-end tests for Causal Eval Bench."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock


@pytest.mark.e2e
class TestFullEvaluationWorkflow:
    """End-to-end tests for complete evaluation workflows."""
    
    def test_complete_evaluation_pipeline(self, mock_file_system, sample_causal_question):
        """Test the complete evaluation pipeline from start to finish."""
        
        # Step 1: Setup evaluation configuration
        config = {
            "model_name": "test-gpt-4",
            "api_key": "test-key",
            "tasks": ["causal_attribution"],
            "domains": ["general"],
            "num_questions": 1,
            "output_format": "json"
        }
        
        # Step 2: Mock test data preparation
        test_data = [sample_causal_question]
        
        # Step 3: Mock evaluation execution
        with patch("causal_eval.evaluation.ModelEvaluator") as mock_evaluator:
            mock_instance = MagicMock()
            mock_instance.evaluate_batch.return_value = {
                "results": [
                    {
                        "question_id": "test_question_001",
                        "score": 0.9,
                        "response": "No, ice cream does not cause swimming accidents. This is correlation, not causation.",
                        "confidence": 0.95,
                        "execution_time": 2.1
                    }
                ],
                "overall_score": 0.9,
                "metadata": {
                    "total_questions": 1,
                    "successful_evaluations": 1,
                    "failed_evaluations": 0
                }
            }
            mock_evaluator.return_value = mock_instance
            
            # Step 4: Execute evaluation
            evaluator = mock_evaluator(config["model_name"], api_key=config["api_key"])
            results = evaluator.evaluate_batch(test_data)
            
            # Step 5: Verify results
            assert results["overall_score"] == 0.9
            assert len(results["results"]) == 1
            assert results["results"][0]["score"] == 0.9
            assert results["metadata"]["successful_evaluations"] == 1
        
        # Step 6: Mock result storage
        results_file = mock_file_system / "evaluation_results.json"
        results_file.write_text(json.dumps(results, indent=2))
        
        # Step 7: Verify result persistence
        saved_results = json.loads(results_file.read_text())
        assert saved_results["overall_score"] == results["overall_score"]
    
    def test_multi_task_evaluation_workflow(self, mock_file_system, sample_causal_question, sample_counterfactual_question):
        """Test evaluation workflow with multiple task types."""
        
        # Setup multi-task configuration
        config = {
            "model_name": "test-claude-3",
            "tasks": ["causal_attribution", "counterfactual"],
            "num_questions_per_task": 1
        }
        
        # Prepare test data for multiple tasks
        test_data = {
            "causal_attribution": [sample_causal_question],
            "counterfactual": [sample_counterfactual_question]
        }
        
        # Mock multi-task evaluation
        with patch("causal_eval.evaluation.BenchmarkRunner") as mock_runner:
            mock_instance = MagicMock()
            mock_instance.run_evaluation.return_value = {
                "overall_score": 0.825,
                "task_scores": {
                    "causal_attribution": 0.9,
                    "counterfactual": 0.75
                },
                "detailed_results": {
                    "causal_attribution": [{"score": 0.9, "question_id": "test_question_001"}],
                    "counterfactual": [{"score": 0.75, "question_id": "test_question_002"}]
                },
                "execution_summary": {
                    "total_time": 45.2,
                    "questions_processed": 2,
                    "avg_time_per_question": 22.6
                }
            }
            mock_runner.return_value = mock_instance
            
            # Execute multi-task evaluation
            runner = mock_runner(config)
            results = runner.run_evaluation(test_data)
            
            # Verify multi-task results
            assert results["overall_score"] == 0.825
            assert "causal_attribution" in results["task_scores"]
            assert "counterfactual" in results["task_scores"]
            assert results["execution_summary"]["questions_processed"] == 2


@pytest.mark.e2e
class TestAPIWorkflow:
    """End-to-end tests for API workflows."""
    
    def test_evaluation_api_endpoint(self, mock_file_system):
        """Test the evaluation API endpoint workflow."""
        
        # Mock API request payload
        api_request = {
            "model_config": {
                "provider": "openai",
                "model_name": "gpt-4",
                "api_key": "test-key"
            },
            "evaluation_config": {
                "tasks": ["causal_attribution"],
                "num_questions": 5,
                "domains": ["medical", "social"],
                "difficulty": "medium"
            },
            "output_config": {
                "format": "json",
                "include_explanations": True,
                "save_responses": True
            }
        }
        
        # Mock API response
        with patch("causal_eval.api.routes.evaluation.run_evaluation") as mock_run_eval:
            mock_run_eval.return_value = {
                "evaluation_id": "eval_123456",
                "status": "completed",
                "results": {
                    "overall_score": 0.82,
                    "task_results": [
                        {
                            "task_type": "causal_attribution",
                            "score": 0.82,
                            "questions_answered": 5,
                            "avg_confidence": 0.87
                        }
                    ]
                },
                "metadata": {
                    "model_used": "gpt-4",
                    "total_time": 67.3,
                    "timestamp": "2025-01-27T12:00:00Z"
                }
            }
            
            # Simulate API call
            response = mock_run_eval(api_request)
            
            # Verify API response structure
            assert "evaluation_id" in response
            assert response["status"] == "completed"
            assert response["results"]["overall_score"] == 0.82
            assert len(response["results"]["task_results"]) == 1
    
    def test_leaderboard_submission_workflow(self, mock_file_system):
        """Test the leaderboard submission workflow."""
        
        # Mock evaluation results for submission
        submission_data = {
            "model_info": {
                "name": "Custom-Model-v1",
                "version": "1.0.0",
                "description": "Custom causal reasoning model",
                "parameters": "7B",
                "training_data": "Custom dataset"
            },
            "evaluation_results": {
                "overall_score": 0.91,
                "task_scores": {
                    "causal_attribution": 0.93,
                    "counterfactual": 0.89,
                    "intervention": 0.91
                },
                "benchmark_version": "0.1.0",
                "evaluation_date": "2025-01-27"
            },
            "verification": {
                "checksum": "abc123def456",
                "evaluation_id": "eval_789012"
            }
        }
        
        # Mock leaderboard submission
        with patch("causal_eval.api.routes.leaderboard.submit_results") as mock_submit:
            mock_submit.return_value = {
                "submission_id": "sub_345678",
                "status": "accepted",
                "leaderboard_position": 3,
                "verification_status": "verified",
                "public_url": "https://leaderboard.causal-eval.org/models/custom-model-v1"
            }
            
            # Submit to leaderboard
            response = mock_submit(submission_data)
            
            # Verify submission response
            assert response["status"] == "accepted"
            assert response["leaderboard_position"] == 3
            assert "public_url" in response


@pytest.mark.e2e
class TestReportGeneration:
    """End-to-end tests for report generation."""
    
    def test_comprehensive_report_generation(self, mock_file_system):
        """Test generation of comprehensive evaluation reports."""
        
        # Mock comprehensive evaluation data
        evaluation_data = {
            "model_info": {
                "name": "test-model",
                "provider": "openai",
                "version": "gpt-4-0125-preview"
            },
            "evaluation_summary": {
                "overall_score": 0.847,
                "total_questions": 50,
                "evaluation_time": 312.5,
                "success_rate": 0.98
            },
            "task_breakdown": {
                "causal_attribution": {"score": 0.89, "questions": 15},
                "counterfactual": {"score": 0.81, "questions": 15},
                "intervention": {"score": 0.84, "questions": 10},
                "confounding": {"score": 0.85, "questions": 10}
            },
            "domain_analysis": {
                "medical": {"score": 0.92, "questions": 12},
                "social": {"score": 0.83, "questions": 13},
                "economic": {"score": 0.79, "questions": 12},
                "scientific": {"score": 0.88, "questions": 13}
            },
            "error_analysis": {
                "common_errors": [
                    "Confusing correlation with causation",
                    "Missing confounding variables",
                    "Incorrect counterfactual reasoning"
                ],
                "improvement_suggestions": [
                    "Strengthen training on causal inference",
                    "Add more examples of spurious correlations",
                    "Improve counterfactual reasoning capabilities"
                ]
            }
        }
        
        # Mock report generation
        with patch("causal_eval.analysis.ReportGenerator") as mock_generator:
            mock_instance = MagicMock()
            
            # Mock different report formats
            mock_instance.generate_json_report.return_value = json.dumps(evaluation_data, indent=2)
            mock_instance.generate_pdf_report.return_value = b"Mock PDF content"
            mock_instance.generate_html_report.return_value = "<html><body>Mock HTML Report</body></html>"
            
            mock_generator.return_value = mock_instance
            
            # Generate reports
            generator = mock_generator(evaluation_data)
            
            # Test JSON report
            json_report = generator.generate_json_report()
            json_data = json.loads(json_report)
            assert json_data["evaluation_summary"]["overall_score"] == 0.847
            
            # Test PDF report
            pdf_report = generator.generate_pdf_report()
            assert isinstance(pdf_report, bytes)
            assert len(pdf_report) > 0
            
            # Test HTML report
            html_report = generator.generate_html_report()
            assert html_report.startswith("<html>")
            assert "Mock HTML Report" in html_report
        
        # Verify report file creation
        reports_dir = mock_file_system / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        # Save reports
        (reports_dir / "evaluation_report.json").write_text(json.dumps(evaluation_data, indent=2))
        (reports_dir / "evaluation_report.html").write_text("<html><body>Test Report</body></html>")
        
        # Verify files exist
        assert (reports_dir / "evaluation_report.json").exists()
        assert (reports_dir / "evaluation_report.html").exists()


@pytest.mark.e2e
@pytest.mark.slow
class TestLongRunningWorkflows:
    """End-to-end tests for long-running workflows."""
    
    def test_large_scale_evaluation(self, performance_test_data):
        """Test large-scale evaluation workflow."""
        
        # Configure large-scale evaluation
        config = {
            "model_name": "test-model",
            "batch_size": 10,
            "parallel_workers": 2,
            "total_questions": len(performance_test_data),
            "timeout_per_question": 30
        }
        
        # Mock large-scale evaluation
        with patch("causal_eval.evaluation.ParallelEvaluator") as mock_evaluator:
            mock_instance = MagicMock()
            mock_instance.evaluate_large_dataset.return_value = {
                "total_processed": config["total_questions"],
                "successful": config["total_questions"] - 2,  # 2 failures
                "failed": 2,
                "overall_score": 0.834,
                "processing_time": 245.7,
                "throughput": config["total_questions"] / 245.7,
                "batches_processed": (config["total_questions"] + config["batch_size"] - 1) // config["batch_size"]
            }
            mock_evaluator.return_value = mock_instance
            
            # Execute large-scale evaluation
            evaluator = mock_evaluator(config)
            results = evaluator.evaluate_large_dataset(performance_test_data)
            
            # Verify large-scale results
            assert results["total_processed"] == len(performance_test_data)
            assert results["successful"] > results["failed"]
            assert results["throughput"] > 0
            assert results["batches_processed"] == 10  # 100 questions / 10 batch_size
    
    def test_continuous_evaluation_simulation(self, mock_file_system):
        """Test continuous evaluation simulation."""
        import time
        
        # Simulate continuous evaluation over time
        evaluation_rounds = 3
        results_history = []
        
        for round_num in range(evaluation_rounds):
            # Simulate evaluation round
            round_result = {
                "round": round_num + 1,
                "timestamp": f"2025-01-27T12:{round_num:02d}:00Z",
                "score": 0.80 + (round_num * 0.02),  # Slight improvement over time
                "questions_evaluated": 20,
                "processing_time": 45.0 + (round_num * 2.0)
            }
            
            results_history.append(round_result)
            
            # Small delay to simulate time passing
            time.sleep(0.01)
        
        # Verify continuous evaluation results
        assert len(results_history) == evaluation_rounds
        assert results_history[0]["score"] < results_history[-1]["score"]  # Performance improvement
        
        # Save continuous evaluation history
        history_file = mock_file_system / "continuous_evaluation_history.json"
        history_file.write_text(json.dumps(results_history, indent=2))
        
        # Verify history persistence
        loaded_history = json.loads(history_file.read_text())
        assert len(loaded_history) == evaluation_rounds
        assert loaded_history[0]["round"] == 1
        assert loaded_history[-1]["round"] == evaluation_rounds