"""
Integration tests for evaluation API endpoints.
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient

from causal_eval.api.app import create_app


class TestEvaluationAPI:
    """Integration tests for evaluation endpoints."""
    
    @pytest.fixture
    def app(self):
        """Create FastAPI app for testing."""
        return create_app()
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)
    
    @pytest_asyncio.fixture
    async def async_client(self, app):
        """Create async test client."""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac
    
    def test_get_task_types(self, client):
        """Test getting available task types."""
        response = client.get("/evaluation/tasks/types")
        
        assert response.status_code == 200
        task_types = response.json()
        
        assert isinstance(task_types, list)
        assert "attribution" in task_types
        assert "counterfactual" in task_types
        assert "intervention" in task_types
    
    def test_get_domains(self, client):
        """Test getting available domains."""
        response = client.get("/evaluation/tasks/domains")
        
        assert response.status_code == 200
        domains = response.json()
        
        assert isinstance(domains, list)
        assert "general" in domains
        assert "medical" in domains
        assert "education" in domains
        assert "business" in domains
    
    def test_get_difficulties(self, client):
        """Test getting available difficulty levels."""
        response = client.get("/evaluation/tasks/difficulties")
        
        assert response.status_code == 200
        difficulties = response.json()
        
        assert isinstance(difficulties, list)
        assert difficulties == ["easy", "medium", "hard"]
    
    def test_generate_task_prompt(self, client):
        """Test generating task prompt."""
        request_data = {
            "task_type": "attribution",
            "domain": "medical",
            "difficulty": "medium"
        }
        
        response = client.post("/evaluation/prompts", json=request_data)
        
        assert response.status_code == 200
        result = response.json()
        
        assert result["task_type"] == "attribution"
        assert result["domain"] == "medical"
        assert result["difficulty"] == "medium"
        assert "prompt" in result
        assert len(result["prompt"]) > 0
        assert "generated_at" in result
        assert "prompt_length" in result
    
    def test_generate_task_prompt_invalid_type(self, client):
        """Test generating prompt with invalid task type."""
        request_data = {
            "task_type": "invalid_task",
            "domain": "general",
            "difficulty": "easy"
        }
        
        response = client.post("/evaluation/prompts", json=request_data)
        
        assert response.status_code == 500
        assert "Prompt generation failed" in response.json()["detail"]
    
    def test_evaluate_single_attribution(self, client):
        """Test single evaluation for attribution task."""
        request_data = {
            "task_type": "attribution",
            "model_response": """
            Relationship Type: spurious
            Confidence Level: 0.85
            Reasoning: Both ice cream sales and swimming accidents increase in summer due to hot weather, which is a confounding variable.
            Potential Confounders: weather, temperature, seasonal factors
            """,
            "domain": "general",
            "difficulty": "easy",
            "model_name": "test-model-v1"
        }
        
        response = client.post("/evaluation/evaluate", json=request_data)
        
        assert response.status_code == 200
        result = response.json()
        
        assert "evaluation_id" in result
        assert "overall_score" in result
        assert "detailed_scores" in result
        assert "confidence" in result
        assert result["task_type"] == "attribution"
        assert result["domain"] == "general"
        assert result["difficulty"] == "easy"
        assert "evaluation_metadata" in result
        assert result["evaluation_metadata"]["model_name"] == "test-model-v1"
    
    def test_evaluate_single_counterfactual(self, client):
        """Test single evaluation for counterfactual task."""
        request_data = {
            "task_type": "counterfactual", 
            "model_response": """
            Predicted Outcome: The student would likely score much lower, probably around 60-70%
            Confidence Level: 0.9
            Reasoning: Study time is a major causal factor in exam performance. Without studying, the student would lack necessary knowledge and preparation.
            Causal Chain: No studying -> No knowledge acquisition -> Poor exam performance
            Key Assumptions: The exam tests material that requires study to learn
            """,
            "domain": "education",
            "difficulty": "medium"
        }
        
        response = client.post("/evaluation/evaluate", json=request_data)
        
        assert response.status_code == 200
        result = response.json()
        
        assert result["task_type"] == "counterfactual"
        assert result["domain"] == "education"
        assert result["difficulty"] == "medium"
        assert result["overall_score"] > 0  # Should get some score for reasonable response
    
    def test_evaluate_single_intervention(self, client):
        """Test single evaluation for intervention task."""
        request_data = {
            "task_type": "intervention",
            "model_response": """
            Predicted Effect: decrease
            Effect Magnitude: 2-4°C decrease
            Time Frame: 15-30 minutes
            Confidence Level: 0.8
            Reasoning: The thermostat directly controls the heating system, so setting it lower will reduce the room temperature through the causal mechanism of reduced heating.
            Potential Side Effects: lower energy consumption, potential discomfort
            Key Assumptions: The heating system is functional and responds to thermostat changes
            """,
            "domain": "home_automation",
            "difficulty": "easy"
        }
        
        response = client.post("/evaluation/evaluate", json=request_data)
        
        assert response.status_code == 200
        result = response.json()
        
        assert result["task_type"] == "intervention"
        assert result["domain"] == "home_automation"
        assert result["overall_score"] > 0
    
    def test_evaluate_batch(self, client):
        """Test batch evaluation."""
        request_data = {
            "evaluations": [
                {
                    "task_type": "attribution",
                    "model_response": "This is a spurious relationship due to confounding factors.",
                    "domain": "general",
                    "difficulty": "easy",
                    "model_name": "test-batch-model"
                },
                {
                    "task_type": "counterfactual",
                    "model_response": "If the student hadn't studied, they would likely score lower.",
                    "domain": "education",
                    "difficulty": "medium",
                    "model_name": "test-batch-model"
                }
            ],
            "session_config": {
                "batch_test": True,
                "timestamp": "2024-01-01T00:00:00Z"
            }
        }
        
        response = client.post("/evaluation/batch", json=request_data)
        
        assert response.status_code == 200
        result = response.json()
        
        assert "session_id" in result
        assert result["total_evaluations"] == 2
        assert result["successful_evaluations"] <= 2
        assert result["failed_evaluations"] >= 0
        assert "aggregate_score" in result
        assert "task_type_scores" in result
        assert "detailed_results" in result
        assert "evaluation_metadata" in result
    
    def test_create_evaluation_session(self, client):
        """Test creating evaluation session."""
        request_data = {
            "model_name": "test-model-v2",
            "model_version": "2.0.1",
            "config": {
                "evaluation_type": "comprehensive",
                "batch_size": 50
            }
        }
        
        response = client.post("/evaluation/sessions", json=request_data)
        
        assert response.status_code == 200
        result = response.json()
        
        assert "session_id" in result
        assert "id" in result
        assert result["model_name"] == "test-model-v2"
        assert result["model_version"] == "2.0.1"
        assert result["status"] == "running"
        assert "created_at" in result
        assert result["config"]["evaluation_type"] == "comprehensive"
    
    def test_get_evaluation_session(self, client):
        """Test getting evaluation session details."""
        # First create a session
        create_request = {
            "model_name": "test-model-v3",
            "config": {"test": True}
        }
        
        create_response = client.post("/evaluation/sessions", json=create_request)
        assert create_response.status_code == 200
        session_id = create_response.json()["session_id"]
        
        # Then get session details
        response = client.get(f"/evaluation/sessions/{session_id}")
        
        assert response.status_code == 200
        result = response.json()
        
        assert "session_info" in result
        assert "task_statistics" in result
        assert "evaluation_statistics" in result
        assert "task_type_breakdown" in result
        assert "domain_breakdown" in result
    
    def test_get_evaluation_session_not_found(self, client):
        """Test getting non-existent evaluation session."""
        response = client.get("/evaluation/sessions/non-existent-session-id")
        
        assert response.status_code == 404
        assert "Session not found" in response.json()["detail"]
    
    def test_get_metrics(self, client):
        """Test getting evaluation metrics."""
        response = client.get("/evaluation/metrics")
        
        assert response.status_code == 200
        result = response.json()
        
        assert "basic_metrics" in result
        assert "aggregate_metrics" in result
        assert "causal_reasoning_profile" in result
        assert "collection_timestamp" in result
        
        # Check basic metrics structure
        basic_metrics = result["basic_metrics"]
        assert "total_evaluations" in basic_metrics
        assert "average_score" in basic_metrics
        assert "domain_breakdown" in basic_metrics
        
        # Check aggregate metrics structure
        aggregate_metrics = result["aggregate_metrics"]
        assert "overall_score" in aggregate_metrics
        assert "confidence_analysis" in aggregate_metrics
        assert "error_analysis" in aggregate_metrics
    
    def test_invalid_task_type(self, client):
        """Test evaluation with invalid task type."""
        request_data = {
            "task_type": "invalid_task_type",
            "model_response": "Some response",
            "domain": "general",
            "difficulty": "easy"
        }
        
        response = client.post("/evaluation/evaluate", json=request_data)
        
        assert response.status_code == 500
        assert "Evaluation failed" in response.json()["detail"]
    
    def test_missing_required_fields(self, client):
        """Test evaluation with missing required fields."""
        request_data = {
            "model_response": "Some response"
            # Missing task_type
        }
        
        response = client.post("/evaluation/evaluate", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_empty_model_response(self, client):
        """Test evaluation with empty model response."""
        request_data = {
            "task_type": "attribution",
            "model_response": "",
            "domain": "general",
            "difficulty": "easy"
        }
        
        response = client.post("/evaluation/evaluate", json=request_data)
        
        assert response.status_code == 200
        result = response.json()
        
        # Should still process but likely get low score
        assert result["overall_score"] >= 0
    
    def test_batch_evaluation_empty_list(self, client):
        """Test batch evaluation with empty list."""
        request_data = {
            "evaluations": [],
            "session_config": {}
        }
        
        response = client.post("/evaluation/batch", json=request_data)
        
        assert response.status_code == 200
        result = response.json()
        
        assert result["total_evaluations"] == 0
        assert result["successful_evaluations"] == 0
        assert result["failed_evaluations"] == 0
        assert result["aggregate_score"] == 0.0
    
    def test_concurrent_evaluations(self, client):
        """Test concurrent evaluation requests."""
        import concurrent.futures
        
        def make_evaluation_request():
            request_data = {
                "task_type": "attribution",
                "model_response": "Test response for concurrent evaluation",
                "domain": "general",
                "difficulty": "easy"
            }
            return client.post("/evaluation/evaluate", json=request_data)
        
        # Submit multiple concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_evaluation_request) for _ in range(5)]
            responses = [future.result() for future in futures]
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200
            result = response.json()
            assert "overall_score" in result
    
    @pytest.mark.asyncio
    async def test_async_evaluation_performance(self, async_client):
        """Test async evaluation performance."""
        import time
        
        request_data = {
            "task_type": "attribution",
            "model_response": "This is a performance test response.",
            "domain": "general",
            "difficulty": "easy"
        }
        
        start_time = time.time()
        response = await async_client.post("/evaluation/evaluate", json=request_data)
        end_time = time.time()
        
        assert response.status_code == 200
        
        # Should complete within reasonable time (adjust threshold as needed)
        execution_time = end_time - start_time
        assert execution_time < 5.0  # 5 seconds max