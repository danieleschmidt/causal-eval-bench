"""
Unit tests for the evaluation engine.
"""

import pytest
import pytest_asyncio
from unittest.mock import Mock, patch, AsyncMock

from causal_eval.core.engine import EvaluationEngine, EvaluationResult, CausalEvaluationRequest
from causal_eval.core.tasks import TaskConfig


class TestEvaluationEngine:
    """Test cases for EvaluationEngine."""
    
    @pytest.fixture
    def engine(self):
        """Create evaluation engine for testing."""
        return EvaluationEngine()
    
    @pytest.fixture
    def sample_task_config(self):
        """Sample task configuration."""
        return {
            "task_type": "attribution",
            "domain": "medical",
            "difficulty": "medium",
            "task_id": "test_task_001"
        }
    
    @pytest.fixture
    def sample_evaluation_request(self):
        """Sample evaluation request."""
        return CausalEvaluationRequest(
            task_type="attribution",
            model_response="The relationship is spurious because both variables are caused by weather.",
            domain="general",
            difficulty="easy",
            task_id="test_request_001"
        )
    
    def test_initialization(self, engine):
        """Test engine initialization."""
        assert isinstance(engine, EvaluationEngine)
        assert engine.config == {}
        assert engine._task_cache == {}
        assert engine.task_registry is None
    
    def test_initialization_with_config(self):
        """Test engine initialization with configuration."""
        config = {"setting1": "value1", "setting2": "value2"}
        engine = EvaluationEngine(config)
        
        assert engine.config == config
    
    def test_create_task_attribution(self, engine):
        """Test creating attribution task."""
        task = engine._create_task("attribution", "medical", "hard")
        
        assert task is not None
        assert task.config.task_id == "attribution_medical_hard"
        assert task.config.domain == "medical"
        assert task.config.difficulty == "hard"
        assert task.config.expected_reasoning_type == "attribution"
    
    def test_create_task_counterfactual(self, engine):
        """Test creating counterfactual task."""
        task = engine._create_task("counterfactual", "education", "easy")
        
        assert task is not None
        assert task.config.task_id == "counterfactual_education_easy"
        assert task.config.domain == "education"
        assert task.config.difficulty == "easy"
    
    def test_create_task_intervention(self, engine):
        """Test creating intervention task."""
        task = engine._create_task("intervention", "business", "medium")
        
        assert task is not None
        assert task.config.task_id == "intervention_business_medium"
        assert task.config.domain == "business"
        assert task.config.difficulty == "medium"
    
    def test_create_task_invalid_type(self, engine):
        """Test creating task with invalid type raises error."""
        with pytest.raises(ValueError, match="Unknown task type"):
            engine._create_task("invalid_task", "general", "medium")
    
    @pytest.mark.asyncio
    async def test_evaluate_basic(self, engine, sample_task_config):
        """Test basic evaluation functionality."""
        model_response = "This is a spurious relationship due to confounding variables."
        
        with patch.object(engine, '_create_task') as mock_create:
            # Mock task behavior
            mock_task = Mock()
            mock_task.generate_prompt = AsyncMock(return_value="Test prompt")
            mock_task.evaluate_response = AsyncMock(return_value={
                "overall_score": 0.85,
                "reasoning_score": 0.8,
                "relationship_score": 0.9,
                "confidence": 0.75
            })
            mock_create.return_value = mock_task
            
            result = await engine.evaluate(model_response, sample_task_config)
            
            assert isinstance(result, EvaluationResult)
            assert result.score == 0.85
            assert result.reasoning_quality == 0.8
            assert result.task_id == "test_task_001"
            assert result.domain == "medical"
            assert "attribution" in result.metadata["task_type"]
    
    @pytest.mark.asyncio
    async def test_evaluate_with_exception(self, engine, sample_task_config):
        """Test evaluation with exception handling."""
        model_response = "Test response"
        
        with patch.object(engine, '_create_task') as mock_create:
            mock_create.side_effect = Exception("Task creation failed")
            
            result = await engine.evaluate(model_response, sample_task_config)
            
            assert isinstance(result, EvaluationResult)
            assert result.score == 0.0
            assert result.reasoning_quality == 0.0
            assert "Evaluation failed" in result.explanation
            assert "error" in result.metadata
    
    @pytest.mark.asyncio
    async def test_evaluate_request(self, engine, sample_evaluation_request):
        """Test evaluate_request method."""
        with patch.object(engine, '_create_task') as mock_create:
            # Mock task behavior
            mock_task = Mock()
            mock_task.generate_prompt = AsyncMock(return_value="Generated prompt for testing")
            mock_task.evaluate_response = AsyncMock(return_value={
                "overall_score": 0.92,
                "reasoning_score": 0.88,
                "relationship_score": 0.95,
                "confidence": 0.8,
                "domain": "general"
            })
            mock_create.return_value = mock_task
            
            result = await engine.evaluate_request(sample_evaluation_request)
            
            assert result["overall_score"] == 0.92
            assert result["task_type"] == "attribution"
            assert result["domain"] == "general"
            assert result["model_response"] == sample_evaluation_request.model_response
            assert "Generated prompt" in result["generated_prompt"]
    
    @pytest.mark.asyncio
    async def test_batch_evaluate(self, engine):
        """Test batch evaluation functionality."""
        evaluations = [
            {
                "model_response": "Response 1",
                "task_config": {"task_type": "attribution", "domain": "medical"}
            },
            {
                "model_response": "Response 2", 
                "task_config": {"task_type": "counterfactual", "domain": "education"}
            }
        ]
        
        with patch.object(engine, '_create_task') as mock_create:
            # Mock task behavior
            mock_task = Mock()
            mock_task.generate_prompt = AsyncMock(return_value="Test prompt")
            mock_task.evaluate_response = AsyncMock(return_value={
                "overall_score": 0.7,
                "reasoning_score": 0.6
            })
            mock_create.return_value = mock_task
            
            results = await engine.batch_evaluate(evaluations)
            
            assert len(results) == 2
            assert all(isinstance(r, EvaluationResult) for r in results)
            assert all(r.score == 0.7 for r in results)
    
    @pytest.mark.asyncio
    async def test_batch_evaluate_with_exception(self, engine):
        """Test batch evaluation with exception in one task."""
        evaluations = [
            {
                "model_response": "Good response",
                "task_config": {"task_type": "attribution", "domain": "medical"}
            },
            {
                "model_response": "Bad response",
                "task_config": {"task_type": "invalid", "domain": "education"}
            }
        ]
        
        with patch.object(engine, '_create_task') as mock_create:
            def side_effect(task_type, domain, difficulty):
                if task_type == "invalid":
                    raise ValueError("Invalid task type")
                
                mock_task = Mock()
                mock_task.generate_prompt = AsyncMock(return_value="Test prompt")
                mock_task.evaluate_response = AsyncMock(return_value={
                    "overall_score": 0.8,
                    "reasoning_score": 0.7
                })
                return mock_task
            
            mock_create.side_effect = side_effect
            
            results = await engine.batch_evaluate(evaluations)
            
            assert len(results) == 2
            assert results[0].score == 0.8  # Successful evaluation
            assert results[1].score == 0.0  # Failed evaluation
            assert "Batch evaluation failed" in results[1].explanation
    
    @pytest.mark.asyncio
    async def test_generate_task_prompt(self, engine):
        """Test task prompt generation."""
        with patch.object(engine, '_create_task') as mock_create:
            mock_task = Mock()
            mock_task.generate_prompt = AsyncMock(return_value="Generated test prompt")
            mock_create.return_value = mock_task
            
            prompt = await engine.generate_task_prompt("attribution", "medical", "hard")
            
            assert prompt == "Generated test prompt"
            mock_create.assert_called_once_with("attribution", "medical", "hard")
    
    def test_get_available_task_types(self, engine):
        """Test getting available task types."""
        task_types = engine.get_available_task_types()
        
        expected_types = ["attribution", "counterfactual", "intervention"]
        assert task_types == expected_types
    
    def test_get_available_domains(self, engine):
        """Test getting available domains."""
        domains = engine.get_available_domains()
        
        expected_domains = [
            "general", "medical", "education", "business", "technology",
            "environmental", "workplace_safety", "urban_planning", 
            "manufacturing", "recreational", "public_safety", "international"
        ]
        assert domains == expected_domains
    
    def test_get_available_difficulties(self, engine):
        """Test getting available difficulties."""
        difficulties = engine.get_available_difficulties()
        
        expected_difficulties = ["easy", "medium", "hard"]
        assert difficulties == expected_difficulties


class TestCausalEvaluationRequest:
    """Test cases for CausalEvaluationRequest model."""
    
    def test_basic_request(self):
        """Test basic request creation."""
        request = CausalEvaluationRequest(
            task_type="attribution",
            model_response="Test response"
        )
        
        assert request.task_type == "attribution"
        assert request.model_response == "Test response"
        assert request.domain == "general"  # Default
        assert request.difficulty == "medium"  # Default
        assert request.task_id is None  # Default
    
    def test_full_request(self):
        """Test request with all fields."""
        request = CausalEvaluationRequest(
            task_type="counterfactual",
            model_response="Detailed response",
            domain="medical",
            difficulty="hard",
            task_id="custom_task_001"
        )
        
        assert request.task_type == "counterfactual"
        assert request.model_response == "Detailed response"
        assert request.domain == "medical"
        assert request.difficulty == "hard"
        assert request.task_id == "custom_task_001"


class TestEvaluationResult:
    """Test cases for EvaluationResult model."""
    
    def test_basic_result(self):
        """Test basic result creation."""
        result = EvaluationResult(
            task_id="test_001",
            domain="medical",
            score=0.85,
            reasoning_quality=0.8,
            explanation="Good performance"
        )
        
        assert result.task_id == "test_001"
        assert result.domain == "medical"
        assert result.score == 0.85
        assert result.reasoning_quality == 0.8
        assert result.explanation == "Good performance"
        assert result.metadata == {}  # Default
    
    def test_result_with_metadata(self):
        """Test result with metadata."""
        metadata = {
            "task_type": "attribution",
            "confidence": 0.9,
            "evaluation_time": 1.5
        }
        
        result = EvaluationResult(
            task_id="test_002",
            domain="education",
            score=0.72,
            reasoning_quality=0.68,
            explanation="Moderate performance",
            metadata=metadata
        )
        
        assert result.metadata == metadata
        assert result.metadata["task_type"] == "attribution"
        assert result.metadata["confidence"] == 0.9