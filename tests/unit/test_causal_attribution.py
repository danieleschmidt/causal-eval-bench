"""
Unit tests for causal attribution task implementation.
"""

import pytest
import pytest_asyncio
from unittest.mock import Mock, patch

from causal_eval.tasks.attribution import CausalAttribution, CausalScenario, AttributionResponse
from causal_eval.core.tasks import TaskConfig


class TestCausalAttribution:
    """Test cases for CausalAttribution task."""
    
    @pytest.fixture
    def task_config(self):
        """Create task configuration for testing."""
        return TaskConfig(
            task_id="test_attribution",
            domain="general",
            difficulty="medium",
            description="Test causal attribution task",
            expected_reasoning_type="attribution"
        )
    
    @pytest.fixture
    def attribution_task(self, task_config):
        """Create CausalAttribution instance for testing."""
        return CausalAttribution(task_config)
    
    @pytest_asyncio.fixture
    async def sample_scenario(self):
        """Sample causal scenario for testing."""
        return CausalScenario(
            context="During summer months, both ice cream sales and swimming pool accidents increase.",
            variable_a="ice cream sales",
            variable_b="swimming pool accidents",
            actual_relationship="spurious",
            confounders=["summer weather", "outdoor activity"],
            domain="recreational",
            explanation="Both variables are caused by a third factor (warm weather) but don't cause each other."
        )
    
    @pytest.mark.asyncio
    async def test_generate_prompt(self, attribution_task):
        """Test prompt generation."""
        prompt = await attribution_task.generate_prompt()
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "Variable A:" in prompt
        assert "Variable B:" in prompt
        assert "relationship" in prompt.lower()
        assert "causal" in prompt.lower()
    
    @pytest.mark.asyncio
    async def test_generate_prompt_creates_scenario(self, attribution_task):
        """Test that prompt generation creates a current scenario."""
        await attribution_task.generate_prompt()
        
        assert hasattr(attribution_task, '_current_scenario')
        assert attribution_task._current_scenario is not None
        assert isinstance(attribution_task._current_scenario, CausalScenario)
    
    @pytest.mark.asyncio
    async def test_evaluate_response_correct_spurious(self, attribution_task):
        """Test evaluation with correct spurious relationship identification."""
        # Set up scenario
        attribution_task._current_scenario = CausalScenario(
            context="Test context",
            variable_a="Variable A",
            variable_b="Variable B",
            actual_relationship="spurious",
            confounders=["weather", "season"],
            domain="test"
        )
        
        # Model response identifying spurious relationship
        response = """
        Relationship Type: spurious
        Confidence Level: 0.9
        Reasoning: Both variables are caused by a third factor (weather) but don't directly cause each other.
        Potential Confounders: weather, seasonal factors
        """
        
        result = await attribution_task.evaluate_response(response)
        
        assert result["overall_score"] > 0.8
        assert result["relationship_score"] == 1.0
        assert result["predicted_relationship"] == "spurious"
        assert result["expected_relationship"] == "spurious"
        assert result["confidence"] == 0.9
    
    @pytest.mark.asyncio
    async def test_evaluate_response_incorrect_relationship(self, attribution_task):
        """Test evaluation with incorrect relationship identification."""
        # Set up scenario
        attribution_task._current_scenario = CausalScenario(
            context="Test context",
            variable_a="Variable A", 
            variable_b="Variable B",
            actual_relationship="spurious",
            confounders=["weather"],
            domain="test"
        )
        
        # Model response with wrong relationship type
        response = """
        Relationship Type: causal
        Confidence Level: 0.7
        Reasoning: Variable A directly causes Variable B.
        Potential Confounders: none identified
        """
        
        result = await attribution_task.evaluate_response(response)
        
        assert result["overall_score"] < 0.5
        assert result["relationship_score"] == 0.0
        assert result["predicted_relationship"] == "causal"
        assert result["expected_relationship"] == "spurious"
    
    @pytest.mark.asyncio
    async def test_evaluate_response_partial_credit(self, attribution_task):
        """Test evaluation with partial credit for related concepts."""
        # Set up scenario
        attribution_task._current_scenario = CausalScenario(
            context="Test context",
            variable_a="Variable A",
            variable_b="Variable B", 
            actual_relationship="spurious",
            confounders=["weather"],
            domain="test"
        )
        
        # Model response with related but not exact match
        response = """
        Relationship Type: correlation
        Confidence Level: 0.6
        Reasoning: The variables are correlated but may not have direct causation.
        Potential Confounders: environmental factors
        """
        
        result = await attribution_task.evaluate_response(response)
        
        assert 0.2 < result["relationship_score"] < 0.5
        assert result["predicted_relationship"] == "correlation"
    
    def test_parse_response_basic(self, attribution_task):
        """Test basic response parsing functionality."""
        response = """
        Relationship Type: causal
        Confidence Level: 0.85
        Reasoning: This is a clear causal relationship because...
        Potential Confounders: age, income, education
        """
        
        parsed = attribution_task._parse_response(response)
        
        assert isinstance(parsed, AttributionResponse)
        assert parsed.relationship_type == "causal"
        assert parsed.confidence == 0.85
        assert "clear causal relationship" in parsed.reasoning
        assert len(parsed.identified_confounders) == 3
        assert "age" in parsed.identified_confounders
    
    def test_parse_response_percentage_confidence(self, attribution_task):
        """Test parsing confidence as percentage."""
        response = """
        Relationship Type: spurious
        Confidence Level: 90%
        Reasoning: This appears to be spurious correlation.
        """
        
        parsed = attribution_task._parse_response(response)
        
        assert parsed.confidence == 0.9
    
    def test_score_relationship_type_exact_match(self, attribution_task):
        """Test relationship type scoring with exact match."""
        score = attribution_task._score_relationship_type("causal", "causal")
        assert score == 1.0
        
        score = attribution_task._score_relationship_type("spurious", "spurious")
        assert score == 1.0
    
    def test_score_relationship_type_partial_match(self, attribution_task):
        """Test relationship type scoring with partial matches."""
        # Related concepts get partial credit
        score = attribution_task._score_relationship_type("causal", "reverse_causal")
        assert score == 0.3
        
        score = attribution_task._score_relationship_type("correlation", "spurious")
        assert score == 0.3
    
    def test_score_relationship_type_no_match(self, attribution_task):
        """Test relationship type scoring with no match."""
        score = attribution_task._score_relationship_type("causal", "spurious")
        assert score == 0.0
    
    def test_score_reasoning_quality_good(self, attribution_task):
        """Test reasoning quality scoring with good reasoning."""
        reasoning = """
        This relationship shows causation because there is a clear mechanism
        where Variable A directly influences Variable B through confounding 
        factors like weather and temperature. The correlation is not spurious.
        """
        scenario = CausalScenario("", "", "", "causal", [], "medical")
        
        score = attribution_task._score_reasoning_quality(reasoning, scenario)
        
        assert score > 0.5
        # Should get points for key concepts: causation, correlation, confound, mechanism
    
    def test_score_reasoning_quality_poor(self, attribution_task):
        """Test reasoning quality scoring with poor reasoning."""
        reasoning = "Yes."
        scenario = CausalScenario("", "", "", "causal", [], "general")
        
        score = attribution_task._score_reasoning_quality(reasoning, scenario)
        
        assert score < 0.3
    
    def test_score_confounder_identification_perfect(self, attribution_task):
        """Test confounder identification scoring with perfect match."""
        identified = ["weather", "temperature", "season"]
        actual = ["weather", "seasonal factors", "temperature"]
        
        score = attribution_task._score_confounder_identification(identified, actual)
        
        assert score > 0.8  # Should get high score for good overlap
    
    def test_score_confounder_identification_no_match(self, attribution_task):
        """Test confounder identification scoring with no matches."""
        identified = ["completely", "wrong", "factors"]
        actual = ["weather", "temperature"]
        
        score = attribution_task._score_confounder_identification(identified, actual)
        
        assert score == 0.0
    
    def test_score_confounder_identification_none_expected(self, attribution_task):
        """Test confounder identification when none are expected."""
        identified = []
        actual = []
        
        score = attribution_task._score_confounder_identification(identified, actual)
        
        assert score == 1.0  # Perfect score when none expected and none identified
    
    def test_load_scenarios(self, attribution_task):
        """Test scenario loading functionality."""
        scenarios = attribution_task._load_scenarios()
        
        assert len(scenarios) > 0
        assert all(isinstance(s, CausalScenario) for s in scenarios)
        
        # Check that all relationship types are covered
        relationship_types = [s.actual_relationship for s in scenarios]
        assert "causal" in relationship_types
        assert "spurious" in relationship_types
        assert "reverse_causal" in relationship_types
    
    def test_domain_filtering(self, task_config):
        """Test domain-specific scenario filtering."""
        # Test with medical domain
        medical_config = TaskConfig(
            task_id="test_medical",
            domain="medical", 
            difficulty="medium",
            description="Medical attribution task",
            expected_reasoning_type="attribution"
        )
        
        medical_task = CausalAttribution(medical_config)
        scenarios = medical_task._load_scenarios()
        
        # Should only have medical domain scenarios
        medical_scenarios = [s for s in scenarios if s.domain == "medical"]
        assert len(medical_scenarios) > 0
        assert all(s.domain == "medical" for s in scenarios)
    
    @pytest.mark.asyncio
    async def test_no_current_scenario_error(self, attribution_task):
        """Test evaluation without current scenario returns error."""
        response = "Some response"
        
        result = await attribution_task.evaluate_response(response)
        
        assert "error" in result
        assert "No current scenario" in result["error"]
    
    @pytest.mark.asyncio
    async def test_multiple_prompt_generations(self, attribution_task):
        """Test that multiple prompt generations work correctly."""
        prompt1 = await attribution_task.generate_prompt()
        scenario1 = attribution_task._current_scenario
        
        prompt2 = await attribution_task.generate_prompt()
        scenario2 = attribution_task._current_scenario
        
        # Scenarios should be different (due to random selection)
        assert isinstance(prompt1, str)
        assert isinstance(prompt2, str)
        assert scenario1 is not scenario2