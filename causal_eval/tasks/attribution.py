"""
Causal Attribution Task: Test ability to distinguish causation from correlation.
"""

import random
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel
import re

from causal_eval.core.tasks import BaseTask, TaskConfig


@dataclass
class CausalScenario:
    """A causal scenario with variables and relationships."""
    
    context: str
    variable_a: str
    variable_b: str
    actual_relationship: str  # "causal", "correlation", "spurious", "reverse_causal"
    confounders: List[str] = field(default_factory=list)
    domain: str = "general"
    explanation: str = ""


class AttributionResponse(BaseModel):
    """Structured response for causal attribution task."""
    
    relationship_type: str  # "causal", "correlation", "spurious", "reverse_causal"
    confidence: float  # 0.0 to 1.0
    reasoning: str
    identified_confounders: List[str] = []


class CausalAttribution(BaseTask):
    """Task for evaluating causal attribution abilities."""
    
    def __init__(self, config: TaskConfig):
        super().__init__(config)
        self.scenarios = self._load_scenarios()
    
    def _load_scenarios(self) -> List[CausalScenario]:
        """Load predefined causal scenarios for testing."""
        scenarios = [
            CausalScenario(
                context="During summer months, both ice cream sales and swimming pool accidents increase.",
                variable_a="ice cream sales",
                variable_b="swimming pool accidents", 
                actual_relationship="spurious",
                confounders=["summer weather", "outdoor activity"],
                domain="recreational",
                explanation="Both variables are caused by a third factor (warm weather) but don't cause each other."
            ),
            CausalScenario(
                context="Students who study more hours tend to get better grades in their courses.",
                variable_a="study hours",
                variable_b="course grades",
                actual_relationship="causal",
                confounders=["student motivation", "prior knowledge"],
                domain="education",
                explanation="Study time directly impacts learning and comprehension, leading to better grades."
            ),
            CausalScenario(
                context="People who carry umbrellas are more likely to experience rain during their day.",
                variable_a="carrying umbrella",
                variable_b="experiencing rain",
                actual_relationship="reverse_causal", 
                confounders=["weather forecast"],
                domain="daily_life",
                explanation="People check weather forecasts and carry umbrellas when rain is predicted."
            ),
            CausalScenario(
                context="Hospitals with more doctors tend to have higher patient mortality rates.",
                variable_a="number of doctors",
                variable_b="patient mortality rate",
                actual_relationship="spurious",
                confounders=["hospital size", "patient severity", "specialty care"],
                domain="medical",
                explanation="Larger hospitals handle more severe cases, requiring more doctors and having higher complexity."
            ),
            CausalScenario(
                context="Regular exercise reduces the risk of cardiovascular disease in adults.",
                variable_a="regular exercise",
                variable_b="cardiovascular disease risk",
                actual_relationship="causal",
                confounders=["diet quality", "genetic factors", "stress levels"],
                domain="medical",
                explanation="Exercise directly improves cardiovascular health through multiple physiological mechanisms."
            ),
            CausalScenario(
                context="Cities with more fire stations tend to have more fires reported.",
                variable_a="number of fire stations",
                variable_b="number of fires reported",
                actual_relationship="reverse_causal",
                confounders=["city size", "population density"],
                domain="public_safety",
                explanation="Cities build more fire stations in response to higher fire frequency and risk."
            ),
            CausalScenario(
                context="Increased advertising spending leads to higher product sales revenue.",
                variable_a="advertising spending",
                variable_b="sales revenue",
                actual_relationship="causal",
                confounders=["product quality", "market competition", "economic conditions"],
                domain="business",
                explanation="Advertising increases brand awareness and customer demand, directly driving sales."
            ),
            CausalScenario(
                context="Countries with higher chocolate consumption have more Nobel Prize winners per capita.",
                variable_a="chocolate consumption",
                variable_b="Nobel Prize winners per capita",
                actual_relationship="spurious",
                confounders=["economic development", "education investment", "research funding"],
                domain="international",
                explanation="Wealthy countries can afford both chocolate and extensive education/research systems."
            )
        ]
        
        # Filter by domain if specified
        if hasattr(self.config, 'domain') and self.config.domain != "general":
            scenarios = [s for s in scenarios if s.domain == self.config.domain]
        
        return scenarios
    
    async def generate_prompt(self) -> str:
        """Generate a causal attribution prompt."""
        scenario = random.choice(self.scenarios)
        
        prompt = f"""
Analyze the following scenario and determine the nature of the relationship between the two variables.

Scenario: {scenario.context}

Variables:
- Variable A: {scenario.variable_a}
- Variable B: {scenario.variable_b}

Question: What is the relationship between Variable A and Variable B?

Please provide your analysis in the following format:
1. Relationship Type: [Choose: causal, correlation, spurious, reverse_causal]
2. Confidence Level: [0.0 to 1.0]
3. Reasoning: [Explain your analysis]
4. Potential Confounders: [List any confounding variables you identify]

Definitions:
- causal: A directly causes B
- correlation: A and B are correlated but no direct causation
- spurious: A and B appear related but are caused by a third factor
- reverse_causal: B causes A (reverse of what might be expected)
"""
        
        # Store scenario for evaluation
        self._current_scenario = scenario
        return prompt.strip()
    
    async def evaluate_response(self, response: str) -> Dict[str, Any]:
        """Evaluate the model's causal attribution response."""
        if not hasattr(self, '_current_scenario'):
            return {"error": "No current scenario available for evaluation"}
        
        scenario = self._current_scenario
        parsed_response = self._parse_response(response)
        
        # Calculate scores
        relationship_score = self._score_relationship_type(
            parsed_response.relationship_type, 
            scenario.actual_relationship
        )
        
        reasoning_score = self._score_reasoning_quality(
            parsed_response.reasoning,
            scenario
        )
        
        confounder_score = self._score_confounder_identification(
            parsed_response.identified_confounders,
            scenario.confounders
        )
        
        # Overall score (weighted average)
        overall_score = (
            relationship_score * 0.5 +
            reasoning_score * 0.3 + 
            confounder_score * 0.2
        )
        
        return {
            "overall_score": overall_score,
            "relationship_score": relationship_score,
            "reasoning_score": reasoning_score,
            "confounder_score": confounder_score,
            "expected_relationship": scenario.actual_relationship,
            "predicted_relationship": parsed_response.relationship_type,
            "confidence": parsed_response.confidence,
            "scenario_domain": scenario.domain,
            "correct_explanation": scenario.explanation,
            "model_reasoning": parsed_response.reasoning,
            "identified_confounders": parsed_response.identified_confounders,
            "actual_confounders": scenario.confounders
        }
    
    def _parse_response(self, response: str) -> AttributionResponse:
        """Parse the model's response into structured format."""
        # Extract relationship type - look for key terms in the response
        response_lower = response.lower()
        
        if "spurious" in response_lower:
            relationship_type = "spurious"
        elif "causal" in response_lower and "reverse" not in response_lower:
            relationship_type = "causal"
        elif "reverse" in response_lower or "reverse_causal" in response_lower:
            relationship_type = "reverse_causal"
        elif "correlation" in response_lower:
            relationship_type = "correlation"
        else:
            # Try structured format
            relationship_match = re.search(
                r"(?:Relationship Type|relationship):\s*(\w+)", 
                response, 
                re.IGNORECASE
            )
            relationship_type = relationship_match.group(1).lower() if relationship_match else "unknown"
        
        # Extract confidence
        confidence_match = re.search(
            r"(?:Confidence|confidence).*?(\d+\.?\d*)", 
            response
        )
        confidence = float(confidence_match.group(1)) if confidence_match else 0.5
        if confidence > 1.0:
            confidence = confidence / 100.0  # Convert percentage to decimal
        
        # Extract reasoning - use the entire response if no structured format
        reasoning_match = re.search(
            r"(?:Reasoning|reasoning):\s*(.+?)(?:\n\d+\.|$)", 
            response, 
            re.IGNORECASE | re.DOTALL
        )
        reasoning = reasoning_match.group(1).strip() if reasoning_match else response.strip()
        
        # Extract confounders - look for weather, temperature, seasonal terms
        confounder_match = re.search(
            r"(?:Confounders?|confounding).*?:\s*(.+?)(?:\n|$)", 
            response, 
            re.IGNORECASE
        )
        confounders_text = confounder_match.group(1) if confounder_match else response
        
        # Look for common confounder terms in the text
        confounders = []
        confounder_terms = ["weather", "temperature", "warm", "hot", "summer", "seasonal", "season"]
        for term in confounder_terms:
            if term in response_lower:
                confounders.append(term)
        
        # Also parse structured list if present
        if confounder_match:
            structured_confounders = [
                c.strip().strip("[]()\"'") 
                for c in re.split(r"[,;]|\band\b", confounders_text)
                if c.strip()
            ]
            confounders.extend(structured_confounders)
        
        return AttributionResponse(
            relationship_type=relationship_type,
            confidence=confidence,
            reasoning=reasoning,
            identified_confounders=confounders
        )
    
    def _score_relationship_type(self, predicted: str, actual: str) -> float:
        """Score the accuracy of relationship type prediction."""
        if predicted.lower() == actual.lower():
            return 1.0
        
        # Partial credit for related concepts
        similar_pairs = [
            ("causal", "reverse_causal"),
            ("correlation", "spurious"),
        ]
        
        for pair in similar_pairs:
            if (predicted.lower() in pair and actual.lower() in pair):
                return 0.3
        
        return 0.0
    
    def _score_reasoning_quality(self, reasoning: str, scenario: CausalScenario) -> float:
        """Score the quality of causal reasoning."""
        if not reasoning:
            return 0.0
        
        score = 0.0
        reasoning_lower = reasoning.lower()
        
        # Check for key concepts
        key_concepts = {
            "causation": 0.2,
            "correlation": 0.2, 
            "confound": 0.2,
            "third factor": 0.15,
            "spurious": 0.15,
            "mechanism": 0.1
        }
        
        for concept, weight in key_concepts.items():
            if concept in reasoning_lower:
                score += weight
        
        # Check if reasoning mentions domain-specific factors
        if scenario.domain == "medical" and any(term in reasoning_lower for term in ["health", "disease", "treatment"]):
            score += 0.1
        elif scenario.domain == "education" and any(term in reasoning_lower for term in ["learning", "knowledge", "study"]):
            score += 0.1
        
        # Check reasoning length (adequate explanation)
        if len(reasoning.split()) >= 10:
            score += 0.1
        
        return min(score, 1.0)
    
    def _score_confounder_identification(self, identified: List[str], actual: List[str]) -> float:
        """Score the identification of confounding variables."""
        if not actual:
            return 1.0 if not identified else 0.8  # No confounders expected
        
        if not identified:
            return 0.0
        
        # Calculate overlap
        identified_lower = [c.lower() for c in identified]
        actual_lower = [c.lower() for c in actual]
        
        correct_identified = 0
        for actual_conf in actual_lower:
            for identified_conf in identified_lower:
                if actual_conf in identified_conf or identified_conf in actual_conf:
                    correct_identified += 1
                    break
        
        precision = correct_identified / len(identified) if identified else 0
        recall = correct_identified / len(actual) if actual else 0
        
        # F1 score
        if precision + recall > 0:
            return 2 * (precision * recall) / (precision + recall)
        return 0.0