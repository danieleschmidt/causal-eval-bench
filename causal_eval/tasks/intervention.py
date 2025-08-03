"""
Causal Intervention Task: Test understanding of intervention effects.
"""

import random
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pydantic import BaseModel
import re

from causal_eval.core.tasks import BaseTask, TaskConfig


@dataclass 
class InterventionScenario:
    """A causal intervention scenario."""
    
    system_description: str
    baseline_state: Dict[str, Any]
    intervention: str
    target_variable: str
    expected_effect: str
    effect_magnitude: Optional[str] = None
    time_frame: str = "immediate"
    domain: str = "general"
    confounding_factors: List[str] = field(default_factory=list)
    side_effects: List[str] = field(default_factory=list)


class InterventionResponse(BaseModel):
    """Structured response for intervention analysis."""
    
    predicted_effect: str
    effect_magnitude: str
    confidence: float
    time_frame: str
    reasoning: str
    side_effects: List[str] = []
    assumptions: List[str] = []


class CausalIntervention(BaseTask):
    """Task for evaluating understanding of causal interventions."""
    
    def __init__(self, config: TaskConfig):
        super().__init__(config)
        self.scenarios = self._load_scenarios()
    
    def _load_scenarios(self) -> List[InterventionScenario]:
        """Load predefined intervention scenarios."""
        scenarios = [
            InterventionScenario(
                system_description="A smart thermostat controls room temperature by turning heating/cooling on and off based on temperature readings.",
                baseline_state={"room_temperature": "20°C", "thermostat_setting": "22°C", "heating": "on"},
                intervention="Manually override the thermostat and set it to 18°C",
                target_variable="room temperature",
                expected_effect="decrease",
                effect_magnitude="2-4°C decrease",
                time_frame="15-30 minutes",
                domain="home_automation",
                confounding_factors=["outside temperature", "insulation quality", "room size"],
                side_effects=["increased energy efficiency", "potential comfort reduction"]
            ),
            InterventionScenario(
                system_description="A social media platform uses an algorithm to determine which posts appear in users' feeds based on engagement metrics.",
                baseline_state={"average_session_time": "45 minutes", "posts_shown": "controversial content", "engagement": "high"},
                intervention="Modify the algorithm to prioritize posts from friends and family over viral content",
                target_variable="user engagement",
                expected_effect="decrease",
                effect_magnitude="15-25% reduction in session time",
                time_frame="1-2 weeks",
                domain="technology",
                confounding_factors=["user adaptation", "content quality", "external events"],
                side_effects=["improved user well-being", "reduced ad revenue", "better relationships"]
            ),
            InterventionScenario(
                system_description="A hospital emergency room has a triage system that prioritizes patients based on severity of condition.",
                baseline_state={"average_wait_time": "90 minutes", "patient_satisfaction": "3.2/5", "staff_efficiency": "75%"},
                intervention="Implement a new fast-track system for minor injuries and complaints",
                target_variable="average wait time",
                expected_effect="decrease",
                effect_magnitude="20-30 minute reduction",
                time_frame="immediate implementation",
                domain="healthcare",
                confounding_factors=["patient volume", "staff availability", "case complexity"],
                side_effects=["improved satisfaction for minor cases", "potential resource strain", "staff training requirements"]
            ),
            InterventionScenario(
                system_description="A city traffic system uses traffic lights with fixed timing patterns during rush hour.",
                baseline_state={"average_commute_time": "35 minutes", "traffic_flow": "stop-and-go", "intersection_delays": "high"},
                intervention="Install adaptive traffic lights that adjust timing based on real-time traffic flow",
                target_variable="average commute time",
                expected_effect="decrease",
                effect_magnitude="10-20% reduction",
                time_frame="3-6 months for full effect",
                domain="urban_planning",
                confounding_factors=["weather conditions", "special events", "road construction"],
                side_effects=["reduced emissions", "improved fuel efficiency", "initial adaptation period"]
            ),
            InterventionScenario(
                system_description="An e-commerce website shows product recommendations based on browsing history and purchase patterns.",
                baseline_state={"conversion_rate": "3.2%", "average_order_value": "$85", "customer_satisfaction": "4.1/5"},
                intervention="Replace algorithm-based recommendations with curated product collections by human experts",
                target_variable="conversion rate",
                expected_effect="mixed",
                effect_magnitude="potential 5-15% change in either direction",
                time_frame="2-4 weeks",
                domain="e_commerce",
                confounding_factors=["seasonal trends", "product availability", "expert selection quality"],
                side_effects=["more diverse product discovery", "higher curation costs", "potential bias reduction"]
            ),
            InterventionScenario(
                system_description="A manufacturing plant operates 24/7 with workers on rotating 8-hour shifts.",
                baseline_state={"production_output": "1000 units/day", "error_rate": "2.5%", "worker_fatigue": "moderate"},
                intervention="Switch to 12-hour shifts with 4 days on, 4 days off schedule",
                target_variable="production output",
                expected_effect="increase",
                effect_magnitude="8-15% increase",
                time_frame="4-8 weeks adaptation period",
                domain="manufacturing",
                confounding_factors=["worker adaptation", "equipment maintenance", "market demand"],
                side_effects=["improved work-life balance", "potential increased fatigue per shift", "reduced handoff errors"]
            ),
            InterventionScenario(
                system_description="A school district assigns students to classes randomly within their grade level.",
                baseline_state={"average_test_scores": "75%", "student_engagement": "moderate", "teacher_workload": "balanced"},
                intervention="Group students by academic ability level (tracking) instead of random assignment",
                target_variable="average test scores",
                expected_effect="mixed",
                effect_magnitude="high achievers +5-10%, low achievers -3-8%",
                time_frame="1 academic year",
                domain="education",
                confounding_factors=["teacher quality", "curriculum design", "peer effects"],
                side_effects=["potential increased inequality", "teacher specialization", "social stratification"]
            ),
            InterventionScenario(
                system_description="A company offers employees unlimited vacation days with manager approval.",
                baseline_state={"average_vacation_days": "18 days/year", "employee_satisfaction": "7.2/10", "productivity": "baseline"},
                intervention="Switch to a mandatory minimum of 25 vacation days with automatic approval",
                target_variable="employee productivity",
                expected_effect="increase",
                effect_magnitude="5-12% improvement",
                time_frame="6-12 months",
                domain="workplace",
                confounding_factors=["work culture", "project deadlines", "individual differences"],
                side_effects=["improved mental health", "better work-life balance", "increased staffing costs"]
            )
        ]
        
        # Filter by domain if specified
        if hasattr(self.config, 'domain') and self.config.domain != "general":
            scenarios = [s for s in scenarios if s.domain == self.config.domain]
        
        return scenarios
    
    async def generate_prompt(self) -> str:
        """Generate a causal intervention prompt."""
        scenario = random.choice(self.scenarios)
        
        # Format baseline state
        baseline_info = ""
        for key, value in scenario.baseline_state.items():
            baseline_info += f"- {key.replace('_', ' ').title()}: {value}\n"
        
        prompt = f"""
Analyze the following causal intervention scenario and predict its effects.

System Description: {scenario.system_description}

Current Baseline State:
{baseline_info.strip()}

Proposed Intervention: {scenario.intervention}

Target Variable: {scenario.target_variable}

Please provide your analysis in the following format:
1. Predicted Effect: [increase/decrease/no change/mixed effects]
2. Effect Magnitude: [quantify the expected change]
3. Time Frame: [when will the effect be observable]
4. Confidence Level: [0.0 to 1.0]
5. Reasoning: [explain the causal mechanism]
6. Potential Side Effects: [list other variables that might be affected]
7. Key Assumptions: [what assumptions are you making]

Consider:
- Direct causal pathways
- Indirect effects and feedback loops
- Time delays in the system
- Potential unintended consequences
- Factors that might interfere with the intervention
"""
        
        # Store scenario for evaluation
        self._current_scenario = scenario
        return prompt.strip()
    
    async def evaluate_response(self, response: str) -> Dict[str, Any]:
        """Evaluate the model's intervention analysis response."""
        if not hasattr(self, '_current_scenario'):
            return {"error": "No current scenario available for evaluation"}
        
        scenario = self._current_scenario
        parsed_response = self._parse_response(response)
        
        # Calculate scores
        effect_score = self._score_effect_prediction(
            parsed_response.predicted_effect,
            scenario.expected_effect
        )
        
        magnitude_score = self._score_magnitude_prediction(
            parsed_response.effect_magnitude,
            scenario.effect_magnitude
        )
        
        reasoning_score = self._score_reasoning_quality(
            parsed_response.reasoning,
            scenario
        )
        
        side_effects_score = self._score_side_effects(
            parsed_response.side_effects,
            scenario.side_effects
        )
        
        time_frame_score = self._score_time_frame(
            parsed_response.time_frame,
            scenario.time_frame
        )
        
        # Overall score (weighted average)
        overall_score = (
            effect_score * 0.3 +
            magnitude_score * 0.2 +
            reasoning_score * 0.25 +
            side_effects_score * 0.15 +
            time_frame_score * 0.1
        )
        
        return {
            "overall_score": overall_score,
            "effect_score": effect_score,
            "magnitude_score": magnitude_score,
            "reasoning_score": reasoning_score,
            "side_effects_score": side_effects_score,
            "time_frame_score": time_frame_score,
            "scenario_domain": scenario.domain,
            "predicted_effect": parsed_response.predicted_effect,
            "expected_effect": scenario.expected_effect,
            "confidence": parsed_response.confidence,
            "intervention": scenario.intervention,
            "target_variable": scenario.target_variable,
            "model_reasoning": parsed_response.reasoning,
            "identified_side_effects": parsed_response.side_effects,
            "expected_side_effects": scenario.side_effects,
            "model_assumptions": parsed_response.assumptions
        }
    
    def _parse_response(self, response: str) -> InterventionResponse:
        """Parse the model's response into structured format."""
        # Extract predicted effect
        effect_match = re.search(
            r"(?:Predicted Effect|effect):\s*([\w\s/]+?)(?:\n\d+\.|$)",
            response,
            re.IGNORECASE
        )
        predicted_effect = effect_match.group(1).strip() if effect_match else "unknown"
        
        # Extract effect magnitude
        magnitude_match = re.search(
            r"(?:Effect Magnitude|magnitude):\s*(.+?)(?:\n\d+\.|$)",
            response,
            re.IGNORECASE | re.DOTALL
        )
        effect_magnitude = magnitude_match.group(1).strip() if magnitude_match else ""
        
        # Extract time frame
        time_match = re.search(
            r"(?:Time Frame|time frame):\s*(.+?)(?:\n\d+\.|$)",
            response,
            re.IGNORECASE | re.DOTALL
        )
        time_frame = time_match.group(1).strip() if time_match else ""
        
        # Extract confidence
        confidence_match = re.search(
            r"(?:Confidence|confidence).*?(\d+\.?\d*)",
            response
        )
        confidence = float(confidence_match.group(1)) if confidence_match else 0.5
        if confidence > 1.0:
            confidence = confidence / 100.0
        
        # Extract reasoning
        reasoning_match = re.search(
            r"(?:Reasoning|reasoning):\s*(.+?)(?:\n\d+\.|$)",
            response,
            re.IGNORECASE | re.DOTALL
        )
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        
        # Extract side effects
        side_effects_match = re.search(
            r"(?:Side Effects|side effects):\s*(.+?)(?:\n\d+\.|$)",
            response,
            re.IGNORECASE | re.DOTALL
        )
        side_effects_text = side_effects_match.group(1) if side_effects_match else ""
        side_effects = [
            effect.strip().strip("[]()\"'-•") 
            for effect in re.split(r"[,;\n]|\d+\.", side_effects_text)
            if effect.strip()
        ]
        
        # Extract assumptions
        assumptions_match = re.search(
            r"(?:Assumptions|assumptions):\s*(.+?)(?:\n\d+\.|$)",
            response,
            re.IGNORECASE | re.DOTALL
        )
        assumptions_text = assumptions_match.group(1) if assumptions_match else ""
        assumptions = [
            assumption.strip().strip("[]()\"'-•") 
            for assumption in re.split(r"[,;\n]|\d+\.", assumptions_text)
            if assumption.strip()
        ]
        
        return InterventionResponse(
            predicted_effect=predicted_effect,
            effect_magnitude=effect_magnitude,
            confidence=confidence,
            time_frame=time_frame,
            reasoning=reasoning,
            side_effects=side_effects,
            assumptions=assumptions
        )
    
    def _score_effect_prediction(self, predicted: str, expected: str) -> float:
        """Score the accuracy of effect direction prediction."""
        predicted_lower = predicted.lower()
        expected_lower = expected.lower()
        
        # Direct match
        if expected_lower in predicted_lower:
            return 1.0
        
        # Check for directional correctness
        increase_terms = ["increase", "rise", "grow", "higher", "more", "up", "improve"]
        decrease_terms = ["decrease", "fall", "drop", "lower", "less", "down", "reduce"]
        mixed_terms = ["mixed", "complex", "varied", "both", "depends"]
        
        predicted_direction = None
        if any(term in predicted_lower for term in increase_terms):
            predicted_direction = "increase"
        elif any(term in predicted_lower for term in decrease_terms):
            predicted_direction = "decrease" 
        elif any(term in predicted_lower for term in mixed_terms):
            predicted_direction = "mixed"
        
        expected_direction = None
        if any(term in expected_lower for term in increase_terms):
            expected_direction = "increase"
        elif any(term in expected_lower for term in decrease_terms):
            expected_direction = "decrease"
        elif any(term in expected_lower for term in mixed_terms):
            expected_direction = "mixed"
        
        if predicted_direction == expected_direction:
            return 0.8
        
        return 0.0
    
    def _score_magnitude_prediction(self, predicted: str, expected: Optional[str]) -> float:
        """Score the accuracy of effect magnitude prediction."""
        if not expected or not predicted:
            return 0.5  # Neutral score if magnitude not specified
        
        predicted_lower = predicted.lower()
        expected_lower = expected.lower()
        
        # Extract percentages from both
        predicted_pct = re.findall(r"(\d+(?:\.\d+)?)%", predicted)
        expected_pct = re.findall(r"(\d+(?:\.\d+)?)%", expected)
        
        if predicted_pct and expected_pct:
            pred_val = float(predicted_pct[0])
            exp_val = float(expected_pct[0])
            
            # Score based on how close the prediction is
            diff = abs(pred_val - exp_val)
            if diff <= 2:
                return 1.0
            elif diff <= 5:
                return 0.8
            elif diff <= 10:
                return 0.6
            elif diff <= 20:
                return 0.4
            else:
                return 0.2
        
        # Check for qualitative magnitude terms
        magnitude_terms = {
            "small": ["small", "minor", "slight", "modest"],
            "moderate": ["moderate", "medium", "significant"],
            "large": ["large", "major", "substantial", "dramatic"]
        }
        
        for category, terms in magnitude_terms.items():
            if (any(term in predicted_lower for term in terms) and 
                any(term in expected_lower for term in terms)):
                return 0.7
        
        return 0.3
    
    def _score_reasoning_quality(self, reasoning: str, scenario: InterventionScenario) -> float:
        """Score the quality of causal reasoning about the intervention."""
        if not reasoning:
            return 0.0
        
        score = 0.0
        reasoning_lower = reasoning.lower()
        
        # Check for intervention-specific concepts
        intervention_concepts = {
            "causal": 0.15,
            "mechanism": 0.15,
            "intervention": 0.1,
            "effect": 0.1,
            "because": 0.1,
            "leads to": 0.1,
            "results in": 0.1,
            "impacts": 0.05,
            "influences": 0.05,
            "feedback": 0.1
        }
        
        for concept, weight in intervention_concepts.items():
            if concept in reasoning_lower:
                score += weight
        
        # Check for consideration of confounding factors
        confounding_mentioned = sum(
            0.05 for factor in scenario.confounding_factors
            if any(word in reasoning_lower for word in factor.lower().split())
        )
        score += min(confounding_mentioned, 0.2)
        
        # Check for domain-specific understanding
        domain_terms = {
            "healthcare": ["patient", "treatment", "medical", "clinical"],
            "technology": ["algorithm", "system", "digital", "platform"],
            "education": ["student", "learning", "academic", "school"],
            "manufacturing": ["production", "efficiency", "worker", "process"]
        }
        
        if scenario.domain in domain_terms:
            domain_score = sum(0.05 for term in domain_terms[scenario.domain]
                             if term in reasoning_lower)
            score += min(domain_score, 0.15)
        
        # Check for consideration of time and complexity
        complexity_indicators = ["time", "gradual", "eventually", "complex", "multiple", "indirect"]
        if any(indicator in reasoning_lower for indicator in complexity_indicators):
            score += 0.1
        
        return min(score, 1.0)
    
    def _score_side_effects(self, predicted: List[str], expected: List[str]) -> float:
        """Score the identification of intervention side effects."""
        if not expected:
            return 1.0 if not predicted else 0.8
        
        if not predicted:
            return 0.3  # Some credit for recognizing complexity is hard
        
        # Calculate semantic overlap
        predicted_lower = [effect.lower() for effect in predicted]
        expected_lower = [effect.lower() for effect in expected]
        
        correct_identified = 0
        for expected_effect in expected_lower:
            for predicted_effect in predicted_lower:
                # Check for word overlap
                expected_words = set(expected_effect.split())
                predicted_words = set(predicted_effect.split())
                
                if len(expected_words & predicted_words) >= 1:
                    correct_identified += 1
                    break
        
        if len(expected) > 0:
            recall = correct_identified / len(expected)
        else:
            recall = 1.0
        
        if len(predicted) > 0:
            precision = correct_identified / len(predicted)
        else:
            precision = 0.0
        
        # F1 score with some bonus for attempting to identify side effects
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
        
        return min(f1_score + 0.2, 1.0)  # Bonus for attempting
    
    def _score_time_frame(self, predicted: str, expected: str) -> float:
        """Score the accuracy of time frame prediction."""
        if not predicted or not expected:
            return 0.5
        
        predicted_lower = predicted.lower()
        expected_lower = expected.lower()
        
        # Time categories
        time_categories = {
            "immediate": ["immediate", "instant", "right away", "now"],
            "short_term": ["minutes", "hours", "days", "week", "short"],
            "medium_term": ["weeks", "months", "medium", "moderate"],
            "long_term": ["years", "long", "eventually"]
        }
        
        predicted_category = None
        expected_category = None
        
        for category, terms in time_categories.items():
            if any(term in predicted_lower for term in terms):
                predicted_category = category
            if any(term in expected_lower for term in terms):
                expected_category = category
        
        if predicted_category == expected_category:
            return 1.0
        
        # Adjacent categories get partial credit
        category_order = ["immediate", "short_term", "medium_term", "long_term"]
        if predicted_category and expected_category:
            pred_idx = category_order.index(predicted_category)
            exp_idx = category_order.index(expected_category)
            
            if abs(pred_idx - exp_idx) == 1:
                return 0.6
            elif abs(pred_idx - exp_idx) == 2:
                return 0.3
        
        return 0.0