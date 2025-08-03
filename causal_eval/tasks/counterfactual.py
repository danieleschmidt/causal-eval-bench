"""
Counterfactual Reasoning Task: Test ability to reason about "what if" scenarios.
"""

import random
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel
import re

from causal_eval.core.tasks import BaseTask, TaskConfig


@dataclass
class CounterfactualScenario:
    """A counterfactual reasoning scenario."""
    
    original_situation: str
    intervention: str
    expected_outcome: str
    domain: str
    difficulty: str  # "easy", "medium", "hard"
    causal_mechanism: str
    alternative_outcomes: List[str] = field(default_factory=list)
    context_variables: List[str] = field(default_factory=list)


class CounterfactualResponse(BaseModel):
    """Structured response for counterfactual reasoning."""
    
    predicted_outcome: str
    confidence: float
    reasoning: str
    causal_chain: List[str] = []
    assumptions: List[str] = []


class CounterfactualReasoning(BaseTask):
    """Task for evaluating counterfactual reasoning abilities."""
    
    def __init__(self, config: TaskConfig):
        super().__init__(config)
        self.scenarios = self._load_scenarios()
    
    def _load_scenarios(self) -> List[CounterfactualScenario]:
        """Load predefined counterfactual scenarios."""
        scenarios = [
            CounterfactualScenario(
                original_situation="Sarah studied for 6 hours and scored 85% on her exam.",
                intervention="What if Sarah had only studied for 2 hours?",
                expected_outcome="Lower exam score, likely 65-75%",
                domain="education",
                difficulty="easy",
                causal_mechanism="Study time directly affects knowledge retention and exam performance",
                alternative_outcomes=["65%", "70%", "75%"],
                context_variables=["study efficiency", "prior knowledge", "exam difficulty"]
            ),
            CounterfactualScenario(
                original_situation="A company increased their advertising budget by 50% and saw a 20% increase in sales.",
                intervention="What if the company had decreased their advertising budget by 30% instead?",
                expected_outcome="Decreased sales, likely 10-15% reduction",
                domain="business",
                difficulty="medium",
                causal_mechanism="Advertising drives brand awareness and customer acquisition",
                alternative_outcomes=["10% decrease", "15% decrease", "5% decrease"],
                context_variables=["market competition", "product quality", "economic conditions"]
            ),
            CounterfactualScenario(
                original_situation="Dr. Martinez prescribed antibiotics for a patient's bacterial infection, and the patient recovered in 7 days.",
                intervention="What if Dr. Martinez had not prescribed antibiotics?",
                expected_outcome="Longer recovery time or potential complications",
                domain="medical",
                difficulty="medium", 
                causal_mechanism="Antibiotics directly target bacterial pathogens",
                alternative_outcomes=["14-21 day recovery", "complications", "hospitalization"],
                context_variables=["immune system strength", "infection severity", "patient age"]
            ),
            CounterfactualScenario(
                original_situation="The government implemented a carbon tax and emissions decreased by 12% over two years.",
                intervention="What if the government had implemented a cap-and-trade system instead?",
                expected_outcome="Different emission reduction pattern, possibly more market-driven",
                domain="environmental",
                difficulty="hard",
                causal_mechanism="Economic incentives drive behavioral change in emissions",
                alternative_outcomes=["8-15% reduction", "more volatile reductions", "industry-specific patterns"],
                context_variables=["economic growth", "technology adoption", "international agreements"]
            ),
            CounterfactualScenario(
                original_situation="A social media algorithm change led to 25% more user engagement but 40% more misinformation spread.",
                intervention="What if the algorithm had prioritized fact-checked content instead?",
                expected_outcome="Lower engagement but reduced misinformation spread",
                domain="technology",
                difficulty="hard",
                causal_mechanism="Algorithm design shapes content distribution and user behavior",
                alternative_outcomes=["15% less engagement", "60% less misinformation", "improved content quality"],
                context_variables=["user preferences", "content creator behavior", "fact-checking resources"]
            ),
            CounterfactualScenario(
                original_situation="A factory installed new safety equipment and workplace accidents decreased by 60%.",
                intervention="What if the factory had only provided additional safety training instead?",
                expected_outcome="Some accident reduction but less dramatic than equipment upgrade",
                domain="workplace_safety",
                difficulty="medium",
                causal_mechanism="Physical safety measures prevent accidents more reliably than behavior change alone",
                alternative_outcomes=["20-30% reduction", "temporary improvement", "gradual decline without reinforcement"],
                context_variables=["worker compliance", "equipment maintenance", "work environment complexity"]
            ),
            CounterfactualScenario(
                original_situation="A city built a new subway line and traffic congestion decreased by 18%.",
                intervention="What if the city had expanded bus service with the same budget instead?",
                expected_outcome="Moderate congestion reduction but likely less than subway",
                domain="urban_planning",
                difficulty="medium",
                causal_mechanism="Public transit capacity and convenience affects mode choice",
                alternative_outcomes=["8-12% reduction", "increased bus ridership", "temporary improvement"],
                context_variables=["route coverage", "service frequency", "user preferences"]
            ),
            CounterfactualScenario(
                original_situation="A student switched from cramming to spaced repetition study and improved retention by 45%.",
                intervention="What if the student had instead doubled their cramming time?",
                expected_outcome="Marginal improvement in short-term but poor long-term retention", 
                domain="education",
                difficulty="hard",
                causal_mechanism="Spaced repetition leverages memory consolidation principles",
                alternative_outcomes=["10% short-term gain", "worse long-term retention", "increased fatigue"],
                context_variables=["sleep quality", "stress levels", "material complexity"]
            )
        ]
        
        # Filter by domain and difficulty if specified
        if hasattr(self.config, 'domain') and self.config.domain != "general":
            scenarios = [s for s in scenarios if s.domain == self.config.domain]
        
        if hasattr(self.config, 'difficulty') and self.config.difficulty != "mixed":
            scenarios = [s for s in scenarios if s.difficulty == self.config.difficulty]
        
        return scenarios
    
    async def generate_prompt(self) -> str:
        """Generate a counterfactual reasoning prompt."""
        scenario = random.choice(self.scenarios)
        
        prompt = f"""
Analyze the following counterfactual scenario and predict what would have happened.

Original Situation: {scenario.original_situation}

Counterfactual Question: {scenario.intervention}

Please provide your analysis in the following format:
1. Predicted Outcome: [What you think would have happened]
2. Confidence Level: [0.0 to 1.0]
3. Reasoning: [Explain your causal analysis step by step]
4. Causal Chain: [List the key steps in the causal process]
5. Key Assumptions: [What assumptions are you making?]

Consider:
- The underlying causal mechanisms
- Relevant context variables that might affect the outcome
- Alternative pathways and their likelihood
- Time-dependent effects
"""
        
        # Store scenario for evaluation
        self._current_scenario = scenario
        return prompt.strip()
    
    async def evaluate_response(self, response: str) -> Dict[str, Any]:
        """Evaluate the model's counterfactual reasoning response."""
        if not hasattr(self, '_current_scenario'):
            return {"error": "No current scenario available for evaluation"}
        
        scenario = self._current_scenario
        parsed_response = self._parse_response(response)
        
        # Calculate scores
        outcome_score = self._score_outcome_prediction(
            parsed_response.predicted_outcome,
            scenario.expected_outcome,
            scenario.alternative_outcomes
        )
        
        reasoning_score = self._score_reasoning_quality(
            parsed_response.reasoning,
            scenario
        )
        
        causal_chain_score = self._score_causal_chain(
            parsed_response.causal_chain,
            scenario.causal_mechanism
        )
        
        assumptions_score = self._score_assumptions(
            parsed_response.assumptions,
            scenario.context_variables
        )
        
        # Overall score (weighted average)
        overall_score = (
            outcome_score * 0.4 +
            reasoning_score * 0.3 +
            causal_chain_score * 0.2 +
            assumptions_score * 0.1
        )
        
        return {
            "overall_score": overall_score,
            "outcome_score": outcome_score,
            "reasoning_score": reasoning_score,
            "causal_chain_score": causal_chain_score,
            "assumptions_score": assumptions_score,
            "scenario_domain": scenario.domain,
            "scenario_difficulty": scenario.difficulty,
            "predicted_outcome": parsed_response.predicted_outcome,
            "expected_outcome": scenario.expected_outcome,
            "confidence": parsed_response.confidence,
            "causal_mechanism": scenario.causal_mechanism,
            "model_reasoning": parsed_response.reasoning,
            "identified_causal_chain": parsed_response.causal_chain,
            "model_assumptions": parsed_response.assumptions,
            "context_variables": scenario.context_variables
        }
    
    def _parse_response(self, response: str) -> CounterfactualResponse:
        """Parse the model's response into structured format."""
        # Extract predicted outcome
        outcome_match = re.search(
            r"(?:Predicted Outcome|outcome):\s*(.+?)(?:\n\d+\.|$)",
            response,
            re.IGNORECASE | re.DOTALL
        )
        predicted_outcome = outcome_match.group(1).strip() if outcome_match else ""
        
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
        
        # Extract causal chain
        chain_match = re.search(
            r"(?:Causal Chain|causal chain):\s*(.+?)(?:\n\d+\.|$)",
            response,
            re.IGNORECASE | re.DOTALL
        )
        chain_text = chain_match.group(1) if chain_match else ""
        causal_chain = [
            step.strip().strip("[]()\"'-•") 
            for step in re.split(r"[,;\n]|\d+\.", chain_text)
            if step.strip()
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
        
        return CounterfactualResponse(
            predicted_outcome=predicted_outcome,
            confidence=confidence,
            reasoning=reasoning,
            causal_chain=causal_chain,
            assumptions=assumptions
        )
    
    def _score_outcome_prediction(self, predicted: str, expected: str, alternatives: List[str]) -> float:
        """Score the accuracy of outcome prediction."""
        if not predicted:
            return 0.0
        
        predicted_lower = predicted.lower()
        expected_lower = expected.lower()
        
        # Check for exact or near match with expected outcome
        if any(word in predicted_lower for word in expected_lower.split() if len(word) > 3):
            return 1.0
        
        # Check for match with alternative outcomes
        for alt in alternatives:
            if any(word in predicted_lower for word in alt.lower().split() if len(word) > 3):
                return 0.7
        
        # Check for directional correctness (increase/decrease)
        direction_words = {
            "increase": ["increase", "rise", "grow", "higher", "more", "up"],
            "decrease": ["decrease", "fall", "drop", "lower", "less", "down", "reduce"]
        }
        
        expected_direction = None
        for direction, words in direction_words.items():
            if any(word in expected_lower for word in words):
                expected_direction = direction
                break
        
        if expected_direction:
            predicted_direction = None
            for direction, words in direction_words.items():
                if any(word in predicted_lower for word in words):
                    predicted_direction = direction
                    break
            
            if predicted_direction == expected_direction:
                return 0.4
        
        return 0.0
    
    def _score_reasoning_quality(self, reasoning: str, scenario: CounterfactualScenario) -> float:
        """Score the quality of counterfactual reasoning."""
        if not reasoning:
            return 0.0
        
        score = 0.0
        reasoning_lower = reasoning.lower()
        
        # Check for counterfactual thinking concepts
        counterfactual_concepts = {
            "causal": 0.2,
            "mechanism": 0.15,
            "because": 0.1,
            "therefore": 0.1,
            "would": 0.1,
            "likely": 0.1,
            "scenario": 0.05,
            "alternative": 0.05,
            "outcome": 0.05,
            "result": 0.05
        }
        
        for concept, weight in counterfactual_concepts.items():
            if concept in reasoning_lower:
                score += weight
        
        # Check for domain-specific reasoning
        domain_terms = {
            "education": ["learn", "knowledge", "study", "retention"],
            "medical": ["treatment", "patient", "health", "recovery"],
            "business": ["sales", "revenue", "customer", "market"],
            "environmental": ["emission", "environment", "policy", "impact"]
        }
        
        if scenario.domain in domain_terms:
            domain_score = sum(0.05 for term in domain_terms[scenario.domain] 
                             if term in reasoning_lower)
            score += min(domain_score, 0.2)
        
        # Check for consideration of complexity
        complexity_indicators = ["however", "but", "although", "consider", "depend", "factor", "complex"]
        if any(indicator in reasoning_lower for indicator in complexity_indicators):
            score += 0.1
        
        # Length and depth check
        if len(reasoning.split()) >= 20:
            score += 0.1
        
        return min(score, 1.0)
    
    def _score_causal_chain(self, causal_chain: List[str], mechanism: str) -> float:
        """Score the identification and sequencing of causal steps."""
        if not causal_chain:
            return 0.0
        
        mechanism_lower = mechanism.lower()
        score = 0.0
        
        # Check if chain elements relate to the mechanism
        relevant_steps = 0
        for step in causal_chain:
            step_lower = step.lower()
            if any(word in step_lower for word in mechanism_lower.split() if len(word) > 3):
                relevant_steps += 1
        
        if causal_chain:
            relevance_score = relevant_steps / len(causal_chain)
            score += relevance_score * 0.6
        
        # Check for logical sequencing words
        sequence_indicators = ["first", "then", "next", "finally", "leads to", "causes", "results in"]
        sequence_score = sum(0.1 for step in causal_chain 
                           for indicator in sequence_indicators 
                           if indicator in step.lower())
        score += min(sequence_score, 0.4)
        
        return min(score, 1.0)
    
    def _score_assumptions(self, assumptions: List[str], context_variables: List[str]) -> float:
        """Score the identification of key assumptions."""
        if not assumptions:
            return 0.5  # Neutral score if no assumptions listed
        
        score = 0.0
        
        # Check for awareness of context variables
        context_lower = [var.lower() for var in context_variables]
        identified_context = 0
        
        for assumption in assumptions:
            assumption_lower = assumption.lower()
            for context_var in context_lower:
                if any(word in assumption_lower for word in context_var.split()):
                    identified_context += 1
                    break
        
        if context_variables:
            context_score = identified_context / len(context_variables)
            score += context_score * 0.7
        
        # Check for general assumption quality
        assumption_quality_terms = ["assume", "given", "constant", "unchanged", "equal", "similar"]
        quality_score = sum(0.1 for assumption in assumptions 
                          for term in assumption_quality_terms 
                          if term in assumption.lower())
        score += min(quality_score, 0.3)
        
        return min(score, 1.0)