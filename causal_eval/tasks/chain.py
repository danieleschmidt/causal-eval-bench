"""
Causal Chain Reasoning Task: Test understanding of multi-step causal processes.
"""

import random
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel
import re

from causal_eval.core.tasks import BaseTask, TaskConfig


@dataclass
class CausalChainScenario:
    """A multi-step causal chain scenario."""
    
    initial_condition: str
    causal_steps: List[Dict[str, str]]  # [{"step": "description", "mechanism": "causal mechanism"}]
    final_outcome: str
    domain: str
    difficulty: str
    chain_length: int
    alternative_paths: List[List[str]] = field(default_factory=list)
    disruption_points: List[str] = field(default_factory=list)


class CausalChainResponse(BaseModel):
    """Structured response for causal chain reasoning."""
    
    predicted_chain: List[str]
    final_outcome: str
    confidence: float
    reasoning: str
    weak_links: List[str] = []
    alternative_explanations: List[str] = []


class CausalChain(BaseTask):
    """Task for evaluating multi-step causal chain reasoning."""
    
    def __init__(self, config: TaskConfig):
        super().__init__(config)
        self.scenarios = self._load_scenarios()
    
    def _load_scenarios(self) -> List[CausalChainScenario]:
        """Load predefined causal chain scenarios."""
        scenarios = [
            CausalChainScenario(
                initial_condition="Heavy rainfall in the mountains",
                causal_steps=[
                    {"step": "Water accumulates in mountain streams", "mechanism": "Gravity causes water flow downhill"},
                    {"step": "Stream flow increases dramatically", "mechanism": "Volume accumulation from multiple tributaries"},
                    {"step": "River levels rise in downstream areas", "mechanism": "Excess water flows into main river system"},
                    {"step": "Riverbanks overflow in flood plains", "mechanism": "Water volume exceeds channel capacity"},
                    {"step": "Agricultural fields become waterlogged", "mechanism": "Surface water cannot drain effectively"}
                ],
                final_outcome="Crop damage and agricultural losses",
                domain="environmental",
                difficulty="medium",
                chain_length=5,
                alternative_paths=[["Dam release", "Controlled flooding", "Managed crop protection"]],
                disruption_points=["Dam construction", "Improved drainage systems", "Flood barriers"]
            ),
            CausalChainScenario(
                initial_condition="Central bank raises interest rates by 2%",
                causal_steps=[
                    {"step": "Bank lending rates increase", "mechanism": "Banks pass through rate changes to maintain margins"},
                    {"step": "Mortgage and loan applications decrease", "mechanism": "Higher borrowing costs reduce demand"},
                    {"step": "Housing demand falls", "mechanism": "Fewer qualified buyers due to affordability"},
                    {"step": "Property prices begin to decline", "mechanism": "Reduced demand creates downward price pressure"},
                    {"step": "Consumer spending decreases", "mechanism": "Wealth effect from lower property values"}
                ],
                final_outcome="Economic slowdown and reduced inflation",
                domain="economics",
                difficulty="hard",
                chain_length=5,
                alternative_paths=[["Corporate investment reduction", "Employment impacts", "Broader economic effects"]],
                disruption_points=["Government stimulus", "Alternative financing", "International factors"]
            ),
            CausalChainScenario(
                initial_condition="Student begins using active recall study method",
                causal_steps=[
                    {"step": "Student practices retrieving information from memory", "mechanism": "Active recall strengthens neural pathways"},
                    {"step": "Memory consolidation improves", "mechanism": "Retrieval practice enhances long-term retention"},
                    {"step": "Test performance increases", "mechanism": "Better retention leads to improved recall during exams"},
                    {"step": "Academic confidence grows", "mechanism": "Success experiences build self-efficacy"},
                    {"step": "Study motivation increases", "mechanism": "Positive reinforcement from improved outcomes"}
                ],
                final_outcome="Sustained academic improvement and better learning habits",
                domain="education",
                difficulty="easy",
                chain_length=5,
                alternative_paths=[["Peer study groups", "Teaching others", "Collaborative learning"]],
                disruption_points=["Poor sleep", "High stress", "Inconsistent practice"]
            ),
            CausalChainScenario(
                initial_condition="Company implements remote work policy",
                causal_steps=[
                    {"step": "Office space requirements decrease", "mechanism": "Fewer employees need daily workspace"},
                    {"step": "Commercial real estate costs reduce", "mechanism": "Downsizing office leases and facilities"},
                    {"step": "Employee commuting expenses eliminate", "mechanism": "No daily travel to central office location"},
                    {"step": "Worker productivity patterns change", "mechanism": "Different home vs office work environments"},
                    {"step": "Work-life balance dynamics shift", "mechanism": "Blurred boundaries between personal and professional time"}
                ],
                final_outcome="Fundamental changes in organizational culture and employee satisfaction",
                domain="workplace",
                difficulty="medium",
                chain_length=5,
                alternative_paths=[["Technology adoption", "Communication changes", "Management adaptation"]],
                disruption_points=["Internet connectivity issues", "Home distractions", "Isolation effects"]
            ),
            CausalChainScenario(
                initial_condition="New antibiotic-resistant bacteria strain emerges",
                causal_steps=[
                    {"step": "Standard antibiotics become ineffective", "mechanism": "Bacterial resistance mechanisms neutralize drug action"},
                    {"step": "Infection treatment options become limited", "mechanism": "Fewer effective medications available to clinicians"},
                    {"step": "Hospital stays lengthen for infected patients", "mechanism": "More time needed to find effective treatments"},
                    {"step": "Healthcare costs increase significantly", "mechanism": "Extended care and specialized treatments are expensive"},
                    {"step": "Hospital capacity becomes strained", "mechanism": "Longer stays reduce bed availability for other patients"}
                ],
                final_outcome="Healthcare system pressure and potential public health crisis",
                domain="medical",
                difficulty="hard",
                chain_length=5,
                alternative_paths=[["Alternative treatment development", "Preventive measures", "Infection control protocols"]],
                disruption_points=["New drug development", "Improved diagnostics", "Vaccination programs"]
            ),
            CausalChainScenario(
                initial_condition="Social media algorithm change promotes engagement over accuracy",
                causal_steps=[
                    {"step": "Sensational and controversial content gets more visibility", "mechanism": "Algorithm rewards high engagement metrics"},
                    {"step": "Users spend more time on emotionally charged posts", "mechanism": "Strong emotional reactions drive continued interaction"},
                    {"step": "Misinformation spreads faster than fact-checked content", "mechanism": "Sensational false claims often more engaging than facts"},
                    {"step": "User beliefs become more polarized", "mechanism": "Echo chambers reinforce existing biases and opinions"},
                    {"step": "Social trust in institutions decreases", "mechanism": "Exposure to conflicting and false information undermines confidence"}
                ],
                final_outcome="Increased social division and decreased trust in reliable information sources",
                domain="technology",
                difficulty="hard",
                chain_length=5,
                alternative_paths=[["Content moderation", "Fact-checking integration", "User education"]],
                disruption_points=["Algorithm transparency", "Regulatory intervention", "Platform accountability"]
            ),
            CausalChainScenario(
                initial_condition="Factory implements just-in-time inventory management",
                causal_steps=[
                    {"step": "Inventory storage costs decrease", "mechanism": "Less warehouse space and carrying costs needed"},
                    {"step": "Supply chain coordination requirements increase", "mechanism": "Precise timing needed for material deliveries"},
                    {"step": "Dependency on supplier reliability grows", "mechanism": "Less buffer stock means higher vulnerability to delays"},
                    {"step": "Production flexibility decreases", "mechanism": "Limited inventory makes rapid changes difficult"},
                    {"step": "Risk of production shutdowns increases", "mechanism": "Any supply disruption can halt manufacturing"}
                ],
                final_outcome="Improved efficiency but increased vulnerability to supply chain disruptions",
                domain="manufacturing",
                difficulty="medium",
                chain_length=5,
                alternative_paths=[["Supplier diversification", "Strategic inventory", "Flexible manufacturing"]],
                disruption_points=["Natural disasters", "Transportation strikes", "Supplier bankruptcy"]
            ),
            CausalChainScenario(
                initial_condition="City implements congestion pricing in downtown area",
                causal_steps=[
                    {"step": "Driving costs in city center increase", "mechanism": "Additional fees make car travel more expensive"},
                    {"step": "Public transit ridership increases", "mechanism": "Cost differential makes alternatives more attractive"},
                    {"step": "Downtown traffic volume decreases", "mechanism": "Price incentive reduces car trips to city center"},
                    {"step": "Air quality in city center improves", "mechanism": "Fewer vehicles reduce pollution emissions"},
                    {"step": "Property values near transit hubs increase", "mechanism": "Improved accessibility and environment boost desirability"}
                ],
                final_outcome="Environmental benefits but potential economic impacts on downtown businesses",
                domain="urban_planning",
                difficulty="medium",
                chain_length=5,
                alternative_paths=[["Electric vehicle adoption", "Delivery optimization", "Remote work increase"]],
                disruption_points=["Public opposition", "Business relocations", "Enforcement challenges"]
            )
        ]
        
        # Filter by domain and difficulty if specified
        if hasattr(self.config, 'domain') and self.config.domain != "general":
            scenarios = [s for s in scenarios if s.domain == self.config.domain]
        
        if hasattr(self.config, 'difficulty') and self.config.difficulty != "mixed":
            scenarios = [s for s in scenarios if s.difficulty == self.config.difficulty]
        
        return scenarios
    
    async def generate_prompt(self) -> str:
        """Generate a causal chain reasoning prompt."""
        scenario = random.choice(self.scenarios)
        
        # Decide whether to show partial chain or full reconstruction
        show_partial = random.choice([True, False])
        
        if show_partial:
            # Show first step and ask to complete the chain
            first_step = scenario.causal_steps[0]["step"]
            prompt = f"""
Analyze the following causal chain scenario and complete the missing steps.

Initial Condition: {scenario.initial_condition}

First Step: {first_step}

Question: What are the subsequent steps in this causal chain that lead to the final outcome?

Please provide your analysis in the following format:
1. Complete Causal Chain: [List all steps from initial condition to final outcome]
2. Final Outcome: [What is the ultimate result of this chain?]
3. Confidence Level: [0.0 to 1.0]
4. Reasoning: [Explain the causal mechanisms at each step]
5. Weak Links: [Identify steps that could be disrupted or are uncertain]
6. Alternative Explanations: [Are there other causal pathways that could explain this outcome?]

Consider:
- The underlying mechanisms that drive each step
- Time delays and feedback effects
- Points where the chain could be interrupted
- Alternative causal pathways
"""
        else:
            # Give initial condition and final outcome, ask to reconstruct chain
            prompt = f"""
Analyze the following causal scenario and reconstruct the complete causal chain.

Initial Condition: {scenario.initial_condition}

Final Outcome: {scenario.final_outcome}

Question: What are the intermediate causal steps that connect the initial condition to the final outcome?

Please provide your analysis in the following format:
1. Complete Causal Chain: [List all steps from initial condition to final outcome]
2. Confidence Level: [0.0 to 1.0]
3. Reasoning: [Explain the causal mechanisms at each step]
4. Weak Links: [Identify steps that could be disrupted or are uncertain]
5. Alternative Explanations: [Are there other causal pathways that could explain this outcome?]

Consider:
- The most plausible causal mechanisms
- Intermediate variables that must change
- Time sequences and dependencies
- Points of vulnerability in the chain
"""
        
        # Store scenario and prompt type for evaluation
        self._current_scenario = scenario
        self._prompt_type = "partial" if show_partial else "reconstruct"
        return prompt.strip()
    
    async def evaluate_response(self, response: str) -> Dict[str, Any]:
        """Evaluate the model's causal chain reasoning response."""
        if not hasattr(self, '_current_scenario'):
            return {"error": "No current scenario available for evaluation"}
        
        scenario = self._current_scenario
        parsed_response = self._parse_response(response)
        
        # Calculate scores
        chain_accuracy_score = self._score_chain_accuracy(
            parsed_response.predicted_chain,
            scenario.causal_steps
        )
        
        outcome_score = self._score_outcome_prediction(
            parsed_response.final_outcome,
            scenario.final_outcome
        )
        
        reasoning_score = self._score_reasoning_quality(
            parsed_response.reasoning,
            scenario
        )
        
        weak_links_score = self._score_weak_links_identification(
            parsed_response.weak_links,
            scenario.disruption_points
        )
        
        alternative_paths_score = self._score_alternative_explanations(
            parsed_response.alternative_explanations,
            scenario.alternative_paths
        )
        
        # Overall score (weighted average)
        overall_score = (
            chain_accuracy_score * 0.35 +
            outcome_score * 0.25 +
            reasoning_score * 0.2 +
            weak_links_score * 0.1 +
            alternative_paths_score * 0.1
        )
        
        return {
            "overall_score": overall_score,
            "chain_accuracy_score": chain_accuracy_score,
            "outcome_score": outcome_score,
            "reasoning_score": reasoning_score,
            "weak_links_score": weak_links_score,
            "alternative_paths_score": alternative_paths_score,
            "scenario_domain": scenario.domain,
            "scenario_difficulty": scenario.difficulty,
            "chain_length": scenario.chain_length,
            "predicted_chain": parsed_response.predicted_chain,
            "expected_steps": [step["step"] for step in scenario.causal_steps],
            "predicted_outcome": parsed_response.final_outcome,
            "expected_outcome": scenario.final_outcome,
            "confidence": parsed_response.confidence,
            "model_reasoning": parsed_response.reasoning,
            "identified_weak_links": parsed_response.weak_links,
            "expected_disruption_points": scenario.disruption_points,
            "alternative_explanations": parsed_response.alternative_explanations,
            "expected_alternatives": scenario.alternative_paths,
            "prompt_type": getattr(self, '_prompt_type', 'unknown')
        }
    
    def _parse_response(self, response: str) -> CausalChainResponse:
        """Parse the model's response into structured format."""
        # Extract causal chain
        chain_match = re.search(
            r"(?:Complete )?Causal Chain:\s*(.+?)(?:\n\d+\.|$)",
            response,
            re.IGNORECASE | re.DOTALL
        )
        chain_text = chain_match.group(1) if chain_match else ""
        
        # Parse individual steps
        predicted_chain = []
        for step_match in re.finditer(r"(?:\d+\.|\-|\•)\s*(.+?)(?=\n(?:\d+\.|\-|\•)|$)", chain_text, re.DOTALL):
            step = step_match.group(1).strip()
            if step:
                predicted_chain.append(step)
        
        # If no numbered list found, split by sentences/lines
        if not predicted_chain:
            predicted_chain = [
                step.strip().strip("[]()\"'-•") 
                for step in re.split(r'[,;\n]|\d+\.', chain_text)
                if step.strip() and len(step.strip()) > 10
            ]
        
        # Extract final outcome
        outcome_match = re.search(
            r"(?:Final Outcome|outcome):\s*(.+?)(?:\n\d+\.|$)",
            response,
            re.IGNORECASE | re.DOTALL
        )
        final_outcome = outcome_match.group(1).strip() if outcome_match else ""
        
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
        
        # Extract weak links
        weak_links_match = re.search(
            r"(?:Weak Links|weak links):\s*(.+?)(?:\n\d+\.|$)",
            response,
            re.IGNORECASE | re.DOTALL
        )
        weak_links_text = weak_links_match.group(1) if weak_links_match else ""
        weak_links = [
            link.strip().strip("[]()\"'-•") 
            for link in re.split(r"[,;\n]|\d+\.", weak_links_text)
            if link.strip()
        ]
        
        # Extract alternative explanations
        alternatives_match = re.search(
            r"(?:Alternative Explanations?|alternatives?):\s*(.+?)(?:\n\d+\.|$)",
            response,
            re.IGNORECASE | re.DOTALL
        )
        alternatives_text = alternatives_match.group(1) if alternatives_match else ""
        alternative_explanations = [
            alt.strip().strip("[]()\"'-•") 
            for alt in re.split(r"[,;\n]|\d+\.", alternatives_text)
            if alt.strip()
        ]
        
        return CausalChainResponse(
            predicted_chain=predicted_chain,
            final_outcome=final_outcome,
            confidence=confidence,
            reasoning=reasoning,
            weak_links=weak_links,
            alternative_explanations=alternative_explanations
        )
    
    def _score_chain_accuracy(self, predicted: List[str], expected: List[Dict[str, str]]) -> float:
        """Score the accuracy of the predicted causal chain."""
        if not predicted or not expected:
            return 0.0
        
        expected_steps = [step["step"].lower() for step in expected]
        predicted_lower = [step.lower() for step in predicted]
        
        # Calculate step-by-step accuracy
        correct_steps = 0
        total_expected = len(expected_steps)
        
        for i, expected_step in enumerate(expected_steps):
            # Check if any predicted step matches this expected step
            for j, predicted_step in enumerate(predicted_lower):
                # Calculate semantic similarity (simple word overlap)
                expected_words = set(expected_step.split())
                predicted_words = set(predicted_step.split())
                
                # Remove common words that don't add meaning
                stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
                expected_words -= stop_words
                predicted_words -= stop_words
                
                if len(expected_words) > 0:
                    overlap = len(expected_words & predicted_words) / len(expected_words)
                    if overlap >= 0.3:  # 30% word overlap threshold
                        correct_steps += 1
                        break
        
        step_accuracy = correct_steps / total_expected if total_expected > 0 else 0
        
        # Bonus for getting the sequence approximately right
        sequence_bonus = 0
        if len(predicted) >= len(expected) * 0.7:  # At least 70% of expected length
            sequence_bonus = 0.1
        
        # Penalty for too many irrelevant steps
        length_penalty = 0
        if len(predicted) > len(expected) * 1.5:  # More than 150% of expected length
            length_penalty = 0.1
        
        return min(step_accuracy + sequence_bonus - length_penalty, 1.0)
    
    def _score_outcome_prediction(self, predicted: str, expected: str) -> float:
        """Score the accuracy of final outcome prediction."""
        if not predicted or not expected:
            return 0.0
        
        predicted_lower = predicted.lower()
        expected_lower = expected.lower()
        
        # Check for word overlap
        predicted_words = set(predicted_lower.split())
        expected_words = set(expected_lower.split())
        
        # Remove stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        predicted_words -= stop_words
        expected_words -= stop_words
        
        if len(expected_words) > 0:
            overlap = len(predicted_words & expected_words) / len(expected_words)
            return min(overlap * 2, 1.0)  # Scale up to reward good matches
        
        return 0.0
    
    def _score_reasoning_quality(self, reasoning: str, scenario: CausalChainScenario) -> float:
        """Score the quality of causal reasoning."""
        if not reasoning:
            return 0.0
        
        score = 0.0
        reasoning_lower = reasoning.lower()
        
        # Check for causal reasoning concepts
        causal_concepts = {
            "causal": 0.1,
            "mechanism": 0.1,
            "because": 0.08,
            "leads to": 0.08,
            "results in": 0.08,
            "causes": 0.08,
            "triggers": 0.05,
            "influences": 0.05,
            "affects": 0.05,
            "chain": 0.05,
            "sequence": 0.05,
            "step": 0.05
        }
        
        for concept, weight in causal_concepts.items():
            if concept in reasoning_lower:
                score += weight
        
        # Check for mechanism explanations (from scenario)
        mechanism_mentions = 0
        for step in scenario.causal_steps:
            mechanism = step["mechanism"].lower()
            mechanism_words = set(mechanism.split()) - {"the", "a", "an", "and", "or", "but", "in", "on", "at"}
            
            for word in mechanism_words:
                if len(word) > 3 and word in reasoning_lower:
                    mechanism_mentions += 1
                    break
        
        mechanism_score = min(mechanism_mentions / len(scenario.causal_steps), 1.0) * 0.3
        score += mechanism_score
        
        # Check for domain-specific understanding
        domain_terms = {
            "environmental": ["water", "flow", "pressure", "natural", "climate"],
            "economics": ["market", "price", "demand", "cost", "economic"],
            "education": ["learning", "knowledge", "study", "retention", "academic"],
            "medical": ["treatment", "patient", "health", "clinical", "medical"],
            "technology": ["algorithm", "system", "data", "digital", "platform"]
        }
        
        if scenario.domain in domain_terms:
            domain_score = sum(0.03 for term in domain_terms[scenario.domain] 
                             if term in reasoning_lower)
            score += min(domain_score, 0.15)
        
        return min(score, 1.0)
    
    def _score_weak_links_identification(self, identified: List[str], expected: List[str]) -> float:
        """Score the identification of weak links in the causal chain."""
        if not expected:
            return 0.8 if not identified else 1.0  # No weak points expected
        
        if not identified:
            return 0.3  # Some credit for difficulty of identifying weak points
        
        # Calculate semantic overlap
        identified_lower = [link.lower() for link in identified]
        expected_lower = [point.lower() for point in expected]
        
        correct_identified = 0
        for expected_point in expected_lower:
            for identified_link in identified_lower:
                expected_words = set(expected_point.split())
                identified_words = set(identified_link.split())
                
                overlap = len(expected_words & identified_words)
                if overlap >= 1:  # At least one word match
                    correct_identified += 1
                    break
        
        if len(expected) > 0:
            recall = correct_identified / len(expected)
        else:
            recall = 1.0
        
        return min(recall + 0.2, 1.0)  # Bonus for attempting identification
    
    def _score_alternative_explanations(self, alternatives: List[str], expected_paths: List[List[str]]) -> float:
        """Score the identification of alternative causal pathways."""
        if not expected_paths:
            return 0.8 if not alternatives else 1.0  # No alternatives expected
        
        if not alternatives:
            return 0.5  # Neutral score for not providing alternatives
        
        # Check if any alternative mentions elements from expected paths
        alternatives_lower = " ".join(alternatives).lower()
        
        path_mentions = 0
        total_paths = len(expected_paths)
        
        for path in expected_paths:
            path_text = " ".join(path).lower()
            path_words = set(path_text.split()) - {"the", "a", "an", "and", "or", "but"}
            
            for word in path_words:
                if len(word) > 3 and word in alternatives_lower:
                    path_mentions += 1
                    break
        
        if total_paths > 0:
            coverage = path_mentions / total_paths
        else:
            coverage = 0
        
        # Bonus for providing thoughtful alternatives even if not exact matches
        creativity_bonus = 0.3 if len(alternatives) > 0 else 0
        
        return min(coverage + creativity_bonus, 1.0)