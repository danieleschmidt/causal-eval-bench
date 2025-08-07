"""
Confounding Analysis Task: Test ability to identify and reason about confounding variables.
"""

import random
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pydantic import BaseModel
import re

from causal_eval.core.tasks import BaseTask, TaskConfig


@dataclass
class ConfoundingScenario:
    """A scenario with potential confounding variables."""
    
    observed_relationship: str
    variable_a: str
    variable_b: str
    true_confounders: List[str]
    red_herring_variables: List[str]  # Variables that seem relevant but aren't confounders
    causal_structure: str  # Description of the actual causal relationships
    domain: str
    difficulty: str
    evidence_strength: str  # "weak", "moderate", "strong"
    alternative_explanations: List[str] = field(default_factory=list)


class ConfoundingResponse(BaseModel):
    """Structured response for confounding analysis."""
    
    is_causal: bool
    confidence: float
    identified_confounders: List[str]
    reasoning: str
    causal_diagram: str = ""
    controlled_analysis: str = ""  # How to control for confounders


class ConfoundingAnalysis(BaseTask):
    """Task for evaluating confounding variable identification and analysis."""
    
    def __init__(self, config: TaskConfig):
        super().__init__(config)
        self.scenarios = self._load_scenarios()
    
    def _load_scenarios(self) -> List[ConfoundingScenario]:
        """Load predefined confounding scenarios."""
        scenarios = [
            ConfoundingScenario(
                observed_relationship="People who drink more coffee have higher productivity at work",
                variable_a="coffee consumption",
                variable_b="work productivity",
                true_confounders=["sleep quality", "work motivation", "job satisfaction", "stress levels"],
                red_herring_variables=["coffee brand", "cup size", "office location"],
                causal_structure="Sleep quality and stress affect both coffee consumption and productivity; motivated people drink more coffee and are more productive",
                domain="workplace",
                difficulty="medium",
                evidence_strength="moderate",
                alternative_explanations=["Reverse causation: productive people drink more coffee to sustain performance"]
            ),
            ConfoundingScenario(
                observed_relationship="Countries with higher chocolate consumption have more Nobel Prize winners per capita",
                variable_a="chocolate consumption per capita",
                variable_b="Nobel Prize winners per capita",
                true_confounders=["economic development", "education investment", "research funding", "cultural values"],
                red_herring_variables=["chocolate quality", "cocoa imports", "candy advertising"],
                causal_structure="Wealthy countries afford both chocolate consumption and extensive educational/research systems that produce Nobel laureates",
                domain="international",
                difficulty="easy",
                evidence_strength="strong",
                alternative_explanations=["Sample size effects in small vs large countries"]
            ),
            ConfoundingScenario(
                observed_relationship="Children who watch more educational TV programs score higher on reading tests",
                variable_a="educational TV viewing hours",
                variable_b="reading test scores",
                true_confounders=["parental involvement", "socioeconomic status", "general intelligence", "home learning environment"],
                red_herring_variables=["TV screen size", "broadcast quality", "viewing time of day"],
                causal_structure="Involved parents both encourage educational TV and support reading development; higher SES families have access to quality programming and books",
                domain="education",
                difficulty="medium",
                evidence_strength="moderate",
                alternative_explanations=["Educational TV actually improves vocabulary and comprehension"]
            ),
            ConfoundingScenario(
                observed_relationship="Patients who receive more expensive medical treatments have better recovery outcomes",
                variable_a="treatment cost",
                variable_b="recovery outcome",
                true_confounders=["disease severity", "patient health status", "hospital quality", "insurance coverage"],
                red_herring_variables=["treatment duration", "medication packaging", "hospital food quality"],
                causal_structure="Sicker patients receive more intensive (expensive) treatments; better hospitals charge more but also provide superior care",
                domain="medical",
                difficulty="hard",
                evidence_strength="strong",
                alternative_explanations=["Expensive treatments may actually be more effective due to newer technology"]
            ),
            ConfoundingScenario(
                observed_relationship="Neighborhoods with more police presence have higher crime rates",
                variable_a="police presence",
                variable_b="crime rates",
                true_confounders=["existing crime levels", "population density", "socioeconomic factors", "urban environment"],
                red_herring_variables=["police uniform color", "patrol car type", "shift schedules"],
                causal_structure="Police are deployed to areas with existing high crime; high-crime areas often have other social challenges",
                domain="public_safety",
                difficulty="medium",
                evidence_strength="strong",
                alternative_explanations=["Police presence may actually cause crime through community tension"]
            ),
            ConfoundingScenario(
                observed_relationship="Social media users who post more frequently report lower life satisfaction",
                variable_a="posting frequency",
                variable_b="life satisfaction",
                true_confounders=["underlying mental health", "social comparison tendency", "real-world social connections", "life circumstances"],
                red_herring_variables=["platform choice", "post length", "follower count"],
                causal_structure="People with lower life satisfaction seek validation through frequent posting; those lacking real-world connections compensate online",
                domain="technology",
                difficulty="hard",
                evidence_strength="moderate",
                alternative_explanations=["Frequent posting may reduce satisfaction through social comparison and addiction"]
            ),
            ConfoundingScenario(
                observed_relationship="Cities with more bike lanes have lower obesity rates",
                variable_a="bike lane density",
                variable_b="obesity rates",
                true_confounders=["urban planning philosophy", "socioeconomic demographics", "climate", "cultural attitudes toward health"],
                red_herring_variables=["bike lane color", "lane width", "signage type"],
                causal_structure="Health-conscious cities invest in both bike infrastructure and have populations that value fitness; wealthier areas can afford bike lanes and have lower obesity",
                domain="urban_planning",
                difficulty="medium",
                evidence_strength="moderate",
                alternative_explanations=["Bike lanes actually encourage cycling and reduce obesity"]
            ),
            ConfoundingScenario(
                observed_relationship="Companies with more diverse leadership teams show higher financial performance",
                variable_a="leadership diversity",
                variable_b="financial performance",
                true_confounders=["company size", "industry type", "geographic markets", "organizational culture", "innovation focus"],
                red_herring_variables=["meeting frequency", "office design", "communication tools"],
                causal_structure="Successful companies in global markets naturally develop diverse leadership; companies focused on innovation value diverse perspectives",
                domain="business",
                difficulty="hard",
                evidence_strength="moderate",
                alternative_explanations=["Diverse teams actually make better decisions leading to improved performance"]
            ),
            ConfoundingScenario(
                observed_relationship="Students who use laptops in class have lower grades than those who take handwritten notes",
                variable_a="laptop use in class",
                variable_b="course grades",
                true_confounders=["study habits", "attention levels", "prior academic performance", "course engagement"],
                red_herring_variables=["laptop brand", "typing speed", "screen brightness"],
                causal_structure="Students with poor attention/study habits choose laptops for convenience; academically weaker students rely on technology rather than developing learning skills",
                domain="education",
                difficulty="medium",
                evidence_strength="strong",
                alternative_explanations=["Laptop use may actually impair learning through multitasking and reduced retention"]
            ),
            ConfoundingScenario(
                observed_relationship="Hospitals with higher nurse-to-patient ratios report more medication errors",
                variable_a="nurse-to-patient ratio",
                variable_b="medication error rates",
                true_confounders=["patient acuity", "hospital type", "reporting practices", "safety culture", "workload complexity"],
                red_herring_variables=["nurse experience", "shift length", "uniform color"],
                causal_structure="ICUs and specialized units have both high nurse ratios and complex patients prone to errors; hospitals with better safety cultures both staff appropriately and report errors honestly",
                domain="medical",
                difficulty="hard",
                evidence_strength="strong",
                alternative_explanations=["More nurses may lead to communication breakdowns and coordination errors"]
            )
        ]
        
        # Filter by domain and difficulty if specified
        if hasattr(self.config, 'domain') and self.config.domain != "general":
            scenarios = [s for s in scenarios if s.domain == self.config.domain]
        
        if hasattr(self.config, 'difficulty') and self.config.difficulty != "mixed":
            scenarios = [s for s in scenarios if s.difficulty == self.config.difficulty]
        
        return scenarios
    
    async def generate_prompt(self) -> str:
        """Generate a confounding analysis prompt."""
        scenario = random.choice(self.scenarios)
        
        # Include some red herrings to test discrimination
        all_variables = scenario.true_confounders + scenario.red_herring_variables
        random.shuffle(all_variables)
        
        prompt = f"""
Analyze the following observed relationship for potential confounding variables.

Observed Relationship: {scenario.observed_relationship}

Variables:
- Variable A: {scenario.variable_a}  
- Variable B: {scenario.variable_b}

Potentially Relevant Variables: {', '.join(all_variables[:6])}

Questions to analyze:
1. Is the relationship between Variable A and Variable B likely to be causal?
2. What confounding variables might explain this relationship?
3. How would you control for these confounders in an analysis?

Please provide your analysis in the following format:
1. Causal Assessment: [Is this likely a causal relationship? Yes/No]
2. Confidence Level: [0.0 to 1.0]
3. Identified Confounders: [List the most important confounding variables]
4. Reasoning: [Explain your causal analysis and why these are confounders]
5. Causal Diagram: [Describe the relationships between all variables]
6. Controlled Analysis: [How would you design a study to control for confounders?]

Consider:
- What third variables might cause both A and B?
- Could the causation run in the reverse direction (B → A)?
- How would you distinguish correlation from causation?
- What would a randomized experiment look like?
"""
        
        # Store scenario for evaluation
        self._current_scenario = scenario
        return prompt.strip()
    
    async def evaluate_response(self, response: str) -> Dict[str, Any]:
        """Evaluate the model's confounding analysis response."""
        if not hasattr(self, '_current_scenario'):
            return {"error": "No current scenario available for evaluation"}
        
        scenario = self._current_scenario
        parsed_response = self._parse_response(response)
        
        # Calculate scores
        causal_assessment_score = self._score_causal_assessment(
            parsed_response.is_causal,
            scenario
        )
        
        confounder_identification_score = self._score_confounder_identification(
            parsed_response.identified_confounders,
            scenario.true_confounders,
            scenario.red_herring_variables
        )
        
        reasoning_score = self._score_reasoning_quality(
            parsed_response.reasoning,
            scenario
        )
        
        causal_diagram_score = self._score_causal_diagram(
            parsed_response.causal_diagram,
            scenario.causal_structure
        )
        
        controlled_analysis_score = self._score_controlled_analysis(
            parsed_response.controlled_analysis
        )
        
        # Overall score (weighted average)
        overall_score = (
            causal_assessment_score * 0.2 +
            confounder_identification_score * 0.3 +
            reasoning_score * 0.25 +
            causal_diagram_score * 0.15 +
            controlled_analysis_score * 0.1
        )
        
        return {
            "overall_score": overall_score,
            "causal_assessment_score": causal_assessment_score,
            "confounder_identification_score": confounder_identification_score,
            "reasoning_score": reasoning_score,
            "causal_diagram_score": causal_diagram_score,
            "controlled_analysis_score": controlled_analysis_score,
            "scenario_domain": scenario.domain,
            "scenario_difficulty": scenario.difficulty,
            "evidence_strength": scenario.evidence_strength,
            "predicted_causal": parsed_response.is_causal,
            "confidence": parsed_response.confidence,
            "identified_confounders": parsed_response.identified_confounders,
            "true_confounders": scenario.true_confounders,
            "red_herring_variables": scenario.red_herring_variables,
            "causal_structure": scenario.causal_structure,
            "model_reasoning": parsed_response.reasoning,
            "model_causal_diagram": parsed_response.causal_diagram,
            "model_controlled_analysis": parsed_response.controlled_analysis,
            "alternative_explanations": scenario.alternative_explanations
        }
    
    def _parse_response(self, response: str) -> ConfoundingResponse:
        """Parse the model's response into structured format."""
        # Extract causal assessment
        causal_match = re.search(
            r"(?:Causal Assessment|causal):\s*(yes|no|true|false|likely|unlikely)",
            response,
            re.IGNORECASE
        )
        is_causal = False
        if causal_match:
            causal_text = causal_match.group(1).lower()
            is_causal = causal_text in ["yes", "true", "likely"]
        
        # Extract confidence
        confidence_match = re.search(
            r"(?:Confidence|confidence).*?(\d+\.?\d*)",
            response
        )
        confidence = float(confidence_match.group(1)) if confidence_match else 0.5
        if confidence > 1.0:
            confidence = confidence / 100.0
        
        # Extract identified confounders
        confounders_match = re.search(
            r"(?:Identified Confounders?|confounders?):\s*(.+?)(?:\n\d+\.|$)",
            response,
            re.IGNORECASE | re.DOTALL
        )
        confounders_text = confounders_match.group(1) if confounders_match else ""
        identified_confounders = [
            conf.strip().strip("[]()\"'-•") 
            for conf in re.split(r"[,;\n]|\d+\.", confounders_text)
            if conf.strip()
        ]
        
        # Extract reasoning
        reasoning_match = re.search(
            r"(?:Reasoning|reasoning):\s*(.+?)(?:\n\d+\.|$)",
            response,
            re.IGNORECASE | re.DOTALL
        )
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        
        # Extract causal diagram
        diagram_match = re.search(
            r"(?:Causal Diagram|diagram):\s*(.+?)(?:\n\d+\.|$)",
            response,
            re.IGNORECASE | re.DOTALL
        )
        causal_diagram = diagram_match.group(1).strip() if diagram_match else ""
        
        # Extract controlled analysis
        controlled_match = re.search(
            r"(?:Controlled Analysis|controlled):\s*(.+?)(?:\n\d+\.|$)",
            response,
            re.IGNORECASE | re.DOTALL
        )
        controlled_analysis = controlled_match.group(1).strip() if controlled_match else ""
        
        return ConfoundingResponse(
            is_causal=is_causal,
            confidence=confidence,
            identified_confounders=identified_confounders,
            reasoning=reasoning,
            causal_diagram=causal_diagram,
            controlled_analysis=controlled_analysis
        )
    
    def _score_causal_assessment(self, predicted: bool, scenario: ConfoundingScenario) -> float:
        """Score the accuracy of causal vs correlational assessment."""
        # Most scenarios in this task are meant to highlight confounding (non-causal relationships)
        # But some might have elements of genuine causation
        
        # For scenarios with strong confounding, correct assessment is "not causal"
        if scenario.evidence_strength in ["strong"] and len(scenario.true_confounders) >= 3:
            expected_causal = False
        # For moderate scenarios, it's more ambiguous
        elif scenario.evidence_strength == "moderate":
            # Give partial credit regardless of answer, but reward sophisticated reasoning
            return 0.7  # Ambiguous cases deserve credit for either answer
        else:
            # Weak confounding might allow for some causal relationship
            expected_causal = True
        
        return 1.0 if predicted == expected_causal else 0.3  # Some credit for attempting assessment
    
    def _score_confounder_identification(self, identified: List[str], true_confounders: List[str], red_herrings: List[str]) -> float:
        """Score the identification of true confounders while avoiding red herrings."""
        if not true_confounders:
            return 1.0 if not identified else 0.8
        
        if not identified:
            return 0.2  # Small credit for difficulty of task
        
        score = 0.0
        
        # Score for correctly identifying true confounders
        identified_lower = [conf.lower() for conf in identified]
        true_lower = [conf.lower() for conf in true_confounders]
        red_herring_lower = [rh.lower() for rh in red_herrings]
        
        true_positives = 0
        for true_conf in true_lower:
            for identified_conf in identified_lower:
                # Check for semantic overlap
                true_words = set(true_conf.split())
                identified_words = set(identified_conf.split())
                
                if len(true_words & identified_words) >= 1:
                    true_positives += 1
                    break
        
        # Score for avoiding red herrings
        false_positives = 0
        for red_herring in red_herring_lower:
            for identified_conf in identified_lower:
                red_words = set(red_herring.split())
                identified_words = set(identified_conf.split())
                
                if len(red_words & identified_words) >= 1:
                    false_positives += 1
                    break
        
        # Calculate precision and recall
        precision = true_positives / len(identified) if identified else 0
        recall = true_positives / len(true_confounders) if true_confounders else 0
        
        # F1 score with penalty for false positives (red herrings)
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
        
        # Penalty for red herrings
        red_herring_penalty = false_positives * 0.1
        
        return max(f1_score - red_herring_penalty, 0.0)
    
    def _score_reasoning_quality(self, reasoning: str, scenario: ConfoundingScenario) -> float:
        """Score the quality of confounding analysis reasoning."""
        if not reasoning:
            return 0.0
        
        score = 0.0
        reasoning_lower = reasoning.lower()
        
        # Check for confounding analysis concepts
        confounding_concepts = {
            "confounder": 0.15,
            "confounding": 0.15,
            "third variable": 0.1,
            "spurious": 0.1,
            "correlation": 0.1,
            "causation": 0.1,
            "causal": 0.08,
            "bias": 0.05,
            "control": 0.05,
            "randomize": 0.05,
            "experiment": 0.05,
            "association": 0.02
        }
        
        for concept, weight in confounding_concepts.items():
            if concept in reasoning_lower:
                score += weight
        
        # Check for mention of specific confounders from the scenario
        confounder_mentions = 0
        for confounder in scenario.true_confounders:
            confounder_words = confounder.lower().split()
            for word in confounder_words:
                if len(word) > 3 and word in reasoning_lower:
                    confounder_mentions += 1
                    break
        
        confounder_score = min(confounder_mentions / len(scenario.true_confounders), 1.0) * 0.2
        score += confounder_score
        
        # Check for consideration of alternative explanations
        alternative_indicators = ["however", "alternatively", "could also", "might be", "possible", "reverse"]
        if any(indicator in reasoning_lower for indicator in alternative_indicators):
            score += 0.1
        
        # Check for domain-specific understanding
        domain_terms = {
            "medical": ["patient", "treatment", "clinical", "diagnosis"],
            "education": ["student", "learning", "academic", "school"],
            "business": ["company", "market", "performance", "economic"],
            "workplace": ["employee", "productivity", "job", "work"]
        }
        
        if scenario.domain in domain_terms:
            domain_score = sum(0.03 for term in domain_terms[scenario.domain] 
                             if term in reasoning_lower)
            score += min(domain_score, 0.1)
        
        return min(score, 1.0)
    
    def _score_causal_diagram(self, diagram: str, expected_structure: str) -> float:
        """Score the accuracy of causal diagram description."""
        if not diagram:
            return 0.3  # Some credit since diagrams are difficult
        
        diagram_lower = diagram.lower()
        structure_lower = expected_structure.lower()
        
        # Check for key structural elements
        structure_words = set(structure_lower.split()) - {"the", "a", "an", "and", "or", "but", "in", "on", "at"}
        diagram_words = set(diagram_lower.split())
        
        word_overlap = len(structure_words & diagram_words)
        if len(structure_words) > 0:
            semantic_score = word_overlap / len(structure_words)
        else:
            semantic_score = 0
        
        # Check for diagram elements (arrows, relationships)
        diagram_elements = ["arrow", "→", "->", "causes", "leads to", "influences", "affects"]
        has_diagram_elements = any(element in diagram_lower for element in diagram_elements)
        
        # Bonus for attempting to show relationships
        diagram_bonus = 0.3 if has_diagram_elements else 0.1
        
        return min(semantic_score + diagram_bonus, 1.0)
    
    def _score_controlled_analysis(self, controlled_analysis: str) -> float:
        """Score the quality of proposed controlled analysis."""
        if not controlled_analysis:
            return 0.4  # Some credit since this is advanced
        
        analysis_lower = controlled_analysis.lower()
        score = 0.0
        
        # Check for research design concepts
        design_concepts = {
            "randomize": 0.2,
            "control": 0.15,
            "experiment": 0.15,
            "match": 0.1,
            "stratify": 0.1,
            "regression": 0.1,
            "instrumental": 0.05,
            "natural experiment": 0.1,
            "longitudinal": 0.05
        }
        
        for concept, weight in design_concepts.items():
            if concept in analysis_lower:
                score += weight
        
        # Check for specific methodological terms
        methods = ["rct", "randomized controlled trial", "propensity score", "difference in difference"]
        if any(method in analysis_lower for method in methods):
            score += 0.15
        
        # Bonus for considering practical constraints
        practical_terms = ["feasible", "ethical", "practical", "cost", "time"]
        if any(term in analysis_lower for term in practical_terms):
            score += 0.1
        
        return min(score, 1.0)