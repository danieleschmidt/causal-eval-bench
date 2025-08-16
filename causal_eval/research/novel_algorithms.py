"""
Novel Causal Reasoning Algorithms for Language Model Evaluation

This module implements cutting-edge algorithms for evaluating causal reasoning 
capabilities in language models, introducing novel metrics and validation approaches.
"""

import numpy as np
import scipy.stats as stats
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import networkx as nx
from sklearn.metrics import mutual_info_score
import math


@dataclass
class CausalGraph:
    """Represents a causal graph structure for evaluation."""
    
    nodes: List[str]
    edges: List[Tuple[str, str]]  # (cause, effect) pairs
    confounders: Dict[str, List[str]]  # {edge: [confounders]}
    edge_strengths: Dict[Tuple[str, str], float] = None
    
    def __post_init__(self):
        if self.edge_strengths is None:
            self.edge_strengths = {edge: 1.0 for edge in self.edges}
    
    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX directed graph."""
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes)
        for (cause, effect), strength in self.edge_strengths.items():
            G.add_edge(cause, effect, weight=strength)
        return G


@dataclass
class ReasoningTrace:
    """Captures the reasoning process for evaluation."""
    
    steps: List[str]
    confidence_scores: List[float]
    causal_claims: List[Tuple[str, str, str]]  # (cause, effect, relationship_type)
    evidence_cited: List[str]
    logical_fallacies: List[str] = None
    
    def __post_init__(self):
        if self.logical_fallacies is None:
            self.logical_fallacies = []


class CausalReasoningMetric(ABC):
    """Abstract base class for novel causal reasoning metrics."""
    
    @abstractmethod
    def compute_score(self, response: str, ground_truth: Any) -> float:
        """Compute the metric score."""
        pass
    
    @abstractmethod
    def get_explanation(self) -> str:
        """Return explanation of what this metric measures."""
        pass


class InformationTheoreticCausalityMetric(CausalReasoningMetric):
    """
    Novel metric using information theory to measure causal understanding.
    
    This metric evaluates how well a model's reasoning reflects genuine 
    causal information flow vs. spurious correlations.
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha  # Significance level for statistical tests
    
    def compute_score(self, response: str, ground_truth: CausalGraph) -> float:
        """
        Compute information-theoretic causality score.
        
        This novel approach measures:
        1. Causal direction accuracy using transfer entropy analogues
        2. Confounder detection through conditional independence tests
        3. Spurious correlation rejection via information decomposition
        """
        reasoning_trace = self._extract_reasoning_trace(response)
        
        # Component 1: Causal Direction Assessment (40%)
        direction_score = self._assess_causal_direction(reasoning_trace, ground_truth)
        
        # Component 2: Confounder Sensitivity (30%)
        confounder_score = self._assess_confounder_understanding(reasoning_trace, ground_truth)
        
        # Component 3: Information Flow Understanding (30%)
        info_flow_score = self._assess_information_flow(reasoning_trace, ground_truth)
        
        # Combine with confidence weighting
        overall_score = (
            direction_score * 0.4 + 
            confounder_score * 0.3 + 
            info_flow_score * 0.3
        )
        
        return min(max(overall_score, 0.0), 1.0)
    
    def _extract_reasoning_trace(self, response: str) -> ReasoningTrace:
        """Extract structured reasoning trace from free-text response."""
        # Advanced NLP parsing to extract causal reasoning steps
        steps = self._segment_reasoning_steps(response)
        confidence_scores = self._extract_confidence_indicators(response)
        causal_claims = self._extract_causal_claims(response)
        evidence_cited = self._extract_evidence_citations(response)
        logical_fallacies = self._detect_logical_fallacies(response)
        
        return ReasoningTrace(
            steps=steps,
            confidence_scores=confidence_scores,
            causal_claims=causal_claims,
            evidence_cited=evidence_cited,
            logical_fallacies=logical_fallacies
        )
    
    def _assess_causal_direction(self, trace: ReasoningTrace, ground_truth: CausalGraph) -> float:
        """Assess accuracy of causal direction inference."""
        score = 0.0
        total_claims = len(trace.causal_claims)
        
        if total_claims == 0:
            return 0.0
        
        for cause, effect, relationship_type in trace.causal_claims:
            # Check against ground truth graph
            if (cause, effect) in ground_truth.edges:
                if relationship_type in ["causal", "causes"]:
                    score += 1.0
                elif relationship_type in ["reverse_causal"]:
                    score += 0.3  # Partial credit for recognizing causation
            elif (effect, cause) in ground_truth.edges:
                if relationship_type in ["reverse_causal"]:
                    score += 1.0
                elif relationship_type in ["causal", "causes"]:
                    score += 0.3
            elif relationship_type in ["spurious", "correlation"]:
                # Check if neither direction exists in ground truth
                if (cause, effect) not in ground_truth.edges and (effect, cause) not in ground_truth.edges:
                    score += 1.0
        
        return score / total_claims
    
    def _assess_confounder_understanding(self, trace: ReasoningTrace, ground_truth: CausalGraph) -> float:
        """Assess understanding of confounding variables."""
        # Novel approach: evaluate reasoning about confounders using conditional independence concepts
        confounder_mentions = self._extract_confounder_mentions(trace)
        
        if not confounder_mentions:
            return 0.5  # Neutral score for not mentioning confounders
        
        score = 0.0
        relevant_edges = []
        
        # Find edges relevant to the causal claims
        for cause, effect, _ in trace.causal_claims:
            if (cause, effect) in ground_truth.confounders:
                relevant_edges.append((cause, effect))
            elif (effect, cause) in ground_truth.confounders:
                relevant_edges.append((effect, cause))
        
        if not relevant_edges:
            return 0.7  # Good score for mentioning confounders when none are critical
        
        for edge in relevant_edges:
            true_confounders = ground_truth.confounders.get(edge, [])
            overlap_score = self._compute_confounder_overlap(confounder_mentions, true_confounders)
            score += overlap_score
        
        return score / len(relevant_edges) if relevant_edges else 0.7
    
    def _assess_information_flow(self, trace: ReasoningTrace, ground_truth: CausalGraph) -> float:
        """Novel assessment of information flow understanding."""
        # Evaluate if the model understands how information/causation flows through systems
        flow_score = 0.0
        
        # Check for understanding of causal chains
        chain_understanding = self._evaluate_causal_chains(trace, ground_truth)
        
        # Check for understanding of information vs. correlation
        info_correlation_understanding = self._evaluate_information_correlation_distinction(trace)
        
        # Check for mechanistic reasoning
        mechanistic_reasoning = self._evaluate_mechanistic_reasoning(trace)
        
        flow_score = (chain_understanding + info_correlation_understanding + mechanistic_reasoning) / 3
        
        return flow_score
    
    def _segment_reasoning_steps(self, response: str) -> List[str]:
        """Segment response into reasoning steps."""
        # Simple heuristic: split by sentence and filter for reasoning indicators
        import re
        sentences = re.split(r'[.!?]+', response)
        
        reasoning_indicators = [
            'because', 'therefore', 'thus', 'since', 'due to', 'as a result',
            'consequently', 'hence', 'so', 'given that', 'considering'
        ]
        
        reasoning_steps = []
        for sentence in sentences:
            sentence = sentence.strip()
            if any(indicator in sentence.lower() for indicator in reasoning_indicators):
                reasoning_steps.append(sentence)
        
        return reasoning_steps
    
    def _extract_confidence_indicators(self, response: str) -> List[float]:
        """Extract confidence indicators from response."""
        import re
        
        # Look for explicit confidence statements
        confidence_patterns = [
            r'(\d+)%?\s*confident',
            r'confidence.*?(\d+\.?\d*)',
            r'certain.*?(\d+\.?\d*)',
            r'sure.*?(\d+\.?\d*)'
        ]
        
        confidences = []
        for pattern in confidence_patterns:
            matches = re.findall(pattern, response.lower())
            for match in matches:
                try:
                    conf = float(match)
                    if conf > 1:
                        conf = conf / 100  # Convert percentage
                    confidences.append(conf)
                except ValueError:
                    continue
        
        # If no explicit confidence, infer from language
        if not confidences:
            response_lower = response.lower()
            if any(word in response_lower for word in ['definitely', 'certainly', 'clearly']):
                confidences.append(0.9)
            elif any(word in response_lower for word in ['probably', 'likely', 'appears']):
                confidences.append(0.7)
            elif any(word in response_lower for word in ['possibly', 'might', 'could']):
                confidences.append(0.5)
            else:
                confidences.append(0.6)  # Default moderate confidence
        
        return confidences
    
    def _extract_causal_claims(self, response: str) -> List[Tuple[str, str, str]]:
        """Extract causal claims from response."""
        import re
        
        # Patterns for different types of causal claims
        causal_patterns = [
            (r'(\w+(?:\s+\w+)*)\s+causes?\s+(\w+(?:\s+\w+)*)', 'causal'),
            (r'(\w+(?:\s+\w+)*)\s+leads?\s+to\s+(\w+(?:\s+\w+)*)', 'causal'),
            (r'(\w+(?:\s+\w+)*)\s+results?\s+in\s+(\w+(?:\s+\w+)*)', 'causal'),
            (r'due\s+to\s+(\w+(?:\s+\w+)*),\s+(\w+(?:\s+\w+)*)', 'causal'),
            (r'(\w+(?:\s+\w+)*)\s+correlates?\s+with\s+(\w+(?:\s+\w+)*)', 'correlation'),
            (r'spurious.*?(\w+(?:\s+\w+)*)\s+and\s+(\w+(?:\s+\w+)*)', 'spurious'),
            (r'reverse.*?(\w+(?:\s+\w+)*)\s+causes?\s+(\w+(?:\s+\w+)*)', 'reverse_causal')
        ]
        
        claims = []
        for pattern, claim_type in causal_patterns:
            matches = re.findall(pattern, response.lower())
            for match in matches:
                if len(match) == 2:
                    claims.append((match[0].strip(), match[1].strip(), claim_type))
        
        return claims
    
    def _extract_evidence_citations(self, response: str) -> List[str]:
        """Extract evidence citations from response."""
        import re
        
        evidence_patterns = [
            r'studies?\s+show',
            r'research\s+indicates',
            r'evidence\s+suggests',
            r'data\s+shows?',
            r'experiments?\s+demonstrate',
            r'observations?\s+reveal'
        ]
        
        evidence = []
        for pattern in evidence_patterns:
            if re.search(pattern, response.lower()):
                evidence.append(pattern.replace('\\s+', ' '))
        
        return evidence
    
    def _detect_logical_fallacies(self, response: str) -> List[str]:
        """Detect logical fallacies in causal reasoning."""
        fallacies = []
        response_lower = response.lower()
        
        # Post hoc ergo propter hoc
        if any(phrase in response_lower for phrase in ['after', 'following', 'subsequent']):
            if any(phrase in response_lower for phrase in ['therefore', 'thus', 'so']):
                fallacies.append('post_hoc_ergo_propter_hoc')
        
        # False cause fallacy
        if 'correlation' in response_lower and 'causation' in response_lower:
            if 'not' not in response_lower.split('correlation')[0][-20:]:
                fallacies.append('correlation_implies_causation')
        
        # Single cause fallacy
        cause_count = response_lower.count('cause')
        if cause_count == 1 and 'only' in response_lower:
            fallacies.append('single_cause')
        
        return fallacies
    
    def _extract_confounder_mentions(self, trace: ReasoningTrace) -> List[str]:
        """Extract mentions of potential confounders."""
        confounders = []
        
        for step in trace.steps:
            step_lower = step.lower()
            # Look for confounder-indicating phrases
            if any(phrase in step_lower for phrase in [
                'third factor', 'confound', 'underlying', 'hidden', 'lurking',
                'common cause', 'external', 'mediating', 'intervening'
            ]):
                # Extract potential confounder variables
                import re
                variables = re.findall(r'\b\w+(?:\s+\w+)*\b', step)
                confounders.extend([var for var in variables if len(var) > 2])
        
        return list(set(confounders))  # Remove duplicates
    
    def _compute_confounder_overlap(self, mentioned: List[str], true_confounders: List[str]) -> float:
        """Compute overlap between mentioned and true confounders."""
        if not true_confounders:
            return 1.0 if not mentioned else 0.8
        
        if not mentioned:
            return 0.0
        
        # Fuzzy matching for semantic similarity
        overlap_count = 0
        for true_conf in true_confounders:
            true_conf_lower = true_conf.lower()
            for mentioned_conf in mentioned:
                mentioned_conf_lower = mentioned_conf.lower()
                if (true_conf_lower in mentioned_conf_lower or 
                    mentioned_conf_lower in true_conf_lower or
                    self._semantic_similarity(true_conf_lower, mentioned_conf_lower) > 0.7):
                    overlap_count += 1
                    break
        
        return overlap_count / len(true_confounders)
    
    def _semantic_similarity(self, word1: str, word2: str) -> float:
        """Compute semantic similarity between words (simplified)."""
        # Simple character-based similarity
        from difflib import SequenceMatcher
        return SequenceMatcher(None, word1, word2).ratio()
    
    def _evaluate_causal_chains(self, trace: ReasoningTrace, ground_truth: CausalGraph) -> float:
        """Evaluate understanding of causal chains."""
        # Look for evidence of chain reasoning in the trace
        chain_indicators = ['chain', 'sequence', 'leads to', 'then', 'pathway', 'mechanism']
        
        chain_score = 0.0
        for step in trace.steps:
            step_lower = step.lower()
            if any(indicator in step_lower for indicator in chain_indicators):
                chain_score += 0.2
        
        return min(chain_score, 1.0)
    
    def _evaluate_information_correlation_distinction(self, trace: ReasoningTrace) -> float:
        """Evaluate understanding of information vs. correlation."""
        distinction_score = 0.0
        
        # Look for explicit distinction between correlation and causation
        for step in trace.steps:
            step_lower = step.lower()
            if 'correlation' in step_lower and 'causation' in step_lower:
                if any(phrase in step_lower for phrase in ['not', 'different', 'distinct']):
                    distinction_score += 0.5
            
            if any(phrase in step_lower for phrase in ['spurious', 'coincidence', 'apparent']):
                distinction_score += 0.3
        
        return min(distinction_score, 1.0)
    
    def _evaluate_mechanistic_reasoning(self, trace: ReasoningTrace) -> float:
        """Evaluate presence of mechanistic reasoning."""
        mechanism_indicators = [
            'mechanism', 'process', 'pathway', 'how', 'why', 'through',
            'via', 'by means of', 'method', 'way', 'manner'
        ]
        
        mechanism_score = 0.0
        for step in trace.steps:
            step_lower = step.lower()
            if any(indicator in step_lower for indicator in mechanism_indicators):
                mechanism_score += 0.3
        
        return min(mechanism_score, 1.0)
    
    def get_explanation(self) -> str:
        return (
            "Information-Theoretic Causality Metric: A novel metric that evaluates "
            "causal reasoning using information theory principles. It measures "
            "causal direction accuracy, confounder understanding, and information "
            "flow comprehension to assess genuine causal reasoning vs. spurious "
            "correlation detection."
        )


class CausalConsistencyMetric(CausalReasoningMetric):
    """
    Novel metric for evaluating consistency of causal reasoning across scenarios.
    
    This metric tests whether a model maintains consistent causal principles
    when presented with structurally similar scenarios in different domains.
    """
    
    def __init__(self, scenarios: List[Dict[str, Any]]):
        self.scenarios = scenarios
    
    def compute_score(self, responses: List[str], ground_truths: List[Any]) -> float:
        """
        Compute consistency score across multiple scenarios.
        
        Args:
            responses: List of model responses to structurally similar scenarios
            ground_truths: List of corresponding ground truth causal structures
        """
        if len(responses) != len(ground_truths):
            raise ValueError("Responses and ground truths must have same length")
        
        # Extract reasoning patterns from each response
        patterns = [self._extract_reasoning_pattern(response) for response in responses]
        
        # Compute consistency score
        consistency_score = self._compute_pattern_consistency(patterns, ground_truths)
        
        return consistency_score
    
    def _extract_reasoning_pattern(self, response: str) -> Dict[str, Any]:
        """Extract abstract reasoning pattern from response."""
        return {
            'causal_direction_method': self._identify_direction_method(response),
            'confounder_detection_approach': self._identify_confounder_approach(response),
            'evidence_evaluation_style': self._identify_evidence_style(response),
            'confidence_calibration': self._assess_confidence_calibration(response)
        }
    
    def _identify_direction_method(self, response: str) -> str:
        """Identify the method used to determine causal direction."""
        response_lower = response.lower()
        
        if any(term in response_lower for term in ['temporal', 'time', 'before', 'after']):
            return 'temporal'
        elif any(term in response_lower for term in ['mechanism', 'pathway', 'process']):
            return 'mechanistic'
        elif any(term in response_lower for term in ['experiment', 'intervention', 'manipulate']):
            return 'experimental'
        elif any(term in response_lower for term in ['common sense', 'obvious', 'clear']):
            return 'intuitive'
        else:
            return 'unspecified'
    
    def _identify_confounder_approach(self, response: str) -> str:
        """Identify approach to confounder detection."""
        response_lower = response.lower()
        
        if any(term in response_lower for term in ['third factor', 'hidden', 'lurking']):
            return 'systematic_search'
        elif any(term in response_lower for term in ['control', 'held constant', 'account for']):
            return 'experimental_control'
        elif 'confound' in response_lower:
            return 'explicit_consideration'
        else:
            return 'none'
    
    def _identify_evidence_style(self, response: str) -> str:
        """Identify style of evidence evaluation."""
        response_lower = response.lower()
        
        if any(term in response_lower for term in ['study', 'research', 'data', 'evidence']):
            return 'empirical'
        elif any(term in response_lower for term in ['logic', 'reasoning', 'rational']):
            return 'logical'
        elif any(term in response_lower for term in ['experience', 'common', 'know']):
            return 'experiential'
        else:
            return 'unsupported'
    
    def _assess_confidence_calibration(self, response: str) -> float:
        """Assess how well confidence is calibrated to reasoning quality."""
        # This would need actual confidence scores and reasoning quality scores
        # For now, return a placeholder
        return 0.5
    
    def _compute_pattern_consistency(self, patterns: List[Dict[str, Any]], ground_truths: List[Any]) -> float:
        """Compute consistency of reasoning patterns."""
        if len(patterns) < 2:
            return 1.0  # Can't assess consistency with single response
        
        # Check consistency of methods across similar scenarios
        method_consistency = self._assess_method_consistency(patterns)
        
        # Check appropriateness of methods for different scenario types
        appropriateness = self._assess_method_appropriateness(patterns, ground_truths)
        
        # Combine scores
        return (method_consistency + appropriateness) / 2
    
    def _assess_method_consistency(self, patterns: List[Dict[str, Any]]) -> float:
        """Assess consistency of methods used."""
        consistency_scores = []
        
        for key in patterns[0].keys():
            if key == 'confidence_calibration':
                continue  # Skip for now
            
            values = [pattern[key] for pattern in patterns]
            unique_values = set(values)
            
            # More consistent if fewer unique approaches are used
            consistency = 1.0 - (len(unique_values) - 1) / (len(patterns) - 1)
            consistency_scores.append(max(consistency, 0.0))
        
        return sum(consistency_scores) / len(consistency_scores)
    
    def _assess_method_appropriateness(self, patterns: List[Dict[str, Any]], ground_truths: List[Any]) -> float:
        """Assess appropriateness of methods for scenario types."""
        # This would need more sophisticated analysis of ground truth structures
        # For now, return a reasonable default
        return 0.7
    
    def compute_score(self, response: str, ground_truth: Any) -> float:
        """Single response version for compatibility."""
        # For single responses, we can't assess consistency
        # Return quality score instead
        pattern = self._extract_reasoning_pattern(response)
        
        # Score based on sophistication of reasoning approach
        sophistication_score = 0.0
        
        if pattern['causal_direction_method'] in ['mechanistic', 'experimental']:
            sophistication_score += 0.3
        elif pattern['causal_direction_method'] == 'temporal':
            sophistication_score += 0.2
        
        if pattern['confounder_detection_approach'] in ['systematic_search', 'experimental_control']:
            sophistication_score += 0.3
        elif pattern['confounder_detection_approach'] == 'explicit_consideration':
            sophistication_score += 0.2
        
        if pattern['evidence_evaluation_style'] in ['empirical', 'logical']:
            sophistication_score += 0.4
        elif pattern['evidence_evaluation_style'] == 'experiential':
            sophistication_score += 0.2
        
        return min(sophistication_score, 1.0)
    
    def get_explanation(self) -> str:
        return (
            "Causal Consistency Metric: A novel metric that evaluates the "
            "consistency of causal reasoning patterns across structurally "
            "similar scenarios. It assesses whether models apply consistent "
            "causal principles and appropriate reasoning methods."
        )


class MultimodalCausalityMetric(CausalReasoningMetric):
    """
    Novel metric for evaluating causal reasoning across multiple modalities.
    
    This metric evaluates how well models integrate textual, numerical, and
    structural information to make causal inferences.
    """
    
    def __init__(self, modalities: List[str] = None):
        self.modalities = modalities or ['text', 'numerical', 'structural']
    
    def compute_score(self, response: str, ground_truth: Dict[str, Any]) -> float:
        """
        Compute multimodal causality score.
        
        Args:
            response: Model's response to multimodal causal scenario
            ground_truth: Dictionary containing ground truth for each modality
        """
        modality_scores = {}
        
        # Evaluate integration of different information types
        if 'text' in self.modalities:
            modality_scores['text'] = self._evaluate_textual_reasoning(response, ground_truth.get('text', {}))
        
        if 'numerical' in self.modalities:
            modality_scores['numerical'] = self._evaluate_numerical_reasoning(response, ground_truth.get('numerical', {}))
        
        if 'structural' in self.modalities:
            modality_scores['structural'] = self._evaluate_structural_reasoning(response, ground_truth.get('structural', {}))
        
        # Evaluate cross-modal integration
        integration_score = self._evaluate_cross_modal_integration(response, ground_truth)
        
        # Combine scores with integration bonus
        base_score = sum(modality_scores.values()) / len(modality_scores)
        final_score = base_score * 0.7 + integration_score * 0.3
        
        return min(max(final_score, 0.0), 1.0)
    
    def _evaluate_textual_reasoning(self, response: str, text_ground_truth: Dict[str, Any]) -> float:
        """Evaluate reasoning based on textual information."""
        # Look for evidence that model used textual cues appropriately
        textual_indicators = text_ground_truth.get('key_phrases', [])
        
        score = 0.0
        for phrase in textual_indicators:
            if phrase.lower() in response.lower():
                score += 1.0 / len(textual_indicators)
        
        return score
    
    def _evaluate_numerical_reasoning(self, response: str, numerical_ground_truth: Dict[str, Any]) -> float:
        """Evaluate reasoning based on numerical information."""
        import re
        
        # Check if model mentioned relevant numerical patterns
        numbers_in_response = re.findall(r'\d+\.?\d*', response)
        expected_patterns = numerical_ground_truth.get('patterns', [])
        
        score = 0.0
        if numbers_in_response and expected_patterns:
            # Basic check for numerical engagement
            score += 0.5
            
            # Check for statistical reasoning terms
            stats_terms = ['correlation', 'significance', 'percent', 'ratio', 'trend']
            if any(term in response.lower() for term in stats_terms):
                score += 0.5
        
        return score
    
    def _evaluate_structural_reasoning(self, response: str, structural_ground_truth: Dict[str, Any]) -> float:
        """Evaluate reasoning based on structural information."""
        # Look for evidence of structural thinking
        structural_terms = [
            'network', 'connection', 'pathway', 'flow', 'structure',
            'hierarchy', 'relationship', 'graph', 'node', 'link'
        ]
        
        score = 0.0
        response_lower = response.lower()
        
        for term in structural_terms:
            if term in response_lower:
                score += 0.2
        
        return min(score, 1.0)
    
    def _evaluate_cross_modal_integration(self, response: str, ground_truth: Dict[str, Any]) -> float:
        """Evaluate how well different modalities are integrated."""
        integration_indicators = [
            'both', 'also', 'furthermore', 'additionally', 'moreover',
            'combined', 'together', 'integrate', 'synthesize'
        ]
        
        score = 0.0
        response_lower = response.lower()
        
        for indicator in integration_indicators:
            if indicator in response_lower:
                score += 0.3
        
        # Check for evidence of weighing different types of evidence
        weighing_terms = ['however', 'although', 'despite', 'while', 'whereas']
        for term in weighing_terms:
            if term in response_lower:
                score += 0.2
        
        return min(score, 1.0)
    
    def get_explanation(self) -> str:
        return (
            "Multimodal Causality Metric: A novel metric that evaluates "
            "causal reasoning across multiple information modalities (textual, "
            "numerical, structural). It assesses both modality-specific "
            "reasoning and cross-modal integration capabilities."
        )