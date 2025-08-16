"""
Baseline Models for Causal Reasoning Evaluation

This module implements sophisticated baseline models and benchmark systems
for rigorous comparison in causal reasoning research.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
import random
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path

from causal_eval.research.novel_algorithms import CausalGraph

logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    """Result from a baseline model evaluation."""
    
    model_name: str
    predicted_relationship: str
    confidence: float
    reasoning: str
    processing_time: float
    metadata: Dict[str, Any]


class CausalReasoningBaseline(ABC):
    """Abstract base class for causal reasoning baseline models."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_trained = False
        
    @abstractmethod
    def predict(self, scenario: str) -> BaselineResult:
        """Make a prediction on a causal scenario."""
        pass
    
    @abstractmethod
    def train(self, training_data: List[Dict[str, Any]]) -> None:
        """Train the baseline model (if applicable)."""
        pass
    
    def get_capability_description(self) -> str:
        """Return description of baseline's capabilities."""
        return f"Baseline model: {self.name}"


class RandomBaseline(CausalReasoningBaseline):
    """
    Random baseline that makes random predictions.
    
    This provides the lowest reasonable baseline - any model should 
    significantly outperform random chance.
    """
    
    def __init__(self, seed: int = 42):
        super().__init__("Random Baseline")
        self.random_state = np.random.RandomState(seed)
    
    def predict(self, scenario: str) -> BaselineResult:
        """Make random prediction."""
        import time
        start_time = time.time()
        
        relationships = ["causal", "spurious", "correlation", "reverse_causal"]
        predicted = self.random_state.choice(relationships)
        confidence = self.random_state.uniform(0.2, 0.8)  # Avoid extreme confidence
        
        reasoning = f"Random prediction: {predicted} (confidence: {confidence:.2f})"
        
        processing_time = time.time() - start_time
        
        return BaselineResult(
            model_name=self.name,
            predicted_relationship=predicted,
            confidence=confidence,
            reasoning=reasoning,
            processing_time=processing_time,
            metadata={"method": "random"}
        )
    
    def train(self, training_data: List[Dict[str, Any]]) -> None:
        """Random baseline doesn't need training."""
        self.is_trained = True
        logger.info("Random baseline marked as trained (no actual training needed)")
    
    def get_capability_description(self) -> str:
        return "Random baseline providing chance-level performance for comparison"


class KeywordBaseline(CausalReasoningBaseline):
    """
    Keyword-based baseline using simple heuristics.
    
    This baseline uses domain knowledge encoded as keyword patterns
    to make causal inferences.
    """
    
    def __init__(self):
        super().__init__("Keyword Heuristic Baseline")
        self.causal_keywords = [
            "causes", "leads to", "results in", "produces", "triggers",
            "due to", "because of", "as a result", "consequently"
        ]
        self.correlation_keywords = [
            "correlates", "associated with", "related to", "linked to"
        ]
        self.spurious_keywords = [
            "coincidence", "both increase", "both decrease", "appear together",
            "third factor", "confounding", "lurking variable"
        ]
        self.temporal_keywords = [
            "before", "after", "then", "subsequently", "following", "preceding"
        ]
    
    def predict(self, scenario: str) -> BaselineResult:
        """Make prediction based on keyword patterns."""
        import time
        start_time = time.time()
        
        scenario_lower = scenario.lower()
        
        # Score different relationship types
        causal_score = sum(1 for kw in self.causal_keywords if kw in scenario_lower)
        correlation_score = sum(1 for kw in self.correlation_keywords if kw in scenario_lower)
        spurious_score = sum(1 for kw in self.spurious_keywords if kw in scenario_lower)
        temporal_score = sum(1 for kw in self.temporal_keywords if kw in scenario_lower)
        
        # Special patterns
        reverse_indicators = ["reverse", "opposite", "backwards"]
        reverse_score = sum(1 for kw in reverse_indicators if kw in scenario_lower)
        
        # Determine prediction
        scores = {
            "causal": causal_score + temporal_score * 0.5,
            "correlation": correlation_score,
            "spurious": spurious_score,
            "reverse_causal": reverse_score + causal_score * 0.3
        }
        
        # If no strong indicators, look for domain-specific patterns
        if max(scores.values()) == 0:
            if any(word in scenario_lower for word in ["ice cream", "summer", "weather"]):
                predicted = "spurious"
                confidence = 0.6
            elif any(word in scenario_lower for word in ["study", "exercise", "medicine"]):
                predicted = "causal"
                confidence = 0.7
            else:
                predicted = "correlation"
                confidence = 0.4
        else:
            predicted = max(scores.items(), key=lambda x: x[1])[0]
            max_score = max(scores.values())
            confidence = min(0.5 + max_score * 0.1, 0.9)
        
        reasoning = f"Keyword analysis - {predicted} indicators: {scores[predicted]:.1f}"
        
        processing_time = time.time() - start_time
        
        return BaselineResult(
            model_name=self.name,
            predicted_relationship=predicted,
            confidence=confidence,
            reasoning=reasoning,
            processing_time=processing_time,
            metadata={"keyword_scores": scores, "method": "keyword_heuristic"}
        )
    
    def train(self, training_data: List[Dict[str, Any]]) -> None:
        """Keyword baseline uses predefined heuristics."""
        self.is_trained = True
        logger.info("Keyword baseline marked as trained (uses predefined heuristics)")
    
    def get_capability_description(self) -> str:
        return "Keyword-based heuristic baseline using causal reasoning indicators"


class MLBaseline(CausalReasoningBaseline):
    """
    Machine learning baseline using traditional NLP and classification.
    
    This baseline represents the current state-of-the-art in traditional
    machine learning approaches to causal reasoning.
    """
    
    def __init__(self, model_type: str = "logistic_regression"):
        super().__init__(f"ML Baseline ({model_type})")
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words='english'
        )
        
        if model_type == "logistic_regression":
            self.classifier = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == "naive_bayes":
            self.classifier = MultinomialNB()
        elif model_type == "random_forest":
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.label_encoder = {
            "causal": 0,
            "correlation": 1, 
            "spurious": 2,
            "reverse_causal": 3
        }
        self.label_decoder = {v: k for k, v in self.label_encoder.items()}
    
    def predict(self, scenario: str) -> BaselineResult:
        """Make ML-based prediction."""
        import time
        start_time = time.time()
        
        if not self.is_trained:
            logger.warning("ML baseline not trained, using fallback prediction")
            return BaselineResult(
                model_name=self.name,
                predicted_relationship="correlation",
                confidence=0.25,
                reasoning="Model not trained - fallback prediction",
                processing_time=time.time() - start_time,
                metadata={"method": "fallback"}
            )
        
        # Vectorize the scenario
        scenario_features = self.vectorizer.transform([scenario])
        
        # Get prediction and confidence
        prediction = self.classifier.predict(scenario_features)[0]
        prediction_proba = self.classifier.predict_proba(scenario_features)[0]
        
        predicted_relationship = self.label_decoder[prediction]
        confidence = float(np.max(prediction_proba))
        
        # Create reasoning explanation
        if hasattr(self.classifier, 'feature_importances_'):
            # For tree-based models, get feature importance
            reasoning = f"ML prediction based on feature importance analysis"
        elif hasattr(self.classifier, 'coef_'):
            # For linear models, get top weighted features
            feature_names = self.vectorizer.get_feature_names_out()
            class_coef = self.classifier.coef_[prediction]
            top_indices = np.argsort(np.abs(class_coef))[-5:]
            top_features = [feature_names[i] for i in top_indices]
            reasoning = f"ML prediction based on key features: {', '.join(top_features)}"
        else:
            reasoning = f"ML prediction using {self.model_type}"
        
        processing_time = time.time() - start_time
        
        return BaselineResult(
            model_name=self.name,
            predicted_relationship=predicted_relationship,
            confidence=confidence,
            reasoning=reasoning,
            processing_time=processing_time,
            metadata={
                "method": "machine_learning",
                "model_type": self.model_type,
                "prediction_proba": prediction_proba.tolist()
            }
        )
    
    def train(self, training_data: List[Dict[str, Any]]) -> None:
        """Train the ML baseline on provided data."""
        logger.info(f"Training ML baseline with {len(training_data)} examples")
        
        texts = []
        labels = []
        
        for example in training_data:
            if 'scenario' in example and 'relationship' in example:
                texts.append(example['scenario'])
                if example['relationship'] in self.label_encoder:
                    labels.append(self.label_encoder[example['relationship']])
        
        if len(texts) < 10:
            logger.warning("Insufficient training data for ML baseline")
            return
        
        # Fit vectorizer and transform texts
        X = self.vectorizer.fit_transform(texts)
        y = np.array(labels)
        
        # Train classifier
        self.classifier.fit(X, y)
        self.is_trained = True
        
        # Log training performance
        train_score = self.classifier.score(X, y)
        logger.info(f"ML baseline training accuracy: {train_score:.3f}")
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        if not self.is_trained:
            logger.warning("Cannot save untrained model")
            return
        
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'label_encoder': self.label_encoder,
            'label_decoder': self.label_decoder,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"ML baseline saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a previously trained model."""
        model_data = joblib.load(filepath)
        
        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        self.label_encoder = model_data['label_encoder']
        self.label_decoder = model_data['label_decoder']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"ML baseline loaded from {filepath}")
    
    def get_capability_description(self) -> str:
        return f"Traditional ML baseline using {self.model_type} with TF-IDF features"


class TemporalReasoningBaseline(CausalReasoningBaseline):
    """
    Baseline that focuses on temporal reasoning for causality.
    
    This baseline represents approaches that primarily use temporal
    information to infer causal relationships.
    """
    
    def __init__(self):
        super().__init__("Temporal Reasoning Baseline")
        self.temporal_patterns = {
            "before_after": [
                r"(\w+)\s+before\s+(\w+)", 
                r"(\w+)\s+then\s+(\w+)",
                r"(\w+)\s+followed by\s+(\w+)"
            ],
            "causal_temporal": [
                r"after\s+(\w+),\s+(\w+)",
                r"when\s+(\w+),\s+(\w+)",
                r"once\s+(\w+),\s+(\w+)"
            ]
        }
    
    def predict(self, scenario: str) -> BaselineResult:
        """Make prediction based on temporal reasoning."""
        import time
        start_time = time.time()
        
        # Look for temporal patterns
        temporal_score = 0
        temporal_evidence = []
        
        for pattern_type, patterns in self.temporal_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, scenario.lower())
                if matches:
                    temporal_score += len(matches)
                    temporal_evidence.append(f"{pattern_type}: {matches}")
        
        # Look for explicit temporal indicators
        temporal_indicators = [
            "before", "after", "then", "subsequently", "following",
            "preceding", "earlier", "later", "first", "second"
        ]
        
        indicator_count = sum(1 for indicator in temporal_indicators 
                            if indicator in scenario.lower())
        
        # Determine prediction based on temporal evidence
        if temporal_score > 2 or indicator_count > 2:
            predicted = "causal"
            confidence = min(0.6 + temporal_score * 0.1, 0.9)
        elif temporal_score > 0 or indicator_count > 0:
            predicted = "correlation"
            confidence = 0.5 + temporal_score * 0.05
        else:
            # No temporal evidence - default to spurious
            predicted = "spurious"
            confidence = 0.4
        
        # Check for reverse temporal indicators
        reverse_patterns = ["result of", "due to", "caused by"]
        if any(pattern in scenario.lower() for pattern in reverse_patterns):
            if predicted == "causal":
                predicted = "reverse_causal"
                confidence *= 0.9
        
        reasoning = f"Temporal analysis - indicators: {indicator_count}, patterns: {temporal_score}"
        if temporal_evidence:
            reasoning += f", evidence: {temporal_evidence[:2]}"
        
        processing_time = time.time() - start_time
        
        return BaselineResult(
            model_name=self.name,
            predicted_relationship=predicted,
            confidence=confidence,
            reasoning=reasoning,
            processing_time=processing_time,
            metadata={
                "temporal_score": temporal_score,
                "indicator_count": indicator_count,
                "temporal_evidence": temporal_evidence,
                "method": "temporal_reasoning"
            }
        )
    
    def train(self, training_data: List[Dict[str, Any]]) -> None:
        """Temporal baseline uses predefined patterns."""
        self.is_trained = True
        logger.info("Temporal baseline marked as trained (uses predefined patterns)")
    
    def get_capability_description(self) -> str:
        return "Temporal reasoning baseline focusing on time-based causal indicators"


class DomainKnowledgeBaseline(CausalReasoningBaseline):
    """
    Baseline that uses domain-specific knowledge for causal reasoning.
    
    This baseline encodes expert knowledge about causality in specific domains.
    """
    
    def __init__(self):
        super().__init__("Domain Knowledge Baseline")
        self.domain_rules = {
            "medical": {
                "causal_indicators": ["medication", "treatment", "therapy", "surgery", "dose"],
                "effect_indicators": ["symptom", "recovery", "improvement", "side effect"],
                "confounders": ["age", "severity", "lifestyle", "genetics"]
            },
            "business": {
                "causal_indicators": ["investment", "marketing", "advertising", "training"],
                "effect_indicators": ["sales", "revenue", "profit", "performance"],
                "confounders": ["market", "competition", "economy", "season"]
            },
            "education": {
                "causal_indicators": ["study", "practice", "teaching", "tutoring"],
                "effect_indicators": ["grade", "score", "performance", "learning"],
                "confounders": ["motivation", "ability", "background", "time"]
            },
            "environmental": {
                "causal_indicators": ["pollution", "emission", "deforestation", "warming"],
                "effect_indicators": ["temperature", "species", "climate", "ecosystem"],
                "confounders": ["natural", "cycles", "solar", "volcanic"]
            }
        }
    
    def predict(self, scenario: str) -> BaselineResult:
        """Make prediction based on domain knowledge."""
        import time
        start_time = time.time()
        
        scenario_lower = scenario.lower()
        
        # Detect domain
        detected_domain = self._detect_domain(scenario_lower)
        
        if detected_domain:
            domain_rules = self.domain_rules[detected_domain]
            
            # Count indicators
            causal_count = sum(1 for indicator in domain_rules["causal_indicators"]
                             if indicator in scenario_lower)
            effect_count = sum(1 for indicator in domain_rules["effect_indicators"]
                             if indicator in scenario_lower)
            confounder_count = sum(1 for conf in domain_rules["confounders"]
                                 if conf in scenario_lower)
            
            # Make prediction based on domain knowledge
            if causal_count > 0 and effect_count > 0:
                if confounder_count > 0:
                    predicted = "spurious"
                    confidence = 0.6 + confounder_count * 0.1
                    reasoning = f"Domain knowledge ({detected_domain}): potential confounders detected"
                else:
                    predicted = "causal"
                    confidence = 0.7 + min(causal_count, effect_count) * 0.1
                    reasoning = f"Domain knowledge ({detected_domain}): clear causal pattern"
            else:
                predicted = "correlation"
                confidence = 0.5
                reasoning = f"Domain knowledge ({detected_domain}): insufficient causal evidence"
        else:
            # No specific domain detected - fall back to general heuristics
            predicted = "correlation"
            confidence = 0.4
            reasoning = "No specific domain detected - general correlation assumption"
        
        processing_time = time.time() - start_time
        
        return BaselineResult(
            model_name=self.name,
            predicted_relationship=predicted,
            confidence=confidence,
            reasoning=reasoning,
            processing_time=processing_time,
            metadata={
                "detected_domain": detected_domain,
                "method": "domain_knowledge"
            }
        )
    
    def _detect_domain(self, scenario: str) -> Optional[str]:
        """Detect the domain of the scenario."""
        domain_keywords = {
            "medical": ["patient", "doctor", "hospital", "medicine", "health", "disease", "treatment"],
            "business": ["company", "sales", "profit", "market", "customer", "revenue", "business"],
            "education": ["student", "teacher", "school", "study", "learn", "grade", "education"],
            "environmental": ["environment", "climate", "pollution", "ecosystem", "species", "warming"]
        }
        
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in scenario)
            domain_scores[domain] = score
        
        max_score = max(domain_scores.values())
        if max_score > 0:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        return None
    
    def train(self, training_data: List[Dict[str, Any]]) -> None:
        """Domain knowledge baseline uses expert-encoded rules."""
        self.is_trained = True
        logger.info("Domain knowledge baseline marked as trained (uses expert rules)")
    
    def get_capability_description(self) -> str:
        return "Domain knowledge baseline using expert-encoded causal patterns"


class BaselineEvaluator:
    """
    Evaluator for baseline models with comprehensive benchmarking.
    
    This class manages multiple baseline models and provides standardized
    evaluation across different baseline approaches.
    """
    
    def __init__(self):
        self.baselines = {
            "random": RandomBaseline(),
            "keyword": KeywordBaseline(),
            "ml_logistic": MLBaseline("logistic_regression"),
            "ml_naive_bayes": MLBaseline("naive_bayes"),
            "ml_random_forest": MLBaseline("random_forest"),
            "temporal": TemporalReasoningBaseline(),
            "domain_knowledge": DomainKnowledgeBaseline()
        }
        
        self.evaluation_history = []
        
    def evaluate_all_baselines(
        self, 
        scenarios: List[str],
        ground_truths: List[str],
        training_data: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Evaluate all baselines on given scenarios."""
        
        # Train baselines if training data provided
        if training_data:
            logger.info("Training baselines with provided data")
            for baseline in self.baselines.values():
                try:
                    baseline.train(training_data)
                except Exception as e:
                    logger.warning(f"Error training {baseline.name}: {e}")
        
        results = {}
        
        for baseline_name, baseline in self.baselines.items():
            logger.info(f"Evaluating baseline: {baseline_name}")
            
            baseline_results = []
            correct_predictions = 0
            total_processing_time = 0
            
            for scenario, ground_truth in zip(scenarios, ground_truths):
                try:
                    result = baseline.predict(scenario)
                    baseline_results.append(result)
                    
                    if result.predicted_relationship == ground_truth:
                        correct_predictions += 1
                    
                    total_processing_time += result.processing_time
                    
                except Exception as e:
                    logger.error(f"Error in {baseline_name} prediction: {e}")
                    # Add error result
                    baseline_results.append(BaselineResult(
                        model_name=baseline.name,
                        predicted_relationship="error",
                        confidence=0.0,
                        reasoning=f"Error: {str(e)}",
                        processing_time=0.0,
                        metadata={"error": str(e)}
                    ))
            
            # Calculate performance metrics
            accuracy = correct_predictions / len(scenarios) if scenarios else 0
            avg_confidence = np.mean([r.confidence for r in baseline_results 
                                    if r.predicted_relationship != "error"])
            avg_processing_time = total_processing_time / len(scenarios) if scenarios else 0
            
            results[baseline_name] = {
                "model": baseline,
                "results": baseline_results,
                "accuracy": accuracy,
                "correct_predictions": correct_predictions,
                "total_predictions": len(scenarios),
                "average_confidence": avg_confidence,
                "average_processing_time": avg_processing_time,
                "capability_description": baseline.get_capability_description()
            }
        
        # Store evaluation history
        evaluation_record = {
            "timestamp": np.datetime64('now'),
            "num_scenarios": len(scenarios),
            "results": results
        }
        self.evaluation_history.append(evaluation_record)
        
        return results
    
    def get_baseline_rankings(self, results: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Get baseline rankings by accuracy."""
        rankings = [(name, data["accuracy"]) for name, data in results.items()]
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def generate_baseline_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive baseline evaluation report."""
        
        rankings = self.get_baseline_rankings(results)
        
        report = {
            "evaluation_summary": {
                "num_baselines": len(results),
                "best_baseline": rankings[0][0] if rankings else None,
                "best_accuracy": rankings[0][1] if rankings else 0,
                "worst_baseline": rankings[-1][0] if rankings else None,
                "worst_accuracy": rankings[-1][1] if rankings else 0,
                "accuracy_range": rankings[0][1] - rankings[-1][1] if len(rankings) > 1 else 0
            },
            "detailed_results": {},
            "performance_analysis": {},
            "recommendations": []
        }
        
        # Detailed results for each baseline
        for baseline_name, data in results.items():
            report["detailed_results"][baseline_name] = {
                "accuracy": data["accuracy"],
                "average_confidence": data["average_confidence"],
                "processing_time": data["average_processing_time"],
                "description": data["capability_description"]
            }
        
        # Performance analysis
        accuracies = [data["accuracy"] for data in results.values()]
        confidences = [data["average_confidence"] for data in results.values()]
        times = [data["average_processing_time"] for data in results.values()]
        
        report["performance_analysis"] = {
            "accuracy_statistics": {
                "mean": float(np.mean(accuracies)),
                "std": float(np.std(accuracies)),
                "min": float(np.min(accuracies)),
                "max": float(np.max(accuracies))
            },
            "confidence_statistics": {
                "mean": float(np.mean(confidences)),
                "std": float(np.std(confidences)),
                "min": float(np.min(confidences)),
                "max": float(np.max(confidences))
            },
            "timing_statistics": {
                "mean": float(np.mean(times)),
                "std": float(np.std(times)),
                "min": float(np.min(times)),
                "max": float(np.max(times))
            }
        }
        
        # Generate recommendations
        best_baseline = rankings[0][0] if rankings else None
        if best_baseline:
            report["recommendations"].append(
                f"Use {best_baseline} as primary baseline for comparison "
                f"(accuracy: {rankings[0][1]:.3f})"
            )
        
        if len(rankings) > 1:
            accuracy_gap = rankings[0][1] - rankings[1][1]
            if accuracy_gap > 0.1:
                report["recommendations"].append(
                    "Large performance gap between best and second-best baseline "
                    "suggests room for improvement"
                )
        
        random_accuracy = results.get("random", {}).get("accuracy", 0.25)
        best_accuracy = rankings[0][1] if rankings else 0
        if best_accuracy - random_accuracy < 0.2:
            report["recommendations"].append(
                "Low performance gap above random baseline suggests "
                "challenging evaluation scenarios"
            )
        
        return report
    
    def save_baselines(self, directory: str) -> None:
        """Save trained baselines to directory."""
        output_dir = Path(directory)
        output_dir.mkdir(exist_ok=True)
        
        for name, baseline in self.baselines.items():
            if isinstance(baseline, MLBaseline) and baseline.is_trained:
                filepath = output_dir / f"{name}_baseline.joblib"
                baseline.save_model(str(filepath))
        
        logger.info(f"Baselines saved to {directory}")
    
    def load_baselines(self, directory: str) -> None:
        """Load trained baselines from directory."""
        input_dir = Path(directory)
        
        for name, baseline in self.baselines.items():
            if isinstance(baseline, MLBaseline):
                filepath = input_dir / f"{name}_baseline.joblib"
                if filepath.exists():
                    baseline.load_model(str(filepath))
        
        logger.info(f"Baselines loaded from {directory}")
    
    def compare_with_baseline(
        self, 
        model_results: List[str],
        ground_truths: List[str],
        baseline_name: str = "best"
    ) -> Dict[str, Any]:
        """Compare model performance with specific baseline."""
        
        if baseline_name == "best":
            # Use the best performing baseline
            scenarios = [f"Scenario {i}" for i in range(len(model_results))]  # Placeholder
            baseline_eval = self.evaluate_all_baselines(scenarios, ground_truths)
            rankings = self.get_baseline_rankings(baseline_eval)
            baseline_name = rankings[0][0] if rankings else "random"
        
        if baseline_name not in self.baselines:
            logger.error(f"Unknown baseline: {baseline_name}")
            return {}
        
        # Calculate model accuracy
        model_accuracy = sum(1 for pred, truth in zip(model_results, ground_truths) 
                           if pred == truth) / len(model_results)
        
        # Get baseline performance (approximate)
        baseline_accuracy = 0.25  # Default random performance
        if hasattr(self, 'evaluation_history') and self.evaluation_history:
            last_eval = self.evaluation_history[-1]["results"]
            if baseline_name in last_eval:
                baseline_accuracy = last_eval[baseline_name]["accuracy"]
        
        performance_gap = model_accuracy - baseline_accuracy
        
        return {
            "model_accuracy": model_accuracy,
            "baseline_accuracy": baseline_accuracy,
            "baseline_name": baseline_name,
            "performance_gap": performance_gap,
            "improvement_factor": model_accuracy / baseline_accuracy if baseline_accuracy > 0 else float('inf'),
            "comparison_summary": self._generate_comparison_summary(
                model_accuracy, baseline_accuracy, baseline_name
            )
        }
    
    def _generate_comparison_summary(
        self, 
        model_accuracy: float, 
        baseline_accuracy: float, 
        baseline_name: str
    ) -> str:
        """Generate human-readable comparison summary."""
        
        performance_gap = model_accuracy - baseline_accuracy
        
        if performance_gap > 0.2:
            return f"Model significantly outperforms {baseline_name} baseline " \
                   f"(+{performance_gap:.1%} accuracy)"
        elif performance_gap > 0.1:
            return f"Model moderately outperforms {baseline_name} baseline " \
                   f"(+{performance_gap:.1%} accuracy)"
        elif performance_gap > 0.05:
            return f"Model slightly outperforms {baseline_name} baseline " \
                   f"(+{performance_gap:.1%} accuracy)"
        elif performance_gap > -0.05:
            return f"Model performs comparably to {baseline_name} baseline " \
                   f"({performance_gap:+.1%} accuracy)"
        else:
            return f"Model underperforms {baseline_name} baseline " \
                   f"({performance_gap:.1%} accuracy)"