"""
CausalEvaluator: Scores model answers on causal reasoning questions.

The evaluator accepts a "model function" — a callable that takes a
CausalQuestion and returns an answer — then aggregates accuracy
per question type and overall.

Supports both exact-match and tolerance-based numeric scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from causal_eval.benchmark.scenario import (
    CausalQuestion,
    CausalScenario,
    QuestionType,
)


# A model function: takes a question, returns an answer
ModelFn = Callable[[CausalQuestion], Any]


@dataclass
class QuestionResult:
    """Result for a single question."""

    question_id: str
    question_type: QuestionType
    model_answer: Any
    ground_truth: Any
    correct: bool
    scenario_id: str = ""


@dataclass
class EvaluationResult:
    """
    Aggregated result from evaluating a model on a benchmark suite.

    Attributes
    ----------
    model_name : str
        Name/label for the model being evaluated.
    total_questions : int
    correct : int
    accuracy : float
        Overall accuracy across all question types.
    per_type_accuracy : Dict[QuestionType, float]
        Accuracy broken down by question type.
    per_type_correct : Dict[QuestionType, int]
    per_type_total : Dict[QuestionType, int]
    question_results : List[QuestionResult]
        Fine-grained per-question results.
    """

    model_name: str
    total_questions: int = 0
    correct: int = 0
    accuracy: float = 0.0
    per_type_accuracy: Dict[QuestionType, float] = field(default_factory=dict)
    per_type_correct: Dict[QuestionType, int] = field(default_factory=dict)
    per_type_total: Dict[QuestionType, int] = field(default_factory=dict)
    question_results: List[QuestionResult] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Model: {self.model_name}",
            f"  Overall accuracy:  {self.accuracy:.1%}  "
            f"({self.correct}/{self.total_questions})",
        ]
        for qt in QuestionType:
            if qt in self.per_type_total and self.per_type_total[qt] > 0:
                acc = self.per_type_accuracy.get(qt, 0.0)
                n = self.per_type_total[qt]
                c = self.per_type_correct[qt]
                lines.append(
                    f"  {qt.value:<16}  {acc:.1%}  ({c}/{n})"
                )
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "total_questions": self.total_questions,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 4),
            "per_type_accuracy": {
                k.value: round(v, 4) for k, v in self.per_type_accuracy.items()
            },
            "per_type_correct": {k.value: v for k, v in self.per_type_correct.items()},
            "per_type_total": {k.value: v for k, v in self.per_type_total.items()},
        }


class CausalEvaluator:
    """
    Evaluates a model (or baseline) on a set of CausalScenarios.

    Usage
    -----
    >>> evaluator = CausalEvaluator()
    >>> result = evaluator.evaluate(scenarios, model_fn, model_name="oracle")
    >>> print(result.summary())

    The model_fn receives a CausalQuestion and should return an answer.
    For numeric probability questions, return a float in [0, 1].
    For attribution string questions, return a string matching the answer.
    """

    def evaluate(
        self,
        scenarios: List[CausalScenario],
        model_fn: ModelFn,
        model_name: str = "model",
    ) -> EvaluationResult:
        """
        Evaluate a model on all questions in the given scenarios.

        Parameters
        ----------
        scenarios : list of CausalScenario
        model_fn : callable (CausalQuestion) -> Any
        model_name : str

        Returns
        -------
        EvaluationResult
        """
        result = EvaluationResult(model_name=model_name)
        per_type_correct: Dict[QuestionType, int] = {qt: 0 for qt in QuestionType}
        per_type_total: Dict[QuestionType, int] = {qt: 0 for qt in QuestionType}
        question_results: List[QuestionResult] = []

        for scenario in scenarios:
            for question in scenario.questions:
                model_answer = model_fn(question)
                correct = question.is_correct(model_answer)

                per_type_correct[question.question_type] += int(correct)
                per_type_total[question.question_type] += 1
                result.total_questions += 1
                result.correct += int(correct)

                question_results.append(
                    QuestionResult(
                        question_id=question.question_id,
                        question_type=question.question_type,
                        model_answer=model_answer,
                        ground_truth=question.answer,
                        correct=correct,
                        scenario_id=scenario.scenario_id,
                    )
                )

        # Aggregate
        result.accuracy = (
            result.correct / result.total_questions if result.total_questions > 0 else 0.0
        )
        for qt in QuestionType:
            result.per_type_correct[qt] = per_type_correct[qt]
            result.per_type_total[qt] = per_type_total[qt]
            if per_type_total[qt] > 0:
                result.per_type_accuracy[qt] = per_type_correct[qt] / per_type_total[qt]
            else:
                result.per_type_accuracy[qt] = 0.0

        result.question_results = question_results
        return result

    def score(
        self,
        scenarios: List[CausalScenario],
        model_fn: ModelFn,
        model_name: str = "model",
    ) -> Dict[str, float]:
        """
        Shorthand that returns a simple dict of per-type accuracy scores.

        Returns
        -------
        dict with keys: "overall", "association", "intervention",
                        "counterfactual", "attribution"
        """
        result = self.evaluate(scenarios, model_fn, model_name)
        scores = {"overall": result.accuracy}
        for qt in QuestionType:
            scores[qt.value] = result.per_type_accuracy.get(qt, 0.0)
        return scores

    # ------------------------------------------------------------------ #
    # Built-in baselines                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def random_baseline(question: CausalQuestion) -> Any:
        """Random baseline: returns 0.5 for numeric, random label for strings."""
        import numpy as np

        if isinstance(question.answer, (int, float)):
            return float(np.random.uniform(0, 1))
        # For attribution questions, guess randomly from metadata if available
        choices = question.metadata.get("choices", ["A", "B"])
        return str(np.random.choice(choices))

    @staticmethod
    def oracle_baseline(question: CausalQuestion) -> Any:
        """Oracle baseline: always returns the ground-truth answer."""
        return question.answer

    @staticmethod
    def majority_baseline(
        questions: List[CausalQuestion],
    ) -> Callable[[CausalQuestion], Any]:
        """
        Majority baseline: always answers with the most common answer.
        Returns a model_fn callable.
        """
        from collections import Counter

        counts: Counter = Counter()
        for q in questions:
            if isinstance(q.answer, (int, float)):
                # Bin probabilities into 0 or 1
                counts[round(float(q.answer))] += 1
            else:
                counts[q.answer] += 1

        majority = counts.most_common(1)[0][0] if counts else 0

        def _fn(q: CausalQuestion) -> Any:
            return majority

        return _fn
