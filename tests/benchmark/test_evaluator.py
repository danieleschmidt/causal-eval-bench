"""
Tests for CausalEvaluator and EvaluationResult.
"""

import pytest
import numpy as np

from causal_eval.benchmark.evaluator import CausalEvaluator, EvaluationResult
from causal_eval.benchmark.scenario import (
    CausalQuestion,
    CausalScenario,
    Difficulty,
    QuestionType,
)


def make_simple_scenario(scenario_id="test", n_of_each=2):
    """Make a scenario with n_of_each questions per type."""
    questions = []
    idx = 1
    for qt in QuestionType:
        for _ in range(n_of_each):
            if qt == QuestionType.ATTRIBUTION:
                q = CausalQuestion(f"q{idx}", qt, "?", "X", metadata={"choices": ["X", "Y"]})
            else:
                q = CausalQuestion(f"q{idx}", qt, "?", 0.5)
            questions.append(q)
            idx += 1
    return CausalScenario(
        scenario_id=scenario_id,
        name="Test",
        description="Test",
        difficulty=Difficulty.SIMPLE,
        graph_edges=[("X", "Y")],
        variable_names=["X", "Y"],
        questions=questions,
    )


class TestCausalEvaluator:
    def test_oracle_perfect_score(self):
        evaluator = CausalEvaluator()
        scenario = make_simple_scenario()
        result = evaluator.evaluate([scenario], CausalEvaluator.oracle_baseline, "oracle")
        assert result.accuracy == 1.0
        assert result.correct == result.total_questions

    def test_random_baseline_below_oracle(self):
        np.random.seed(0)
        evaluator = CausalEvaluator()
        scenarios = [make_simple_scenario(f"s{i}") for i in range(5)]
        random_result = evaluator.evaluate(scenarios, CausalEvaluator.random_baseline, "random")
        oracle_result = evaluator.evaluate(scenarios, CausalEvaluator.oracle_baseline, "oracle")
        assert random_result.accuracy < oracle_result.accuracy

    def test_per_type_accuracy_all_correct(self):
        evaluator = CausalEvaluator()
        scenario = make_simple_scenario()
        result = evaluator.evaluate([scenario], CausalEvaluator.oracle_baseline, "oracle")
        for qt in QuestionType:
            assert result.per_type_accuracy[qt] == 1.0

    def test_per_type_accuracy_all_wrong(self):
        evaluator = CausalEvaluator()
        scenario = make_simple_scenario()

        def always_wrong(q):
            if isinstance(q.answer, (int, float)):
                return q.answer + 1.0  # clearly wrong
            return "definitely_wrong"

        result = evaluator.evaluate([scenario], always_wrong, "wrong")
        assert result.accuracy == 0.0
        for qt in QuestionType:
            assert result.per_type_accuracy[qt] == 0.0

    def test_score_returns_dict(self):
        evaluator = CausalEvaluator()
        scenario = make_simple_scenario()
        scores = evaluator.score([scenario], CausalEvaluator.oracle_baseline, "oracle")
        assert "overall" in scores
        assert "association" in scores
        assert "intervention" in scores
        assert "counterfactual" in scores
        assert "attribution" in scores
        assert scores["overall"] == 1.0

    def test_empty_scenarios_zero_accuracy(self):
        evaluator = CausalEvaluator()
        result = evaluator.evaluate([], CausalEvaluator.oracle_baseline, "oracle")
        assert result.accuracy == 0.0
        assert result.total_questions == 0

    def test_question_results_collected(self):
        evaluator = CausalEvaluator()
        scenario = make_simple_scenario()
        result = evaluator.evaluate([scenario], CausalEvaluator.oracle_baseline, "oracle")
        assert len(result.question_results) == len(scenario.questions)
        for qr in result.question_results:
            assert qr.correct is True

    def test_partial_correct(self):
        evaluator = CausalEvaluator()
        q_correct = CausalQuestion("q1", QuestionType.ASSOCIATION, "?", 0.5)
        q_wrong = CausalQuestion("q2", QuestionType.INTERVENTION, "?", 0.5)
        scenario = CausalScenario(
            scenario_id="partial",
            name="Partial",
            description="",
            difficulty=Difficulty.SIMPLE,
            graph_edges=[],
            variable_names=["X"],
            questions=[q_correct, q_wrong],
        )

        def half_correct(q):
            if q.question_id == "q1":
                return 0.5  # correct
            return 0.0  # wrong (far from 0.5)

        result = evaluator.evaluate([scenario], half_correct, "half")
        assert result.correct == 1
        assert result.total_questions == 2
        assert abs(result.accuracy - 0.5) < 0.01

    def test_majority_baseline_factory(self):
        questions = [
            CausalQuestion("q1", QuestionType.ATTRIBUTION, "?", "X"),
            CausalQuestion("q2", QuestionType.ATTRIBUTION, "?", "X"),
            CausalQuestion("q3", QuestionType.ATTRIBUTION, "?", "Y"),
        ]
        scenario = CausalScenario(
            scenario_id="maj",
            name="Majority",
            description="",
            difficulty=Difficulty.SIMPLE,
            graph_edges=[],
            variable_names=["X", "Y"],
            questions=questions,
        )
        majority_fn = CausalEvaluator.majority_baseline(questions)
        evaluator = CausalEvaluator()
        result = evaluator.evaluate([scenario], majority_fn, "majority")
        # Majority answer is "X" (2/3 questions), so 2 correct
        assert result.correct == 2


class TestEvaluationResultSummary:
    def test_summary_contains_model_name(self):
        result = EvaluationResult(model_name="test_model", total_questions=10, correct=7)
        result.accuracy = 0.7
        for qt in QuestionType:
            result.per_type_accuracy[qt] = 0.7
            result.per_type_total[qt] = 2
            result.per_type_correct[qt] = 1
        summary = result.summary()
        assert "test_model" in summary
        assert "70.0%" in summary

    def test_to_dict_keys(self):
        result = EvaluationResult(model_name="m", total_questions=5, correct=4)
        result.accuracy = 0.8
        d = result.to_dict()
        assert "model_name" in d
        assert "accuracy" in d
        assert "per_type_accuracy" in d
