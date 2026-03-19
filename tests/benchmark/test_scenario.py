"""
Tests for CausalScenario and CausalQuestion.
"""

import pytest
import numpy as np

from causal_eval.benchmark.scenario import (
    CausalQuestion,
    CausalScenario,
    Difficulty,
    QuestionType,
)


class TestCausalQuestion:
    def test_numeric_correct_within_tolerance(self):
        q = CausalQuestion("q1", QuestionType.ASSOCIATION, "P(Y>1)?", 0.8, answer_tolerance=0.05)
        assert q.is_correct(0.8)
        assert q.is_correct(0.83)
        assert q.is_correct(0.77)

    def test_numeric_wrong_outside_tolerance(self):
        q = CausalQuestion("q1", QuestionType.ASSOCIATION, "P(Y>1)?", 0.8, answer_tolerance=0.05)
        assert not q.is_correct(0.74)
        assert not q.is_correct(0.86)

    def test_string_correct(self):
        q = CausalQuestion("q2", QuestionType.ATTRIBUTION, "Who caused Y?", "X")
        assert q.is_correct("X")
        assert q.is_correct("x")
        assert q.is_correct("  X  ")

    def test_string_wrong(self):
        q = CausalQuestion("q2", QuestionType.ATTRIBUTION, "Who caused Y?", "X")
        assert not q.is_correct("Z")
        assert not q.is_correct("Y")

    def test_to_dict(self):
        q = CausalQuestion("q1", QuestionType.INTERVENTION, "do(X)?", 0.5)
        d = q.to_dict()
        assert d["question_id"] == "q1"
        assert d["question_type"] == "intervention"
        assert d["answer"] == 0.5

    def test_non_numeric_model_answer_for_numeric_truth(self):
        q = CausalQuestion("q1", QuestionType.ASSOCIATION, "?", 0.5)
        assert not q.is_correct("not_a_number")
        assert not q.is_correct(None)


class TestCausalScenario:
    def make_scenario(self):
        questions = [
            CausalQuestion("q1", QuestionType.ASSOCIATION, "assoc?", 0.5),
            CausalQuestion("q2", QuestionType.INTERVENTION, "interv?", 0.4),
            CausalQuestion("q3", QuestionType.COUNTERFACTUAL, "cf?", 0.7),
            CausalQuestion("q4", QuestionType.ATTRIBUTION, "attr?", "X"),
        ]
        return CausalScenario(
            scenario_id="test_s",
            name="Test Scenario",
            description="Test",
            difficulty=Difficulty.SIMPLE,
            graph_edges=[("X", "Y")],
            variable_names=["X", "Y"],
            questions=questions,
        )

    def test_questions_by_type(self):
        s = self.make_scenario()
        assert len(s.questions_by_type(QuestionType.ASSOCIATION)) == 1
        assert len(s.questions_by_type(QuestionType.ATTRIBUTION)) == 1

    def test_summary_contains_name(self):
        s = self.make_scenario()
        summary = s.summary()
        assert "Test Scenario" in summary
        assert "simple" in summary

    def test_to_dict_structure(self):
        s = self.make_scenario()
        d = s.to_dict()
        assert d["scenario_id"] == "test_s"
        assert d["difficulty"] == "simple"
        assert len(d["questions"]) == 4
        assert d["graph_edges"] == [("X", "Y")]

    def test_to_json(self):
        import json
        s = self.make_scenario()
        j = s.to_json()
        parsed = json.loads(j)
        assert parsed["scenario_id"] == "test_s"

    def test_get_data_df_requires_data(self):
        s = self.make_scenario()
        s.data = np.array([[1.0, 2.0], [3.0, 4.0]])
        s.data_columns = ["X", "Y"]
        df = s.get_data_df()
        assert list(df.columns) == ["X", "Y"]
        assert len(df) == 2
