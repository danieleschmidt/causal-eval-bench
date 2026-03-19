"""
Tests for the pre-built benchmark scenarios.
Verifies scenario structure and sanity-checks ground-truth answers via simulation.
"""

import pytest
import numpy as np

from causal_eval.benchmark.scenarios import load_all_scenarios, load_by_difficulty
from causal_eval.benchmark.scenario import Difficulty, QuestionType
from causal_eval.benchmark.evaluator import CausalEvaluator


class TestScenarioRegistry:
    def test_load_all_returns_12_scenarios(self):
        scenarios = load_all_scenarios()
        assert len(scenarios) == 12

    def test_all_scenario_ids_unique(self):
        scenarios = load_all_scenarios()
        ids = [s.scenario_id for s in scenarios]
        assert len(ids) == len(set(ids))

    def test_each_scenario_has_all_four_question_types(self):
        scenarios = load_all_scenarios()
        for scenario in scenarios:
            types_present = {q.question_type for q in scenario.questions}
            for qt in QuestionType:
                assert qt in types_present, (
                    f"Scenario {scenario.scenario_id} missing question type {qt}"
                )

    def test_difficulty_distribution(self):
        scenarios = load_all_scenarios()
        by_diff = {d: [] for d in Difficulty}
        for s in scenarios:
            by_diff[s.difficulty].append(s)
        assert len(by_diff[Difficulty.SIMPLE]) >= 4
        assert len(by_diff[Difficulty.MEDIUM]) >= 4
        assert len(by_diff[Difficulty.HARD]) >= 4

    def test_all_scenarios_have_data(self):
        scenarios = load_all_scenarios()
        for s in scenarios:
            assert s.data is not None, f"{s.scenario_id} has no data"
            assert s.data.shape[0] > 100, f"{s.scenario_id} has too few samples"
            assert len(s.data_columns) == s.data.shape[1]

    def test_all_scenarios_have_graph_edges(self):
        scenarios = load_all_scenarios()
        for s in scenarios:
            assert len(s.graph_edges) > 0, f"{s.scenario_id} has no edges"

    def test_load_by_difficulty_simple(self):
        simple = load_by_difficulty(Difficulty.SIMPLE)
        assert all(s.difficulty == Difficulty.SIMPLE for s in simple)
        assert len(simple) >= 4

    def test_oracle_scores_100_on_all_scenarios(self):
        """Oracle baseline must be perfect — validates ground-truth answers are self-consistent."""
        evaluator = CausalEvaluator()
        scenarios = load_all_scenarios()
        result = evaluator.evaluate(scenarios, CausalEvaluator.oracle_baseline, "oracle")
        assert result.accuracy == 1.0, (
            f"Oracle baseline should be 100% accurate. "
            f"Got {result.accuracy:.1%}. Failing questions: "
            + str([qr.question_id for qr in result.question_results if not qr.correct])
        )

    def test_scenario_summaries_non_empty(self):
        scenarios = load_all_scenarios()
        for s in scenarios:
            summary = s.summary()
            assert len(summary) > 10
            assert s.name in summary


class TestScenario01SimpleChain:
    """Sanity-check key properties of S01."""

    @pytest.fixture
    def scenario(self):
        from causal_eval.benchmark.scenarios import scenario_01_simple_chain
        return scenario_01_simple_chain()

    def test_has_two_variables(self, scenario):
        assert len(scenario.variable_names) == 2

    def test_one_edge_X_to_Y(self, scenario):
        assert ("X", "Y") in scenario.graph_edges
        assert len(scenario.graph_edges) == 1

    def test_association_intervention_same_no_confounder(self, scenario):
        """With no confounder, association and intervention answers should match."""
        assoc_q = scenario.questions_by_type(QuestionType.ASSOCIATION)[0]
        interv_q = scenario.questions_by_type(QuestionType.INTERVENTION)[0]
        assert abs(assoc_q.answer - interv_q.answer) < 0.01


class TestScenario04SpuriousCorrelation:
    """S04: X,Y both caused by Z — do(X) should not affect Y."""

    @pytest.fixture
    def scenario(self):
        from causal_eval.benchmark.scenarios import scenario_04_no_causal_effect
        return scenario_04_no_causal_effect()

    def test_intervention_answer_near_zero(self, scenario):
        interv_q = scenario.questions_by_type(QuestionType.INTERVENTION)[0]
        assert abs(float(interv_q.answer) - 0.0) < 0.1

    def test_attribution_answer_is_Z(self, scenario):
        attr_q = scenario.questions_by_type(QuestionType.ATTRIBUTION)[0]
        assert "Z" in str(attr_q.answer)


class TestScenario05Confounding:
    """S05: Confounded estimate should differ from interventional estimate."""

    @pytest.fixture
    def scenario(self):
        from causal_eval.benchmark.scenarios import scenario_05_confounding
        return scenario_05_confounding()

    def test_assoc_intervention_differ(self, scenario):
        assoc_q = scenario.questions_by_type(QuestionType.ASSOCIATION)[0]
        interv_q = scenario.questions_by_type(QuestionType.INTERVENTION)[0]
        # Observational estimate should be larger (positive confounding)
        assert float(assoc_q.answer) > float(interv_q.answer) + 0.3


class TestScenario09MultiPath:
    """S09: Direct + indirect effects sum to total."""

    @pytest.fixture
    def scenario(self):
        from causal_eval.benchmark.scenarios import scenario_09_multi_path
        return scenario_09_multi_path()

    def test_total_effect_is_sum(self, scenario):
        # Total = direct + indirect = 0.3 + 0.25 = 0.55
        total_q = scenario.questions_by_type(QuestionType.ASSOCIATION)[0]
        assert abs(float(total_q.answer) - 0.55) < 0.05
