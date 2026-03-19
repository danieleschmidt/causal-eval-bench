"""
BenchmarkSuite: Orchestrates evaluation of models across all scenarios.

Usage:
    from causal_eval.benchmark import BenchmarkSuite, CausalEvaluator
    from causal_eval.benchmark.scenarios import load_all_scenarios

    suite = BenchmarkSuite(load_all_scenarios())
    results = suite.run_all([
        ("random",  CausalEvaluator.random_baseline),
        ("oracle",  CausalEvaluator.oracle_baseline),
    ])
    suite.print_leaderboard(results)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from causal_eval.benchmark.evaluator import CausalEvaluator, EvaluationResult
from causal_eval.benchmark.scenario import CausalScenario, Difficulty, QuestionType


class BenchmarkSuite:
    """
    Manages a collection of CausalScenarios and runs evaluations.

    Parameters
    ----------
    scenarios : list of CausalScenario
    """

    def __init__(self, scenarios: List[CausalScenario]):
        self.scenarios = scenarios
        self._evaluator = CausalEvaluator()

    # ------------------------------------------------------------------ #
    # Evaluation                                                           #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        model_fn: Callable,
        model_name: str = "model",
        scenarios: Optional[List[CausalScenario]] = None,
    ) -> EvaluationResult:
        """Evaluate a single model on all (or given) scenarios."""
        target = scenarios if scenarios is not None else self.scenarios
        return self._evaluator.evaluate(target, model_fn, model_name)

    def run_all(
        self,
        models: List[Tuple[str, Callable]],
        scenarios: Optional[List[CausalScenario]] = None,
    ) -> Dict[str, EvaluationResult]:
        """
        Run multiple models and collect results.

        Parameters
        ----------
        models : list of (name, model_fn) pairs
        scenarios : optional subset of scenarios

        Returns
        -------
        dict of model_name → EvaluationResult
        """
        results = {}
        for name, fn in models:
            results[name] = self.evaluate(fn, name, scenarios)
        return results

    # ------------------------------------------------------------------ #
    # Reporting                                                            #
    # ------------------------------------------------------------------ #

    def print_leaderboard(
        self,
        results: Dict[str, EvaluationResult],
        show_per_type: bool = True,
    ) -> None:
        """Pretty-print a leaderboard table."""
        print("\n" + "=" * 65)
        print("  CAUSAL EVAL BENCH — LEADERBOARD")
        print("=" * 65)

        # Sort by overall accuracy descending
        sorted_models = sorted(results.items(), key=lambda x: x[1].accuracy, reverse=True)

        header = f"{'Model':<20} {'Overall':>8}"
        if show_per_type:
            for qt in QuestionType:
                header += f"  {qt.value[:8]:>8}"
        print(header)
        print("-" * 65)

        for rank, (name, result) in enumerate(sorted_models, 1):
            row = f"{rank}. {name:<17} {result.accuracy:>7.1%}"
            if show_per_type:
                for qt in QuestionType:
                    acc = result.per_type_accuracy.get(qt, 0.0)
                    row += f"  {acc:>7.1%}"
            print(row)

        print("=" * 65)
        if show_per_type:
            print(
                f"{'Col headers':>20}  overall"
                + "".join(f"  {qt.value[:8]:>8}" for qt in QuestionType)
            )

    def print_scenario_summary(self) -> None:
        """Print a summary of all loaded scenarios."""
        print("\n" + "=" * 65)
        print("  BENCHMARK SCENARIOS")
        print("=" * 65)
        for s in self.scenarios:
            diff_label = {
                Difficulty.SIMPLE: "[ SIMPLE ]",
                Difficulty.MEDIUM: "[ MEDIUM ]",
                Difficulty.HARD:   "[  HARD  ]",
            }[s.difficulty]
            q_counts = []
            for qt in QuestionType:
                n = len(s.questions_by_type(qt))
                if n:
                    q_counts.append(f"{qt.value[:4]}={n}")
            print(
                f"  {s.scenario_id}  {diff_label}  {s.name[:40]:<40}"
                f"  [{', '.join(q_counts)}]"
            )
        print(f"\n  Total: {len(self.scenarios)} scenarios, "
              f"{sum(len(s.questions) for s in self.scenarios)} questions")
        print("=" * 65)

    def print_sample_questions(self, n_per_type: int = 1) -> None:
        """Print sample questions from each type."""
        print("\n" + "=" * 65)
        print("  SAMPLE QUESTIONS (by type)")
        print("=" * 65)
        seen: Dict[QuestionType, int] = {qt: 0 for qt in QuestionType}
        for scenario in self.scenarios:
            for q in scenario.questions:
                if seen[q.question_type] < n_per_type:
                    print(f"\n  [{q.question_type.value.upper()}]  ({scenario.name})")
                    print(f"  Q: {q.text}")
                    print(f"  A: {q.answer}")
                    seen[q.question_type] += 1
        print()

    # ------------------------------------------------------------------ #
    # Filtering                                                            #
    # ------------------------------------------------------------------ #

    def by_difficulty(self, difficulty: Difficulty) -> "BenchmarkSuite":
        return BenchmarkSuite([s for s in self.scenarios if s.difficulty == difficulty])

    def by_tag(self, tag: str) -> "BenchmarkSuite":
        return BenchmarkSuite([s for s in self.scenarios if tag in s.tags])

    def __len__(self) -> int:
        return len(self.scenarios)

    def __repr__(self) -> str:
        return f"BenchmarkSuite(n_scenarios={len(self.scenarios)})"
