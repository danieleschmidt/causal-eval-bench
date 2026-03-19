#!/usr/bin/env python3
"""
Causal Eval Bench — Demo Script

Demonstrates:
  1. Loading all 12 benchmark scenarios
  2. Printing scenario summaries
  3. Showing sample questions from each type (Pearl's Ladder)
  4. Evaluating two baselines: random and oracle
  5. Printing a leaderboard

Run with:
    ~/anaconda3/bin/python3 demo_benchmark.py
"""

import sys
import numpy as np

# Make sure the package is importable from the repo root
sys.path.insert(0, ".")

from causal_eval.benchmark import (
    BenchmarkSuite,
    CausalEvaluator,
    load_all_scenarios,
)
from causal_eval.benchmark.scenario import QuestionType


def main():
    print("\n🔬 Causal Eval Bench — Evaluation Framework Demo")
    print("   Testing genuine causal reasoning beyond correlation\n")

    # ------------------------------------------------------------------ #
    # 1. Load scenarios                                                    #
    # ------------------------------------------------------------------ #
    print("Loading benchmark scenarios...")
    np.random.seed(42)
    scenarios = load_all_scenarios()
    suite = BenchmarkSuite(scenarios)
    print(f"  ✓ Loaded {len(scenarios)} scenarios with "
          f"{sum(len(s.questions) for s in scenarios)} questions total\n")

    # ------------------------------------------------------------------ #
    # 2. Scenario summary                                                  #
    # ------------------------------------------------------------------ #
    suite.print_scenario_summary()

    # ------------------------------------------------------------------ #
    # 3. Sample questions                                                  #
    # ------------------------------------------------------------------ #
    suite.print_sample_questions(n_per_type=2)

    # ------------------------------------------------------------------ #
    # 4. Evaluate baselines                                                #
    # ------------------------------------------------------------------ #
    print("\n📊 Running evaluations...")

    evaluator = CausalEvaluator()

    # Random baseline
    np.random.seed(0)
    random_result = evaluator.evaluate(
        scenarios,
        CausalEvaluator.random_baseline,
        model_name="random",
    )

    # Oracle baseline
    oracle_result = evaluator.evaluate(
        scenarios,
        CausalEvaluator.oracle_baseline,
        model_name="oracle",
    )

    # Stubborn baseline (always answer 0.5 for numeric)
    def stubborn_half(q):
        if isinstance(q.answer, (int, float)):
            return 0.5
        return "no"

    stubborn_result = evaluator.evaluate(
        scenarios,
        stubborn_half,
        model_name="stubborn-0.5",
    )

    results = {
        "random": random_result,
        "oracle": oracle_result,
        "stubborn-0.5": stubborn_result,
    }

    # ------------------------------------------------------------------ #
    # 5. Individual summaries                                              #
    # ------------------------------------------------------------------ #
    for name, result in results.items():
        print(f"\n{result.summary()}")

    # ------------------------------------------------------------------ #
    # 6. Leaderboard                                                       #
    # ------------------------------------------------------------------ #
    suite.print_leaderboard(results)

    # ------------------------------------------------------------------ #
    # 7. Difficulty breakdown for oracle (sanity check)                    #
    # ------------------------------------------------------------------ #
    print("\n📈 Oracle accuracy by difficulty:")
    from causal_eval.benchmark.scenario import Difficulty
    for diff in Difficulty:
        diff_scenarios = suite.by_difficulty(diff).scenarios
        if diff_scenarios:
            diff_result = evaluator.evaluate(
                diff_scenarios,
                CausalEvaluator.oracle_baseline,
                model_name=f"oracle-{diff.value}",
            )
            print(f"  {diff.value:<8}  {diff_result.accuracy:.1%}  "
                  f"({diff_result.correct}/{diff_result.total_questions} questions)")

    # ------------------------------------------------------------------ #
    # 8. Show key causal insights                                          #
    # ------------------------------------------------------------------ #
    print("\n\n💡 Key causal distinctions this benchmark tests:")
    insights = [
        ("S04 Spurious correlation",
         "X,Y correlated, but E[Y|do(X=2)]=0 (both caused by Z)"),
        ("S05 Confounding bias",
         "E[Y|X=1]=1.0 (biased) vs E[Y|do(X=1)]=0.4 (true effect)"),
        ("S06 Collider bias",
         "X,Y independent but negatively corr. within hospitals (C conditioned)"),
        ("S08 Front-door criterion",
         "Unobserved U blocks back-door; front-door via M identifies effect"),
        ("S10 IV estimation",
         "IV recovers 0.6 despite unobserved confounder"),
        ("S09 Direct vs indirect",
         "Total=0.55; direct=0.3 (X→Y); indirect=0.25 (X→M→Y)"),
    ]
    for title, insight in insights:
        print(f"  • {title}: {insight}")

    print("\n✅ Demo complete!\n")


if __name__ == "__main__":
    main()
