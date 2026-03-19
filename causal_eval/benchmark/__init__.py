"""
Causal Eval Bench - Core Benchmark Framework

This module provides the scientific core of the benchmark:
  - CausalScenario: a test case with SCM, observational data, and questions
  - DataGenerator: creates synthetic SCM-based datasets
  - CausalEvaluator: scores model answers across 4 reasoning types
  - BenchmarkSuite: collects and runs a full benchmark

Question types (Pearl's Ladder of Causation):
  1. Association:      P(Y | X)                       - "what is?"
  2. Intervention:     P(Y | do(X=x))                 - "what if we do?"
  3. Counterfactual:   P(Y_x | X=x', Y=y')            - "what would have been?"
  4. Attribution:      What caused Y?                  - "why did Y happen?"
"""

from causal_eval.benchmark.scenario import (
    CausalScenario,
    CausalQuestion,
    QuestionType,
    Difficulty,
)
from causal_eval.benchmark.data_generator import DataGenerator, SCMNode
from causal_eval.benchmark.evaluator import CausalEvaluator, EvaluationResult
from causal_eval.benchmark.suite import BenchmarkSuite
from causal_eval.benchmark.scenarios import load_all_scenarios

__all__ = [
    "CausalScenario",
    "CausalQuestion",
    "QuestionType",
    "Difficulty",
    "DataGenerator",
    "SCMNode",
    "CausalEvaluator",
    "EvaluationResult",
    "BenchmarkSuite",
    "load_all_scenarios",
]
