# Causal Eval Bench

**Evaluation framework for testing genuine causal reasoning in LLMs — beyond correlation.**

[![Tests](https://img.shields.io/badge/tests-55%20passed-brightgreen)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-86%25-green)](htmlcov/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

---

## Motivation

Language models can appear to reason causally while merely exploiting statistical patterns.
A model that has memorized "smoking correlates with cancer" might answer causal questions
correctly for the wrong reason. Causal Eval Bench tests whether models can distinguish:

| Level | Question type | Example |
|-------|--------------|---------|
| Rung 1 | **Association** | What is P(Y \| X)? |
| Rung 2 | **Intervention** | What is P(Y \| **do(X)**=x)? |
| Rung 3 | **Counterfactual** | What would Y have been if X had been x instead? |
| +1 | **Attribution** | What caused Y? |

These correspond to Pearl's Ladder of Causation (Pearl 2009).

---

## Architecture

```
causal_eval/benchmark/
├── scenario.py        # CausalScenario, CausalQuestion, QuestionType, Difficulty
├── data_generator.py  # DataGenerator, SCMNode — forward/interventional/counterfactual sampling
├── evaluator.py       # CausalEvaluator — scores model answers, baselines
├── scenarios.py       # 12 pre-built benchmark scenarios (S01–S12)
└── suite.py           # BenchmarkSuite — orchestration + leaderboard
```

### Core Classes

#### `CausalScenario`
A benchmark test case grounded in a Structural Causal Model (SCM):
- **Causal graph** (DAG edges)
- **Observational data** sampled from the SCM
- **Questions** with ground-truth answers across all 4 question types

#### `DataGenerator`
Implements forward sampling from an SCM using the [Abduction–Action–Prediction] framework:
- **Observational**: `gen.sample(n=1000)`
- **Interventional**: `gen.do({"X": 1.0}, n=1000)` — mutilates the DAG
- **Counterfactual**: `gen.counterfactual(observed={"X":0,"Y":0.3}, interventions={"X":2})` — twin-network

#### `CausalEvaluator`
Scores a model function on a set of scenarios:
```python
def my_model(question: CausalQuestion) -> float | str:
    ...  # return predicted answer

result = evaluator.evaluate(scenarios, my_model, model_name="gpt-4")
print(result.summary())
```

---

## Benchmark Scenarios

12 scenarios across three difficulty levels:

| ID | Difficulty | Name | Key concept |
|----|-----------|------|------------|
| S01 | Simple | Simple Linear Chain X→Y | Direct cause, no confounders |
| S02 | Simple | Binary Treatment→Outcome | Vaccine effect, necessity/sufficiency |
| S03 | Simple | Mediation X→M→Y | Full mediation, indirect effect |
| S04 | Simple | Spurious Correlation Z→X, Z→Y | Common cause, no X→Y edge |
| S05 | Medium | Classic Confounding | Back-door path, biased estimate |
| S06 | Medium | Collider Bias X→C←Y | Berkson's paradox, selection bias |
| S07 | Medium | Non-linear Threshold (Sigmoid) | Non-linear causal reasoning |
| S08 | Medium | Front-Door Criterion | Unobserved confounder, identification |
| S09 | Hard | Multi-Path (Direct + Mediated) | NDE vs NIE, path decomposition |
| S10 | Hard | Instrumental Variable | Unobserved confounder, IV estimation |
| S11 | Hard | Lagged Effect (Time Series) | Granger causality, temporal reasoning |
| S12 | Hard | Complex 5-variable DAG | Multi-path + confounder |

### Key Causal Distinctions

```
S04: E[Y|X=2]  = 1.9  ≠  E[Y|do(X=2)] = 0.0   (spurious correlation)
S05: E[Y|X=1]  = 1.0  ≠  E[Y|do(X=1)] = 0.4   (confounding bias)
S08: E[Y|X=1]  biased     E[Y|do(X=1)] = 0.28  (front-door criterion)
S06: Corr(X,Y) = 0.0   Corr(X,Y|C) < 0         (collider bias)
S09: Total=0.55, Direct=0.30, Indirect=0.25     (path decomposition)
```

---

## Quick Start

```python
from causal_eval.benchmark import (
    BenchmarkSuite,
    CausalEvaluator,
    load_all_scenarios,
)

# Load all 12 scenarios
scenarios = load_all_scenarios()
suite = BenchmarkSuite(scenarios)
suite.print_scenario_summary()

# Define your model
def my_llm_model(question):
    # question.text contains the human-readable question
    # question.question_type is QuestionType.ASSOCIATION/INTERVENTION/COUNTERFACTUAL/ATTRIBUTION
    # question.metadata contains variable names, intervention values, etc.
    return your_llm_call(question.text)

# Evaluate
evaluator = CausalEvaluator()
result = evaluator.evaluate(scenarios, my_llm_model, model_name="my-llm")
print(result.summary())

# Compare multiple models
results = suite.run_all([
    ("random",  CausalEvaluator.random_baseline),
    ("oracle",  CausalEvaluator.oracle_baseline),
    ("my-llm",  my_llm_model),
])
suite.print_leaderboard(results)
```

### Running the Demo

```bash
python3 demo_benchmark.py
```

Output (excerpt):
```
=================================================================
  CAUSAL EVAL BENCH — LEADERBOARD
=================================================================
Model                 Overall  associat  interven  counterf  attribut
-----------------------------------------------------------------
1. oracle             100.0%   100.0%   100.0%   100.0%   100.0%
2. stubborn-0.5        16.0%    16.7%     0.0%    16.7%    33.3%
3. random              14.0%     8.3%     0.0%    16.7%    33.3%
=================================================================
```

---

## Building Your Own Scenarios

```python
from causal_eval.benchmark.data_generator import DataGenerator, SCMNode
from causal_eval.benchmark.scenario import CausalScenario, CausalQuestion, QuestionType, Difficulty
import numpy as np

# Define the SCM
gen = DataGenerator(seed=42)
gen.add_node(SCMNode(
    "X", [],
    equation=lambda pa, n: n,
    noise_fn=lambda n: np.random.normal(0, 1, n),
))
gen.add_node(SCMNode(
    "Y", ["X"],
    equation=lambda pa, n: 0.5 * pa["X"] + n,
    noise_fn=lambda n: np.random.normal(0, 0.5, n),
))

# Sample observational data
data = gen.sample(n=5000)
arr, cols = gen.to_array(data)

# Compute ground truths
e_y_do_x1 = gen.interventional_mean("Y", {"X": 1.0}, n=50_000)  # = 0.5

# Build scenario
scenario = CausalScenario(
    scenario_id="MY_S01",
    name="My Custom Scenario",
    description="...",
    difficulty=Difficulty.SIMPLE,
    graph_edges=gen.edges,
    variable_names=gen.variable_names,
    data=arr,
    data_columns=cols,
    questions=[
        CausalQuestion(
            "q1", QuestionType.INTERVENTION,
            "What is E[Y | do(X=1)]?",
            answer=0.5,
        ),
    ],
)
```

---

## Running Tests

```bash
# All benchmark tests
python3 -m pytest tests/benchmark/ -v

# Quick run without coverage
python3 -m pytest tests/benchmark/ -v --no-cov
```

---

## Installation

```bash
# Core dependencies only (no FastAPI/DB/Redis required)
pip install numpy scipy networkx

# Or use the full environment
pip install -e ".[all]"
```

---

## Relation to causal-discovery-toolkit

This repo is designed to complement
[causal-discovery-toolkit](https://github.com/danieleschmidt/causal-discovery-toolkit).
The toolkit discovers causal structure from data; Causal Eval Bench tests whether
LLMs can *reason* about that structure.

The `CausalGraph` from the toolkit (Pearl do-calculus, d-separation, Markov blankets)
can be used directly with the `DataGenerator` in this repo.

---

## References

- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.
- Pearl, J., Glymour, M., & Jewell, N. P. (2016). *Causal Inference in Statistics: A Primer*. Wiley.
- Peters, J., Janzing, D., & Schölkopf, B. (2017). *Elements of Causal Inference*. MIT Press.
- Spirtes, P., Glymour, C., & Scheines, R. (2000). *Causation, Prediction, and Search* (2nd ed.). MIT Press.

---

## License

MIT — see [LICENSE](LICENSE).
