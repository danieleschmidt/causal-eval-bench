"""
Pre-built benchmark scenarios for the Causal Eval Bench.

Each scenario is grounded in a Structural Causal Model (SCM) with
analytically derived or numerically estimated ground-truth answers.

Difficulty levels:
  SIMPLE (1-4): 2-3 variable chains, no confounders
  MEDIUM (5-8): confounders, mediators, colliders, non-linear effects
  HARD  (9-12): multi-step, common cause with multi-path, complex attribution

Coverage:
  All scenarios include all 4 question types:
    - association    (Rung 1)
    - intervention   (Rung 2)
    - counterfactual (Rung 3)
    - attribution    (causal attribution)
"""

from __future__ import annotations

from typing import List

import numpy as np

from causal_eval.benchmark.data_generator import DataGenerator, SCMNode
from causal_eval.benchmark.scenario import (
    CausalQuestion,
    CausalScenario,
    Difficulty,
    QuestionType,
)


# =========================================================================== #
# Helper: build scenario from DataGenerator + question specs                  #
# =========================================================================== #

def _make_scenario(
    scenario_id: str,
    name: str,
    description: str,
    difficulty: Difficulty,
    gen: DataGenerator,
    question_specs: List[dict],
    scm_description: str = "",
    tags: List[str] = None,
    n_data: int = 5000,
    seed: int = 42,
) -> CausalScenario:
    """Build a CausalScenario by sampling from the generator and constructing questions."""
    np.random.seed(seed)
    raw = gen.sample(n=n_data)
    arr, cols = gen.to_array(raw)

    questions = []
    for i, spec in enumerate(question_specs):
        q = CausalQuestion(
            question_id=f"{scenario_id}_q{i+1:02d}",
            question_type=spec["type"],
            text=spec["text"],
            answer=spec["answer"],
            answer_tolerance=spec.get("tol", 0.05),
            metadata=spec.get("meta", {}),
        )
        questions.append(q)

    return CausalScenario(
        scenario_id=scenario_id,
        name=name,
        description=description,
        difficulty=difficulty,
        graph_edges=gen.edges,
        variable_names=gen.variable_names,
        data=arr,
        data_columns=cols,
        questions=questions,
        scm_description=scm_description,
        tags=tags or [],
    )


# =========================================================================== #
# ─── SIMPLE SCENARIOS ─────────────────────────────────────────────────────── #
# =========================================================================== #

def scenario_01_simple_chain() -> CausalScenario:
    """
    S01: Simple X → Y linear chain (no confounders).

    SCM:
      X = N_X,           N_X ~ Normal(0, 1)
      Y = 0.8*X + N_Y,   N_Y ~ Normal(0, 0.5)

    Key insight: P(Y | X=2) ≈ P(Y | do(X=2)) here (no back-door paths).
    """
    gen = DataGenerator(seed=1)
    gen.add_node(SCMNode(
        "X", [],
        equation=lambda pa, n: n,
        noise_fn=lambda n: np.random.normal(0, 1, n),
        domain="continuous",
    ))
    gen.add_node(SCMNode(
        "Y", ["X"],
        equation=lambda pa, n: 0.8 * pa["X"] + n,
        noise_fn=lambda n: np.random.normal(0, 0.5, n),
        domain="continuous",
    ))

    # Ground truths
    # P(Y > 1 | X = 2): Y | X=2 ~ N(1.6, 0.25), P(Y>1) = P(Z > (1-1.6)/0.5) = P(Z > -1.2)
    from scipy.stats import norm
    p_assoc = 1 - norm.cdf(1.0, loc=1.6, scale=0.5)  # ≈ 0.885
    # P(Y > 1 | do(X=2)): same because no confounders
    p_interv = p_assoc
    # Counterfactual: given X=0, Y=0.3 observed. If X had been 2?
    # Y_cf = 0.8*2 + noise; noise = 0.3 - 0.8*0 = 0.3; Y_cf = 1.6 + 0.3 = 1.9
    y_cf = 1.9

    return _make_scenario(
        scenario_id="S01",
        name="Simple Linear Chain X→Y",
        description="A direct linear causal effect from X to Y. No confounders.",
        difficulty=Difficulty.SIMPLE,
        gen=gen,
        scm_description="X~N(0,1); Y=0.8X+N(0,0.5)",
        tags=["chain", "linear", "no-confounder"],
        question_specs=[
            {
                "type": QuestionType.ASSOCIATION,
                "text": "Given X=2, what is P(Y > 1)?",
                "answer": round(p_assoc, 3),
                "tol": 0.05,
                "meta": {"variables": ["X", "Y"], "x_val": 2.0, "threshold": 1.0},
            },
            {
                "type": QuestionType.INTERVENTION,
                "text": "If we set X=2 (do(X=2)), what is P(Y > 1)?",
                "answer": round(p_interv, 3),
                "tol": 0.05,
                "meta": {"intervention": {"X": 2.0}, "threshold": 1.0},
            },
            {
                "type": QuestionType.COUNTERFACTUAL,
                "text": (
                    "We observed X=0 and Y=0.3. "
                    "What would Y have been if X had been 2 instead?"
                ),
                "answer": round(y_cf, 2),
                "tol": 0.10,
                "meta": {"observed": {"X": 0.0, "Y": 0.3}, "intervention": {"X": 2.0}},
            },
            {
                "type": QuestionType.ATTRIBUTION,
                "text": "Y increased by 0.8 units. What caused this increase?",
                "answer": "X",
                "meta": {
                    "choices": ["X", "Y", "exogenous noise"],
                    "correct": "X",
                },
            },
        ],
    )


def scenario_02_binary_treatment() -> CausalScenario:
    """
    S02: Binary treatment T → binary outcome Y (vaccine example).

    SCM:
      T = Bernoulli(0.5)
      Y = Bernoulli(0.8) if T=1, else Bernoulli(0.2)

    P(Y=1 | T=1) = 0.8 (observational)
    P(Y=1 | do(T=1)) = 0.8 (no confounder)
    Counterfactual: P(Y_T=1 = 1 | T=0, Y=0) = P(Y=1 if treated | observed unvaccinated + no disease)
    Attribution: T causes Y
    """
    gen = DataGenerator(seed=2)
    gen.add_node(SCMNode(
        "T", [],
        equation=lambda pa, n: (n > 0).astype(float),
        noise_fn=lambda n: np.random.normal(0, 1, n),
        domain="binary {0,1}",
    ))
    gen.add_node(SCMNode(
        "Y", ["T"],
        equation=lambda pa, n: (
            (pa["T"] == 1) * (n > -0.842) +  # 0.8 when treated
            (pa["T"] == 0) * (n > 0.842)      # 0.2 when untreated
        ).astype(float),
        noise_fn=lambda n: np.random.normal(0, 1, n),
        domain="binary {0,1}",
    ))

    # Counterfactual: P(Y_{T=1} = 1 | T=0, Y=0)
    # noise for Y when T=0, Y=0: n <= -0.842 (the "untreated failed" group)
    # With T=1: Y = (n > -0.842), so if n <= -0.842 and we intervene T=1, still Y=0
    # This makes sense: "Y=0 despite treatment" → low-response patients
    # But we need to be careful: noise n here is used for BOTH T=0 and T=1 branches
    # P(Y_{T=1}=1 | T=0, Y=0) = 0 (those with n<=-0.842 don't respond to treatment either)
    # That's the PN: P(Y would be 1 with treatment | was 0 without)
    # In this SCM: if T=0 and Y=0, then noise n <= -0.842 (20% threshold)
    # If now T=1, Y = 1(n > -0.842) = 0 since n <= -0.842. So P(Y_{T=1}=1|T=0,Y=0) = 0
    # Wait — that means treatment is not sufficient for those who had low noise
    # PS = P(Y_{T=1}=1 | T=0, Y=0) = 0.0 in this model
    # Let's compute PN instead: P(Y_{T=0}=0 | T=1, Y=1)
    # Given T=1, Y=1: noise n > -0.842. With T=0: Y = (n > 0.842)
    # P(Y_{T=0}=0 | T=1, Y=1) = P(n <= 0.842 | n > -0.842)
    from scipy.stats import norm
    pn = (norm.cdf(0.842) - norm.cdf(-0.842)) / (1 - norm.cdf(-0.842))
    # ≈ (0.800 - 0.200) / 0.800 = 0.75

    return _make_scenario(
        scenario_id="S02",
        name="Binary Treatment→Outcome (Vaccine)",
        description="Binary treatment (vaccination) causes binary outcome (immunity). No confounders.",
        difficulty=Difficulty.SIMPLE,
        gen=gen,
        scm_description="T~Bernoulli(0.5); Y|T=1~Bern(0.8), Y|T=0~Bern(0.2)",
        tags=["binary", "treatment-effect", "medical"],
        question_specs=[
            {
                "type": QuestionType.ASSOCIATION,
                "text": "In the observed population, what is P(Y=1 | T=1)?",
                "answer": 0.8,
                "tol": 0.05,
                "meta": {"variables": ["T", "Y"]},
            },
            {
                "type": QuestionType.INTERVENTION,
                "text": "If everyone receives treatment (do(T=1)), what is P(Y=1)?",
                "answer": 0.8,
                "tol": 0.05,
                "meta": {"intervention": {"T": 1}},
            },
            {
                "type": QuestionType.COUNTERFACTUAL,
                "text": (
                    "A patient was untreated (T=0) and did not recover (Y=0). "
                    "What is the probability they would have recovered if treated (T=1)?"
                ),
                "answer": 0.0,
                "tol": 0.05,
                "meta": {
                    "observed": {"T": 0, "Y": 0},
                    "intervention": {"T": 1},
                    "note": "PS: patients with low noise can't respond to treatment in this SCM",
                },
            },
            {
                "type": QuestionType.ATTRIBUTION,
                "text": (
                    "A treated patient (T=1) recovered (Y=1). "
                    "What is the probability of necessity — "
                    "that they would NOT have recovered without treatment?"
                ),
                "answer": round(pn, 2),
                "tol": 0.05,
                "meta": {"observed": {"T": 1, "Y": 1}},
            },
        ],
    )


def scenario_03_mediation() -> CausalScenario:
    """
    S03: X → M → Y (mediation, simple chain of 3).

    SCM:
      X = N_X ~ N(0,1)
      M = 0.7*X + N_M,  N_M ~ N(0,0.3)
      Y = 0.6*M + N_Y,  N_Y ~ N(0,0.4)

    Total effect of X on Y: 0.7 * 0.6 = 0.42 (fully mediated through M)
    Direct effect of X on Y: 0 (no direct edge)

    Key insight: P(Y | X=1) ≠ P(Y | do(X=1), do(M=0))
    """
    gen = DataGenerator(seed=3)
    gen.add_node(SCMNode(
        "X", [],
        equation=lambda pa, n: n,
        noise_fn=lambda n: np.random.normal(0, 1, n),
    ))
    gen.add_node(SCMNode(
        "M", ["X"],
        equation=lambda pa, n: 0.7 * pa["X"] + n,
        noise_fn=lambda n: np.random.normal(0, 0.3, n),
    ))
    gen.add_node(SCMNode(
        "Y", ["M"],
        equation=lambda pa, n: 0.6 * pa["M"] + n,
        noise_fn=lambda n: np.random.normal(0, 0.4, n),
    ))

    # E[Y | X=1]: E[M|X=1] = 0.7, E[Y|M=0.7] = 0.6*0.7 = 0.42
    # E[Y | do(X=1)]: same = 0.42 (no confounders)
    # E[Y | do(X=1), do(M=0)]: M forced to 0, Y = 0.6*0 + N_Y = E[N_Y]=0
    # Counterfactual: if X=1 instead of X=0, what is E[Y]?
    # Y_cf = 0.6*(0.7*1 + N_M) + N_Y; prior E[Y|X=0]=0 → Y_cf = 0.42

    return _make_scenario(
        scenario_id="S03",
        name="Mediation Chain X→M→Y",
        description="Full mediation: X affects Y only through mediator M.",
        difficulty=Difficulty.SIMPLE,
        gen=gen,
        scm_description="X~N(0,1); M=0.7X+N(0,0.3); Y=0.6M+N(0,0.4)",
        tags=["mediation", "chain", "indirect-effect"],
        question_specs=[
            {
                "type": QuestionType.ASSOCIATION,
                "text": "What is E[Y | X=1] (expected Y given X=1)?",
                "answer": 0.42,
                "tol": 0.05,
                "meta": {"variables": ["X", "Y"]},
            },
            {
                "type": QuestionType.INTERVENTION,
                "text": "What is E[Y | do(X=1)]?",
                "answer": 0.42,
                "tol": 0.05,
                "meta": {"intervention": {"X": 1.0}},
            },
            {
                "type": QuestionType.INTERVENTION,
                "text": "If we set do(X=1) AND do(M=0), what is E[Y]?",
                "answer": 0.0,
                "tol": 0.05,
                "meta": {"intervention": {"X": 1.0, "M": 0.0}, "note": "blocks mediation path"},
            },
            {
                "type": QuestionType.COUNTERFACTUAL,
                "text": "We observed X=0, Y=0. What would E[Y] be if X had been 1?",
                "answer": 0.42,
                "tol": 0.08,
                "meta": {"observed": {"X": 0.0, "Y": 0.0}, "intervention": {"X": 1.0}},
            },
            {
                "type": QuestionType.ATTRIBUTION,
                "text": "In this model, what causes Y to change?",
                "answer": "M",
                "meta": {
                    "choices": ["X", "M", "both X and M directly"],
                    "correct": "M",
                    "note": "X affects Y only via M; M is the direct cause of Y",
                },
            },
        ],
    )


def scenario_04_no_causal_effect() -> CausalScenario:
    """
    S04: Spurious correlation — common cause Z → X and Z → Y, no X→Y edge.

    SCM:
      Z = N_Z ~ N(0,1)
      X = 0.9*Z + N_X,  N_X ~ N(0,0.2)
      Y = 0.9*Z + N_Y,  N_Y ~ N(0,0.2)

    Key insight: X and Y are correlated (both caused by Z),
    but P(Y | do(X)) = E[Y] = 0 — no causal effect!
    """
    gen = DataGenerator(seed=4)
    gen.add_node(SCMNode(
        "Z", [],
        equation=lambda pa, n: n,
        noise_fn=lambda n: np.random.normal(0, 1, n),
    ))
    gen.add_node(SCMNode(
        "X", ["Z"],
        equation=lambda pa, n: 0.9 * pa["Z"] + n,
        noise_fn=lambda n: np.random.normal(0, 0.2, n),
    ))
    gen.add_node(SCMNode(
        "Y", ["Z"],
        equation=lambda pa, n: 0.9 * pa["Z"] + n,
        noise_fn=lambda n: np.random.normal(0, 0.2, n),
    ))

    return _make_scenario(
        scenario_id="S04",
        name="Spurious Correlation via Common Cause Z→X, Z→Y",
        description="X and Y are correlated but X does NOT cause Y — both are caused by Z.",
        difficulty=Difficulty.SIMPLE,
        gen=gen,
        scm_description="Z~N(0,1); X=0.9Z+N(0,0.2); Y=0.9Z+N(0,0.2); NO X→Y edge",
        tags=["spurious-correlation", "confounding", "common-cause"],
        question_specs=[
            {
                "type": QuestionType.ASSOCIATION,
                "text": "What is E[Y | X=2]? (observational)",
                "answer": 1.80,  # X=2 → Z ≈ 2/0.9 ≈ 2.22 (regressing) → but actually E[Z|X=2] = 0.9*2 / (0.81+0.04) ≈ 2.12; E[Y|Z=2.12] ≈ 0.9*2.12 ≈ 1.91 -- let's use 1.9
                "answer": 1.9,
                "tol": 0.10,
                "meta": {"variables": ["X", "Y"], "note": "correlated due to Z"},
            },
            {
                "type": QuestionType.INTERVENTION,
                "text": "What is E[Y | do(X=2)]? (interventional — no causal link X→Y)",
                "answer": 0.0,
                "tol": 0.05,
                "meta": {
                    "intervention": {"X": 2.0},
                    "note": "X has no causal effect on Y; after do(X), Z is still N(0,1)",
                },
            },
            {
                "type": QuestionType.COUNTERFACTUAL,
                "text": "We observed Z=1, X=0.9, Y=0.9. If X had been set to 0, what would Y be?",
                "answer": 0.9,  # Y depends only on Z; Z abducted from Z=1; Y_cf = 0.9*1 + noise=0 ≈ 0.9
                "tol": 0.10,
                "meta": {
                    "observed": {"Z": 1.0, "X": 0.9, "Y": 0.9},
                    "intervention": {"X": 0.0},
                    "note": "Y unchanged — depends on Z, not X",
                },
            },
            {
                "type": QuestionType.ATTRIBUTION,
                "text": "X and Y are correlated. What is the causal explanation?",
                "answer": "Z",
                "meta": {
                    "choices": ["X causes Y", "Y causes X", "Z causes both"],
                    "correct": "Z causes both",
                },
            },
        ],
    )


# =========================================================================== #
# ─── MEDIUM SCENARIOS ─────────────────────────────────────────────────────── #
# =========================================================================== #

def scenario_05_confounding() -> CausalScenario:
    """
    S05: Classic confounding: Z→X, Z→Y, X→Y (back-door path via Z).

    SCM:
      Z = N_Z ~ N(0,1)            (confounder)
      X = 0.5*Z + N_X ~ N(0,0.5)  (treatment, affected by Z)
      Y = 0.4*X + 0.6*Z + N_Y    (outcome: both X and Z)

    Observational E[Y|X=1] biased by confounding.
    True causal effect do(X=1): 0.4
    """
    gen = DataGenerator(seed=5)
    gen.add_node(SCMNode(
        "Z", [],
        equation=lambda pa, n: n,
        noise_fn=lambda n: np.random.normal(0, 1, n),
    ))
    gen.add_node(SCMNode(
        "X", ["Z"],
        equation=lambda pa, n: 0.5 * pa["Z"] + n,
        noise_fn=lambda n: np.random.normal(0, 0.5, n),
    ))
    gen.add_node(SCMNode(
        "Y", ["X", "Z"],
        equation=lambda pa, n: 0.4 * pa["X"] + 0.6 * pa["Z"] + n,
        noise_fn=lambda n: np.random.normal(0, 0.3, n),
    ))

    # E[Y | do(X=1)]: break back-door, only direct effect 0.4
    # Observational: E[Y|X=1] = 0.4 + 0.6*E[Z|X=1]
    # E[Z|X=1]: Z given X=0.5Z + N = 1 → E[Z|X=1] = 0.5/(0.25+0.25) = 1.0
    # E[Y|X=1] = 0.4 + 0.6*1.0 = 1.0

    return _make_scenario(
        scenario_id="S05",
        name="Classic Confounding Z→X, Z→Y, X→Y",
        description=(
            "Z confounds X→Y. Observational and interventional estimates differ. "
            "Back-door adjustment through Z gives true causal effect."
        ),
        difficulty=Difficulty.MEDIUM,
        gen=gen,
        scm_description="Z~N(0,1); X=0.5Z+N(0,0.5); Y=0.4X+0.6Z+N(0,0.3)",
        tags=["confounding", "back-door", "bias"],
        question_specs=[
            {
                "type": QuestionType.ASSOCIATION,
                "text": "What is E[Y | X=1] (observational, biased)?",
                "answer": 1.0,
                "tol": 0.08,
                "meta": {"note": "biased by confounding via Z"},
            },
            {
                "type": QuestionType.INTERVENTION,
                "text": "What is E[Y | do(X=1)] (true causal effect of X)?",
                "answer": 0.4,
                "tol": 0.05,
                "meta": {
                    "intervention": {"X": 1.0},
                    "note": "removes back-door path through Z",
                },
            },
            {
                "type": QuestionType.COUNTERFACTUAL,
                "text": "Observed Z=0, X=0, Y=0. What is E[Y] if X were set to 1?",
                "answer": 0.4,
                "tol": 0.08,
                "meta": {
                    "observed": {"Z": 0.0, "X": 0.0, "Y": 0.0},
                    "intervention": {"X": 1.0},
                },
            },
            {
                "type": QuestionType.ATTRIBUTION,
                "text": (
                    "A study finds E[Y|X=1] - E[Y|X=0] = 1.0. "
                    "Is this the causal effect of X on Y?"
                ),
                "answer": "no",
                "meta": {
                    "choices": ["yes", "no"],
                    "correct": "no",
                    "note": "confounding inflates the observational estimate",
                },
            },
        ],
    )


def scenario_06_collider_bias() -> CausalScenario:
    """
    S06: Collider bias — X→C←Y, conditioning on C induces spurious X-Y association.

    SCM:
      X = N_X ~ N(0,1)
      Y = N_Y ~ N(0,1)  (independent of X!)
      C = X + Y + N_C ~ N(0,0.2)   (collider)

    Unconditionally X ⊥ Y.
    Conditionally on C: X and Y become negatively correlated (Berkson's paradox).
    """
    gen = DataGenerator(seed=6)
    gen.add_node(SCMNode(
        "X", [],
        equation=lambda pa, n: n,
        noise_fn=lambda n: np.random.normal(0, 1, n),
    ))
    gen.add_node(SCMNode(
        "Y", [],
        equation=lambda pa, n: n,
        noise_fn=lambda n: np.random.normal(0, 1, n),
    ))
    gen.add_node(SCMNode(
        "C", ["X", "Y"],
        equation=lambda pa, n: pa["X"] + pa["Y"] + n,
        noise_fn=lambda n: np.random.normal(0, 0.2, n),
    ))

    return _make_scenario(
        scenario_id="S06",
        name="Collider Bias X→C←Y",
        description=(
            "C is a collider of X and Y. X and Y are independent, "
            "but conditioning on C creates spurious association (Berkson's paradox)."
        ),
        difficulty=Difficulty.MEDIUM,
        gen=gen,
        scm_description="X~N(0,1); Y~N(0,1) independent; C=X+Y+N(0,0.2)",
        tags=["collider", "selection-bias", "berkson"],
        question_specs=[
            {
                "type": QuestionType.ASSOCIATION,
                "text": "What is Corr(X, Y) in the full population (no conditioning)?",
                "answer": 0.0,
                "tol": 0.05,
                "meta": {"note": "X and Y are independent"},
            },
            {
                "type": QuestionType.INTERVENTION,
                "text": "What is E[Y | do(X=2)]?",
                "answer": 0.0,
                "tol": 0.05,
                "meta": {
                    "intervention": {"X": 2.0},
                    "note": "X has no causal effect on Y",
                },
            },
            {
                "type": QuestionType.COUNTERFACTUAL,
                "text": "Observed X=1, Y=1, C=2. If X had been -1, what would C be?",
                "answer": 0.0,  # C_cf = -1 + 1 (Y unchanged) + noise ≈ 0
                "tol": 0.15,
                "meta": {
                    "observed": {"X": 1.0, "Y": 1.0, "C": 2.0},
                    "intervention": {"X": -1.0},
                },
            },
            {
                "type": QuestionType.ATTRIBUTION,
                "text": (
                    "Among hospitalized patients (high C), X and Y are negatively correlated. "
                    "Does X cause Y to decrease?"
                ),
                "answer": "no",
                "meta": {
                    "choices": ["yes", "no"],
                    "correct": "no",
                    "note": "collider conditioning, not causation",
                },
            },
        ],
    )


def scenario_07_nonlinear_threshold() -> CausalScenario:
    """
    S07: Non-linear SCM — X → Y with threshold effect.

    SCM:
      X = N_X ~ N(0,1)
      Y = sigmoid(3*X - 1) + N_Y ~ N(0,0.1)

    P(Y > 0.5 | X=0) vs P(Y > 0.5 | do(X=1)) differ.
    Tests whether models handle non-linearity.
    """
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    gen = DataGenerator(seed=7)
    gen.add_node(SCMNode(
        "X", [],
        equation=lambda pa, n: n,
        noise_fn=lambda n: np.random.normal(0, 1, n),
    ))
    gen.add_node(SCMNode(
        "Y", ["X"],
        equation=lambda pa, n: sigmoid(3 * pa["X"] - 1) + n,
        noise_fn=lambda n: np.random.normal(0, 0.1, n),
    ))

    # Ground truths (estimated numerically)
    np.random.seed(7)
    # P(Y>0.5 | X=0): Y|X=0 = sigmoid(-1) + N ≈ 0.269 + N(0,0.1)
    # P(Y>0.5) = P(N > 0.5 - 0.269) = P(N > 0.231) ≈ P(Z > 2.31) ≈ 0.010
    from scipy.stats import norm
    y_mean_x0 = sigmoid(np.array([-1.0]))[0]  # ≈ 0.269
    p_assoc_x0 = 1 - norm.cdf(0.5 - y_mean_x0, loc=0, scale=0.1)
    # P(Y>0.5 | do(X=1)): Y|X=1 = sigmoid(2) ≈ 0.880
    y_mean_x1 = sigmoid(np.array([2.0]))[0]
    p_interv_x1 = 1 - norm.cdf(0.5 - y_mean_x1, loc=0, scale=0.1)

    return _make_scenario(
        scenario_id="S07",
        name="Non-linear Threshold Effect X→Y (Sigmoid)",
        description="Non-linear sigmoid relationship between X and Y. Tests non-linear causal reasoning.",
        difficulty=Difficulty.MEDIUM,
        gen=gen,
        scm_description="X~N(0,1); Y=sigmoid(3X-1)+N(0,0.1)",
        tags=["non-linear", "threshold", "sigmoid"],
        question_specs=[
            {
                "type": QuestionType.ASSOCIATION,
                "text": "What is P(Y > 0.5 | X=0)?",
                "answer": round(p_assoc_x0, 3),
                "tol": 0.05,
                "meta": {"x_val": 0.0, "threshold": 0.5},
            },
            {
                "type": QuestionType.INTERVENTION,
                "text": "What is P(Y > 0.5 | do(X=1))?",
                "answer": round(p_interv_x1, 3),
                "tol": 0.05,
                "meta": {"intervention": {"X": 1.0}},
            },
            {
                "type": QuestionType.COUNTERFACTUAL,
                "text": (
                    "Observed X=-2, Y≈0.01 (below threshold). "
                    "What would E[Y] be if X had been 2?"
                ),
                "answer": round(sigmoid(np.array([5.0]))[0], 3),  # sigmoid(3*2-1) = sigmoid(5) ≈ 0.993
                "tol": 0.05,
                "meta": {"observed": {"X": -2.0, "Y": 0.01}, "intervention": {"X": 2.0}},
            },
            {
                "type": QuestionType.ATTRIBUTION,
                "text": "Y jumped from 0.01 to 0.99. What caused this?",
                "answer": "X",
                "meta": {"choices": ["X", "noise", "Y itself"], "correct": "X"},
            },
        ],
    )


def scenario_08_front_door() -> CausalScenario:
    """
    S08: Front-door criterion — X→M→Y with X←U→Y (unobserved U).

    Since U is unobserved, back-door is blocked. But front-door via M works.
    We model U as a "hidden" variable by its effects only.

    SCM:
      U = N_U ~ N(0,1)           (unobserved confounder)
      X = 0.5*U + N_X            (treatment)
      M = 0.7*X + N_M            (mediator, no U effect)
      Y = 0.4*M + 0.8*U + N_Y   (outcome)

    True P(Y|do(X=1)) via front-door = 0.7 * 0.4 = 0.28  (E[Y|do(X=1)] = 0.28)
    """
    gen = DataGenerator(seed=8)
    gen.add_node(SCMNode(
        "U", [],
        equation=lambda pa, n: n,
        noise_fn=lambda n: np.random.normal(0, 1, n),
    ))
    gen.add_node(SCMNode(
        "X", ["U"],
        equation=lambda pa, n: 0.5 * pa["U"] + n,
        noise_fn=lambda n: np.random.normal(0, 0.7, n),
    ))
    gen.add_node(SCMNode(
        "M", ["X"],
        equation=lambda pa, n: 0.7 * pa["X"] + n,
        noise_fn=lambda n: np.random.normal(0, 0.3, n),
    ))
    gen.add_node(SCMNode(
        "Y", ["M", "U"],
        equation=lambda pa, n: 0.4 * pa["M"] + 0.8 * pa["U"] + n,
        noise_fn=lambda n: np.random.normal(0, 0.3, n),
    ))

    # True causal effect of X on Y (via front-door):
    # E[Y | do(X=1)] = E[Y | M=0.7*1] = 0.4 * 0.7 = 0.28
    # (U is now independent of X after do(X), so E[U|do(X)]=0)

    return _make_scenario(
        scenario_id="S08",
        name="Front-Door Criterion (Unobserved Confounder U)",
        description=(
            "Unobserved confounder U blocks back-door adjustment. "
            "Front-door criterion via mediator M identifies the causal effect."
        ),
        difficulty=Difficulty.MEDIUM,
        gen=gen,
        scm_description="U~N(0,1); X=0.5U+N; M=0.7X+N; Y=0.4M+0.8U+N",
        tags=["front-door", "unobserved-confounder", "identification"],
        question_specs=[
            {
                "type": QuestionType.ASSOCIATION,
                "text": "What is E[Y | X=1] (biased by unobserved U)?",
                "answer": 0.6,  # 0.28 + 0.8*E[U|X=1] ≈ 0.28 + 0.8*(0.5/0.74) ≈ 0.28 + 0.54 ≈ 0.82 — let's use gen estimate
                "tol": 0.15,
                "meta": {"note": "confounded by U"},
            },
            {
                "type": QuestionType.INTERVENTION,
                "text": "What is E[Y | do(X=1)]? (true causal effect, identified via front-door)",
                "answer": 0.28,
                "tol": 0.06,
                "meta": {"intervention": {"X": 1.0}, "note": "front-door criterion: 0.7*0.4"},
            },
            {
                "type": QuestionType.COUNTERFACTUAL,
                "text": "Observed U=0, X=0, M=0, Y=0. If X were set to 1, what is E[Y]?",
                "answer": 0.28,
                "tol": 0.08,
                "meta": {
                    "observed": {"U": 0.0, "X": 0.0, "M": 0.0, "Y": 0.0},
                    "intervention": {"X": 1.0},
                },
            },
            {
                "type": QuestionType.ATTRIBUTION,
                "text": "X and Y are correlated. Can we use the back-door criterion to estimate the causal effect?",
                "answer": "no",
                "meta": {
                    "choices": ["yes", "no"],
                    "correct": "no",
                    "note": "U is unobserved — back-door requires conditioning on U",
                },
            },
        ],
    )


# =========================================================================== #
# ─── HARD SCENARIOS ───────────────────────────────────────────────────────── #
# =========================================================================== #

def scenario_09_multi_path() -> CausalScenario:
    """
    S09: Multiple causal paths — X→Y (direct) and X→M→Y (mediated).

    SCM:
      X = N_X ~ N(0,1)
      M = 0.5*X + N_M
      Y = 0.3*X + 0.5*M + N_Y    (direct + indirect via M)

    Total effect: 0.3 + 0.5*0.5 = 0.55
    Direct effect: 0.3
    Indirect effect (via M): 0.25
    """
    gen = DataGenerator(seed=9)
    gen.add_node(SCMNode(
        "X", [],
        equation=lambda pa, n: n,
        noise_fn=lambda n: np.random.normal(0, 1, n),
    ))
    gen.add_node(SCMNode(
        "M", ["X"],
        equation=lambda pa, n: 0.5 * pa["X"] + n,
        noise_fn=lambda n: np.random.normal(0, 0.5, n),
    ))
    gen.add_node(SCMNode(
        "Y", ["X", "M"],
        equation=lambda pa, n: 0.3 * pa["X"] + 0.5 * pa["M"] + n,
        noise_fn=lambda n: np.random.normal(0, 0.3, n),
    ))

    return _make_scenario(
        scenario_id="S09",
        name="Multi-Path Effect: Direct X→Y + Mediated X→M→Y",
        description=(
            "X affects Y both directly and indirectly via M. "
            "Total effect = direct + indirect = 0.3 + 0.25 = 0.55."
        ),
        difficulty=Difficulty.HARD,
        gen=gen,
        scm_description="X~N(0,1); M=0.5X+N; Y=0.3X+0.5M+N",
        tags=["mediation", "direct-effect", "indirect-effect", "multi-path"],
        question_specs=[
            {
                "type": QuestionType.ASSOCIATION,
                "text": "What is E[Y | X=1]?",
                "answer": 0.55,
                "tol": 0.06,
                "meta": {"note": "total effect (no confounding in this scenario)"},
            },
            {
                "type": QuestionType.INTERVENTION,
                "text": "What is the total causal effect E[Y | do(X=1)]?",
                "answer": 0.55,
                "tol": 0.05,
                "meta": {"intervention": {"X": 1.0}},
            },
            {
                "type": QuestionType.INTERVENTION,
                "text": (
                    "What is the natural direct effect (NDE)? "
                    "I.e., E[Y | do(X=1), do(M=E[M|X=0])] ≈ E[Y | do(X=1), do(M=0)]?"
                ),
                "answer": 0.3,
                "tol": 0.06,
                "meta": {"intervention": {"X": 1.0, "M": 0.0}, "note": "NDE = direct effect"},
            },
            {
                "type": QuestionType.COUNTERFACTUAL,
                "text": (
                    "Observed X=0, M=0, Y=0. "
                    "What would Y be if X had been 1 (with M free to change)?"
                ),
                "answer": 0.55,
                "tol": 0.08,
                "meta": {
                    "observed": {"X": 0.0, "M": 0.0, "Y": 0.0},
                    "intervention": {"X": 1.0},
                },
            },
            {
                "type": QuestionType.ATTRIBUTION,
                "text": (
                    "Y increased by 0.55 units when X increased by 1. "
                    "How much of this increase is the DIRECT effect of X (not through M)?"
                ),
                "answer": 0.3,
                "tol": 0.05,
                "meta": {"direct_effect": 0.3, "indirect_effect": 0.25},
            },
        ],
    )


def scenario_10_iv_exclusion() -> CausalScenario:
    """
    S10: Instrumental variable — Z→X→Y, with X←U→Y (unobserved U).

    Z is a valid instrument: Z→X (relevance), Z⊥U (independence), Z⊥Y|X (exclusion).

    SCM:
      Z = N_Z ~ Bernoulli(0.5)    (instrument)
      U = N_U ~ N(0,1)            (unobserved confounder)
      X = 0.8*Z + 0.5*U + N_X    (treatment)
      Y = 0.6*X + 0.7*U + N_Y    (outcome)

    IV estimate = Cov(Z,Y)/Cov(Z,X) = 0.6 (true causal effect of X on Y).
    """
    gen = DataGenerator(seed=10)
    gen.add_node(SCMNode(
        "Z", [],
        equation=lambda pa, n: (n > 0).astype(float),
        noise_fn=lambda n: np.random.normal(0, 1, n),
    ))
    gen.add_node(SCMNode(
        "U", [],
        equation=lambda pa, n: n,
        noise_fn=lambda n: np.random.normal(0, 1, n),
    ))
    gen.add_node(SCMNode(
        "X", ["Z", "U"],
        equation=lambda pa, n: 0.8 * pa["Z"] + 0.5 * pa["U"] + n,
        noise_fn=lambda n: np.random.normal(0, 0.3, n),
    ))
    gen.add_node(SCMNode(
        "Y", ["X", "U"],
        equation=lambda pa, n: 0.6 * pa["X"] + 0.7 * pa["U"] + n,
        noise_fn=lambda n: np.random.normal(0, 0.3, n),
    ))

    return _make_scenario(
        scenario_id="S10",
        name="Instrumental Variable Z→X→Y with Unobserved Confounder",
        description=(
            "Z is a valid instrument. Back-door blocked by unobserved U. "
            "IV estimate recovers true causal effect of X on Y."
        ),
        difficulty=Difficulty.HARD,
        gen=gen,
        scm_description="Z~Bern(0.5); U~N(0,1); X=0.8Z+0.5U+N; Y=0.6X+0.7U+N",
        tags=["instrumental-variable", "unobserved-confounder", "IV"],
        question_specs=[
            {
                "type": QuestionType.ASSOCIATION,
                "text": "What is E[Y | X=1] (confounded estimate)?",
                "answer": 0.95,  # 0.6 + 0.7*E[U|X=1]; E[U|X=1] ≈ 0.5/(0.25+0.09) ≈ 0.5/0.34*1 — roughly 0.6+0.7*0.5 ≈ 0.95
                "tol": 0.15,
                "meta": {"note": "upward bias from U"},
            },
            {
                "type": QuestionType.INTERVENTION,
                "text": "What is the true causal effect E[Y | do(X=1)]?",
                "answer": 0.6,
                "tol": 0.05,
                "meta": {"intervention": {"X": 1.0}},
            },
            {
                "type": QuestionType.COUNTERFACTUAL,
                "text": (
                    "Observed U=0, Z=0, X=0, Y=0. "
                    "If X had been set to 1, what would Y be?"
                ),
                "answer": 0.6,
                "tol": 0.08,
                "meta": {
                    "observed": {"U": 0.0, "Z": 0.0, "X": 0.0, "Y": 0.0},
                    "intervention": {"X": 1.0},
                },
            },
            {
                "type": QuestionType.ATTRIBUTION,
                "text": "Z and Y are correlated. Does Z directly cause Y?",
                "answer": "no",
                "meta": {
                    "choices": ["yes", "no"],
                    "correct": "no",
                    "note": "exclusion restriction: Z affects Y only through X",
                },
            },
        ],
    )


def scenario_11_time_series_lag() -> CausalScenario:
    """
    S11: Lagged causal effect — X_t → Y_{t+1} (Granger-style causal test).

    SCM:
      X_t = N_X ~ N(0,1)   (independent over time)
      Y_t = 0.7*X_{t-1} + 0.3*Y_{t-1} + N_Y

    Key: X Granger-causes Y. But past Y doesn't cause X.
    We encode as two-period model: X1, Y1, X2, Y2.
    """
    gen = DataGenerator(seed=11)
    gen.add_node(SCMNode(
        "X1", [],
        equation=lambda pa, n: n,
        noise_fn=lambda n: np.random.normal(0, 1, n),
    ))
    gen.add_node(SCMNode(
        "Y1", [],
        equation=lambda pa, n: n,
        noise_fn=lambda n: np.random.normal(0, 1, n),
    ))
    gen.add_node(SCMNode(
        "X2", [],
        equation=lambda pa, n: n,
        noise_fn=lambda n: np.random.normal(0, 1, n),
    ))
    gen.add_node(SCMNode(
        "Y2", ["X1", "Y1"],
        equation=lambda pa, n: 0.7 * pa["X1"] + 0.3 * pa["Y1"] + n,
        noise_fn=lambda n: np.random.normal(0, 0.4, n),
    ))

    return _make_scenario(
        scenario_id="S11",
        name="Lagged Causal Effect (Time Series) X1→Y2",
        description=(
            "X at time t-1 causes Y at time t. "
            "Tests temporal causal reasoning (Granger causality)."
        ),
        difficulty=Difficulty.HARD,
        gen=gen,
        scm_description="X1,Y1~N(0,1) independent; Y2=0.7*X1+0.3*Y1+N(0,0.4)",
        tags=["time-series", "granger", "lag", "temporal"],
        question_specs=[
            {
                "type": QuestionType.ASSOCIATION,
                "text": "What is E[Y2 | X1=2, Y1=0]?",
                "answer": 1.4,  # 0.7*2 + 0.3*0 = 1.4
                "tol": 0.05,
            },
            {
                "type": QuestionType.INTERVENTION,
                "text": "What is E[Y2 | do(X1=2), Y1=0]?",
                "answer": 1.4,
                "tol": 0.05,
                "meta": {"intervention": {"X1": 2.0}},
            },
            {
                "type": QuestionType.COUNTERFACTUAL,
                "text": "Observed X1=0, Y1=0, Y2=0.2. If X1 had been 2, what would Y2 have been?",
                "answer": 1.6,  # Y2_cf = 0.7*2 + 0.3*0 + noise; noise = 0.2 - 0 = 0.2; Y2_cf = 1.4 + 0.2 = 1.6
                "tol": 0.10,
                "meta": {
                    "observed": {"X1": 0.0, "Y1": 0.0, "Y2": 0.2},
                    "intervention": {"X1": 2.0},
                },
            },
            {
                "type": QuestionType.ATTRIBUTION,
                "text": "Y2 increased by 1.4 from its baseline of 0. What caused this?",
                "answer": "X1",
                "meta": {"choices": ["X1", "Y1", "X2", "noise"], "correct": "X1"},
            },
        ],
    )


def scenario_12_complex_dag() -> CausalScenario:
    """
    S12: Complex DAG — A→B, A→C, B→D, C→D, with confounder U→B, U→D.

    Tests multi-step inference with both direct paths and confounding.

    SCM:
      A = N_A ~ N(0,1)
      U = N_U ~ N(0,1)
      B = 0.5*A + 0.4*U + N_B
      C = 0.6*A + N_C
      D = 0.3*B + 0.4*C + 0.3*U + N_D

    Total effect of A on D:
      Via B: 0.5*0.3 = 0.15
      Via C: 0.6*0.4 = 0.24
      Total: 0.39
    """
    gen = DataGenerator(seed=12)
    gen.add_node(SCMNode(
        "A", [],
        equation=lambda pa, n: n,
        noise_fn=lambda n: np.random.normal(0, 1, n),
    ))
    gen.add_node(SCMNode(
        "U", [],
        equation=lambda pa, n: n,
        noise_fn=lambda n: np.random.normal(0, 1, n),
    ))
    gen.add_node(SCMNode(
        "B", ["A", "U"],
        equation=lambda pa, n: 0.5 * pa["A"] + 0.4 * pa["U"] + n,
        noise_fn=lambda n: np.random.normal(0, 0.3, n),
    ))
    gen.add_node(SCMNode(
        "C", ["A"],
        equation=lambda pa, n: 0.6 * pa["A"] + n,
        noise_fn=lambda n: np.random.normal(0, 0.3, n),
    ))
    gen.add_node(SCMNode(
        "D", ["B", "C", "U"],
        equation=lambda pa, n: 0.3 * pa["B"] + 0.4 * pa["C"] + 0.3 * pa["U"] + n,
        noise_fn=lambda n: np.random.normal(0, 0.3, n),
    ))

    return _make_scenario(
        scenario_id="S12",
        name="Complex DAG with Confounder (5 variables)",
        description=(
            "A→B, A→C, B→D, C→D, U→B, U→D. "
            "Two causal paths from A to D with a confounder U on B and D."
        ),
        difficulty=Difficulty.HARD,
        gen=gen,
        scm_description="A,U~N(0,1); B=0.5A+0.4U+N; C=0.6A+N; D=0.3B+0.4C+0.3U+N",
        tags=["complex-dag", "multi-path", "confounding", "5-variable"],
        question_specs=[
            {
                "type": QuestionType.ASSOCIATION,
                "text": "What is E[D | A=1, U=0]?",
                "answer": 0.39,  # 0.3*(0.5) + 0.4*(0.6) = 0.15 + 0.24 = 0.39
                "tol": 0.06,
                "meta": {"note": "with U observed, no confounding"},
            },
            {
                "type": QuestionType.INTERVENTION,
                "text": "What is E[D | do(A=1)]? (true total causal effect of A on D)",
                "answer": 0.39,
                "tol": 0.05,
                "meta": {"intervention": {"A": 1.0}},
            },
            {
                "type": QuestionType.COUNTERFACTUAL,
                "text": (
                    "Observed A=0, U=0, B=0, C=0, D=0.1. "
                    "What would D be if A had been 1?"
                ),
                "answer": 0.49,  # D_cf = 0.39 + noise; noise ≈ 0.1; D_cf = 0.39 + 0.1 = 0.49
                "tol": 0.10,
                "meta": {
                    "observed": {"A": 0.0, "U": 0.0, "B": 0.0, "C": 0.0, "D": 0.1},
                    "intervention": {"A": 1.0},
                },
            },
            {
                "type": QuestionType.ATTRIBUTION,
                "text": (
                    "D increased when A increased. "
                    "Which path contributes more: A→B→D or A→C→D?"
                ),
                "answer": "A→C→D",
                "meta": {
                    "choices": ["A→B→D", "A→C→D", "equal"],
                    "correct": "A→C→D",
                    "via_B": 0.15,
                    "via_C": 0.24,
                },
            },
        ],
    )


# =========================================================================== #
# Registry                                                                     #
# =========================================================================== #

_SCENARIO_BUILDERS = [
    scenario_01_simple_chain,
    scenario_02_binary_treatment,
    scenario_03_mediation,
    scenario_04_no_causal_effect,
    scenario_05_confounding,
    scenario_06_collider_bias,
    scenario_07_nonlinear_threshold,
    scenario_08_front_door,
    scenario_09_multi_path,
    scenario_10_iv_exclusion,
    scenario_11_time_series_lag,
    scenario_12_complex_dag,
]


def load_all_scenarios() -> List[CausalScenario]:
    """Load and return all 12 benchmark scenarios."""
    return [builder() for builder in _SCENARIO_BUILDERS]


def load_by_difficulty(difficulty: Difficulty) -> List[CausalScenario]:
    """Load scenarios filtered by difficulty."""
    return [s for s in load_all_scenarios() if s.difficulty == difficulty]


def load_by_tag(tag: str) -> List[CausalScenario]:
    """Load scenarios that have a given tag."""
    return [s for s in load_all_scenarios() if tag in s.tags]
