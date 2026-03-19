"""
CausalScenario: A benchmark test case based on a Structural Causal Model.

Each scenario contains:
  - A causal graph (DAG)
  - Observational data generated from the SCM
  - Questions requiring causal reasoning, with ground-truth answers

References:
    Pearl, J. (2009). Causality. Cambridge University Press.
    Pearl, J., Glymour, M., & Jewell, N. P. (2016). Causal Inference in Statistics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class QuestionType(str, Enum):
    """
    Pearl's Ladder of Causation — three rungs + attribution.

    ASSOCIATION:      Rung 1 — observational "seeing" — P(Y|X)
    INTERVENTION:     Rung 2 — interventional "doing" — P(Y|do(X))
    COUNTERFACTUAL:   Rung 3 — imagining — P(Y_x | X=x', Y=y')
    ATTRIBUTION:      Causal attribution / responsibility — "what caused Y?"
    """

    ASSOCIATION = "association"
    INTERVENTION = "intervention"
    COUNTERFACTUAL = "counterfactual"
    ATTRIBUTION = "attribution"


class Difficulty(str, Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class CausalQuestion:
    """
    A single question in a CausalScenario.

    Attributes
    ----------
    question_id : str
        Unique identifier within the scenario.
    question_type : QuestionType
        Which rung of the ladder this question belongs to.
    text : str
        Human-readable question text.
    answer : Any
        Ground-truth answer (numeric probability or string label).
    answer_tolerance : float
        For numeric answers, accepted absolute error (default 0.05).
    metadata : dict
        Extra info (e.g., which variables are involved, query details).
    """

    question_id: str
    question_type: QuestionType
    text: str
    answer: Any  # float probability or string
    answer_tolerance: float = 0.05
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_correct(self, model_answer: Any) -> bool:
        """
        Check if a model answer is correct.

        For numeric answers, compares within tolerance.
        For string answers, compares case-insensitively.
        """
        if isinstance(self.answer, (int, float)):
            try:
                val = float(model_answer)
                return abs(val - float(self.answer)) <= self.answer_tolerance
            except (TypeError, ValueError):
                return False
        # String answer
        return str(model_answer).strip().lower() == str(self.answer).strip().lower()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question_type": self.question_type.value,
            "text": self.text,
            "answer": self.answer,
            "answer_tolerance": self.answer_tolerance,
            "metadata": self.metadata,
        }


@dataclass
class CausalScenario:
    """
    A benchmark test case grounded in a Structural Causal Model.

    Attributes
    ----------
    scenario_id : str
        Unique identifier.
    name : str
        Human-readable name (e.g. "Smoking → Cancer with Tar confounder").
    description : str
        Longer explanation of the causal structure.
    difficulty : Difficulty
        Simple / Medium / Hard.
    graph_edges : List[Tuple[str, str]]
        Directed edges (cause → effect) representing the causal DAG.
    variable_names : List[str]
        All variable names in the SCM.
    data : Optional[np.ndarray]
        Observational dataset, shape (n_samples, n_variables).
    data_columns : List[str]
        Column names corresponding to data axes.
    questions : List[CausalQuestion]
        Questions that require causal (not merely correlational) reasoning.
    scm_description : str
        Human-readable description of the structural equations.
    tags : List[str]
        Domain tags (e.g. "medical", "confounding", "mediation").
    """

    scenario_id: str
    name: str
    description: str
    difficulty: Difficulty
    graph_edges: List[Tuple[str, str]]
    variable_names: List[str]
    data: Optional[np.ndarray] = None
    data_columns: List[str] = field(default_factory=list)
    questions: List[CausalQuestion] = field(default_factory=list)
    scm_description: str = ""
    tags: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Accessors                                                            #
    # ------------------------------------------------------------------ #

    def questions_by_type(self, qt: QuestionType) -> List[CausalQuestion]:
        return [q for q in self.questions if q.question_type == qt]

    def get_data_df(self):
        """Return data as a pandas DataFrame (requires pandas)."""
        import pandas as pd  # optional

        cols = self.data_columns or [f"V{i}" for i in range(self.data.shape[1])]
        return pd.DataFrame(self.data, columns=cols)

    def summary(self) -> str:
        lines = [
            f"Scenario: {self.name} [{self.difficulty.value}]",
            f"  ID:       {self.scenario_id}",
            f"  Edges:    {self.graph_edges}",
            f"  Variables:{self.variable_names}",
            f"  Questions:{len(self.questions)} "
            f"(assoc={len(self.questions_by_type(QuestionType.ASSOCIATION))}, "
            f"interv={len(self.questions_by_type(QuestionType.INTERVENTION))}, "
            f"cf={len(self.questions_by_type(QuestionType.COUNTERFACTUAL))}, "
            f"attr={len(self.questions_by_type(QuestionType.ATTRIBUTION))})",
        ]
        if self.scm_description:
            lines.append(f"  SCM:      {self.scm_description}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "description": self.description,
            "difficulty": self.difficulty.value,
            "graph_edges": list(self.graph_edges),
            "variable_names": self.variable_names,
            "questions": [q.to_dict() for q in self.questions],
            "scm_description": self.scm_description,
            "tags": self.tags,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
