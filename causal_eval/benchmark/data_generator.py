"""
DataGenerator: Creates synthetic causal datasets from Structural Causal Models (SCMs).

A Structural Causal Model (SCM) is defined by:
  - A set of variables V
  - A DAG G over V
  - For each non-root variable Vi: a structural equation
        Vi = fi(pa(Vi), Ni)
    where pa(Vi) are Vi's parents and Ni is an independent noise term.

This module generates:
  1. Observational data: standard forward sampling from the SCM
  2. Interventional data: do(X=x) — mutilate incoming edges to X, set X=x
  3. Counterfactual data: abduction → action → prediction (twin-network)

References:
    Peters, J., Mooij, J. M., Janzing, D., & Schölkopf, B. (2014).
    Causal discovery with continuous additive noise models.
    Journal of Machine Learning Research, 15(1), 2009-2053.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SCMNode:
    """
    A single node in a Structural Causal Model.

    Parameters
    ----------
    name : str
        Variable name.
    parents : List[str]
        Names of parent variables (empty for exogenous roots).
    equation : Callable
        f(parent_values_dict, noise) → float/int
        Called during forward sampling.
    noise_fn : Callable
        noise_fn(n_samples) → np.ndarray  — generates exogenous noise.
    domain : str
        Domain / unit description (e.g. "binary", "continuous [0,1]").
    """

    name: str
    parents: List[str]
    equation: Callable[[Dict[str, np.ndarray], np.ndarray], np.ndarray]
    noise_fn: Callable[[int], np.ndarray] = field(
        default_factory=lambda: lambda n: np.random.normal(0, 1, n)
    )
    domain: str = "continuous"


class DataGenerator:
    """
    Generates observational, interventional, and counterfactual datasets
    from a Structural Causal Model.

    Usage
    -----
    >>> gen = DataGenerator()
    >>> gen.add_node(SCMNode("X", [], lambda pa, n: n, lambda n: np.random.normal(0,1,n)))
    >>> gen.add_node(SCMNode("Y", ["X"], lambda pa, n: 0.8*pa["X"] + n))
    >>> data = gen.sample(n=1000)
    >>> int_data = gen.do({"X": 1.0}, n=1000)
    """

    def __init__(self, seed: Optional[int] = None):
        self.nodes: Dict[str, SCMNode] = {}
        self._topo_order: Optional[List[str]] = None
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)

    # ------------------------------------------------------------------ #
    # Graph construction                                                   #
    # ------------------------------------------------------------------ #

    def add_node(self, node: SCMNode) -> "DataGenerator":
        self.nodes[node.name] = node
        self._topo_order = None  # invalidate cache
        return self

    def _topological_order(self) -> List[str]:
        """Kahn's algorithm for topological sort."""
        if self._topo_order is not None:
            return self._topo_order

        in_degree: Dict[str, int] = {n: 0 for n in self.nodes}
        children: Dict[str, List[str]] = {n: [] for n in self.nodes}

        for name, node in self.nodes.items():
            for p in node.parents:
                if p not in self.nodes:
                    raise ValueError(f"Parent '{p}' of '{name}' not found in SCM.")
                in_degree[name] += 1
                children[p].append(name)

        queue = [n for n, d in in_degree.items() if d == 0]
        order: List[str] = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in children[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(order) != len(self.nodes):
            raise ValueError("SCM graph contains a cycle — not a valid DAG.")

        self._topo_order = order
        return order

    @property
    def variable_names(self) -> List[str]:
        return self._topological_order()

    @property
    def edges(self) -> List[Tuple[str, str]]:
        edges = []
        for name, node in self.nodes.items():
            for p in node.parents:
                edges.append((p, name))
        return edges

    # ------------------------------------------------------------------ #
    # Sampling                                                             #
    # ------------------------------------------------------------------ #

    def sample(
        self,
        n: int = 1000,
        interventions: Optional[Dict[str, Any]] = None,
        noise_overrides: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Forward-sample n observations from the SCM.

        Parameters
        ----------
        n : int
            Number of samples.
        interventions : dict, optional
            do(X=x) interventions — removes structural equation for X and
            sets it to the given constant value.
        noise_overrides : dict, optional
            For counterfactual queries — inject pre-computed noise values.

        Returns
        -------
        dict : variable_name → np.ndarray of shape (n,)
        """
        order = self._topological_order()
        data: Dict[str, np.ndarray] = {}

        for name in order:
            node = self.nodes[name]

            # Intervention: do(name = value)
            if interventions and name in interventions:
                val = interventions[name]
                data[name] = np.full(n, float(val))
                continue

            # Generate noise
            if noise_overrides and name in noise_overrides:
                noise = noise_overrides[name]
            else:
                noise = node.noise_fn(n)

            # Parent values dict
            parent_vals = {p: data[p] for p in node.parents}

            # Evaluate structural equation
            data[name] = node.equation(parent_vals, noise)

        return data

    def to_array(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        """Convert sampled dict to (array, column_names)."""
        cols = self._topological_order()
        arr = np.column_stack([data[c] for c in cols])
        return arr, cols

    # ------------------------------------------------------------------ #
    # Interventional distribution P(Y | do(X=x))                          #
    # ------------------------------------------------------------------ #

    def do(
        self,
        interventions: Dict[str, Any],
        n: int = 1000,
    ) -> Dict[str, np.ndarray]:
        """
        Sample from interventional distribution P(· | do(X=x, ...)).

        This mutilates the DAG: incoming edges to intervened variables are
        removed, and those variables are fixed to the given values.
        """
        return self.sample(n=n, interventions=interventions)

    def interventional_mean(
        self,
        target: str,
        interventions: Dict[str, Any],
        n: int = 10_000,
    ) -> float:
        """Estimate E[target | do(interventions)] via Monte Carlo."""
        data = self.do(interventions, n=n)
        return float(np.mean(data[target]))

    def interventional_prob(
        self,
        target: str,
        target_condition: Callable[[np.ndarray], np.ndarray],
        interventions: Dict[str, Any],
        n: int = 10_000,
    ) -> float:
        """Estimate P(target_condition | do(interventions)) via Monte Carlo."""
        data = self.do(interventions, n=n)
        return float(np.mean(target_condition(data[target])))

    # ------------------------------------------------------------------ #
    # Counterfactual: twin-network approach                                #
    # ------------------------------------------------------------------ #

    def counterfactual(
        self,
        observed: Dict[str, float],
        interventions: Dict[str, Any],
        n: int = 10_000,
    ) -> Dict[str, np.ndarray]:
        """
        Estimate counterfactual P(Y_{do(X=x)} | obs) via twin-network method.

        Steps (Pearl, 2009, Section 7.1):
          1. Abduction: infer noise N* consistent with observed evidence
          2. Action: apply do(interventions) in the modified model
          3. Prediction: forward-sample with N* and interventions

        For additive noise models (Y = f(pa) + N), abduction is:
          N_Y* = Y_obs - f(pa_obs)

        This implementation uses a Gaussian noise abduction approximation.
        """
        order = self._topological_order()

        # Step 1: Abduct noise for each observed variable
        # We sample many noise instances and weight by P(evidence | noise)
        # For simplicity with additive noise: solve for noise directly.
        abducted_noise: Dict[str, np.ndarray] = {}

        # First, compute the "most likely" noise for each node given observations
        # by back-solving the structural equations.
        factual_data: Dict[str, float] = {}
        for name in order:
            node = self.nodes[name]
            parent_vals = {p: np.array([factual_data[p]]) for p in node.parents}
            if name in observed:
                # Back-solve: noise = observed - equation(parents, 0)
                # (assumes additive noise: eq(pa, noise) = g(pa) + noise)
                deterministic = node.equation(parent_vals, np.array([0.0]))[0]
                inferred_noise = observed[name] - deterministic
                abducted_noise[name] = np.full(n, inferred_noise)
                factual_data[name] = observed[name]
            else:
                # Not observed — sample normally (will be integrated out)
                abducted_noise[name] = node.noise_fn(n)
                # Use mean for the running factual
                noise_mean = node.noise_fn(1000)
                factual_data[name] = float(
                    np.mean(node.equation(
                        {p: np.full(1000, factual_data[p]) for p in node.parents},
                        noise_mean,
                    ))
                )

        # Step 2+3: Action + Prediction — forward sample with abducted noise
        return self.sample(
            n=n,
            interventions=interventions,
            noise_overrides=abducted_noise,
        )

    def counterfactual_prob(
        self,
        target: str,
        target_condition: Callable[[np.ndarray], np.ndarray],
        observed: Dict[str, float],
        interventions: Dict[str, Any],
        n: int = 10_000,
    ) -> float:
        """
        Estimate P(Y_{x} condition | evidence) via twin-network counterfactual.
        """
        cf_data = self.counterfactual(observed, interventions, n=n)
        return float(np.mean(target_condition(cf_data[target])))

    # ------------------------------------------------------------------ #
    # Association P(Y | X=x)                                              #
    # ------------------------------------------------------------------ #

    def conditional_prob(
        self,
        target: str,
        target_condition: Callable[[np.ndarray], np.ndarray],
        condition_var: str,
        condition_fn: Callable[[np.ndarray], np.ndarray],
        n: int = 100_000,
    ) -> float:
        """
        Estimate P(target_condition | condition_fn(condition_var)) observationally.
        """
        data = self.sample(n=n)
        mask = condition_fn(data[condition_var])
        if mask.sum() == 0:
            return float("nan")
        return float(np.mean(target_condition(data[target][mask])))

    # ------------------------------------------------------------------ #
    # Attribution helpers                                                  #
    # ------------------------------------------------------------------ #

    def probability_of_necessity(
        self,
        cause_var: str,
        cause_val: float,
        effect_var: str,
        effect_condition: Callable[[np.ndarray], np.ndarray],
        n: int = 10_000,
    ) -> float:
        """
        PN (Probability of Necessity):
          P(Y_{X=0} = 0 | X=cause_val, Y=effect_val)
        Estimates how necessary the cause was for the effect.
        """
        observed = {cause_var: cause_val}
        # Counterfactual: what if cause had been 0?
        cf_data = self.counterfactual(
            observed={cause_var: cause_val},
            interventions={cause_var: 0.0},
            n=n,
        )
        return float(np.mean(~effect_condition(cf_data[effect_var]).astype(bool)))

    def probability_of_sufficiency(
        self,
        cause_var: str,
        cause_val: float,
        effect_var: str,
        effect_condition: Callable[[np.ndarray], np.ndarray],
        n: int = 10_000,
    ) -> float:
        """
        PS (Probability of Sufficiency):
          P(Y_{X=x} = 1 | X=0, Y=0)
        """
        cf_data = self.counterfactual(
            observed={cause_var: 0.0},
            interventions={cause_var: cause_val},
            n=n,
        )
        return float(np.mean(effect_condition(cf_data[effect_var])))

    # ------------------------------------------------------------------ #
    # Utilities                                                            #
    # ------------------------------------------------------------------ #

    def clone(self) -> "DataGenerator":
        """Return a deep copy of this generator."""
        return copy.deepcopy(self)
