"""
Tests for DataGenerator and SCMNode.
"""

import pytest
import numpy as np

from causal_eval.benchmark.data_generator import DataGenerator, SCMNode


@pytest.fixture
def simple_chain_gen():
    """X → Y SCM."""
    gen = DataGenerator(seed=42)
    gen.add_node(SCMNode(
        "X", [],
        equation=lambda pa, n: n,
        noise_fn=lambda n: np.random.normal(0, 1, n),
    ))
    gen.add_node(SCMNode(
        "Y", ["X"],
        equation=lambda pa, n: 0.8 * pa["X"] + n,
        noise_fn=lambda n: np.random.normal(0, 0.5, n),
    ))
    return gen


@pytest.fixture
def confounder_gen():
    """Z → X, Z → Y SCM (no X→Y)."""
    gen = DataGenerator(seed=42)
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
    return gen


class TestDataGeneratorBasics:
    def test_sample_returns_all_variables(self, simple_chain_gen):
        data = simple_chain_gen.sample(n=100)
        assert "X" in data and "Y" in data
        assert data["X"].shape == (100,)
        assert data["Y"].shape == (100,)

    def test_topological_order_correct(self, simple_chain_gen):
        order = simple_chain_gen.variable_names
        assert order.index("X") < order.index("Y")

    def test_cycle_detection(self):
        gen = DataGenerator()
        gen.add_node(SCMNode("A", ["B"], equation=lambda pa, n: pa["B"] + n))
        gen.add_node(SCMNode("B", ["A"], equation=lambda pa, n: pa["A"] + n))
        with pytest.raises(ValueError, match="cycle"):
            gen.sample(n=10)

    def test_missing_parent_raises(self):
        gen = DataGenerator()
        gen.add_node(SCMNode("Y", ["X"], equation=lambda pa, n: pa["X"] + n))
        with pytest.raises(ValueError, match="Parent 'X'"):
            gen.sample(n=10)

    def test_to_array_shape(self, simple_chain_gen):
        data = simple_chain_gen.sample(n=200)
        arr, cols = simple_chain_gen.to_array(data)
        assert arr.shape == (200, 2)
        assert set(cols) == {"X", "Y"}

    def test_edges_property(self, simple_chain_gen):
        edges = simple_chain_gen.edges
        assert ("X", "Y") in edges

    def test_clone_independence(self, simple_chain_gen):
        clone = simple_chain_gen.clone()
        clone.add_node(SCMNode("Z", [], equation=lambda pa, n: n))
        assert "Z" not in simple_chain_gen.nodes


class TestObservationalSampling:
    def test_linear_relationship_slope(self, simple_chain_gen):
        """E[Y | X] ≈ 0.8 * X in the simple chain."""
        np.random.seed(42)
        data = simple_chain_gen.sample(n=50_000)
        # OLS slope should be ≈ 0.8
        X, Y = data["X"], data["Y"]
        slope = np.cov(X, Y)[0, 1] / np.var(X)
        assert abs(slope - 0.8) < 0.05, f"Expected slope ≈ 0.8, got {slope:.3f}"

    def test_confounder_induces_correlation(self, confounder_gen):
        """X and Y should be correlated even though no X→Y edge."""
        np.random.seed(42)
        data = confounder_gen.sample(n=50_000)
        corr = np.corrcoef(data["X"], data["Y"])[0, 1]
        assert corr > 0.7, f"Expected high correlation, got {corr:.3f}"


class TestInterventional:
    def test_do_removes_dependence_on_parents(self, simple_chain_gen):
        """After do(X=2), X should be fixed at 2, not from its SCM."""
        np.random.seed(42)
        data = simple_chain_gen.do({"X": 2.0}, n=10_000)
        assert np.allclose(data["X"], 2.0), "X should be fixed at 2.0"

    def test_do_no_effect_when_no_causal_link(self, confounder_gen):
        """do(X=2) should not affect Y in the confounder model."""
        np.random.seed(42)
        baseline = np.mean(confounder_gen.sample(n=50_000)["Y"])
        intervened = np.mean(confounder_gen.do({"X": 2.0}, n=50_000)["Y"])
        # Y depends on Z, not X. After do(X=2), E[Y] ≈ E[0.9*Z] = 0
        assert abs(intervened - baseline) < 0.1, (
            f"do(X=2) should not shift E[Y] significantly; "
            f"baseline={baseline:.3f}, intervened={intervened:.3f}"
        )

    def test_interventional_mean(self, simple_chain_gen):
        """E[Y | do(X=1)] ≈ 0.8 * 1 = 0.8."""
        np.random.seed(42)
        mean_y = simple_chain_gen.interventional_mean("Y", {"X": 1.0}, n=50_000)
        assert abs(mean_y - 0.8) < 0.05, f"Expected ≈ 0.8, got {mean_y:.3f}"

    def test_interventional_prob(self, simple_chain_gen):
        """P(Y > 0 | do(X=1)) ≈ P(Z > (0-0.8)/0.5) ≈ 0.945."""
        from scipy.stats import norm
        expected = 1 - norm.cdf(0.0, loc=0.8, scale=0.5)
        np.random.seed(42)
        prob = simple_chain_gen.interventional_prob(
            "Y", lambda y: y > 0, {"X": 1.0}, n=50_000
        )
        assert abs(prob - expected) < 0.02, f"Expected ≈ {expected:.3f}, got {prob:.3f}"


class TestCounterfactual:
    def test_counterfactual_noise_abduction(self, simple_chain_gen):
        """
        For additive noise SCM Y = 0.8X + N:
        Given observed X=0, Y=0.3 (so N_Y = 0.3),
        counterfactual Y with X=2 should be 0.8*2 + 0.3 = 1.9.
        """
        np.random.seed(42)
        cf_data = simple_chain_gen.counterfactual(
            observed={"X": 0.0, "Y": 0.3},
            interventions={"X": 2.0},
            n=1,
        )
        assert abs(cf_data["Y"][0] - 1.9) < 0.01, (
            f"Expected Y_cf ≈ 1.9, got {cf_data['Y'][0]:.3f}"
        )

    def test_counterfactual_no_change_without_intervention(self, simple_chain_gen):
        """If intervention matches observed, Y_cf ≈ Y_obs."""
        np.random.seed(42)
        cf_data = simple_chain_gen.counterfactual(
            observed={"X": 1.0, "Y": 1.1},
            interventions={"X": 1.0},
            n=1,
        )
        assert abs(cf_data["Y"][0] - 1.1) < 0.01


class TestAttribution:
    def test_pn_necessary_cause(self, simple_chain_gen):
        """
        In a clean causal chain X→Y, X is a necessary cause.
        PN should be positive.
        """
        np.random.seed(42)
        pn = simple_chain_gen.probability_of_necessity(
            cause_var="X",
            cause_val=2.0,
            effect_var="Y",
            effect_condition=lambda y: y > 1.0,
            n=5_000,
        )
        assert 0.3 < pn < 1.0, f"Expected PN in (0.3, 1), got {pn:.3f}"

    def test_ps_sufficient_cause(self, simple_chain_gen):
        """PS should be positive for a real causal link."""
        np.random.seed(42)
        ps = simple_chain_gen.probability_of_sufficiency(
            cause_var="X",
            cause_val=2.0,
            effect_var="Y",
            effect_condition=lambda y: y > 1.0,
            n=5_000,
        )
        assert ps > 0.5, f"Expected PS > 0.5, got {ps:.3f}"
