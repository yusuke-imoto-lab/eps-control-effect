r"""Tests for :mod:`epscontrol`.

The reference values come from Example 1 and Figure 3 of

    Y. Imoto and T. Yokoyama, *Multi-parameter persistence in dynamical systems
    for maximizing effects of control inputs*,

evaluated at :math:`x = (0, 6)^{\mathsf T}` of the branching system of Figure 2.
"""

from __future__ import annotations

import numpy as np
import pytest

import epscontrol as ec


@pytest.fixture(scope="module")
def paper():
    return ec.datasets.paper_example()


@pytest.fixture(scope="module")
def prob_l1(paper):
    return ec.build_problem(paper, p=1.0)


@pytest.fixture(scope="module")
def prob_linf(paper):
    return ec.build_problem(paper, p=np.inf)


# --------------------------------------------------------------- paper values
def test_dataset_shape(paper):
    assert paper.shape == (20, 2)
    assert list(paper.obs["h"].dropna()) == [6.0, 5.0, 2.0, 1.0, 0.0]


@pytest.mark.parametrize(
    "eps, expected", [(0.0, 6.0), (1.0, 5.0), (2.0, 2.0)]
)
def test_example1_l1_effect(prob_l1, eps, expected):
    """Example 1(1): h^{0,1,2-l^1}_f(x) = 6, 5, 2."""
    assert prob_l1.control_effect(eps)[0] == expected


@pytest.mark.parametrize(
    "eps, expected", [(0.0, 6.0), (1.0, 1.0), (2.0, 0.0)]
)
def test_example1_linf_effect(prob_linf, eps, expected):
    """Example 1(2): h^{0,1,2-l^inf}_f(x) = 6, 1, 0."""
    assert prob_linf.control_effect(eps)[0] == expected


@pytest.mark.parametrize(
    "eps, expected", [(0.0, 0.0), (1.0, -1.0), (2.0, -4.0)]
)
def test_example1_l1_difference(prob_l1, eps, expected):
    """Example 1(3): h^{Delta 0,1,2-l^1}_f(x) = 0, -1, -4."""
    assert prob_l1.effect_difference(eps)[0] == expected


@pytest.mark.parametrize(
    "eps, expected", [(0.0, 0.0), (1.0, -5.0), (2.0, -6.0)]
)
def test_example1_linf_difference(prob_linf, eps, expected):
    """Example 1(4): h^{Delta 0,1,2-l^inf}_f(x) = 0, -5, -6."""
    assert prob_linf.effect_difference(eps)[0] == expected


@pytest.mark.parametrize(
    "p, eps, expected",
    [
        (1.0, 1.0, [5.0, 6.0]),
        (1.0, 2.0, [2.0, 5.0, 6.0]),
        (np.inf, 1.0, [1.0, 2.0, 5.0, 6.0]),
        (np.inf, 2.0, [0.0, 1.0, 2.0, 5.0, 6.0]),
    ],
)
def test_figure3b_reachable_ranges(paper, p, eps, expected):
    """Figure 3b: the reachable ranges R^{l^p}_{h,f}(eps, x)."""
    prob = ec.build_problem(paper, p=p)
    assert prob.reachable_range(0, eps).tolist() == expected


@pytest.mark.parametrize(
    "p, r, expected", [(1.0, 5.0, 1.0), (1.0, 2.0, 2.0), (np.inf, 5.0, 1.0), (np.inf, 2.0, 1.0)]
)
def test_figure3b_controllability(paper, p, r, expected):
    """Figure 3b: C^{l^p}_{h,f}(x, r) = [1, inf], [2, inf], [1, inf], [1, inf]."""
    prob = ec.build_problem(paper, p=p)
    assert prob.sublevel_controllability(0, r) == expected


def test_dom_h_values_are_fixed(prob_linf, paper):
    """h_f = h on dom h (Lemma 3.4), because dom h consists of (c, eps)-fixed points."""
    h = paper.obs["h"].to_numpy(dtype=float)
    dom_h = np.isfinite(h)
    for eps in (0.0, 1.0, 5.0):
        got = prob_linf.control_effect(eps)[dom_h]
        assert np.allclose(got, h[dom_h])


def test_hypotheses_hold(paper):
    flags = ec.check_hypotheses(paper)
    assert all(flags.values()), flags


# ------------------------------------------------------------ general theory
@pytest.mark.parametrize("p", [1.0, 2.0, np.inf])
def test_lemma37_monotone_in_eps(paper, p):
    """Lemma 3.7: h^{eps'} <= h^{eps} <= h^{0} <= h^{-0} <= h^{-eps} <= h^{-eps'}."""
    prob = ec.build_problem(paper, p=p)
    chain = [
        prob.control_effect(3.0),
        prob.control_effect(1.0),
        prob.control_effect(0.0),
        prob.control_effect(-0.0),
        prob.control_effect(-1.0),
        prob.control_effect(-3.0),
    ]
    for lo, hi in zip(chain, chain[1:]):
        assert np.all(lo <= hi)


def test_lemma39_monotone_in_p(paper):
    """Lemma 3.9: h^{eps-l^{p'}} <= h^{eps-l^p} <= h^{-eps-l^p} <= h^{-eps-l^{p'}}."""
    probs = [ec.build_problem(paper, p=p) for p in (1.0, 2.0, np.inf)]
    for eps in (0.5, 1.0, 2.5):
        pos = [pr.control_effect(eps) for pr in probs]
        neg = [pr.control_effect(-eps) for pr in probs]
        for lo, hi in zip(pos, pos[1:]):
            assert np.all(hi <= lo)
        for lo, hi in zip(neg, neg[1:]):
            assert np.all(lo <= hi)
        assert np.all(pos[-1] <= neg[0])


def test_lemma36_zero_branches_agree(paper):
    """Lemma 3.6: h^{0-l^p}_f = h^{-0-l^p}_f when dom h and dom f are disjoint."""
    prob = ec.build_problem(paper, p=2.0)
    assert np.allclose(prob.control_effect(0.0), prob.control_effect(-0.0))


def test_lemma25_control_cost_monotone(prob_linf):
    """Lemma 2.5: C^{l^p}_{h,f}(x, r) is non-increasing in r."""
    C = prob_linf.control_cost()
    assert np.all(C[:, 1:] <= C[:, :-1])


def test_large_eps_reaches_global_minimum(paper):
    """With an unbounded budget every point attains min(Im h)."""
    prob = ec.build_problem(paper, p=1.0)
    got = prob.control_effect(np.inf)
    dom_f = prob.dom_f
    assert np.all(got[dom_f] == float(np.nanmin(paper.obs["h"])))


# ---------------------------------------------------------- minimizing paths
@pytest.mark.parametrize("p", [1.0, 2.0, np.inf])
@pytest.mark.parametrize("eps", [0.0, 1.0, 2.0, 4.0])
def test_theorem312_minimizing_path(paper, p, eps):
    """Theorem 3.12: the reconstructed path attains the effect value."""
    prob = ec.build_problem(paper, p=p)
    target = prob.control_effect(eps)
    for x in range(prob.n):
        if not np.isfinite(target[x]):
            continue
        path = prob.minimizing_path(x, eps)
        # (1) terminal value equals the effect function
        assert path["value"] == pytest.approx(target[x])
        # the path respects the budget
        assert path["epsilon"] <= eps + 1e-9
        # (2) N(gamma) meets dom h only at the terminal state
        inner = path["states"][:-1]
        assert not prob.dom_h[inner].any()
        # (3) every state of N(gamma) attains the same value with the budget
        # still available there. For p = inf the residual budget is eps
        # throughout, which is assertion (3) verbatim.
        for state, rem in zip(path["states"], path["residual"]):
            assert prob.control_effect(rem)[state] == pytest.approx(path["value"])
        if np.isinf(p):
            assert np.allclose(target[path["states"]], target[x])
        # the path is consistent with succ and cost
        for k, j in enumerate(path["jumps"]):
            assert prob.cost[path["states"][k], j] == pytest.approx(path["costs"][k])
            assert prob.succ[j] == path["states"][k + 1]


def test_minimizing_path_unreachable_raises():
    succ = np.array([-1, -1])
    cost = np.array([[0.0, np.inf], [np.inf, 0.0]])
    h = np.array([np.nan, 1.0])
    prob = ec.ControlProblem(succ, cost, h, layer=np.array([0, 0]), p=np.inf)
    with pytest.raises(ValueError, match="no epsilon"):
        prob.minimizing_path(0, 0.0)


# ------------------------------------------------------------------ solvers
def test_solver_selection(paper):
    """Time-aligned costs give a layered DAG; unrestricted costs do not."""
    assert ec.build_problem(paper, p=np.inf).solver == "layered"
    assert (
        ec.build_problem(
            paper.copy(), p=np.inf, time_aligned=False, cost_matrix_key="none"
        ).solver
        == "dijkstra"
    )


@pytest.mark.parametrize("p", [1.0, 2.0, np.inf])
def test_solvers_agree(paper, p):
    """The layered and Dijkstra solvers give identical control costs."""
    layered = ec.build_problem(paper, p=p)
    general = ec.ControlProblem(
        succ=layered.succ, cost=layered.cost, h=layered.h, layer=None, p=p
    )
    assert general.solver == "dijkstra"
    assert np.allclose(
        layered.control_cost_exact(), general.control_cost_exact(), equal_nan=True
    )


def test_reachable_class_matches_range(prob_linf):
    """[x]^{eps-l^p}_f intersected with dom h reproduces the reachable range."""
    for eps in (0.0, 1.0, 2.0):
        cls = prob_linf.reachable_class(0, eps)
        got = np.unique(prob_linf.h[cls & prob_linf.dom_h])
        assert got.tolist() == prob_linf.reachable_range(0, eps).tolist()


def test_reachable_domain_matches_effect(prob_l1):
    """D^{l^p}_{h,f}(eps, r) = {x | h_f^{eps}(x) <= r}."""
    for eps in (0.0, 1.0, 3.0):
        for r in prob_l1.levels:
            mask = prob_l1.reachable_domain(eps, r)
            assert np.array_equal(mask, prob_l1.control_effect(eps) <= r)


# -------------------------------------------------------------- AnnData API
def test_from_ensemble_layout():
    a = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    b = np.array([[0.0, 1.0], [1.0, 1.0]])
    adata = ec.from_ensemble([a, b], values=[4.0, 9.0], var_names=["x1", "x2"])
    assert adata.shape == (5, 2)
    assert adata.obs["seq_id"].tolist() == [0, 0, 0, 1, 1]
    assert adata.obs["time"].tolist() == [0, 1, 2, 0, 1]
    assert np.allclose(adata.obs["h"].to_numpy()[[2, 4]], [4.0, 9.0])
    assert np.isnan(adata.obs["h"].to_numpy()[[0, 1, 3]]).all()
    assert list(adata.var_names) == ["x1", "x2"]


def test_from_ensemble_ragged_and_errors():
    with pytest.raises(ValueError, match="feature dimension"):
        ec.from_ensemble([np.zeros((2, 2)), np.zeros((2, 3))], values=[0.0, 0.0])
    with pytest.raises(ValueError, match="one entry per sequence"):
        ec.from_ensemble([np.zeros((2, 2))], values=[0.0, 1.0])


def test_control_effect_writes_obs(paper):
    adata = paper.copy()
    ec.control_effect(adata, [0.0, 1.0], p=1.0)
    assert "control_effect_eps0_p1" in adata.obs
    assert "control_effect_eps1_p1" in adata.obs
    assert adata.obs["control_effect_eps1_p1"].iloc[0] == 5.0
    assert adata.uns["epscontrol"]["control_effect"]["solver"] == "layered"


def test_control_effect_negative_key(paper):
    adata = paper.copy()
    ec.control_effect(adata, [-1.0], p=np.inf)
    assert "control_effect_eps-1_pinf" in adata.obs


def test_effect_difference_writes_obs(paper):
    adata = paper.copy()
    ec.effect_difference(adata, 2.0, p=np.inf, key_added="delta")
    assert adata.obs["delta"].iloc[0] == -6.0


def test_control_cost_writes_obsm(paper):
    adata = paper.copy()
    ec.control_cost(adata, p=1.0)
    assert adata.obsm["control_cost"].shape == (20, 5)
    assert np.allclose(adata.uns["epscontrol"]["levels"], [0.0, 1.0, 2.0, 5.0, 6.0])


def test_copy_semantics(paper):
    out = ec.control_effect(paper, 0.0, p=1.0, copy=True)
    assert out is not paper
    assert "control_effect_eps0_p1" in out.obs
    assert "control_effect_eps0_p1" not in paper.obs


def test_minimizing_path_via_anndata(paper):
    out = ec.minimizing_path(paper, 0, 2.0, p=np.inf)
    assert out["value"] == 0.0
    assert len(out["obs_names"]) == len(out["states"])


def test_problem_reuse_is_consistent(paper):
    adata = paper.copy()
    prob = ec.build_problem(adata, p=2.0)
    ec.control_effect(adata, 1.0, problem=prob, key_added="reused")
    ec.control_effect(adata, 1.0, p=2.0, key_added="rebuilt")
    assert np.allclose(adata.obs["reused"], adata.obs["rebuilt"])


def test_explicit_cost_matrix_is_used(paper):
    adata = paper.copy()
    n = adata.n_obs
    adata.uns["cost_matrix"] = np.full((n, n), np.inf)
    np.fill_diagonal(adata.uns["cost_matrix"], 0.0)
    prob = ec.build_problem(adata, p=np.inf)
    # No control is possible, so every eps gives the uncontrolled value.
    assert np.allclose(prob.control_effect(5.0), prob.control_effect(0.0))


def test_time_key_absent_uses_row_order():
    a = np.array([[0.0, 0.0], [1.0, 0.0]])
    adata = ec.from_ensemble([a], values=[1.0])
    del adata.obs["time"]
    prob = ec.build_problem(adata, p=np.inf)
    assert prob.succ.tolist() == [1, -1]


def test_missing_h_raises():
    a = np.array([[0.0, 0.0], [1.0, 0.0]])
    adata = ec.from_ensemble([a])
    with pytest.raises(ValueError, match="no terminal point"):
        ec.build_problem(adata)


def test_unknown_obs_label_raises(paper):
    with pytest.raises(KeyError):
        ec.minimizing_path(paper, "no-such-observation", 1.0)


# -------------------------------------------------------------- validation
def test_nonzero_diagonal_rejected():
    cost = np.array([[1.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match=r"c\(x, x\) = 0"):
        ec.ControlProblem(np.array([1, -1]), cost, np.array([np.nan, 0.0]))


def test_negative_cost_rejected():
    cost = np.array([[0.0, -1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="non-negative"):
        ec.ControlProblem(np.array([1, -1]), cost, np.array([np.nan, 0.0]))


def test_bad_shapes_rejected():
    with pytest.raises(ValueError, match="cost must have shape"):
        ec.ControlProblem(np.array([1, -1]), np.zeros((3, 3)), np.array([np.nan, 0.0]))
    with pytest.raises(ValueError, match="p must lie"):
        ec.ControlProblem(
            np.array([-1]), np.zeros((1, 1)), np.array([0.0]), p=0.5
        )


def test_hypotheses_detect_violation():
    """A cycle in f breaks X = ⊔_n f^{-n}(dom h)."""
    succ = np.array([1, 0, -1])
    cost = np.zeros((3, 3))
    cost[:] = np.inf
    np.fill_diagonal(cost, 0.0)
    prob = ec.ControlProblem(succ, cost, np.array([np.nan, np.nan, 5.0]))
    flags = prob.hypotheses()
    assert flags["disjoint_domains"]
    assert not flags["preimage_decomposition"]


# ------------------------------------------------------------- filtrations
def test_theorem43_filtration(paper):
    prob = ec.build_problem(paper, p=np.inf)
    filt = ec.filtration(prob, ps=(1.0, 2.0, np.inf), n_epsilons=7)
    assert filt.is_filtration()
    assert filt.is_filtration_in_p()
    assert filt.masks.shape == (3, 7, filt.deltas.size, prob.n)


def test_theorem44_difference_filtration(paper):
    prob = ec.build_problem(paper, p=np.inf)
    filt = ec.filtration(prob, ps=(1.0, np.inf), n_epsilons=6, kind="difference")
    assert filt.kind == "difference"
    assert filt.is_filtration()


def test_filtration_sizes_and_frame(paper):
    prob = ec.build_problem(paper, p=1.0)
    filt = ec.filtration(prob, ps=(1.0,), epsilons=(0.0, 2.0), deltas=(0.0, 6.0))
    sizes = filt.sizes
    assert sizes.shape == (1, 2, 2)
    # The largest budget and threshold include every point.
    assert sizes[0, 1, 1] == prob.n
    df = filt.to_frame()
    assert list(df.columns) == ["p", "eps", "delta", "size"]
    assert len(df) == 4


def test_filtration_betti0(paper):
    prob = ec.build_problem(paper, p=np.inf)
    filt = ec.filtration(prob, ps=(np.inf,), epsilons=(0.0,), deltas=(6.0,))
    b0 = filt.betti0(prob)
    # At eps = 0 and delta = max(Im h) every orbit is included and they are the
    # five ensemble members, each a connected chain.
    assert b0[0, 0, 0] == 5


def test_filtration_mask_lookup(paper):
    prob = ec.build_problem(paper, p=1.0)
    filt = ec.filtration(prob, ps=(1.0,), epsilons=(0.0, 1.0), deltas=(5.0,))
    assert filt.mask(1.0, 1.0, 5.0)[0]
    assert not filt.mask(1.0, 0.0, 5.0)[0]


def test_filtration_rejects_unsorted(paper):
    prob = ec.build_problem(paper, p=1.0)
    with pytest.raises(ValueError, match="ascending"):
        ec.filtration(prob, ps=(np.inf, 1.0))


# --------------------------------------------------------------- datasets
@pytest.fixture(scope="module")
def chaotic():
    return ec.datasets.chaotic_competition(n_members=12, seed=1)


def test_competition_attractor_is_chaotic():
    """The pre-shift system has a positive largest Lyapunov exponent.

    Reference value ~0.0203 from Vano et al., Nonlinearity 19 (2006), 2391.
    """
    A = ec.datasets.competition_matrix
    r = ec.datasets.competition_growth_rates
    dt, n, skip = 0.01, 120_000, 20_000
    x = np.array([0.3, 0.3, 0.3, 0.3])
    v = np.eye(4)[0]
    total, count = 0.0, 0
    for i in range(n):
        J = r[:, None] * (
            np.eye(4) * (1.0 - A @ x)[:, None] - np.outer(x, np.ones(4)) * A
        )
        v = v + dt * (J @ v)
        x = ec.datasets._lv_step(x, A, r, dt)
        nv = np.linalg.norm(v)
        if nv > 0:
            v /= nv
            if i >= skip:
                total += np.log(nv)
                count += 1
    lle = total / (count * dt)
    assert 0.01 < lle < 0.03, lle


def test_competition_post_shift_is_bistable():
    """Both {1,2} and {1,3} are stable against invasion after the shift."""
    A = ec.datasets.competition_matrix_post
    r = ec.datasets.competition_growth_rates
    expected = {1: 0.7143, 2: 0.2941}
    for j, x1_star in expected.items():
        idx = np.array([0, j])
        xs = np.linalg.solve(A[np.ix_(idx, idx)], np.ones(2))
        assert xs.min() > 0
        assert xs[0] == pytest.approx(x1_star, abs=1e-3)
        x = np.zeros(4)
        x[idx] = xs
        J = r[:, None] * (
            np.eye(4) * (1.0 - A @ x)[:, None] - np.outer(x, np.ones(4)) * A
        )
        assert np.max(np.linalg.eigvals(J).real) < 1e-9


def test_chaotic_competition_two_outcomes(chaotic):
    """The regime shift splits the ensemble between two alternative states."""
    levels = np.unique(chaotic.obs["h"].dropna().to_numpy())
    assert levels.size == 2
    assert sorted(np.round(levels, 4)) == [0.2857, 0.7059]


def test_chaotic_competition_hypotheses(chaotic):
    """The supplied cost matrix makes all four hypotheses hold."""
    assert all(ec.check_hypotheses(chaotic).values())


def test_chaotic_competition_control_window(chaotic):
    """Securing the good outcome gets much more expensive after the shift."""
    prob = ec.build_problem(chaotic, p=np.inf)
    t = chaotic.obs["t"].to_numpy()
    t_switch = chaotic.uns["competition"]["t_switch"]
    cost = prob.control_cost()[:, 0]
    early = np.median(cost[t < 0.4 * t_switch])
    late = np.median(cost[t > t_switch + 60.0])
    assert np.isfinite(early) and np.isfinite(late)
    assert late > 10 * early, (early, late)


def test_chaotic_competition_observed_species_only(chaotic):
    """The states are delay vectors of one scalar record, not the full system."""
    m, _ = chaotic.uns["competition"]["embedding"]
    assert chaotic.n_vars == m
    assert chaotic.uns["competition"]["true_states"].shape[-1] == 4
    assert chaotic.X.shape[1] < 4


def test_competition_post3_has_three_stable_states():
    """The three-outcome matrix supports {1,2}, {1,3} and {1,4}, all stable."""
    from itertools import combinations

    A = ec.datasets.competition_matrix_post3
    r = ec.datasets.competition_growth_rates
    levels = []
    for k in range(1, 5):
        for S in combinations(range(4), k):
            idx = np.array(S)
            try:
                xs = np.linalg.solve(A[np.ix_(idx, idx)], np.ones(k))
            except np.linalg.LinAlgError:
                continue
            if (xs <= 1e-12).any():
                continue
            x = np.zeros(4)
            x[idx] = xs
            J = r[:, None] * (
                np.eye(4) * (1.0 - A @ x)[:, None] - np.outer(x, np.ones(4)) * A
            )
            if np.linalg.eigvals(J).real.max() < 1e-9:
                levels.append((tuple(i + 1 for i in S), x[0]))
    assert {s for s, _ in levels} == {(1, 2), (1, 3), (1, 4)}
    x1 = sorted(round(v, 3) for _, v in levels)
    assert x1 == [0.165, 0.567, 0.832]
    assert min(np.diff(x1)) > 0.2, "outcomes must be well separated"


def test_chaotic_competition_three_outcomes():
    """With the three-outcome endpoint, Im h has three values."""
    adata = ec.datasets.chaotic_competition(
        n_members=30, seed=7, A_post=ec.datasets.competition_matrix_post3
    )
    levels = np.unique(adata.obs["h"].dropna().to_numpy())
    assert levels.size == 3
    assert all(ec.check_hypotheses(adata).values())
    prob = ec.build_problem(adata, p=np.inf)
    assert prob.levels.size == 3
    # The chaotic phase feeds every basin, so no outcome is vanishingly rare.
    _, counts = np.unique(adata.obs["h"].dropna().to_numpy(), return_counts=True)
    assert counts.min() >= 3, counts


def test_chaotic_competition_graded_control_window():
    """Better outcomes cost more, and all of them get harder after the shift."""
    adata = ec.datasets.chaotic_competition(
        n_members=30, seed=7, A_post=ec.datasets.competition_matrix_post3
    )
    prob = ec.build_problem(adata, p=np.inf)
    t = adata.obs["t"].to_numpy()
    t_switch = adata.uns["competition"]["t_switch"]
    cost = prob.control_cost()
    early = t < 0.4 * t_switch
    late = t > t_switch + 60.0
    # Securing the best outcome is the most expensive at any time (Lemma 2.5).
    assert np.all(cost[:, 0] >= cost[:, 1] - 1e-12)
    assert np.all(cost[:, 1] >= cost[:, 2] - 1e-12)
    # And it becomes far more expensive once the shift has passed.
    best_early = np.median(cost[early, 0])
    best_late = np.median(cost[late, 0])
    assert best_late > 10 * best_early, (best_early, best_late)


def test_plot_effect_heatmap(chaotic):
    """The heatmap grid is built, monotone, and rejects unusable input."""
    import matplotlib

    matplotlib.use("Agg")
    prob = ec.build_problem(chaotic, p=np.inf)
    fig, axes = ec.plot_effect_heatmap(
        chaotic, prob, ps=(1.0, np.inf), epsilons=(0.0, 0.01, 0.05)
    )
    assert axes.shape == (2, 3)
    # every panel drew one image
    assert all(len(ax.images) == 1 for ax in axes.flat)
    matplotlib.pyplot.close(fig)

    # ragged sequences cannot share a time axis
    ragged = ec.from_ensemble(
        [np.zeros((3, 2)), np.zeros((5, 2))], values=[0.0, 1.0]
    )
    with pytest.raises(ValueError, match="differing lengths"):
        ec.plot_effect_heatmap(ragged, ec.build_problem(ragged))


def test_plot_effect_heatmap_rejects_many_levels():
    """A discrete palette is refused when h has too many values."""
    import matplotlib

    matplotlib.use("Agg")
    seqs = [np.array([[float(i), 0.0], [float(i), 1.0]]) for i in range(15)]
    adata = ec.from_ensemble(seqs, values=np.arange(15.0))
    prob = ec.build_problem(adata, p=np.inf)
    assert prob.levels.size == 15
    with pytest.raises(ValueError, match="distinct values"):
        ec.plot_effect_heatmap(adata, prob)


def test_heatmap_panels_are_monotone(chaotic):
    """Panels improve to the right (Lemma 3.7) and downward (Lemma 3.9)."""
    prob = ec.build_problem(chaotic, p=np.inf)
    ps, epsilons = (1.0, 2.0, np.inf), (0.0, 0.005, 0.01, 0.05)
    grids = {
        (p, e): (prob if prob.p == p else prob.with_p(p)).control_effect(e)
        for p in ps
        for e in epsilons
    }
    for p in ps:
        for a, b in zip(epsilons, epsilons[1:]):
            assert np.all(grids[(p, b)] <= grids[(p, a)] + 1e-12)
    for e in epsilons:
        for a, b in zip(ps, ps[1:]):
            assert np.all(grids[(b, e)] <= grids[(a, e)] + 1e-12)
    # with no control the norm is irrelevant
    for p in ps[1:]:
        assert np.array_equal(grids[(ps[0], 0.0)], grids[(p, 0.0)])


def test_version_is_consistent():
    """__version__, pyproject and CITATION.cff must agree."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text()
    declared = re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1)
    assert ec.__version__ == declared, (ec.__version__, declared)
    cff = root / "CITATION.cff"
    if cff.exists():
        cff_version = re.search(r"^version: (\S+)", cff.read_text(), re.M).group(1)
        assert cff_version == declared, (cff_version, declared)


def test_public_api_is_documented():
    """Every exported name appears in the README, so the docs cannot drift."""
    from pathlib import Path

    readme_path = Path(__file__).resolve().parents[1] / "README.md"
    if not readme_path.exists():
        pytest.skip("README not present")
    readme = readme_path.read_text()
    undocumented = [n for n in ec.__all__ if n not in readme]
    undocumented += [
        f"datasets.{n}" for n in ec.datasets.__all__ if n not in readme
    ]
    assert not undocumented, undocumented


def test_stable_states_public_api():
    """stable_states is importable and needs no notebook-local variables."""
    st = ec.datasets.stable_states(ec.datasets.competition_matrix_post3)
    assert {s for s, _ in st} == {(1, 2), (1, 3), (1, 4)}
    assert sorted(round(float(x[0]), 3) for _, x in st) == [0.165, 0.567, 0.832]
    # r defaults to the package growth rates
    st2 = ec.datasets.stable_states(
        ec.datasets.competition_matrix_post3, r=ec.datasets.competition_growth_rates
    )
    assert [s for s, _ in st] == [s for s, _ in st2]
    # the chaotic pre-shift system has no invasion-resistant equilibrium
    assert ec.datasets.stable_states(ec.datasets.competition_matrix) == []
    with pytest.raises(ValueError, match="square"):
        ec.datasets.stable_states(np.ones((2, 3)))


def test_tutorial_cells_are_self_contained():
    """Every code cell of the ecology tutorial runs top-to-bottom without
    depending on names it never binds (beyond builtins and its own imports)."""
    import ast
    import builtins
    import json
    from pathlib import Path

    nb_path = Path(__file__).resolve().parents[1] / "examples" / "tutorial_ecology.ipynb"
    if not nb_path.exists():  # sdist without examples
        pytest.skip("tutorial notebook not present")
    nb = json.loads(nb_path.read_text())
    defined = set(dir(builtins))
    missing = {}
    for i, cell in enumerate(
        [c for c in nb["cells"] if c["cell_type"] == "code"], start=1
    ):
        src = "".join(cell["source"])
        tree = ast.parse(src)
        used, binds = set(), set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Name):
                (used if isinstance(n.ctx, ast.Load) else binds).add(n.id)
            elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                binds.add(n.name)
            elif isinstance(n, ast.alias):
                binds.add((n.asname or n.name).split(".")[0])
            elif isinstance(n, ast.arg):
                binds.add(n.arg)
            elif isinstance(n, ast.comprehension) and isinstance(n.target, ast.Name):
                binds.add(n.target.id)
        gap = used - defined - binds
        if gap:
            missing[i] = sorted(gap)
        defined |= binds
    assert not missing, f"cells referencing undefined names: {missing}"


def test_simulate_competition_shapes():
    traj, times = ec.datasets.simulate_competition(
        np.array([0.3, 0.3, 0.3, 0.3]), t_max=20.0, sample_every=1.0
    )
    assert traj.shape == (times.size, 4)
    ens, times2 = ec.datasets.simulate_competition(
        np.full((5, 4), 0.3), t_max=20.0, sample_every=1.0
    )
    assert ens.shape == (times2.size, 5, 4)
    assert (ens >= 0).all()


def test_branching_ensemble():
    adata = ec.datasets.branching_ensemble(n_members=8, n_steps=5, seed=1)
    assert adata.shape == (40, 2)
    prob = ec.build_problem(adata, p=np.inf)
    assert all(ec.check_hypotheses(adata, problem=prob).values())
    eff0 = prob.control_effect(0.0)
    eff1 = prob.control_effect(1.0)
    assert np.all(eff1 <= eff0)


# ---------------------------------------------------------------- plotting
def test_plots_run(paper):
    import matplotlib

    matplotlib.use("Agg")
    adata = paper.copy()
    prob = ec.build_problem(adata, p=np.inf)
    ec.control_effect(adata, 1.0, problem=prob, key_added="eff")
    ec.effect_difference(adata, 1.0, problem=prob, key_added="diff")
    fig, ax = ec.plot_ensemble(adata)
    ec.plot_effect(adata, "eff", ax=ax)
    ec.plot_effect(adata, "diff", center=0.0)
    path = ec.minimizing_path(adata, 0, 1.0, problem=prob)
    ec.plot_minimizing_path(adata, path, ax=ax)
    ec.plot_effect_curve(prob, 0, ps=(1.0, np.inf), n_points=9)
    filt = ec.filtration(prob, ps=(1.0, np.inf), n_epsilons=5)
    ec.plot_filtration_sizes(filt)
    matplotlib.pyplot.close("all")
