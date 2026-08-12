r"""Reference datasets, including the worked example of the paper.

:func:`paper_example` reproduces the branching system of Figures 2 and 3 of
Imoto--Yokoyama, for which the paper states closed-form values (Example 1); the
test suite checks the library against them.  :func:`branching_ensemble` is a
larger synthetic ensemble for tutorials.
"""

from __future__ import annotations

import numpy as np

from ._data import from_ensemble

__all__ = [
    "paper_example",
    "branching_ensemble",
    "competition_growth_rates",
    "competition_matrix",
    "competition_matrix_post",
    "competition_matrix_post3",
    "simulate_competition",
    "stable_states",
    "chaotic_competition",
]


def _paper_dynamics(state: np.ndarray) -> np.ndarray:
    r"""One step of the map of Figure 2a.

    .. math::

        f\bigl((x_1, x_2)^{\mathsf T}\bigr) =
        \begin{cases}
            (x_1 + 3,\; x_2 - 1)^{\mathsf T} & x_1 \ge 3 \wedge x_2 < 5,\\
            (x_1 + 3,\; x_2)^{\mathsf T} & \text{otherwise.}
        \end{cases}
    """
    x1, x2 = float(state[0]), float(state[1])
    return np.array([x1 + 3.0, x2 - 1.0 if (x1 >= 3.0 and x2 < 5.0) else x2])


def paper_example(*, starts: tuple[float, ...] = (6.0, 5.0, 4.0, 3.0, 2.0), n_steps: int = 3):
    r"""The branching example of Figures 2 and 3 of Imoto--Yokoyama.

    Five orbits start at :math:`(0, x_2)^{\mathsf T}` for
    :math:`x_2 \in \{6, 5, 4, 3, 2\}` and are iterated three times under the map
    of :func:`_paper_dynamics`.  The dynamics translate to the right, and drop
    by one unit once :math:`x_1 \ge 3` and :math:`x_2 < 5`, so the ensemble
    splits into an upper branch that keeps its height and a lower branch that
    descends.  The evaluation function is the height of the terminal state,
    :math:`h((x_1, x_2)^{\mathsf T}) = x_2` on :math:`x_1 = 9`, and the default
    time-aligned Euclidean cost reproduces the paper's

    .. math::

        c\bigl((x_1, x_2)^{\mathsf T}, (x_1', x_2')^{\mathsf T}\bigr) =
        \begin{cases}
            \lvert x_2 - x_2' \rvert & x_1 = x_1',\\
            \infty & \text{otherwise.}
        \end{cases}

    Parameters
    ----------
    starts
        Initial heights :math:`x_2` at :math:`x_1 = 0`.
    n_steps
        Number of applications of :math:`f`; the default 3 lands the terminal
        states on :math:`x_1 = 9`, where :math:`h` is defined.

    Returns
    -------
    anndata.AnnData
        20 observations: 5 sequences of 4 time points, with ``obs['h']`` holding
        the terminal heights ``[6, 5, 2, 1, 0]``.

    Examples
    --------
    >>> import numpy as np, epscontrol as ec
    >>> adata = ec.datasets.paper_example()
    >>> prob = ec.build_problem(adata, p=1.0)
    >>> [float(prob.control_effect(e)[0]) for e in (0.0, 1.0, 2.0)]
    [6.0, 5.0, 2.0]
    >>> prob_inf = ec.build_problem(adata, p=np.inf)
    >>> [float(prob_inf.control_effect(e)[0]) for e in (0.0, 1.0, 2.0)]
    [6.0, 1.0, 0.0]

    Notes
    -----
    Example 1 of the paper records exactly these values for
    :math:`x = (0, 6)^{\mathsf T}`, together with the differences
    :math:`h_f^{\Delta 1\text{-}\ell^1}(x) = -1`,
    :math:`h_f^{\Delta 2\text{-}\ell^1}(x) = -4`,
    :math:`h_f^{\Delta 1\text{-}\ell^\infty}(x) = -5` and
    :math:`h_f^{\Delta 2\text{-}\ell^\infty}(x) = -6`.
    """
    sequences, values = [], []
    for x2 in starts:
        traj = [np.array([0.0, float(x2)])]
        for _ in range(int(n_steps)):
            traj.append(_paper_dynamics(traj[-1]))
        arr = np.vstack(traj)
        sequences.append(arr)
        values.append(float(arr[-1, 1]))
    return from_ensemble(
        sequences,
        values=values,
        seq_names=[f"x2={x2:g}" for x2 in starts],
        var_names=["x1", "x2"],
    )


def branching_ensemble(
    *,
    n_members: int = 60,
    n_steps: int = 12,
    noise: float = 0.06,
    drift: float = 0.35,
    seed: int = 0,
):
    r"""A synthetic ensemble whose members fall into two outcome branches.

    Each member starts near :math:`x_2 = 0` and drifts to the right; once past
    the branch point its vertical velocity is pushed up or down, so the ensemble
    separates into a high-value and a low-value branch.  The evaluation function
    of a member is the squared vertical offset of its terminal state, so small
    values are the desirable outcome and control that nudges a member onto the
    upper branch pays off.

    Parameters
    ----------
    n_members
        Number of ensemble members.
    n_steps
        Number of time points per member.
    noise
        Standard deviation of the per-step vertical perturbation.
    drift
        Strength of the branch separation.
    seed
        Seed of :func:`numpy.random.default_rng`.

    Returns
    -------
    anndata.AnnData
        ``n_members * n_steps`` observations, with ``obs['h']`` defined on the
        terminal state of each member.
    """
    rng = np.random.default_rng(int(seed))
    sequences, values = [], []
    for i in range(int(n_members)):
        y = float(rng.normal(0.0, 0.25))
        side = 1.0 if rng.random() < 0.5 else -1.0
        traj = []
        for t in range(int(n_steps)):
            x = t / (n_steps - 1) * 3.0
            traj.append([x, y])
            gate = 1.0 / (1.0 + np.exp(-(x - 1.2) * 4.0))
            y = y + side * drift * gate / n_steps * 6.0 + float(rng.normal(0.0, noise))
        arr = np.asarray(traj, dtype=np.float64)
        sequences.append(arr)
        values.append(float(arr[-1, 1] ** 2))
    return from_ensemble(sequences, values=values, var_names=["x1", "x2"])


# --------------------------------------------------------------------------
# Four-species competitive Lotka--Volterra with a regime shift
# --------------------------------------------------------------------------

#: Intrinsic growth rates of the chaotic four-species competition system of
#: Vano, Wildenberg, Anderson, Noel & Sprott, *Chaos in low-dimensional
#: Lotka--Volterra models of competition*, Nonlinearity **19** (2006), 2391.
competition_growth_rates = np.array([1.0, 0.72, 1.53, 1.27])

#: Interaction matrix of the same system.  The resulting attractor is chaotic;
#: the largest Lyapunov exponent reported in that paper is ~0.0203, which
#: :func:`simulate_competition` reproduces to ~0.020.
competition_matrix = np.array(
    [
        [1.00, 1.09, 1.52, 0.00],
        [0.00, 1.00, 0.44, 1.36],
        [2.33, 0.00, 1.00, 0.47],
        [1.21, 0.51, 0.35, 1.00],
    ]
)

#: Interaction matrix after the regime shift used by
#: :func:`chaotic_competition`.  Species 1 is niche-differentiated from each
#: competitor while species 2, 3 and 4 exclude one another, so the system is
#: bistable: the pairs :math:`\{1, 2\}` and :math:`\{1, 3\}` are both stable
#: against invasion, with species 1 settling at 0.714 and 0.294 respectively.
competition_matrix_post = np.array(
    [
        [1.00, 0.50, 0.80, 0.30],
        [0.60, 1.00, 1.45, 1.30],
        [0.40, 1.35, 1.00, 1.40],
        [0.90, 1.25, 1.30, 1.00],
    ]
)

#: A post-shift matrix with **three** alternative stable states, selected so the
#: outcomes are well separated and the chaotic attractor feeds all three basins.
#: States sampled along the attractor and switched immediately split about
#: 35/33/33 between the three; an ensemble that keeps evolving chaotically until
#: the shift is less even (typically 20-50% per outcome, varying with the seed),
#: because the members are correlated by the shared dynamics.  Species 1
#: competes weakly with each of the
#: others while species 2, 3 and 4 exclude one another, so the stable pairs are
#: :math:`\{1,2\}`, :math:`\{1,3\}` and :math:`\{1,4\}`, with species 1 settling
#: at 0.832, 0.165 and 0.567 respectively.  Pass
#: ``A_post=competition_matrix_post3`` to :func:`chaotic_competition` for an
#: evaluation function whose image has three values instead of two.
competition_matrix_post3 = np.array(
    [
        [1.00, 0.33, 0.91, 0.64],
        [0.59, 1.00, 1.60, 1.64],
        [0.50, 1.30, 1.00, 1.27],
        [0.57, 1.60, 1.07, 1.00],
    ]
)


def _lv_rhs(x: np.ndarray, A: np.ndarray, r: np.ndarray) -> np.ndarray:
    r"""Competitive Lotka--Volterra field
    :math:`\dot x_i = r_i x_i (1 - \sum_j A_{ij} x_j)`, vectorized over rows."""
    return r * x * (1.0 - x @ A.T)


def _lv_step(x: np.ndarray, A: np.ndarray, r: np.ndarray, dt: float) -> np.ndarray:
    """One classical fourth-order Runge--Kutta step."""
    k1 = _lv_rhs(x, A, r)
    k2 = _lv_rhs(x + 0.5 * dt * k1, A, r)
    k3 = _lv_rhs(x + 0.5 * dt * k2, A, r)
    k4 = _lv_rhs(x + dt * k3, A, r)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_competition(
    x0: np.ndarray,
    *,
    A: np.ndarray | None = None,
    A_post: np.ndarray | None = None,
    t_switch: float = np.inf,
    t_max: float = 700.0,
    dt: float = 0.02,
    sample_every: float = 1.0,
    r: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Integrate the competitive Lotka--Volterra system, optionally with a shift.

    Parameters
    ----------
    x0
        Initial abundances, shape ``(4,)`` or ``(M, 4)`` for an ensemble of
        ``M`` members integrated together.
    A, A_post
        Interaction matrices before and after ``t_switch``; default to
        :data:`competition_matrix` and :data:`competition_matrix_post`.
    t_switch
        Time of the regime shift.  The default ``inf`` never switches, giving
        the pure chaotic attractor.
    t_max, dt, sample_every
        Integration horizon, Runge--Kutta step, and sampling interval.
    r
        Intrinsic growth rates; defaults to :data:`competition_growth_rates`.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Trajectory of shape ``(n_samples, M, 4)`` (or ``(n_samples, 4)`` when
        ``x0`` is one-dimensional) and the sample times.

    Notes
    -----
    Abundances are clipped at zero after each step: extinction is absorbing,
    and the unclipped equations would otherwise let a numerically negative
    abundance grow.
    """
    A = competition_matrix if A is None else np.asarray(A, dtype=np.float64)
    A_post = (
        competition_matrix_post if A_post is None else np.asarray(A_post, dtype=np.float64)
    )
    r = competition_growth_rates if r is None else np.asarray(r, dtype=np.float64)

    x = np.atleast_2d(np.asarray(x0, dtype=np.float64)).copy()
    squeeze = np.ndim(x0) == 1
    stride = max(1, int(round(sample_every / dt)))

    out = [x.copy()]
    times = [0.0]
    for i in range(int(t_max / dt)):
        Acur = A if i * dt < t_switch else A_post
        x = _lv_step(x, Acur, r, dt)
        np.clip(x, 0.0, None, out=x)
        if (i + 1) % stride == 0:
            out.append(x.copy())
            times.append((i + 1) * dt)
    traj = np.asarray(out)
    return (traj[:, 0, :] if squeeze else traj), np.asarray(times)


def stable_states(
    A: np.ndarray, r: np.ndarray | None = None, tol: float = 1e-9
) -> list[tuple[tuple[int, ...], np.ndarray]]:
    r"""Equilibria of a competitive Lotka--Volterra system that resist invasion.

    Enumerates every subset of surviving species, solves the interior
    equilibrium of that subset, and keeps the ones whose Jacobian in the
    **full** system has no eigenvalue with positive real part -- i.e. the
    species left out cannot invade.  These are the alternative stable states
    that a trajectory can settle into, so their number is the number of values
    the terminal evaluation function can take.

    Parameters
    ----------
    A
        Interaction matrix, shape ``(n, n)``.
    r
        Intrinsic growth rates, shape ``(n,)``.  Defaults to
        :data:`competition_growth_rates`.  They do not change *which* states are
        stable for a competitive system, but are needed for the Jacobian.
    tol
        A state is accepted when the largest real part of the eigenvalues is
        below ``tol``.

    Returns
    -------
    list of (tuple, numpy.ndarray)
        One entry per stable state: the 1-based indices of the surviving
        species, and the full abundance vector (zeros for the extinct ones).

    Examples
    --------
    >>> import epscontrol as ec
    >>> for surv, x in ec.datasets.stable_states(ec.datasets.competition_matrix_post3):
    ...     print(surv, round(float(x[0]), 3))
    (1, 2) 0.832
    (1, 3) 0.165
    (1, 4) 0.567

    Use this when designing your own post-shift regime: the number of stable
    states is the number of outcome levels, and their spread in the observed
    coordinate is what makes the levels distinguishable.
    """
    from itertools import combinations

    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError(f"A must be square, got {A.shape}")
    r = competition_growth_rates if r is None else np.asarray(r, dtype=np.float64)
    if r.size != n:
        raise ValueError(f"r must have length {n}, got {r.size}")

    out = []
    for k in range(1, n + 1):
        for S in combinations(range(n), k):
            idx = np.array(S)
            try:
                xs = np.linalg.solve(A[np.ix_(idx, idx)], np.ones(k))
            except np.linalg.LinAlgError:
                continue
            if (xs <= 1e-12).any():
                continue
            x = np.zeros(n)
            x[idx] = xs
            J = r[:, None] * (
                np.eye(n) * (1.0 - A @ x)[:, None] - np.outer(x, np.ones(n)) * A
            )
            if np.linalg.eigvals(J).real.max() < tol:
                out.append((tuple(i + 1 for i in S), x))
    return out


def _stable_equilibria(A: np.ndarray, r: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Abundance vectors of :func:`stable_states`, as an array."""
    states = stable_states(A, r=r, tol=tol)
    n = np.asarray(A).shape[0]
    return np.asarray([x for _, x in states]) if states else np.empty((0, n))


def _delay_embed(series: np.ndarray, m: int, tau: int) -> np.ndarray:
    r"""Delay-coordinate embedding
    :math:`(y_t, y_{t-\tau}, \dots, y_{t-(m-1)\tau})` of a scalar series."""
    n = series.size - (m - 1) * tau
    if n <= 0:
        raise ValueError(
            f"series of length {series.size} is too short for m={m}, tau={tau}"
        )
        # pragma: no cover
    lo = (m - 1) * tau
    return np.stack([series[lo - k * tau : lo - k * tau + n] for k in range(m)], axis=1)


def chaotic_competition(
    *,
    n_members: int = 60,
    t_switch: float = 500.0,
    t_max: float = 700.0,
    burn_in: float = 500.0,
    obs_interval: float = 5.0,
    embedding_dim: int = 3,
    embedding_lag: int = 1,
    noise: float = 0.005,
    seed: int = 0,
    observed_species: int = 0,
    A_post: np.ndarray | None = None,
    snap_tol: float = 0.02,
):
    r"""An ensemble of chaotic competition trajectories with a regime shift.

    Four species compete under the Lotka--Volterra system of Vano et al. (2006),
    whose attractor is chaotic (largest Lyapunov exponent ~0.02).  Ensemble
    members are states sampled along that attractor, perturbed by observational
    noise.  At ``t_switch`` the interaction matrix changes to
    :data:`competition_matrix_post` -- a press disturbance, ecologically -- and
    the system becomes bistable.  Which of the two alternative stable states a
    member reaches is decided by where chaos had carried it when the shift
    arrived, so nearby members diverge to different fates.

    Only **one** species is observed, at a coarse interval, and the state is
    reconstructed from that scalar record by delay-coordinate embedding.  The
    evaluation function is the terminal shortfall of the observed species,
    :math:`h = 1 - x_{\text{obs}}(\infty)`, so small :math:`h` is the desirable
    outcome and :math:`\operatorname{Im} h` has exactly two values.

    Parameters
    ----------
    n_members
        Number of ensemble members.
    t_switch, t_max
        Time of the regime shift and of the end of the record.
    burn_in
        Length of the preliminary run used to place sampling on the attractor.
    obs_interval
        Interval between observations of the single observed species.
    embedding_dim, embedding_lag
        Parameters :math:`m` and :math:`\tau` of the delay embedding.
    noise
        Standard deviation of the observational perturbation applied to the
        sampled attractor states.
    seed
        Seed of :func:`numpy.random.default_rng`.
    observed_species
        Index of the observed species; the default 0 is species 1.
    A_post
        Interaction matrix after the shift.  Defaults to
        :data:`competition_matrix_post`, which is bistable and gives an
        evaluation function with two values.  Pass
        :data:`competition_matrix_post3` for three alternative stable states,
        and therefore three values of :math:`h`.
    snap_tol
        Terminal abundances are snapped to the nearest stable equilibrium of the
        post-shift system, so that :math:`\\operatorname{Im} h` contains one
        level per alternative stable state rather than numerical duplicates.  A
        member further than ``snap_tol`` from every equilibrium has not settled
        and raises :class:`RuntimeError`.

    Returns
    -------
    anndata.AnnData
        ``n_members`` sequences of delay vectors.  ``obs['h']`` holds the
        terminal shortfall of the observed species, ``obs['t']`` the
        observation time, and ``obs['phase']`` is ``'chaotic'`` before
        ``t_switch`` and ``'post_shift'`` after.  ``uns`` records the true
        four-species trajectory (``'true_states'``), the interaction matrices,
        and the settings, so the reconstruction can be compared with the state
        it was reconstructed from.

    Notes
    -----
    ``uns['cost_matrix']`` is set so that control jumps stay within a time layer
    **and** cannot leave a terminal state: the outcome is evaluated once the
    trajectory has settled, when no further intervention is possible.  This
    makes :math:`\operatorname{dom} h` consist of
    :math:`(c, \varepsilon)`-fixed points, the hypothesis of Theorem 3.12 and
    Theorems 4.1/4.2, so ``check_hypotheses`` reports all four flags true.

    Examples
    --------
    >>> import numpy as np, epscontrol as ec
    >>> adata = ec.datasets.chaotic_competition(n_members=8, seed=1)
    >>> sorted(np.unique(adata.obs["h"].dropna()).round(4).tolist())
    [0.2857, 0.7059]
    >>> all(ec.check_hypotheses(adata).values())
    True
    """
    from sklearn.metrics import pairwise_distances

    rng = np.random.default_rng(int(seed))
    A_pre = competition_matrix
    A_post = (
        competition_matrix_post
        if A_post is None
        else np.asarray(A_post, dtype=np.float64)
    )

    # A long orbit on the chaotic attractor, used as the sampling pool.
    pool, _ = simulate_competition(
        np.array([0.3, 0.3, 0.3, 0.3]),
        t_switch=np.inf,
        t_max=burn_in + 3000.0,
        dt=0.02,
        sample_every=0.2,
    )
    pool = pool[int(burn_in / 0.2) :]

    idx = rng.choice(pool.shape[0], size=int(n_members), replace=False)
    x0 = np.clip(pool[idx] + rng.normal(0.0, noise, size=(int(n_members), 4)), 1e-5, None)

    traj, times = simulate_competition(
        x0, A_post=A_post, t_switch=t_switch, t_max=t_max, dt=0.02,
        sample_every=obs_interval,
    )

    # Observe one species; reconstruct the state by delay embedding.
    series = traj[:, :, int(observed_species)].T  # (M, n_obs)
    emb = np.stack(
        [_delay_embed(s, embedding_dim, embedding_lag) for s in series]
    )  # (M, L, m)
    t_emb = times[(embedding_dim - 1) * embedding_lag :]

    # Terminal values are snapped to the equilibria of the post-shift system.
    # A member that has not fully relaxed would otherwise contribute a level of
    # its own, inflating Im h with numerical duplicates of one true outcome.
    x_end = traj[-1, :, int(observed_species)]
    equil = _stable_equilibria(A_post, r=competition_growth_rates)
    if equil.size:
        obs_levels = equil[:, int(observed_species)]
        gap = np.abs(x_end[:, None] - obs_levels[None, :])
        nearest = obs_levels[np.argmin(gap, axis=1)]
        stray = gap.min(axis=1) > snap_tol
        if stray.any():
            raise RuntimeError(
                f"{int(stray.sum())} member(s) had not settled to within "
                f"{snap_tol} of a stable state by t_max={t_max}; increase t_max "
                "or raise snap_tol."
            )
        x_end = nearest
    h = np.round(1.0 - x_end, 6)

    lag = embedding_lag * obs_interval
    adata = from_ensemble(
        [emb[i] for i in range(emb.shape[0])],
        values=h,
        var_names=[f"x{observed_species + 1}(t-{k * lag:g})" for k in range(embedding_dim)],
        obs={
            "t": np.tile(t_emb, emb.shape[0]),
            "phase": np.where(
                np.tile(t_emb, emb.shape[0]) < t_switch, "chaotic", "post_shift"
            ),
        },
    )

    # Time-aligned cost, with terminal states held fixed (see Notes).
    layer = adata.obs["time"].to_numpy()
    cost = pairwise_distances(np.asarray(adata.X))
    cost[layer[:, None] != layer[None, :]] = np.inf
    cost[np.isfinite(adata.obs["h"].to_numpy()), :] = np.inf
    np.fill_diagonal(cost, 0.0)
    adata.uns["cost_matrix"] = cost

    adata.uns["competition"] = {
        "true_states": traj.transpose(1, 0, 2)[:, (embedding_dim - 1) * embedding_lag :],
        "times": t_emb,
        "A_pre": A_pre,
        "A_post": A_post,
        "growth_rates": competition_growth_rates,
        "t_switch": float(t_switch),
        "observed_species": int(observed_species),
        "obs_interval": float(obs_interval),
        "embedding": (int(embedding_dim), int(embedding_lag)),
    }
    return adata
