r"""AnnData interface: ensembles of time series with terminal evaluations.

The input convention follows
`eps-attracting-basin <https://github.com/yusuke-imoto-lab/eps-attracting-basin>`_:

* ``adata.X`` -- the states, one row per observed time point of one ensemble
  member (``n_obs x d``).
* ``adata.obs[seq_key]`` -- the sequence (ensemble member) each row belongs to.
* ``adata.obs[time_key]`` -- the order of the row within its sequence.
  Optional; if absent, the order of appearance in ``adata`` is used.
* ``adata.obs[h_key]`` -- the evaluation function :math:`h`, defined at the
  terminal time of each sequence and ``NaN`` elsewhere.
* ``adata.uns["cost_matrix"]`` -- optional precomputed cost matrix
  :math:`c(x, y)`; otherwise built from ``adata.X``.
* ``adata.obsm[plot_key]`` -- optional 2-D coordinates for plotting.

Given an ensemble :math:`\{(y^{(i)}_1, \dots, y^{(i)}_{n_i})\}_{i=1}^{I}`, the
dynamical system is the shift along each sequence,
:math:`f(y^{(i)}_t) = y^{(i)}_{t+1}`, so
:math:`\operatorname{dom} f` is every non-terminal point and
:math:`\operatorname{dom} h` every terminal point.  This realizes
:math:`X = \operatorname{dom} f \sqcup \operatorname{dom} h =
\bigsqcup_{n \ge 0} f^{-n}(\operatorname{dom} h)`, the hypothesis of
Theorems 4.1 and 4.2.

The default cost is Euclidean *within a time layer* and :math:`\infty` across
layers: a control input may switch between ensemble members observed at the
same time, which is the cost function used in Figures 2 and 3 of the paper.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from ._core import ControlProblem

__all__ = [
    "from_ensemble",
    "build_problem",
    "control_effect",
    "effect_difference",
    "control_cost",
    "reachable_range",
    "reachable_domain",
    "minimizing_path",
    "check_hypotheses",
]

DEFAULT_SEQ_KEY = "seq_id"
DEFAULT_TIME_KEY = "time"
DEFAULT_H_KEY = "h"
COST_MATRIX_KEY = "cost_matrix"
PROBLEM_KEY = "epscontrol"


def _require_anndata():
    try:
        import anndata  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ImportError(
            "anndata is required for the AnnData interface; "
            "install it with `pip install anndata`."
        ) from exc
    return anndata


def _eps_tag(eps: float) -> str:
    """Filename-safe tag for a value of the control parameter."""
    eps = float(eps)
    sign = "-" if eps < 0 or (eps == 0 and np.signbit(eps)) else ""
    mag = abs(eps)
    body = "inf" if np.isinf(mag) else f"{mag:g}"
    return f"{sign}{body}"


def _p_tag(p: float) -> str:
    """Filename-safe tag for the norm parameter."""
    p = float(p)
    return "inf" if np.isinf(p) else f"{p:g}"


def result_key(eps: float, p: float, prefix: str = "control_effect") -> str:
    r"""Name under which a result for :math:`(\varepsilon, p)` is stored.

    ``result_key(1.0, np.inf)`` gives ``'control_effect_eps1_pinf'``, matching
    :math:`h_f^{1\text{-}\ell^\infty}`.
    """
    return f"{prefix}_eps{_eps_tag(eps)}_p{_p_tag(p)}"


# --------------------------------------------------------------- construction
def from_ensemble(
    sequences: Sequence[np.ndarray],
    values: Sequence[float] | np.ndarray | None = None,
    *,
    seq_names: Sequence[Any] | None = None,
    seq_key: str = DEFAULT_SEQ_KEY,
    time_key: str = DEFAULT_TIME_KEY,
    h_key: str = DEFAULT_H_KEY,
    obs: Mapping[str, Sequence[Any]] | pd.DataFrame | None = None,
    var_names: Sequence[str] | None = None,
):
    r"""Build an :class:`anndata.AnnData` from an ensemble of time series.

    Parameters
    ----------
    sequences
        One array per ensemble member, of shape ``(n_i, d)``.  All members must
        share the feature dimension ``d`` but may differ in length.
    values
        The evaluation function at the terminal time of each member, i.e. the
        quality of the outcome that member reaches.  Length ``I``.  ``NaN`` is
        allowed and marks a member whose outcome is not evaluated.  When
        ``None``, the ``h`` column is created empty and must be filled before
        computing.
    seq_names
        Labels for the members; defaults to ``0, 1, ..., I - 1``.
    seq_key, time_key, h_key
        Column names written into ``.obs``.
    obs
        Extra per-observation columns, each of total length
        :math:`\sum_i n_i` in the same row order as the concatenated
        ``sequences``.
    var_names
        Names of the ``d`` features.

    Returns
    -------
    anndata.AnnData
        With ``n_obs = sum(len(s) for s in sequences)`` rows ordered member by
        member and, within a member, by increasing time.

    Examples
    --------
    >>> import numpy as np, epscontrol as ec
    >>> a = np.array([[0.0, 0.0], [1.0, 0.0]])
    >>> b = np.array([[0.0, 1.0], [1.0, 1.0]])
    >>> adata = ec.from_ensemble([a, b], values=[3.0, 1.0])
    >>> adata.obs["seq_id"].tolist()
    [0, 0, 1, 1]
    >>> adata.obs["time"].tolist()
    [0, 1, 0, 1]
    >>> adata.obs["h"].tolist()
    [nan, 3.0, nan, 1.0]
    """
    anndata = _require_anndata()
    if len(sequences) == 0:
        raise ValueError("sequences must contain at least one time series.")
    arrays = [np.atleast_2d(np.asarray(s, dtype=np.float64)) for s in sequences]
    dims = {a.shape[1] for a in arrays}
    if len(dims) != 1:
        raise ValueError(f"all sequences must share the feature dimension; got {dims}")
    lengths = [a.shape[0] for a in arrays]
    if min(lengths) < 1:
        raise ValueError("every sequence must contain at least one time point.")

    if seq_names is None:
        seq_names = list(range(len(arrays)))
    elif len(seq_names) != len(arrays):
        raise ValueError("seq_names must have one entry per sequence.")

    X = np.vstack(arrays)
    seq_col = np.concatenate([np.full(n, s) for s, n in zip(seq_names, lengths)])
    time_col = np.concatenate([np.arange(n, dtype=np.int64) for n in lengths])

    h_col = np.full(X.shape[0], np.nan)
    if values is not None:
        values = np.asarray(values, dtype=np.float64).ravel()
        if values.size != len(arrays):
            raise ValueError(
                f"values must have one entry per sequence: expected "
                f"{len(arrays)}, got {values.size}"
            )
        ends = np.cumsum(lengths) - 1
        h_col[ends] = values

    obs_df = pd.DataFrame(
        {seq_key: seq_col, time_key: time_col, h_key: h_col},
        index=pd.RangeIndex(X.shape[0]).astype(str),
    )
    if obs is not None:
        extra = pd.DataFrame(dict(obs))
        if len(extra) != X.shape[0]:
            raise ValueError(
                f"obs columns must have length {X.shape[0]}, got {len(extra)}"
            )
        extra.index = obs_df.index
        obs_df = pd.concat([obs_df, extra], axis=1)

    var = None
    if var_names is not None:
        var = pd.DataFrame(index=pd.Index(list(var_names), name=None))
    return anndata.AnnData(X=X, obs=obs_df, var=var)


def _order_within_sequences(
    obs: pd.DataFrame, seq_key: str, time_key: str | None
) -> tuple[np.ndarray, np.ndarray]:
    """Return the successor array and the time layer of each observation."""
    n = obs.shape[0]
    seq = obs[seq_key].to_numpy()
    if time_key is not None and time_key in obs.columns:
        time = obs[time_key].to_numpy()
        if not np.issubdtype(np.asarray(time).dtype, np.number):
            time = pd.factorize(pd.Series(time).sort_values().index)[0]
            time = obs[time_key].rank(method="dense").to_numpy() - 1
        layer = pd.Series(time).rank(method="dense").to_numpy().astype(np.int64) - 1
        order_within = np.asarray(time, dtype=np.float64)
    else:
        layer = np.empty(n, dtype=np.int64)
        order_within = np.empty(n, dtype=np.float64)
        for s in pd.unique(seq):
            idx = np.nonzero(seq == s)[0]
            layer[idx] = np.arange(idx.size)
            order_within[idx] = np.arange(idx.size)

    succ = np.full(n, -1, dtype=np.int64)
    for s in pd.unique(seq):
        idx = np.nonzero(seq == s)[0]
        idx = idx[np.argsort(order_within[idx], kind="stable")]
        if np.unique(order_within[idx]).size != idx.size:
            raise ValueError(
                f"sequence {s!r} has duplicate {time_key!r} values; the time "
                "column must order the points of a sequence uniquely."
            )
        succ[idx[:-1]] = idx[1:]
    return succ, layer


def _build_cost(
    X: np.ndarray,
    layer: np.ndarray,
    cost: str | Callable[[np.ndarray], np.ndarray],
    metric: str,
    time_aligned: bool,
) -> np.ndarray:
    if callable(cost):
        C = np.array(cost(X), dtype=np.float64)
    elif cost == "distance":
        from sklearn.metrics import pairwise_distances

        C = pairwise_distances(X, metric=metric).astype(np.float64)
    else:
        raise ValueError(
            f"cost must be 'distance' or a callable X -> cost matrix, got {cost!r}"
        )
    if C.shape != (X.shape[0], X.shape[0]):
        raise ValueError(
            f"cost matrix must have shape {(X.shape[0], X.shape[0])}, got {C.shape}"
        )
    if time_aligned:
        C = np.where(layer[:, None] == layer[None, :], C, np.inf)
    np.fill_diagonal(C, 0.0)
    return C


def build_problem(
    adata,
    *,
    p: float = np.inf,
    seq_key: str = DEFAULT_SEQ_KEY,
    time_key: str | None = DEFAULT_TIME_KEY,
    h_key: str = DEFAULT_H_KEY,
    cost: str | Callable[[np.ndarray], np.ndarray] = "distance",
    cost_matrix_key: str = COST_MATRIX_KEY,
    metric: str = "euclidean",
    time_aligned: bool = True,
    store_cost: bool = True,
    validate: bool = True,
) -> ControlProblem:
    r"""Assemble a :class:`~epscontrol.ControlProblem` from an AnnData object.

    Parameters
    ----------
    adata
        Annotated data matrix following the convention documented in this
        module.
    p
        Norm parameter :math:`p \in [1, \infty]`.  ``np.inf`` bounds each
        individual control input, ``1`` bounds their sum.
    seq_key, time_key, h_key
        Columns of ``adata.obs`` holding the sequence label, the within-sequence
        order, and the evaluation function.  ``time_key=None`` uses the row
        order of each sequence.
    cost
        ``'distance'`` builds the cost from ``adata.X`` with ``metric``; a
        callable receives ``adata.X`` and must return an ``n x n`` matrix.
        Ignored when ``adata.uns[cost_matrix_key]`` already exists.
    cost_matrix_key
        Key in ``adata.uns`` for a precomputed cost matrix; the built matrix is
        written back there when ``store_cost`` is true.
    metric
        Metric passed to :func:`sklearn.metrics.pairwise_distances`.
    time_aligned
        Forbid control jumps between different time layers by setting those
        costs to :math:`\infty`.  This keeps the transition graph a DAG and
        enables the fast layered solver.
    store_cost
        Write the constructed cost matrix into ``adata.uns[cost_matrix_key]``.
    validate
        Passed to :class:`~epscontrol.ControlProblem`.

    Returns
    -------
    ControlProblem
        Row order of ``adata`` is preserved, so every returned array indexes
        ``adata.obs`` directly.
    """
    for key in (seq_key, h_key):
        if key not in adata.obs.columns:
            raise KeyError(f"{key!r} not found in adata.obs columns")
    if time_key is not None and time_key not in adata.obs.columns:
        time_key = None

    succ, layer = _order_within_sequences(adata.obs, seq_key, time_key)
    X = np.asarray(
        adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X, dtype=np.float64
    )
    h = adata.obs[h_key].to_numpy(dtype=np.float64)

    if cost_matrix_key in adata.uns:
        C = np.array(adata.uns[cost_matrix_key], dtype=np.float64)
        if C.shape != (X.shape[0], X.shape[0]):
            raise ValueError(
                f"adata.uns[{cost_matrix_key!r}] has shape {C.shape}, expected "
                f"{(X.shape[0], X.shape[0])}"
            )
        np.fill_diagonal(C, 0.0)
    else:
        C = _build_cost(X, layer, cost, metric, time_aligned)
        if store_cost:
            adata.uns[cost_matrix_key] = C

    if not np.isfinite(h[succ < 0]).any():
        raise ValueError(
            f"no terminal point carries a value of {h_key!r}; the evaluation "
            "function must be defined at the end of at least one sequence."
        )
    return ControlProblem(
        succ=succ, cost=C, h=h, layer=layer, p=p, validate=validate
    )


def _get_problem(adata, problem, kwargs) -> ControlProblem:
    if problem is not None:
        if problem.n != adata.n_obs:
            raise ValueError(
                f"problem has {problem.n} points but adata has {adata.n_obs} "
                "observations."
            )
        return problem
    return build_problem(adata, **kwargs)


# ------------------------------------------------------------------- analyses
def control_effect(
    adata,
    eps: float | Sequence[float],
    *,
    p: float = np.inf,
    key_added: str | None = None,
    problem: ControlProblem | None = None,
    copy: bool = False,
    **kwargs,
):
    r"""Compute the control effect function and store it in ``adata.obs``.

    For each requested :math:`\varepsilon` this evaluates
    :math:`h_f(p, \varepsilon, \cdot)` (Definition 18) at every observation:
    the best value of :math:`h` reachable under a control budget of
    :math:`\varepsilon` in the :math:`\ell^p` norm for :math:`\varepsilon \ge 0`,
    and the worst such value for :math:`\varepsilon < 0`.

    Parameters
    ----------
    adata
        Annotated data matrix.
    eps
        One control budget or several.  Negative values (including ``-0.0``)
        select the :math:`-\varepsilon`-control effect function of
        Definition 14.
    p
        Norm parameter.
    key_added
        Column name in ``adata.obs``.  Defaults to
        ``'control_effect_eps{eps}_p{p}'`` per value of ``eps``.
    problem
        A prebuilt :class:`~epscontrol.ControlProblem` to reuse; skips
        reconstruction and re-solving.
    copy
        Return a modified copy instead of writing in place.
    **kwargs
        Forwarded to :func:`build_problem` when ``problem`` is ``None``.

    Returns
    -------
    anndata.AnnData or None
        The modified object when ``copy`` is true, otherwise ``None``.

    Notes
    -----
    Values are exact, not swept on an :math:`\varepsilon` grid: the underlying
    solver computes the minimal control magnitude reaching each level of
    :math:`h` by shortest paths.  Points from which
    :math:`\operatorname{dom} h` cannot be reached receive ``np.inf``.
    """
    adata = adata.copy() if copy else adata
    prob = _get_problem(adata, problem, dict(kwargs, p=p))
    eps_list = np.atleast_1d(np.asarray(eps, dtype=np.float64))
    keys = []
    for e in eps_list:
        name = (
            result_key(e, prob.p)
            if key_added is None or eps_list.size > 1
            else key_added
        )
        adata.obs[name] = prob.control_effect(e)
        keys.append(name)
    adata.uns.setdefault(PROBLEM_KEY, {})
    adata.uns[PROBLEM_KEY]["control_effect"] = {
        "keys": keys,
        "eps": eps_list.tolist(),
        "p": float(prob.p),
        "solver": prob.solver,
    }
    return adata if copy else None


def effect_difference(
    adata,
    eps: float | Sequence[float],
    *,
    p: float = np.inf,
    key_added: str | None = None,
    problem: ControlProblem | None = None,
    copy: bool = False,
    **kwargs,
):
    r"""Compute the effect function of the difference and store it in ``adata.obs``.

    :math:`h^{\Delta}_f(p, \varepsilon, \cdot) = h_f(p, \varepsilon, \cdot)
    - h_f(p, 0, \cdot)` (Definitions 15 and 16): how much the attainable
    outcome improves relative to no control.  Non-positive for
    :math:`\varepsilon \ge 0`, so a large negative value marks a state where
    control pays off most.

    Parameters
    ----------
    adata, eps, p, key_added, problem, copy, **kwargs
        As in :func:`control_effect`; the default column name is
        ``'effect_difference_eps{eps}_p{p}'``.
    """
    adata = adata.copy() if copy else adata
    prob = _get_problem(adata, problem, dict(kwargs, p=p))
    eps_list = np.atleast_1d(np.asarray(eps, dtype=np.float64))
    keys = []
    for e in eps_list:
        name = (
            result_key(e, prob.p, prefix="effect_difference")
            if key_added is None or eps_list.size > 1
            else key_added
        )
        adata.obs[name] = prob.effect_difference(e)
        keys.append(name)
    adata.uns.setdefault(PROBLEM_KEY, {})
    adata.uns[PROBLEM_KEY]["effect_difference"] = {
        "keys": keys,
        "eps": eps_list.tolist(),
        "p": float(prob.p),
    }
    return adata if copy else None


def control_cost(
    adata,
    *,
    p: float = np.inf,
    key_added: str = "control_cost",
    problem: ControlProblem | None = None,
    copy: bool = False,
    **kwargs,
):
    r"""Store the :math:`h`-sublevel controllability values in ``adata.obsm``.

    Writes an ``n_obs x len(levels)`` array whose ``[i, k]`` entry is
    :math:`C^{\ell^p}_{h, f}(x_i, r_k)`, the minimal control magnitude that
    drives :math:`x_i` into :math:`h^{-1}([-\infty, r_k])` (§ 2.2.2).  The
    levels :math:`r_k = \operatorname{Im} h` are stored in
    ``adata.uns['epscontrol']['levels']``.
    """
    adata = adata.copy() if copy else adata
    prob = _get_problem(adata, problem, dict(kwargs, p=p))
    adata.obsm[key_added] = prob.control_cost()
    adata.uns.setdefault(PROBLEM_KEY, {})
    adata.uns[PROBLEM_KEY]["levels"] = prob.levels
    adata.uns[PROBLEM_KEY]["control_cost"] = {"key": key_added, "p": float(prob.p)}
    return adata if copy else None


def reachable_range(
    adata,
    x: int | str,
    eps: float,
    *,
    p: float = np.inf,
    problem: ControlProblem | None = None,
    **kwargs,
) -> np.ndarray:
    r"""The reachable range :math:`R^{\ell^p}_{h, f}(\varepsilon, x)` (§ 2.2.2).

    Parameters
    ----------
    x
        Positional index or ``adata.obs_names`` label of the initial state.
    eps
        Control budget; its magnitude is used.

    Returns
    -------
    numpy.ndarray
        Ascending values of :math:`h` attainable from ``x`` under
        :math:`\varepsilon`-:math:`\ell^p` control.
    """
    prob = _get_problem(adata, problem, dict(kwargs, p=p))
    return prob.reachable_range(_resolve_obs(adata, x), eps)


def reachable_domain(
    adata,
    eps: float,
    r: float,
    *,
    p: float = np.inf,
    problem: ControlProblem | None = None,
    **kwargs,
) -> np.ndarray:
    r"""The reachable domain :math:`D^{\ell^p}_{h, f}(\varepsilon, r)`
    as a boolean mask over ``adata.obs`` (§ 2.2.2)."""
    prob = _get_problem(adata, problem, dict(kwargs, p=p))
    return prob.reachable_domain(eps, r)


def minimizing_path(
    adata,
    x: int | str,
    eps: float,
    *,
    p: float = np.inf,
    problem: ControlProblem | None = None,
    **kwargs,
) -> dict:
    r"""A minimizing :math:`\varepsilon`-:math:`\ell^p`-controlled path from ``x``.

    Thin wrapper around :meth:`ControlProblem.minimizing_path` that resolves
    ``x`` against ``adata.obs_names`` and adds the observation labels of the
    visited states.

    Returns
    -------
    dict
        The keys of :meth:`ControlProblem.minimizing_path` plus
        ``'obs_names'`` (labels of ``states``) and ``'jump_obs_names'``.
    """
    prob = _get_problem(adata, problem, dict(kwargs, p=p))
    out = prob.minimizing_path(_resolve_obs(adata, x), eps)
    names = np.asarray(adata.obs_names)
    out["obs_names"] = names[out["states"]]
    out["jump_obs_names"] = names[out["jumps"]]
    return out


def check_hypotheses(
    adata,
    *,
    eps: float = 0.0,
    p: float = np.inf,
    problem: ControlProblem | None = None,
    **kwargs,
) -> dict[str, bool]:
    r"""Report the paper's structural hypotheses for the assembled problem.

    See :meth:`ControlProblem.hypotheses`.  All four flags are true for an
    ensemble built by :func:`from_ensemble` whose members each carry a terminal
    evaluation, with the default time-aligned cost.
    """
    prob = _get_problem(adata, problem, dict(kwargs, p=p))
    return prob.hypotheses(eps)


def _resolve_obs(adata, x: int | str) -> int:
    """Positional index of an observation given a label or an index."""
    if isinstance(x, (int, np.integer)):
        if not -adata.n_obs <= int(x) < adata.n_obs:
            raise IndexError(f"observation index {x} out of range for {adata.n_obs}")
        return int(x) % adata.n_obs
    loc = adata.obs_names.get_indexer([x])[0]
    if loc < 0:
        raise KeyError(f"{x!r} not found in adata.obs_names")
    return int(loc)
