r"""Three-parameter filtrations associated with a control effect function.

Section 4.1 of Imoto--Yokoyama associates with a control effect function the
family of sublevel sets

.. math::

    H^{\ell^p}_{f, \varepsilon, \delta}
        = \{\, x \in X \mid h_f(p, \varepsilon, x) \le \delta \,\}, \qquad
    H^{\ell^p}_{f, \Delta\varepsilon, \delta}
        = \{\, x \in X \mid h^{\Delta}_f(p, \varepsilon, x) \le \delta \,\},

and Theorems 4.3 and 4.4 state that both are (multi-parameter) filtrations:
monotone in :math:`\delta` by construction, monotone in :math:`\varepsilon`
because :math:`h_f` is weakly decreasing in its second argument
(Proposition 3.10), and monotone in :math:`p` on the branch
:math:`\varepsilon \ge 0` by the monotonicity of the :math:`\ell^p` norms
(Lemma 3.9).

The three axes are exactly the three factors the paper sets out to relate: the
cost norm :math:`p`, the strength of control :math:`\varepsilon`, and the
resulting value :math:`\delta`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from ._core import ControlProblem

__all__ = ["Filtration", "filtration"]


@dataclass
class Filtration:
    r"""A three-parameter filtration indexed by :math:`(p, \varepsilon, \delta)`.

    Attributes
    ----------
    masks
        Boolean array of shape ``(len(ps), len(epsilons), len(deltas), n)``;
        ``masks[a, b, c]`` is the indicator of
        :math:`H^{\ell^{p_a}}_{f, \varepsilon_b, \delta_c}`.
    ps, epsilons, deltas
        The parameter values, ascending.
    kind
        ``'value'`` for :math:`H^{\ell^p}_{f, \varepsilon, \delta}`,
        ``'difference'`` for :math:`H^{\ell^p}_{f, \Delta\varepsilon, \delta}`.
    values
        The effect function itself, of shape
        ``(len(ps), len(epsilons), n)``.
    """

    masks: np.ndarray
    ps: np.ndarray
    epsilons: np.ndarray
    deltas: np.ndarray
    kind: Literal["value", "difference"]
    values: np.ndarray

    # ------------------------------------------------------------------ access
    def mask(self, p: float, eps: float, delta: float) -> np.ndarray:
        """Indicator of the filtration member at the nearest stored parameters."""
        a = int(np.argmin(np.abs(self.ps - p)))
        b = int(np.argmin(np.abs(self.epsilons - eps)))
        c = int(np.argmin(np.abs(self.deltas - delta)))
        return self.masks[a, b, c]

    @property
    def sizes(self) -> np.ndarray:
        r"""Cardinalities :math:`\lvert H^{\ell^p}_{f, \varepsilon, \delta}
        \rvert`, of shape ``(len(ps), len(epsilons), len(deltas))``."""
        return self.masks.sum(axis=-1)

    def to_frame(self) -> pd.DataFrame:
        """Long-form table with one row per parameter triple."""
        P, E, D = np.meshgrid(
            self.ps, self.epsilons, self.deltas, indexing="ij"
        )
        return pd.DataFrame(
            {
                "p": P.ravel(),
                "eps": E.ravel(),
                "delta": D.ravel(),
                "size": self.sizes.ravel(),
            }
        )

    # -------------------------------------------------------------- validation
    def is_filtration(self) -> bool:
        r"""Verify monotonicity in :math:`\delta` and in :math:`\varepsilon`.

        Corresponds to assertion (1) of Theorems 4.3/4.4, checked numerically on
        the stored parameter grid.  Monotonicity in :math:`p` (assertion (2))
        holds on the branch :math:`\varepsilon \ge 0`; use
        :meth:`is_filtration_in_p` for that axis.
        """
        return bool(
            np.all(np.diff(self.masks.astype(np.int8), axis=2) >= 0)
            and np.all(np.diff(self.masks.astype(np.int8), axis=1) >= 0)
        )

    def is_filtration_in_p(self) -> bool:
        r"""Verify monotonicity along the :math:`p` axis for
        :math:`\varepsilon \ge 0` (assertion (2) of Theorems 4.3/4.4)."""
        keep = self.epsilons >= 0
        if not keep.any():
            return True
        sub = self.masks[:, keep].astype(np.int8)
        return bool(np.all(np.diff(sub, axis=0) >= 0))

    # -------------------------------------------------------------- topology
    def betti0(self, problem: ControlProblem) -> np.ndarray:
        r"""Zeroth Betti numbers of the filtration members.

        Each member is given the graph structure it inherits from the dynamics:
        vertices are its points and edges join :math:`x` to :math:`f(x)`
        whenever both lie in the member.  The returned array of shape
        ``(len(ps), len(epsilons), len(deltas))`` counts connected components,
        i.e. the number of distinct controllable branches captured at each
        parameter triple.
        """
        succ = problem.succ
        has = succ >= 0
        src = np.nonzero(has)[0]
        dst = succ[src]
        out = np.zeros(self.sizes.shape, dtype=np.int64)
        for a in range(self.masks.shape[0]):
            for b in range(self.masks.shape[1]):
                for c in range(self.masks.shape[2]):
                    m = self.masks[a, b, c]
                    keep = m[src] & m[dst]
                    out[a, b, c] = _count_components(
                        int(m.sum()), np.nonzero(m)[0], src[keep], dst[keep]
                    )
        return out


def _count_components(
    n_vertices: int, vertices: np.ndarray, src: np.ndarray, dst: np.ndarray
) -> int:
    """Number of connected components of a subgraph, by union-find."""
    if n_vertices == 0:
        return 0
    pos = {int(v): i for i, v in enumerate(vertices)}
    parent = list(range(n_vertices))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    count = n_vertices
    for u, v in zip(src, dst):
        ru, rv = find(pos[int(u)]), find(pos[int(v)])
        if ru != rv:
            parent[ru] = rv
            count -= 1
    return count


def filtration(
    problem: ControlProblem,
    *,
    ps: Sequence[float] = (1.0, 2.0, np.inf),
    epsilons: Sequence[float] | None = None,
    deltas: Sequence[float] | None = None,
    kind: Literal["value", "difference"] = "value",
    n_epsilons: int = 11,
) -> Filtration:
    r"""Build the three-parameter filtration of a control problem.

    Parameters
    ----------
    problem
        The control problem; its own ``p`` is ignored in favour of ``ps``.
    ps
        Norm parameters, ascending.
    epsilons
        Control budgets, ascending.  Defaults to ``n_epsilons`` values from
        :math:`0` to the largest finite control cost occurring in the problem,
        which is the point beyond which nothing new becomes reachable.
    deltas
        Value thresholds, ascending.  Defaults to :math:`\operatorname{Im} h`
        for ``kind='value'``, and to the distinct attained differences for
        ``kind='difference'``.
    kind
        ``'value'`` filters by :math:`h_f`, ``'difference'`` by
        :math:`h^{\Delta}_f`.
    n_epsilons
        Number of default control budgets.

    Returns
    -------
    Filtration

    Examples
    --------
    >>> import numpy as np, epscontrol as ec
    >>> adata = ec.from_ensemble(
    ...     [np.array([[0.0, 0.0], [1.0, 0.0]]),
    ...      np.array([[0.0, 1.0], [1.0, 1.0]])], values=[3.0, 1.0])
    >>> prob = ec.build_problem(adata)
    >>> filt = ec.filtration(prob, ps=(1.0, np.inf), epsilons=(0.0, 1.0))
    >>> filt.is_filtration() and filt.is_filtration_in_p()
    True
    """
    ps = np.asarray(ps, dtype=np.float64).ravel()
    if ps.size and np.any(np.diff(ps) < 0):
        raise ValueError("ps must be ascending.")
    if np.any(ps < 1):
        raise ValueError("ps must lie in [1, inf].")

    probs = [problem if problem.p == p else problem.with_p(p) for p in ps]

    if epsilons is None:
        finite = np.concatenate(
            [pr.control_cost_exact()[np.isfinite(pr.control_cost_exact())] for pr in probs]
        )
        top = float(finite.max()) if finite.size else 1.0
        epsilons = np.linspace(0.0, top if top > 0 else 1.0, int(n_epsilons))
    epsilons = np.asarray(epsilons, dtype=np.float64).ravel()
    if epsilons.size and np.any(np.diff(epsilons) < 0):
        raise ValueError("epsilons must be ascending.")

    values = np.stack(
        [
            np.stack(
                [
                    pr.effect_difference(e) if kind == "difference" else pr.control_effect(e)
                    for e in epsilons
                ]
            )
            for pr in probs
        ]
    )

    if deltas is None:
        finite = values[np.isfinite(values)]
        deltas = np.unique(finite) if finite.size else np.array([0.0])
    deltas = np.asarray(deltas, dtype=np.float64).ravel()
    if deltas.size and np.any(np.diff(deltas) < 0):
        raise ValueError("deltas must be ascending.")

    masks = values[:, :, None, :] <= deltas[None, None, :, None]
    return Filtration(
        masks=masks,
        ps=ps,
        epsilons=epsilons,
        deltas=deltas,
        kind=kind,
        values=values,
    )
