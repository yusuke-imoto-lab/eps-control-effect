r"""Core solvers for control effect functions on a finite state space.

Implements the constructions of

    Y. Imoto and T. Yokoyama, *Multi-parameter persistence in dynamical
    systems for maximizing effects of control inputs*.

Setting
-------
A finite set :math:`X` carries

* a partial map (dynamical system) :math:`f : X \rightharpoonup X`,
* a cost function :math:`c : X^2 \to [0, \infty]` with :math:`c(x, x) = 0`
  (Definition 4),
* a partial evaluation function :math:`h : X \rightharpoonup [-\infty, \infty]`,
* a norm parameter :math:`p \in [1, \infty]`.

An :math:`\varepsilon`-:math:`\ell^p`-controlled path of length :math:`n` from
the initial state :math:`x` to the terminal state :math:`f(x_{n-1})` is a
sequence :math:`(x; x_0, \dots, x_{n-1}; f(x_{n-1}))` with
:math:`x_i \in \operatorname{dom} f`,

.. math::

    \varepsilon_0 = c(x, x_0), \qquad
    \varepsilon_i = c(f(x_{i-1}), x_i), \qquad
    \varepsilon \ge \bigl\lVert (\varepsilon_i)_{i=0}^{n-1} \bigr\rVert_p

(Definition 7).  Each step is one control jump followed by one application of
the dynamics.

Computation
-----------
Everything is derived from the array of *exact-level control costs*

.. math::

    C_{\mathrm{exact}}(x, r) = \inf \Bigl\{\, \varepsilon \;\Bigm|\;
        x \overset{\exists}{\rightharpoonup}_{f,\varepsilon\text{-}\ell^p}
        h^{-1}(r) \,\Bigr\}, \qquad r \in \operatorname{Im} h ,

which is finite in number because :math:`X` is finite.  From it,

.. math::

    h_f^{\varepsilon\text{-}\ell^p}(x)
      = \min \{\, r \mid C_{\mathrm{exact}}(x, r) \le \varepsilon \,\},
    \qquad
    h_f^{-\varepsilon\text{-}\ell^p}(x)
      = \max \{\, r \mid C_{\mathrm{exact}}(x, r) \le \varepsilon \,\}

(Definitions 13 and 14), the reachable range is
:math:`R^{\ell^p}_{h,f}(\varepsilon, x) = \{\, r \mid
C_{\mathrm{exact}}(x, r) \le \varepsilon \,\}` (§ 2.2.2), and the
:math:`h`-sublevel controllability value is the running minimum
:math:`C^{\ell^p}_{h,f}(x, r) = \min_{r' \le r}
C_{\mathrm{exact}}(x, r')`.  Because the exact costs are computed as
shortest-path values rather than swept on an :math:`\varepsilon` grid, all
returned values are exact; the recursion of Theorem 4.1/4.2 is the same
statement read level by level.

Two solvers compute :math:`C_{\mathrm{exact}}`:

``layered``
    Used when every admissible control jump stays inside one time layer, so the
    transition graph is a DAG graded by time.  This holds for ensemble
    (multiple time-series) input with the default time-aligned cost, and for
    the setting of Figures 2 and 3 of the paper.  Exact backward dynamic
    programming, vectorized over levels.

``dijkstra``
    General fallback for cost matrices that connect different time layers.
    Min-plus (finite :math:`p`) or min-max (:math:`p = \infty`) Dijkstra on the
    reversed transition graph.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

__all__ = ["ControlProblem"]

# Cap on the temporary array used by the layered solver, in elements.
_MAX_CHUNK_ELEMENTS = 20_000_000


def _aggregate(head: np.ndarray, tail: np.ndarray, p: float) -> np.ndarray:
    """Combine a first-jump weight with the weight of the remaining path."""
    if np.isinf(p):
        return np.maximum(head, tail)
    return head + tail


@dataclass
class ControlProblem:
    r"""A finite control problem :math:`(X, f, c, h, p)`.

    Parameters
    ----------
    succ
        Integer array of shape ``(n,)``.  ``succ[i]`` is the index of
        :math:`f(x_i)`, or ``-1`` when
        :math:`x_i \notin \operatorname{dom} f`.
    cost
        Float array of shape ``(n, n)``: ``cost[i, j]`` is :math:`c(x_i, x_j)`,
        with ``np.inf`` marking a forbidden jump.  The diagonal must vanish.
    h
        Float array of shape ``(n,)``.  ``np.nan`` marks points outside
        :math:`\operatorname{dom} h`.
    layer
        Optional integer array of shape ``(n,)`` giving the time layer of each
        point.  When present and consistent with ``cost``, the layered solver
        is used.
    p
        Norm parameter :math:`p \in [1, \infty]`.  ``np.inf`` reproduces the
        :math:`\varepsilon`-controlled paths of Imoto--Yokoyama, *Filtrations
        indexed by attracting levels and their applications*, Chaos **36**
        (2026), no. 5, 053144, and ``p = 1`` their
        :math:`\varepsilon_\Sigma`-controlled paths (Remark 1).
    validate
        Check array shapes and the cost-function axioms on construction.  The
        paper's structural hypotheses are reported separately by
        :meth:`hypotheses` and are not enforced.

    Examples
    --------
    Two length-2 sequences that branch, with :math:`h` on their endpoints:

    >>> import numpy as np
    >>> succ = np.array([1, -1, 3, -1])
    >>> cost = np.array([[0., np.inf, 1., np.inf],
    ...                  [np.inf, 0., np.inf, 1.],
    ...                  [1., np.inf, 0., np.inf],
    ...                  [np.inf, 1., np.inf, 0.]])
    >>> h = np.array([np.nan, 10.0, np.nan, 2.0])
    >>> prob = ControlProblem(succ, cost, h, layer=np.array([0, 1, 0, 1]))
    >>> float(prob.control_effect(0.0)[0])
    10.0
    >>> float(prob.control_effect(1.0)[0])
    2.0
    """

    succ: np.ndarray
    cost: np.ndarray
    h: np.ndarray
    layer: np.ndarray | None = None
    p: float = np.inf
    validate: bool = True
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    # ------------------------------------------------------------------ setup
    def __post_init__(self) -> None:
        self.succ = np.asarray(self.succ, dtype=np.int64).ravel()
        self.cost = np.array(self.cost, dtype=np.float64, copy=True)
        self.h = np.array(self.h, dtype=np.float64, copy=True).ravel()
        if self.layer is not None:
            self.layer = np.asarray(self.layer, dtype=np.int64).ravel()
        self.p = float(self.p)

        n = self.succ.size
        if self.cost.shape != (n, n):
            raise ValueError(f"cost must have shape ({n}, {n}), got {self.cost.shape}")
        if self.h.size != n:
            raise ValueError(f"h must have length {n}, got {self.h.size}")
        if self.layer is not None and self.layer.size != n:
            raise ValueError(f"layer must have length {n}, got {self.layer.size}")
        if not self.p >= 1.0:
            raise ValueError(f"p must lie in [1, inf], got {self.p}")

        if self.validate:
            self._validate()

    def _validate(self) -> None:
        n = self.n
        if np.isnan(self.cost).any():
            raise ValueError(
                "cost must not contain NaN; use np.inf for a forbidden jump."
            )
        if np.any(self.cost < 0.0):
            raise ValueError(
                "cost must be non-negative-valued (Definition 6). The theory "
                "permits negative costs, but the shortest-path solvers used "
                "here require c >= 0."
            )
        diag = np.diag(self.cost)
        if not np.allclose(diag, 0.0):
            bad = int(np.argmax(np.abs(diag)))
            raise ValueError(
                "a cost function must satisfy c(x, x) = 0 (Definition 4); "
                f"cost[{bad}, {bad}] = {diag[bad]}"
            )
        if np.any((self.succ < -1) | (self.succ >= n)):
            raise ValueError("succ entries must lie in [-1, n).")
        if self.levels.size == 0:
            raise ValueError("h is nowhere defined; dom h must be non-empty.")

    # ------------------------------------------------------------- properties
    @property
    def n(self) -> int:
        """Number of points of :math:`X`."""
        return self.succ.size

    @property
    def dom_f(self) -> np.ndarray:
        r"""Boolean mask of :math:`\operatorname{dom} f`."""
        return self.succ >= 0

    @property
    def dom_h(self) -> np.ndarray:
        r"""Boolean mask of :math:`\operatorname{dom} h`."""
        return ~np.isnan(self.h)

    @property
    def levels(self) -> np.ndarray:
        r"""Sorted distinct values of :math:`\operatorname{Im} h`."""
        if "levels" not in self._cache:
            self._cache["levels"] = np.unique(self.h[self.dom_h])
        return self._cache["levels"]

    @property
    def solver(self) -> Literal["layered", "dijkstra"]:
        """Which solver this problem uses."""
        return "layered" if self._is_layered() else "dijkstra"

    def _is_layered(self) -> bool:
        if "is_layered" in self._cache:
            return self._cache["is_layered"]
        ok = False
        if self.layer is not None:
            jump = np.isfinite(self.cost) & self.dom_f[None, :]
            src, mid = np.nonzero(jump)
            ok = bool(
                np.all(self.layer[src] == self.layer[mid])
                and np.all(self.layer[self.succ[mid]] > self.layer[mid])
            )
        self._cache["is_layered"] = ok
        return ok

    # ------------------------------------------------- structural hypotheses
    def fixed_points(self, eps: float) -> np.ndarray:
        r"""Boolean mask of the :math:`(c, \varepsilon)`-fixed points
        :math:`E_c(\varepsilon)` (Definition 11).

        A point :math:`x` is :math:`(c, \varepsilon)`-fixed when
        :math:`c(x, \cdot)^{-1}([-\infty, \varepsilon]) \setminus \{x\}
        = \emptyset`, i.e. no control of magnitude at most
        :math:`\varepsilon` moves it anywhere else.
        """
        eps = float(eps)
        reachable = self.cost <= eps
        np.fill_diagonal(reachable, False)
        return ~reachable.any(axis=1)

    def hypotheses(self, eps: float = 0.0) -> dict[str, bool]:
        r"""Report the paper's structural hypotheses for this problem.

        Returns a dict with the keys

        ``disjoint_domains``
            :math:`\operatorname{dom} f \cap \operatorname{dom} h = \emptyset`.
        ``covers_X``
            :math:`X = \operatorname{dom} f \sqcup \operatorname{dom} h`.
        ``preimage_decomposition``
            :math:`X = \bigsqcup_{n \ge 0} f^{-n}(\operatorname{dom} h)`, i.e.
            every orbit enters :math:`\operatorname{dom} h` in finite time.
        ``dom_h_fixed``
            :math:`\operatorname{dom} h \subseteq E_c(\varepsilon) \setminus
            \operatorname{dom} f`, the hypothesis of Theorem 3.12 and
            Theorems 4.1/4.2.

        The solvers do not require these; they are the conditions under which
        the theorems of the paper apply.
        """
        dom_f, dom_h = self.dom_f, self.dom_h
        reached = dom_h.copy()
        for _ in range(self.n):
            if reached.all():
                break
            nxt = np.where(dom_f, self.succ, 0)
            grown = reached | (dom_f & reached[nxt])
            if np.array_equal(grown, reached):
                break
            reached = grown
        return {
            "disjoint_domains": bool(not (dom_f & dom_h).any()),
            "covers_X": bool((dom_f | dom_h).all() and not (dom_f & dom_h).any()),
            "preimage_decomposition": bool(reached.all()),
            "dom_h_fixed": bool(
                np.all(self.fixed_points(eps)[dom_h]) and not (dom_f & dom_h).any()
            ),
        }

    # ------------------------------------------------------- weight semiring
    def _weights(self) -> np.ndarray:
        r"""Jump weights in the semiring used by the solver.

        For :math:`p = \infty` the weight is the cost itself and paths compose
        by ``max``; for finite :math:`p` the weight is :math:`c^p` and paths
        compose by ``+``, so that the accumulated weight is
        :math:`\lVert \cdot \rVert_p^p`.
        """
        if np.isinf(self.p):
            return self.cost
        with np.errstate(over="ignore"):
            return np.where(np.isinf(self.cost), np.inf, self.cost**self.p)

    def _to_epsilon(self, w: np.ndarray) -> np.ndarray:
        """Convert accumulated semiring weights back to control magnitudes."""
        if np.isinf(self.p):
            return w
        with np.errstate(invalid="ignore", over="ignore"):
            return np.where(np.isinf(w), np.inf, w ** (1.0 / self.p))

    def _to_weight(self, eps: np.ndarray | float) -> np.ndarray:
        """Inverse of :meth:`_to_epsilon`."""
        eps = np.asarray(eps, dtype=np.float64)
        if np.isinf(self.p):
            return eps
        with np.errstate(over="ignore"):
            return np.where(np.isinf(eps), np.inf, eps**self.p)

    # --------------------------------------------------------- control costs
    def control_cost_exact(self) -> np.ndarray:
        r"""Minimal control magnitude reaching each *exact* level of :math:`h`.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n, len(self.levels))`` whose ``[i, k]`` entry is
            the smallest :math:`\varepsilon` admitting an
            :math:`\varepsilon`-:math:`\ell^p`-controlled path from
            :math:`x_i` into :math:`h^{-1}(r_k)`, and ``np.inf`` when no such
            path exists.
        """
        if "exact" not in self._cache:
            targets = self.h[:, None] == self.levels[None, :]
            w = (
                self._solve_layered(targets)
                if self._is_layered()
                else self._solve_dijkstra(targets)
            )
            self._cache["exact"] = self._to_epsilon(w)
        return self._cache["exact"]

    def control_cost(self) -> np.ndarray:
        r"""The :math:`h`-sublevel controllability values
        :math:`C^{\ell^p}_{h, f}(x, r)` (§ 2.2.2).

        Returns an array of shape ``(n, len(self.levels))`` whose ``[i, k]``
        entry is the smallest :math:`\varepsilon` admitting an
        :math:`\varepsilon`-:math:`\ell^p`-controlled path from :math:`x_i`
        into :math:`h^{-1}([-\infty, r_k])`.  Non-increasing along the level
        axis, in agreement with Lemma 2.5.
        """
        if "sublevel" not in self._cache:
            self._cache["sublevel"] = np.minimum.accumulate(
                self.control_cost_exact(), axis=1
            )
        return self._cache["sublevel"]

    def superlevel_control_cost(self) -> np.ndarray:
        r"""Minimal control magnitude reaching :math:`h^{-1}([r, \infty])`.

        The counterpart of :meth:`control_cost` used by the
        :math:`-\varepsilon`-control effect function; non-decreasing along the
        level axis.
        """
        if "superlevel" not in self._cache:
            self._cache["superlevel"] = np.minimum.accumulate(
                self.control_cost_exact()[:, ::-1], axis=1
            )[:, ::-1]
        return self._cache["superlevel"]

    # ------------------------------------------------------- solver: layered
    def _solve_layered(self, targets: np.ndarray) -> np.ndarray:
        """Backward dynamic programming on the time-graded transition DAG."""
        n, L = self.n, targets.shape[1]
        weights = self._weights()
        W = np.where(targets, 0.0, np.inf)
        if L == 0:
            return W
        idx_all = np.arange(n)
        for lay in np.unique(self.layer)[::-1]:
            in_layer = self.layer == lay
            A = idx_all[in_layer]
            B = idx_all[in_layer & self.dom_f]
            if A.size == 0 or B.size == 0:
                continue
            sub = weights[np.ix_(A, B)]
            movable = np.isfinite(sub).any(axis=1)
            if not movable.any():
                continue
            A, sub = A[movable], sub[movable]
            tail = W[self.succ[B], :]
            chunk = max(1, _MAX_CHUNK_ELEMENTS // max(1, A.size * B.size))
            for k0 in range(0, L, chunk):
                k1 = min(L, k0 + chunk)
                cand = _aggregate(sub[:, :, None], tail[None, :, k0:k1], self.p)
                W[A, k0:k1] = np.minimum(W[A, k0:k1], cand.min(axis=1))
        return W

    # ------------------------------------------------------ solver: dijkstra
    def _reverse_graph(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""CSR arrays of the reversed transition graph.

        The transition graph has an edge :math:`x \to f(x_0)` of weight
        :math:`w(c(x, x_0))` for every :math:`x_0 \in \operatorname{dom} f`
        with :math:`c(x, x_0) < \infty`.
        """
        if "reverse_graph" not in self._cache:
            weights = self._weights()
            mask = np.isfinite(weights) & self.dom_f[None, :]
            src, mid = np.nonzero(mask)
            dst = self.succ[mid]
            w = weights[src, mid]
            order = np.argsort(dst, kind="stable")
            dst, src, w = dst[order], src[order], w[order]
            indptr = np.zeros(self.n + 1, dtype=np.int64)
            np.add.at(indptr, dst + 1, 1)
            self._cache["reverse_graph"] = (
                np.cumsum(indptr),
                src.astype(np.int64),
                w,
            )
        return self._cache["reverse_graph"]

    def _solve_dijkstra(self, targets: np.ndarray) -> np.ndarray:
        indptr, indices, data = self._reverse_graph()
        bottleneck = bool(np.isinf(self.p))
        W = np.full((self.n, targets.shape[1]), np.inf)
        for k in range(targets.shape[1]):
            W[:, k] = _dijkstra_semiring(
                indptr, indices, data, np.nonzero(targets[:, k])[0], self.n, bottleneck
            )
        return W

    # ------------------------------------------------------ effect functions
    def control_effect(self, eps: float) -> np.ndarray:
        r"""The control effect function
        :math:`h_f(p, \varepsilon, \cdot)` (Definition 18).

        Parameters
        ----------
        eps
            A value of the parameter domain
            :math:`[-\infty, -0] \sqcup [0, \infty]`.  A non-negative
            ``eps`` returns the :math:`\varepsilon`-control effect function
            :math:`h_f^{\varepsilon\text{-}\ell^p} =
            \inf R^{\ell^p}_{h, f}(\varepsilon, \cdot)` (Definition 13); a
            negative ``eps`` -- including the signed zero ``-0.0`` -- returns
            the :math:`-\varepsilon`-control effect function
            :math:`h_f^{-\varepsilon\text{-}\ell^p} =
            \sup R^{\ell^p}_{h, f}(\varepsilon, \cdot)` (Definition 14).

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n,)``; ``np.inf`` where
            :math:`\operatorname{dom} h` is unreachable.
        """
        eps = float(eps)
        negative = eps < 0 or (eps == 0 and bool(np.signbit(eps)))
        lv = self.levels
        if not negative:
            ok = self.control_cost() <= eps
            return np.where(ok.any(axis=1), lv[np.argmax(ok, axis=1)], np.inf)
        ok = self.superlevel_control_cost() <= abs(eps)
        rev = ok[:, ::-1]
        out = np.where(rev.any(axis=1), lv[::-1][np.argmax(rev, axis=1)], np.inf)
        # Definition 14 is infinite exactly where dom h is unreachable at eps = 0.
        return np.where(np.isfinite(self.control_effect(0.0)), out, np.inf)

    def effect_difference(self, eps: float) -> np.ndarray:
        r"""The effect function of the difference
        :math:`h_f^{\Delta}(p, \varepsilon, \cdot)` (Definitions 15 and 16).

        :math:`h_f^{\Delta\varepsilon\text{-}\ell^p} =
        h_f^{\varepsilon\text{-}\ell^p} - h_f^{0\text{-}\ell^p}`, using the
        conventions :math:`\infty - \infty := 0` and
        :math:`r - \infty := -\infty`.  Non-positive for ``eps >= 0`` and
        non-negative for ``eps < 0``.
        """
        base = self.control_effect(0.0)
        cur = self.control_effect(eps)
        out = np.empty(self.n, dtype=np.float64)
        both_inf = np.isinf(cur) & np.isinf(base)
        base_inf = np.isinf(base) & ~np.isinf(cur)
        rest = ~(both_inf | base_inf)
        out[both_inf] = 0.0
        out[base_inf] = -np.inf
        out[rest] = cur[rest] - base[rest]
        return out

    # ------------------------------------------------------- reachability API
    def reachable_range(self, x: int, eps: float) -> np.ndarray:
        r"""The :math:`(\varepsilon, h)`-reachable range
        :math:`R^{\ell^p}_{h, f}(\varepsilon, x)` (§ 2.2.2).

        Returns the ascending array of values of :math:`h` attained by terminal
        states of :math:`\varepsilon`-:math:`\ell^p`-controlled paths from
        :math:`x_x`, i.e.
        :math:`h([x]^{\varepsilon\text{-}\ell^p}_f \cap \operatorname{dom} h)`.
        """
        return self.levels[self.control_cost_exact()[int(x)] <= abs(float(eps))]

    def reachable_class(self, x: int, eps: float) -> np.ndarray:
        r"""Boolean mask of :math:`[x]^{\varepsilon\text{-}\ell^p}_f`
        (Definition 8): the points reachable from :math:`x_x` by an
        :math:`\varepsilon`-:math:`\ell^p`-controlled path."""
        eps = abs(float(eps))
        budget = self._to_weight(eps)
        weights = self._weights()
        n = self.n
        best = np.full(n, np.inf)
        heap: list[tuple[float, int]] = [(0.0, int(x))]
        seen = np.zeros(n, dtype=bool)
        combine = np.fmax if np.isinf(self.p) else np.add
        while heap:
            d, u = heapq.heappop(heap)
            if d > best[u]:
                continue
            row = np.where(self.dom_f, weights[u], np.inf)
            nd = combine(d, row)
            cand = np.nonzero(nd <= budget)[0]
            for j in cand:
                v = int(self.succ[j])
                if nd[j] < best[v]:
                    best[v] = nd[j]
                    seen[v] = True
                    heapq.heappush(heap, (float(nd[j]), v))
        return seen

    def reachable_domain(self, eps: float, r: float) -> np.ndarray:
        r"""The :math:`(\varepsilon, h)`-reachable domain
        :math:`D^{\ell^p}_{h, f}(\varepsilon, r)` as a boolean mask (§ 2.2.2):
        the points from which :math:`h^{-1}([-\infty, r])` is reachable under
        :math:`\varepsilon`-:math:`\ell^p` control."""
        k = int(np.searchsorted(self.levels, float(r), side="right")) - 1
        if k < 0:
            return np.zeros(self.n, dtype=bool)
        return self.control_cost()[:, k] <= abs(float(eps))

    def sublevel_controllability(self, x: int, r: float) -> float:
        r"""The value :math:`C^{\ell^p}_{h, f}(x, r) =
        \inf \mathcal{C}^{\ell^p}_{h, f}(x, r)` (§ 2.2.2), with
        :math:`\inf \emptyset = \infty`."""
        k = int(np.searchsorted(self.levels, float(r), side="right")) - 1
        if k < 0:
            return np.inf
        return float(self.control_cost()[int(x), k])

    # ------------------------------------------------------ minimizing paths
    def minimizing_path(self, x: int, eps: float) -> dict:
        r"""Reconstruct a "minimizing" :math:`\varepsilon`-:math:`\ell^p`-controlled path.

        Returns a path :math:`\gamma = (x; x_0, \dots, x_{n-1}; f(x_{n-1}))`
        whose terminal state attains
        :math:`h(f(x_{n-1})) = h_f^{\varepsilon\text{-}\ell^p}(x)`, as
        guaranteed by Theorem 3.12.

        Parameters
        ----------
        x
            Index of the initial state.
        eps
            Control budget; the magnitude is used, so ``-eps`` and ``eps``
            give the same path.

        Returns
        -------
        dict
            ``states``
                Indices :math:`(x, f(x_0), \dots, f(x_{n-1}))`, i.e. the set
                :math:`N(\gamma)` of Definition 9 in path order.
            ``jumps``
                Indices :math:`(x_0, \dots, x_{n-1})` of the jump targets.
            ``costs``
                The control magnitudes
                :math:`(\varepsilon_0, \dots, \varepsilon_{n-1})`.
            ``epsilon``
                :math:`\lVert (\varepsilon_i) \rVert_p` for the path, which is
                at most ``abs(eps)``.
            ``residual``
                Budget still available at each state of ``states``:
                ``(eps^p - sum of spent^p)^(1/p)`` for finite :math:`p`, and
                ``abs(eps)`` throughout for :math:`p = \infty`.
            ``value``
                :math:`h` at the terminal state.

        Raises
        ------
        ValueError
            If :math:`h_f^{\varepsilon\text{-}\ell^p}(x)` is infinite, i.e. no
            controlled path from :math:`x` reaches
            :math:`\operatorname{dom} h`.

        Notes
        -----
        Assertion (3) of Theorem 3.12 -- that every :math:`x' \in N(\gamma)`
        satisfies :math:`h_f^{\varepsilon\text{-}\ell^p}(x') = h(f(x_{n-1}))` --
        rests on the inclusion
        :math:`[x']^{\varepsilon\text{-}\ell^p}_f \subseteq
        [x]^{\varepsilon\text{-}\ell^p}_f`, which requires that prefixing the
        controlled path from :math:`x` to :math:`x'` keep the budget: true for
        :math:`p = \infty`, where concatenation composes by ``max``, and for
        zero-cost prefixes.  For finite :math:`p` the prefix has already spent
        part of the budget, and the correct statement uses the budget left at
        :math:`x'`, i.e. :math:`h_f^{\varrho\text{-}\ell^p}(x') =
        h(f(x_{n-1}))` for the corresponding entry :math:`\varrho` of
        ``residual``.
        """
        x = int(x)
        eps = abs(float(eps))
        target = float(self.control_effect(eps)[x])
        if not np.isfinite(target):
            raise ValueError(
                f"no epsilon-l^p-controlled path from index {x} into dom h for "
                f"epsilon = {eps}: the control effect value is infinite."
            )
        k = int(np.searchsorted(self.levels, target))
        W = self._to_weight(self.control_cost_exact()[:, k])
        # Cost of the remainder after jumping to x_j and applying f once.
        tail = np.where(self.dom_f, W[np.where(self.dom_f, self.succ, 0)], np.inf)
        weights = self._weights()

        states, jumps, costs = [x], [], []
        u = x
        for _ in range(self.n + 1):
            if self.dom_h[u] and self.h[u] == target and np.isclose(W[u], 0.0):
                break
            cand = np.where(self.dom_f, _aggregate(weights[u], tail, self.p), np.inf)
            j = int(np.argmin(cand))
            if not np.isfinite(cand[j]) or not np.isclose(
                cand[j], W[u], rtol=1e-9, atol=1e-12
            ):
                raise RuntimeError(  # pragma: no cover - solver invariant
                    "failed to reconstruct a minimizing path: the control cost "
                    "array and the cost matrix are inconsistent."
                )
            jumps.append(j)
            costs.append(float(self.cost[u, j]))
            u = int(self.succ[j])
            states.append(u)
        else:  # pragma: no cover - guarded by the solver invariants
            raise RuntimeError("minimizing path did not terminate.")

        c = np.asarray(costs, dtype=np.float64)
        if c.size == 0:
            total = 0.0
            residual = np.array([eps])
        elif np.isinf(self.p):
            total = float(c.max())
            residual = np.full(c.size + 1, eps)
        else:
            total = float((c**self.p).sum() ** (1.0 / self.p))
            spent = np.concatenate([[0.0], np.cumsum(c**self.p)])
            residual = np.maximum(eps**self.p - spent, 0.0) ** (1.0 / self.p)
        return {
            "states": np.asarray(states, dtype=np.int64),
            "jumps": np.asarray(jumps, dtype=np.int64),
            "costs": c,
            "epsilon": total,
            "residual": residual,
            "value": float(self.h[u]),
        }

    # ---------------------------------------------------------- filtrations
    def filtration_value(self, eps: float, delta: float) -> np.ndarray:
        r"""Boolean mask of :math:`H^{\ell^p}_{f, \varepsilon, \delta} =
        \{x \in X \mid h_f(p, \varepsilon, x) \le \delta\}` (§ 4.1)."""
        return self.control_effect(eps) <= float(delta)

    def filtration_difference(self, eps: float, delta: float) -> np.ndarray:
        r"""Boolean mask of :math:`H^{\ell^p}_{f, \Delta\varepsilon, \delta} =
        \{x \in X \mid h^{\Delta}_f(p, \varepsilon, x) \le \delta\}`
        (§ 4.1.1)."""
        return self.effect_difference(eps) <= float(delta)

    def with_p(self, p: float) -> "ControlProblem":
        """A copy of this problem with a different norm parameter ``p``."""
        return ControlProblem(
            succ=self.succ,
            cost=self.cost,
            h=self.h,
            layer=self.layer,
            p=p,
            validate=False,
        )


def _dijkstra_semiring(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    sources: Sequence[int] | np.ndarray,
    n: int,
    bottleneck: bool,
) -> np.ndarray:
    """Multi-source Dijkstra on a CSR graph, min-plus or min-max."""
    dist = np.full(n, np.inf)
    sources = np.asarray(sources, dtype=np.int64)
    if sources.size == 0:
        return dist
    dist[sources] = 0.0
    heap = [(0.0, int(s)) for s in sources]
    heapq.heapify(heap)
    done = np.zeros(n, dtype=bool)
    while heap:
        d, u = heapq.heappop(heap)
        if done[u]:
            continue
        done[u] = True
        for e in range(int(indptr[u]), int(indptr[u + 1])):
            v = int(indices[e])
            if done[v]:
                continue
            w = float(data[e])
            nd = max(d, w) if bottleneck else d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist
