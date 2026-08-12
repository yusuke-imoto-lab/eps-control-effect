r"""Visualization of control effect functions and their filtrations.

Every function returns the ``(fig, ax)`` it drew on, so figures can be composed
and saved by the caller; none of them calls ``plt.show``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ._core import ControlProblem
from ._data import DEFAULT_H_KEY, DEFAULT_SEQ_KEY, _resolve_obs
from ._persistence import Filtration

__all__ = [
    "plot_ensemble",
    "plot_effect",
    "plot_effect_heatmap",
    "plot_minimizing_path",
    "plot_effect_curve",
    "plot_filtration_sizes",
]


def _coords(adata, plot_key: str | None) -> np.ndarray:
    if plot_key is None:
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] < 2:
            raise ValueError(
                "at least two features are needed for a 2-D plot; pass "
                "plot_key to use coordinates from adata.obsm."
            )
        return X[:, :2]
    return np.asarray(adata.obsm[plot_key], dtype=np.float64)[:, :2]


def _axes(ax, figsize):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        return fig, ax
    return ax.get_figure(), ax


def plot_ensemble(
    adata,
    *,
    plot_key: str | None = None,
    seq_key: str = DEFAULT_SEQ_KEY,
    h_key: str = DEFAULT_H_KEY,
    ax=None,
    figsize: tuple[float, float] = (6.0, 4.5),
    cmap: str = "viridis",
    pointsize: float = 28.0,
    linewidth: float = 1.0,
    show_colorbar: bool = True,
):
    r"""Plot the ensemble trajectories, with terminal states coloured by :math:`h`.

    Parameters
    ----------
    adata
        Annotated data matrix following the :mod:`epscontrol` convention.
    plot_key
        Key in ``adata.obsm`` holding 2-D coordinates; the first two columns of
        ``adata.X`` are used when ``None``.
    seq_key, h_key
        Columns of ``adata.obs`` holding the sequence label and the evaluation
        function.

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    import pandas as pd

    pts = _coords(adata, plot_key)
    fig, ax = _axes(ax, figsize)
    seq = adata.obs[seq_key].to_numpy()
    for s in pd.unique(seq):
        m = seq == s
        ax.plot(
            pts[m, 0],
            pts[m, 1],
            marker="o",
            ms=3,
            lw=linewidth,
            color="0.65",
            zorder=1,
        )
    h = adata.obs[h_key].to_numpy(dtype=np.float64)
    term = np.isfinite(h)
    sc = ax.scatter(
        pts[term, 0],
        pts[term, 1],
        c=h[term],
        cmap=cmap,
        s=pointsize,
        edgecolor="k",
        linewidths=0.5,
        zorder=3,
    )
    if show_colorbar and term.any():
        fig.colorbar(sc, ax=ax, label=r"$h$ (terminal evaluation)")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.grid(ls=":", lw=0.5, zorder=0)
    return fig, ax


def plot_effect(
    adata,
    key: str,
    *,
    plot_key: str | None = None,
    ax=None,
    figsize: tuple[float, float] = (6.0, 4.5),
    cmap: str = "coolwarm",
    center: float | None = None,
    pointsize: float = 40.0,
    inf_color: str = "0.8",
    label: str | None = None,
    show_colorbar: bool = True,
):
    r"""Scatter the observations coloured by a stored effect function.

    Parameters
    ----------
    key
        Column of ``adata.obs`` written by :func:`~epscontrol.control_effect` or
        :func:`~epscontrol.effect_difference`.
    center
        Value mapped to the middle of a diverging colormap; ``0`` is a natural
        choice for a difference effect function.
    inf_color
        Colour for observations with an infinite value, i.e. those from which
        :math:`\operatorname{dom} h` is unreachable.

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    from matplotlib.colors import TwoSlopeNorm

    pts = _coords(adata, plot_key)
    if key not in adata.obs.columns:
        raise KeyError(f"{key!r} not found in adata.obs columns")
    vals = adata.obs[key].to_numpy(dtype=np.float64)
    fig, ax = _axes(ax, figsize)

    bad = ~np.isfinite(vals)
    if bad.any():
        ax.scatter(
            pts[bad, 0],
            pts[bad, 1],
            color=inf_color,
            s=pointsize,
            edgecolor="k",
            linewidths=0.4,
            zorder=2,
            label=r"$\infty$",
        )
    good = ~bad
    norm = None
    if center is not None and good.any():
        lo, hi = float(np.min(vals[good])), float(np.max(vals[good]))
        if lo < center < hi:
            norm = TwoSlopeNorm(vmin=lo, vcenter=center, vmax=hi)
    sc = ax.scatter(
        pts[good, 0],
        pts[good, 1],
        c=vals[good],
        cmap=cmap,
        norm=norm,
        s=pointsize,
        edgecolor="k",
        linewidths=0.4,
        zorder=3,
    )
    if show_colorbar and good.any():
        fig.colorbar(sc, ax=ax, label=label if label is not None else key)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.grid(ls=":", lw=0.5, zorder=0)
    if bad.any():
        ax.legend(loc="best", frameon=False, fontsize=8)
    return fig, ax


def plot_effect_heatmap(
    adata,
    problem: ControlProblem,
    *,
    ps: Sequence[float] = (1.0, 2.0, np.inf),
    epsilons: Sequence[float] = (0.0, 0.005, 0.01, 0.05),
    seq_key: str = DEFAULT_SEQ_KEY,
    time_key: str = "t",
    sort_by_outcome: bool = True,
    colors: Sequence[str] | None = None,
    t_mark: float | None = None,
    figsize: tuple[float, float] | None = None,
    axes=None,
    show_legend: bool = True,
    title: str | None = None,
):
    r"""Grid of heatmaps of :math:`h_f^{\varepsilon\text{-}\ell^p}` over time and trajectory.

    One panel per :math:`(p, \varepsilon)` pair: the horizontal axis is time,
    the vertical axis indexes the sequences (ensemble members), and the colour
    of a cell is the best outcome still reachable from that state under a
    control budget :math:`\varepsilon` in the :math:`\ell^p` norm.  Sorting the
    rows by terminal outcome makes the uncontrolled panel
    (:math:`\varepsilon = 0`) resolve into solid bands, so any other colour
    appearing inside a band is a state whose fate control can still change.

    Requires an ensemble in which every sequence is observed at the same times,
    which is what :func:`~epscontrol.from_ensemble` produces from equal-length
    members.

    Parameters
    ----------
    adata
        Annotated data matrix; its rows must be ordered as
        :func:`~epscontrol.build_problem` received them.
    problem
        The control problem.  Its own ``p`` is ignored in favour of ``ps``.
    ps, epsilons
        Norm parameters (one row of panels each) and control budgets (one
        column each), ascending.
    seq_key
        Column of ``adata.obs`` identifying the sequence.
    time_key
        Column of ``adata.obs`` holding the observation time, used for the
        horizontal axis.  Falls back to the within-sequence index when absent.
    sort_by_outcome
        Order the rows by the terminal value of :math:`h`, then by the cost of
        securing the best outcome.  Set ``False`` to keep the original order.
    colors
        One colour per level of :math:`\operatorname{Im} h`, ascending (so the
        first is the best outcome).  Defaults to a categorical palette.  Pass
        the same sequence used elsewhere in a figure set so that an outcome
        keeps its colour across figures.
    t_mark
        Time of a vertical reference line, e.g. a regime shift.  Read from
        ``adata.uns['competition']['t_switch']`` when available.
    figsize, axes
        Figure size, or an existing array of axes of shape
        ``(len(ps), len(epsilons))`` to draw into.
    show_legend, title
        Draw the outcome legend; overall figure title.

    Returns
    -------
    (matplotlib.figure.Figure, numpy.ndarray)
        The figure and the array of axes.

    Raises
    ------
    ValueError
        If :math:`h` takes more than 12 distinct values (a discrete palette
        would be unreadable — use :func:`plot_effect` instead), or if the
        sequences are not observed at a common set of times.

    Notes
    -----
    The panels are monotone by construction: moving right along a row can only
    improve the reachable outcome (Lemma 3.7) and, at fixed
    :math:`\varepsilon \ge 0`, so can moving down a column (Lemma 3.9).  The
    :math:`\varepsilon = 0` column is identical across rows, since with no
    control the norm is irrelevant.
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch
    import pandas as pd

    levels = problem.levels
    if levels.size > 12:
        raise ValueError(
            f"h takes {levels.size} distinct values; a discrete heatmap is only "
            "readable for a handful of outcomes. Use plot_effect instead."
        )

    seq = adata.obs[seq_key].to_numpy()
    members = pd.unique(seq)
    rows = [np.nonzero(seq == m)[0] for m in members]
    lengths = {len(r) for r in rows}
    if len(lengths) != 1:
        raise ValueError(
            f"sequences have differing lengths {sorted(lengths)}; a heatmap "
            "needs a common set of observation times."
        )
    idx = np.stack(rows)  # (n_members, n_times)

    if time_key in adata.obs.columns:
        times = adata.obs[time_key].to_numpy(dtype=np.float64)[idx[0]]
    else:
        times = np.arange(idx.shape[1], dtype=np.float64)

    ps = np.asarray(ps, dtype=np.float64).ravel()
    epsilons = np.asarray(epsilons, dtype=np.float64).ravel()

    h = adata.obs["h"].to_numpy(dtype=np.float64) if "h" in adata.obs else problem.h
    fate = np.array([np.nanmax(h[r]) for r in rows])
    if sort_by_outcome:
        base = problem.control_cost()[:, 0][idx[:, 0]]
        order = np.lexsort((base, fate))
    else:
        order = np.arange(idx.shape[0])
    idx, fate = idx[order], fate[order]

    grids = {}
    for p in ps:
        prob = problem if problem.p == p else problem.with_p(p)
        for e in epsilons:
            grids[(p, e)] = prob.control_effect(e)[idx]

    if colors is None:
        base_cycle = ["tab:blue", "tab:orange", "tab:red", "tab:green",
                      "tab:purple", "tab:brown", "tab:pink", "tab:olive",
                      "tab:cyan", "tab:gray", "gold", "navy"]
        colors = base_cycle[: levels.size]
    cmap = ListedColormap(list(colors))
    edges = np.concatenate(
        [[-np.inf], (levels[:-1] + levels[1:]) / 2.0, [np.inf]]
    )
    norm = BoundaryNorm(edges, cmap.N)

    if t_mark is None:
        t_mark = (adata.uns.get("competition", {}) or {}).get("t_switch")

    nr, nc = ps.size, epsilons.size
    if axes is None:
        import matplotlib.pyplot as plt

        if figsize is None:
            figsize = (3.2 * nc + 1.0, 2.4 * nr + 0.9)
        fig, axes = plt.subplots(nr, nc, figsize=figsize, sharex=True, sharey=True,
                                 squeeze=False)
    else:
        axes = np.atleast_2d(axes)
        fig = axes.flat[0].get_figure()

    n_mem = idx.shape[0]
    extent = [times[0], times[-1], n_mem - 0.5, -0.5]
    bounds = [int(np.searchsorted(fate, l, side="right")) for l in levels[:-1]]

    for i, p in enumerate(ps):
        for j, e in enumerate(epsilons):
            ax = axes[i, j]
            ax.imshow(grids[(p, e)], aspect="auto", cmap=cmap, norm=norm,
                      extent=extent, interpolation="nearest")
            if t_mark is not None:
                ax.axvline(float(t_mark), color="k", lw=0.9, ls=":")
            if sort_by_outcome:
                for b in bounds:
                    ax.axhline(b - 0.5, color="k", lw=0.7, alpha=0.5)
            if i == 0:
                lab = rf"$\varepsilon = {e:g}$"
                ax.set_title(lab + ("  (no control)" if e == 0 else ""))
            if j == 0:
                pl = r"$p=\infty$" if np.isinf(p) else rf"$p={p:g}$"
                ax.set_ylabel(f"{pl}\ntrajectory ID")
            if i == nr - 1:
                ax.set_xlabel("time")

    if show_legend:
        handles = [
            Patch(facecolor=cmap(k), label=rf"$h = {levels[k]:.3g}$")
            for k in range(levels.size)
        ]
        fig.legend(
            handles=handles, loc="lower center", ncol=min(levels.size, 4),
            frameon=False, bbox_to_anchor=(0.5, 0.005),
            title=r"best outcome still reachable   (lower = better)",
        )
    if title is not None:
        fig.suptitle(title, y=0.985)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.91,
                        bottom=0.145 if show_legend else 0.09,
                        wspace=0.06, hspace=0.16)
    return fig, axes


def plot_minimizing_path(
    adata,
    path: dict,
    *,
    plot_key: str | None = None,
    ax=None,
    figsize: tuple[float, float] = (6.0, 4.5),
    jump_color: str = "tab:red",
    flow_color: str = "tab:blue",
    linewidth: float = 1.6,
    annotate: bool = True,
):
    r"""Overlay a minimizing controlled path on a 2-D plot.

    Control jumps :math:`x \to x_i` are drawn in ``jump_color`` (dashed) and
    applications of the dynamics :math:`x_i \to f(x_i)` in ``flow_color``.

    Parameters
    ----------
    path
        The dict returned by :func:`~epscontrol.minimizing_path`.

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    pts = _coords(adata, plot_key)
    fig, ax = _axes(ax, figsize)
    states, jumps, costs = path["states"], path["jumps"], path["costs"]
    for k, j in enumerate(jumps):
        a, b, c = pts[states[k]], pts[j], pts[states[k + 1]]
        if costs[k] > 0:
            ax.annotate(
                "",
                xy=b,
                xytext=a,
                arrowprops=dict(
                    arrowstyle="->", color=jump_color, ls="--", lw=linewidth
                ),
                zorder=5,
            )
            if annotate:
                mid = 0.5 * (a + b)
                ax.annotate(
                    rf"$\varepsilon_{{{k}}}={costs[k]:g}$",
                    xy=mid,
                    color=jump_color,
                    fontsize=8,
                    zorder=6,
                )
        ax.annotate(
            "",
            xy=c,
            xytext=b,
            arrowprops=dict(arrowstyle="->", color=flow_color, lw=linewidth),
            zorder=5,
        )
    ax.scatter(
        pts[states[0], 0],
        pts[states[0], 1],
        marker="s",
        s=70,
        facecolor="none",
        edgecolor="k",
        lw=1.4,
        zorder=7,
    )
    ax.scatter(
        pts[states[-1], 0],
        pts[states[-1], 1],
        marker="*",
        s=160,
        color="gold",
        edgecolor="k",
        lw=0.6,
        zorder=7,
    )
    return fig, ax


def plot_effect_curve(
    problem: ControlProblem,
    x: int,
    *,
    ps: Sequence[float] = (1.0, 2.0, np.inf),
    epsilons: Sequence[float] | None = None,
    signed: bool = True,
    ax=None,
    figsize: tuple[float, float] = (6.0, 4.0),
    n_points: int = 81,
):
    r"""Plot :math:`\varepsilon \mapsto h_f(p, \varepsilon, x)` for several :math:`p`.

    The curve is weakly decreasing in :math:`\varepsilon` (Proposition 3.10) and,
    at fixed :math:`\varepsilon \ge 0`, weakly decreasing in :math:`p`
    (Lemma 3.9), so a larger :math:`p` and a larger budget can only improve the
    attainable value.

    Parameters
    ----------
    problem
        The control problem; its ``p`` is ignored in favour of ``ps``.
    x
        Index of the initial state.
    signed
        Also draw the :math:`-\varepsilon` branch of Definition 14 on the
        negative half-axis, giving the full parameter domain
        :math:`[-\infty, -0] \sqcup [0, \infty]`.

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    fig, ax = _axes(ax, figsize)
    if epsilons is None:
        probs = [problem.with_p(p) for p in ps]
        finite = np.concatenate(
            [pr.control_cost_exact().ravel() for pr in probs]
        )
        finite = finite[np.isfinite(finite)]
        top = float(finite.max()) if finite.size else 1.0
        epsilons = np.linspace(0.0, top if top > 0 else 1.0, int(n_points))
    epsilons = np.asarray(epsilons, dtype=np.float64)

    for p in ps:
        pr = problem if problem.p == p else problem.with_p(p)
        label = r"$p=\infty$" if np.isinf(p) else rf"$p={p:g}$"
        up = np.array([pr.control_effect(e)[x] for e in epsilons])
        ax.step(epsilons, up, where="post", label=label)
        if signed:
            dn = np.array([pr.control_effect(-e if e > 0 else -0.0)[x] for e in epsilons])
            ax.step(
                -epsilons,
                dn,
                where="pre",
                ls="--",
                color=ax.lines[-1].get_color(),
            )
    ax.axvline(0.0, color="k", lw=0.6)
    ax.set_xlabel(r"control budget $\varepsilon$")
    ax.set_ylabel(r"$h_f(p, \varepsilon, x)$")
    ax.legend(frameon=False)
    ax.grid(ls=":", lw=0.5)
    return fig, ax


def plot_filtration_sizes(
    filt: Filtration,
    *,
    p_index: int = -1,
    ax=None,
    figsize: tuple[float, float] = (6.0, 4.0),
    cmap: str = "magma",
):
    r"""Heatmap of :math:`\lvert H^{\ell^p}_{f, \varepsilon, \delta} \rvert`.

    Shows one slice of the three-parameter filtration at ``filt.ps[p_index]``:
    how many states admit an outcome of value at most :math:`\delta` under a
    control budget :math:`\varepsilon`.

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    fig, ax = _axes(ax, figsize)
    sizes = filt.sizes[p_index]
    im = ax.pcolormesh(
        filt.deltas,
        filt.epsilons,
        sizes,
        cmap=cmap,
        shading="nearest",
    )
    p = filt.ps[p_index]
    fig.colorbar(im, ax=ax, label=r"$|H_{f,\varepsilon,\delta}|$")
    ax.set_xlabel(r"value threshold $\delta$")
    ax.set_ylabel(r"control budget $\varepsilon$")
    ax.set_title(r"$p=\infty$" if np.isinf(p) else rf"$p={p:g}$")
    return fig, ax
