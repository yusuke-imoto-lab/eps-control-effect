r"""Control effect functions and multi-parameter persistence for ensembles.

``epscontrol`` implements the constructions of

    Y. Imoto and T. Yokoyama, *Multi-parameter persistence in dynamical systems
    for maximizing effects of control inputs*, arXiv:2606.05577 (2026),
    https://doi.org/10.48550/arXiv.2606.05577.

Given an ensemble of time series and an evaluation function defined only at the
terminal time of each member, the library extends that partial function to the
whole state space as the :math:`\varepsilon`-control effect function

.. math::

    h_f^{\varepsilon\text{-}\ell^p}(x)
        = \inf h\bigl([x]^{\varepsilon\text{-}\ell^p}_f
          \cap \operatorname{dom} h\bigr),

the best outcome reachable from :math:`x` when control inputs of total
:math:`\ell^p` magnitude at most :math:`\varepsilon` may be applied along the
orbit; the :math:`-\varepsilon` branch takes the supremum instead and gives the
worst outcome an adversarial control of the same budget can force.  The
associated sublevel sets form a three-parameter filtration in
:math:`(p, \varepsilon, \delta)` -- cost norm, control strength, resulting
value -- ready for multi-parameter persistence.

Typical use
-----------
>>> import numpy as np, epscontrol as ec
>>> adata = ec.datasets.paper_example()
>>> prob = ec.build_problem(adata, p=np.inf)
>>> ec.control_effect(adata, [0.0, 1.0, 2.0], problem=prob)
>>> float(adata.obs.loc[adata.obs_names[0], "control_effect_eps1_pinf"])
1.0
"""

from ._core import ControlProblem
from ._data import (
    build_problem,
    check_hypotheses,
    control_cost,
    control_effect,
    effect_difference,
    from_ensemble,
    minimizing_path,
    reachable_domain,
    reachable_range,
    result_key,
)
from ._persistence import Filtration, filtration
from ._plotting import (
    plot_effect,
    plot_effect_curve,
    plot_effect_heatmap,
    plot_ensemble,
    plot_filtration_sizes,
    plot_minimizing_path,
)
from . import datasets

__version__ = "0.2.0"

__all__ = [
    "ControlProblem",
    "Filtration",
    "build_problem",
    "check_hypotheses",
    "control_cost",
    "control_effect",
    "datasets",
    "effect_difference",
    "filtration",
    "from_ensemble",
    "minimizing_path",
    "plot_effect",
    "plot_effect_curve",
    "plot_effect_heatmap",
    "plot_ensemble",
    "plot_filtration_sizes",
    "plot_minimizing_path",
    "reachable_domain",
    "reachable_range",
    "result_key",
    "__version__",
]
