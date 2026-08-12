# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added

- `plot_effect_heatmap` — the control effect function drawn without aggregating:
  one cell per (trajectory, time), one panel per $(p, \varepsilon)$. Rows sorted
  by terminal outcome make the uncontrolled panel resolve into solid bands, so a
  foreign colour inside a band marks a state whose fate control can still change.
  Shows two things the aggregated curves hide: leverage is distributed unevenly
  across trajectories, and the choice of norm only matters before the shift.
- Section 9 of the ecology tutorial, built on the above.

## [0.2.0]

### Added

- `datasets.chaotic_competition` — an ensemble of chaotic four-species
  competition trajectories that undergo a regime shift and settle into
  alternative stable states. Only one species is observed, at a coarse
  interval, and the state is reconstructed by delay-coordinate embedding, so
  the evaluation function is genuinely partial and the observable genuinely
  low-dimensional.
- `datasets.simulate_competition` — integrator for the competitive
  Lotka–Volterra system, for a single state or a whole ensemble, with an
  optional switch of the interaction matrix.
- `datasets.stable_states` — the equilibria of a competitive system that resist
  invasion by the species left out. The number of stable states is the number
  of levels of the evaluation function, so this is the tool for designing a
  post-shift regime with a prescribed number of outcomes.
- `datasets.competition_growth_rates`, `datasets.competition_matrix` — the
  chaotic system of Vano et al., *Nonlinearity* **19** (2006), 2391; largest
  Lyapunov exponent ≈ 0.02, reproduced to 0.0201 by the test suite.
- `datasets.competition_matrix_post` — bistable post-shift regime (two outcome
  levels), and `datasets.competition_matrix_post3` — tristable post-shift
  regime (three outcome levels).
- `examples/tutorial_ecology.ipynb` — applied tutorial on the above: transient
  chaos, a press disturbance, priority effects, and the resulting *control
  window*, in which the cost of securing a good outcome rises roughly 100-fold
  across the transition and begins rising before the disturbance is visible in
  the data.

### Changed

- Terminal abundances in `chaotic_competition` are snapped to the exact
  equilibria of the post-shift system (tolerance `snap_tol`, default 0.02).
  Without this, members that had not fully relaxed contributed numerical
  duplicates of a single true outcome, inflating the image of the evaluation
  function; a member further than `snap_tol` from every equilibrium now raises
  `RuntimeError` rather than silently adding a level.
- `README.md` documents the new dataset API and `result_key`.

### Fixed

- Section 8 of the ecology tutorial referenced the growth rates `r` bound five
  sections earlier, so the cell failed with `NameError` when run on its own. It
  now calls `datasets.stable_states`, and a regression test checks that no cell
  of either tutorial references a name it has not bound.

## [0.1.0]

### Added

- `ControlProblem` — the finite control problem $(X, f, c, h, p)$, with exact
  solvers for the control effect function: backward dynamic programming on the
  time-graded transition DAG, and min-plus / min-max Dijkstra in the general
  case. Values are exact rather than swept on an ε grid.
- AnnData interface — `from_ensemble`, `build_problem`, `control_effect`,
  `effect_difference`, `control_cost`, `reachable_range`, `reachable_domain`,
  `minimizing_path`, `check_hypotheses`, `result_key`.
- Multi-parameter persistence — `filtration` and `Filtration`, the
  three-parameter filtration in (cost norm, control strength, resulting value)
  of Theorems 4.3 and 4.4, with zeroth Betti numbers.
- Plotting — `plot_ensemble`, `plot_effect`, `plot_minimizing_path`,
  `plot_effect_curve`, `plot_filtration_sizes`.
- `datasets.paper_example` and `datasets.branching_ensemble`.
- `examples/tutorial.ipynb`, and a test suite that checks the library against
  Example 1 and Figure 3b of Imoto–Yokoyama, together with the monotonicity
  statements of Lemmas 2.5, 3.6, 3.7 and 3.9.
