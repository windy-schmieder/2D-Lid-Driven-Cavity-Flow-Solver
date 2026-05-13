# MAE 4100 Course Project 2 Final README

This project contains a Python implementation of a two-dimensional incompressible lid-driven cavity solver, plus notebooks used for validation, plotting, mesh studies, parameter studies, Reynolds number studies, and OpenFOAM comparison. Most project work is in `start-code/`, where `main_solver.py` drives the simulations and `req_functions.py` contains the numerical methods used by the solver.

The main shared file is `start-code/req_functions.py`. The notebooks usually patch or configure `main_solver.py` in memory, run one or more cases, then save contours, residual histories, centerline profiles, cache files, or summary CSV files.

## Assignment Phase Definitions

The assignment PDF defines Phase II as vectorizing the density-based artificial-compressibility solver, not the fractional-step method. The fractional-step method is the pressure-based solver in Section 4: compulsory for MAE 5100 and bonus for MAE 4100. In this README, "Phase II" refers to the coupled solver vectorization workflow, while "fractional step" refers to the optional pressure-based solver path.

## Main Function Library: `req_functions.py`

`req_functions.py` is the numerical-methods library for the project. It stores the reusable pieces of the solver so that the notebooks can switch between coupled, fractional-step, serial, vectorized, and accelerated versions without rewriting the entire CFD algorithm.

The flow-field array convention used throughout the file is:

- `u[:, :, 0]`: pressure
- `u[:, :, 1]`: x-velocity
- `u[:, :, 2]`: y-velocity

Most functions operate only on interior nodes and then rely on boundary-condition functions to update the ghost or boundary values. Several functions include a `vectorize` option so the same operation can be run with either explicit Python loops or NumPy array operations.

### Time Step And Stabilization

`compute_time_step(...)` calculates a local pseudo-time step based on convection, artificial-compressibility wave speeds, viscosity, grid spacing, CFL number, and the preconditioning parameter `rkappa`. It returns both the local `dt` array and the minimum time step used for convergence reporting.

`Compute_Artificial_Viscosity(...)` computes fourth-order artificial viscosity coefficients in the x and y directions. These coefficients stabilize the pressure field by damping high-frequency oscillations. The scaling depends on the maximum local pseudo-acoustic wave speeds and the chosen artificial viscosity constants `Cx` and `Cy`.

### Boundary Conditions

`set_boundary_conditions(...)` is a wrapper that chooses between the physical cavity boundary conditions and the manufactured-solution boundary conditions.

`bndry(...)` applies the standard lid-driven cavity boundaries. The side and bottom walls are no-slip, the top lid has x-velocity `uinf`, the y-velocity is zero on all walls, and pressure is extrapolated from the interior.

`bndrymms(...)` applies manufactured-solution velocity values on the boundaries for MMS verification. Pressure is still extrapolated from the interior, which keeps the pressure treatment consistent with the standard solver.

### Coupled Solver Updates

`point_Jacobi(...)` performs an explicit point-Jacobi update of pressure and velocity. It evaluates central finite differences for pressure gradients, velocity gradients, diffusion terms, source terms, and fourth-order pressure dissipation. Because it uses `uold` for the update, all interior points are advanced from the previous iteration state.

`SGS_forward_sweep(...)` and `SGS_backward_sweep(...)` implement the baseline symmetric Gauss-Seidel coupled solver. The forward sweep updates points from lower-left to upper-right, and the backward sweep reverses the order. Unlike Jacobi, Gauss-Seidel immediately uses updated values during the sweep, which usually improves convergence.

`SGS_forward_sweep_acc(...)` and `SGS_backward_sweep_acc(...)` are Numba-accelerated versions of the coupled SGS sweeps. They use diagonal wavefront ordering so points on the same diagonal can be processed in parallel while preserving the main Gauss-Seidel dependency structure.

### Convergence And Error Measurement

`check_iterative_convergence(...)` computes normalized residuals for pressure, x-velocity, and y-velocity by comparing the current solution to the previous iteration. It also writes residual history lines to the solver log and prints periodic convergence updates.

`discretization_error_norms(...)` computes L1, L2, and L-infinity error norms for MMS cases. When the manufactured-solution flag `imms` is not enabled, it returns zero arrays.

### Fractional-Step Solver Functions

The fractional-step method splits the velocity update, pressure solve, and velocity correction into separate stages.

`compute_intermediate_velocity(...)` predicts velocity without the pressure-gradient projection. It includes convection, diffusion, and momentum source terms.

`solve_PPE_Jacobi(...)` solves the pressure Poisson equation using Jacobi iterations. It updates pressure until either the pressure tolerance is reached or the maximum iteration count is met.

`solve_PPE_SGS(...)` solves the pressure Poisson equation using symmetric Gauss-Seidel iterations. This is the main pressure solver used in the optional pressure-based fractional-step studies.

`solve_PPE_SGS_acc(...)` is the Numba-accelerated pressure Poisson solver. Like the accelerated coupled sweeps, it uses diagonal wavefront ordering to expose parallel work while keeping the Gauss-Seidel-style update.

`correct_velocity(...)` applies the pressure-gradient correction after the PPE solve so the predicted velocity field is projected toward a divergence-free field.

## Notebook Guide

### `start-code/phase1_plotting.ipynb`

This notebook runs the default Phase I version of `main_solver.py` and collects the standard output plots. It adds vertical centerline velocity and pressure profiles for the Phase I deliverable.

Main outputs are saved in `start-code/Phase I Plots/`, including velocity contours, pressure contours, residual plots, and centerline profile figures.

### `start-code/phase2_plotting.ipynb`

This notebook runs the assignment Phase II vectorized-code workflow. It keeps `main_solver.py` on the coupled artificial-compressibility SGS solver path, enables `vectorize = True`, then saves contours, residual histories, and both vertical and horizontal centerline profiles.

Main outputs are saved in `start-code/Phase II Plots/`.

### `start-code/phase2_vectorized_validation.ipynb`

This notebook validates the assignment Phase II vectorized implementation against the serial implementation. It runs the same coupled SGS case with `vectorize=False` and `vectorize=True`, overlays centerline profiles, reports maximum differences between the two results, and compares timing.

Main outputs are saved in `start-code/Phase II Vectorization Validation/`.

### `start-code/phase4_fractional_step_plotting.ipynb`

This notebook preserves the optional pressure-based fractional-step workflow from Section 4 of the assignment. It runs `main_solver.py` with `solver_method = 'fractional_step'`, uses the PPE SGS pressure solve, and saves the same contour and centerline profile plots as the Phase II plotting notebook.

Main outputs are saved in `start-code/Section 4 Fractional Step Plots/`.

### `start-code/phase3_plotting.ipynb`

This notebook performs Phase III code verification using the method of manufactured solutions. It enables `imms = 1`, runs a mesh sequence at `Re = 10`, computes L1, L2, and L-infinity errors, and estimates the observed order of accuracy.

The mesh sequence is typically `9`, `17`, `33`, `65`, and `129` nodes. The notebook primarily displays tables and log-log error plots inline.

### `start-code/phase3_mesh_comparison_Re10.ipynb`

This notebook compares cavity solutions at `Re = 10` across several mesh sizes. It uses the coupled solver with accelerated SGS sweeps and studies how the solution changes as the grid is refined.

Typical cases are `9x9`, `17x17`, `33x33`, and `65x65`, with parameters such as `rkappa = 0.5`, `C4 = 0.01`, and `CFL = 0.5`. Outputs and cached case data are stored in `start-code/Phase III Mesh Comparison Re10/`.

### `start-code/phase3_geometry_comparison_17x17.ipynb`

This notebook compares different cavity aspect ratios on a `17x17` grid at `Re = 10`. The cases include a square cavity, a wide cavity, and a tall cavity. It is useful for showing how geometry changes the flow structure while keeping the node count fixed.

Outputs are saved in `start-code/Phase III Geometry Comparison 17x17/`, including comparison plots, cached case data, `geometry_summary.csv`, and geometry flow metrics.

### `start-code/phase3_kappa.ipynb`

This notebook studies sensitivity to the artificial-compressibility preconditioning parameter `rkappa`. It uses the standard coupled SGS path and compares convergence and solution behavior across several `kappa` values.

It is generally used for report-quality `kappa` plots, often on a finer grid such as `65x65` at `Re = 10`. Outputs are saved in `start-code/Phase III Kappa Study/`.

### `start-code/phase3_kappa_accelerated.ipynb`

This notebook is a faster screening version of the `kappa` study. It uses similar plotting and summary logic but is set up for quicker runs, commonly on `33x33` at `Re = 10` with a smaller set of `kappa` values.

Outputs are saved under a baseline kappa-study folder, such as `start-code/Phase III Kappa Study Baseline/`.

### `start-code/phase3_kappa_accelerated_65x65.ipynb`

This notebook performs the higher-resolution accelerated `kappa` study. It uses a `65x65` grid at `Re = 10` and the accelerated coupled SGS implementation to make the larger cases practical.

Outputs are saved in `start-code/Phase III Kappa Study Accelerated 65x65/`.

### `start-code/phase3_c4_accelerated_17x17.ipynb`

This notebook studies sensitivity to the fourth-order artificial viscosity coefficient `C4`, which is applied through `Cx` and `Cy` in `Compute_Artificial_Viscosity(...)`. The notebook name includes `17x17`, but the current study setup is oriented around a faster `33x33` accelerated case.

Typical parameters are `Re = 10`, `rkappa = 0.5`, and several `C4` values. Outputs are saved in `start-code/Phase III C4 Study Accelerated 33x33/`.

### `start-code/phase3_accelerated_sgs_validation.ipynb`

This notebook validates the accelerated coupled SGS sweeps against the baseline SGS sweeps. It compares convergence behavior, field differences, and MMS norms so the accelerated functions can be trusted before being used in longer parameter studies.

The notebook is mainly a validation notebook and focuses on printed results and DataFrame summaries instead of a large plot directory.

### `start-code/Phase3_reynolds_Re500_Re1000_coupled_standard.ipynb`

This notebook runs higher-Reynolds-number cavity cases using the standard coupled SGS solver. It focuses on `Re = 500` and `Re = 1000` rather than the lower-Reynolds-number verification cases.

The current setup uses a `33x33` grid for `Re = 500` and a `65x65` grid for `Re = 1000`, with smaller CFL values for stability. Outputs are saved in `start-code/Phase III Reynolds Re500 Re1000 Coupled Standard/`, including per-case folders, cache files, `summary_partial.csv`, centerline comparisons, and contour comparisons. Because the `65x65`, `Re = 1000` case is expensive, this notebook may contain only the completed partial summary until the long case finishes.

### `start-code/Phase3_reynolds_Re1000_33x33_coupled_standard.ipynb`

This notebook is a practical single-case high-Reynolds-number run using the standard coupled SGS solver. It keeps `solver_method = 'coupled'`, patches `main_solver.py` only in memory, and runs `Re = 1000` on a `33x33` grid with a smaller CFL value.

Outputs are saved in `start-code/Phase III Reynolds Re1000 33x33 Coupled Standard/`, including `summary.csv`, cached case data, centerline profiles, field contours, and `flow_metrics_Re1000_33x33.csv`. This case is useful when the finer `65x65`, `Re = 1000` run is too expensive to complete, but it should be described as a coarser practical result in the report.

### `start-code/Phase3_ghia_literature_comparison_Re500_bracket.ipynb`

This notebook creates a literature benchmark comparison against the classic lid-driven-cavity centerline data from Ghia, Ghia, and Shin. Because Ghia et al. do not tabulate `Re = 500`, the present `Re = 500`, `33x33` coupled-SGS solution is compared qualitatively against the neighboring published `Re = 400` and `Re = 1000` profiles.

The notebook also plots the present `Re = 100`, `17x17` solver result against the matching Ghia `Re = 100` data as a lower-Reynolds-number consistency check. Outputs are saved in `start-code/Phase III Ghia Literature Comparison Re500 Bracket/`, including `ghia_bracket_vertical_centerline_u_Re500.png`, `ghia_bracket_horizontal_centerline_v_Re500.png`, and a `case_cache/` folder for any regenerated solver fields.

### `start-code/Phase3_reynolds_comparison_mesh_refinement_accelerated.ipynb`

This notebook compares Reynolds numbers over a mesh-refinement map. It uses the optional pressure-based fractional-step method with the accelerated SGS pressure Poisson solver, so it should be interpreted as part of the advanced/fractional-step solver analysis rather than the required Phase II vectorization task.

The study maps `Re = 10`, `100`, `500`, and `1000` to increasingly fine meshes, commonly `9x9`, `17x17`, `33x33`, and `65x65`. Outputs are saved in `start-code/Phase III Reynolds Comparison Mesh Refinement Accelerated/`, along with cached cases, CSV summaries, and Reynolds-number comparison plots.

### `openfoam-cavity-Re100/openfoam_results_plotting.ipynb`

This notebook post-processes an OpenFOAM lid-driven cavity simulation at `Re = 100`. It reads the completed OpenFOAM field files, parses velocity and pressure, shifts pressure to match the course reference convention, and creates contour and centerline plots in the same style as the Python solver outputs.

Outputs are saved in `openfoam-cavity-Re100/OpenFOAM Plots/`.

## Suggested Reading Order

Start with `req_functions.py` to understand the numerical building blocks. Then read the notebooks in this order:

1. `phase1_plotting.ipynb` for the baseline coupled solver output.
2. `phase2_plotting.ipynb` for the vectorized coupled SGS Phase II workflow.
3. `phase2_vectorized_validation.ipynb` to confirm vectorized and serial results agree.
4. `phase3_plotting.ipynb` for MMS verification and observed-order checks.
5. `phase3_accelerated_sgs_validation.ipynb` before relying on accelerated SGS runs.
6. Phase III study notebooks for mesh, geometry, `kappa`, `C4`, and Reynolds-number comparisons.
7. `Phase3_ghia_literature_comparison_Re500_bracket.ipynb` for the Ghia benchmark comparison and report wording.
8. `phase4_fractional_step_plotting.ipynb` and other fractional-step notebooks only for the Section 4 / bonus pressure-based solver analysis.
9. `openfoam_results_plotting.ipynb` for the OpenFOAM comparison workflow.

## Output Organization

Most notebooks save results into descriptive folders under `start-code/`, usually beginning with `Phase I`, `Phase II`, or `Phase III`. Long-running Phase III notebooks often also create a `case_cache/` subfolder so previously computed cases can be reused for plotting without rerunning the full solver.

Common output files include:

- `ucontour.png`, `vcontour.png`, and `pcontour.png`
- residual-history plots
- vertical and horizontal centerline velocity plots
- Ghia benchmark centerline comparison plots
- pressure centerline plots
- comparison contour figures
- `summary.csv` or related summary CSV files
- cached `.npz` and metadata files for multi-case studies

## Dependencies

The project notebooks and solver rely mainly on:

- Python
- NumPy
- Matplotlib
- Pandas, for notebook summaries and comparison tables
- Numba, for accelerated SGS and pressure-solver functions
- Jupyter Notebook or JupyterLab

The OpenFOAM notebook assumes the OpenFOAM case has already been run and that the expected field and sampling files are present in `openfoam-cavity-Re100/`.
