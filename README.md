# 2D Lid-Driven Cavity Flow Solver

**MAE 4100/5100 Course Project 2 — Numerical Methods and AI for CFD**

A Python implementation of a two-dimensional incompressible Navier–Stokes solver for the classic lid-driven cavity problem, built from scratch on a finite-difference discretization. The project implements and compares two solution strategies:

- **Coupled artificial-compressibility solver** — the density-based pseudo-time approach with time-derivative preconditioning, solved with point Jacobi or symmetric Gauss-Seidel (SGS) iterations.
- **Fractional-step (pressure-projection) solver** — the pressure-based approach: intermediate velocity prediction, a pressure Poisson equation (PPE) solve, and a divergence-free velocity correction.

Both paths support serial loops, NumPy-vectorized kernels, and Numba-accelerated SGS sweeps using diagonal wavefront ordering for parallelism. The solver is verified with the Method of Manufactured Solutions (MMS), validated against the Ghia, Ghia & Shin (1982) benchmark data, and cross-checked against an OpenFOAM simulation of the same cavity.

<p align="center">
  <img src="start-code/Phase%20I%20Plots/ucontour.png" width="45%" alt="u-velocity contour">
  <img src="start-code/Phase%20III%20Ghia%20Literature%20Comparison%20Re500%20Bracket/ghia_bracket_vertical_centerline_u_Re500.png" width="45%" alt="Ghia benchmark comparison">
</p>

## Problem Description

Incompressible flow in a square cavity driven by a lid moving at constant velocity `uinf`, with no-slip side and bottom walls. The solver marches in pseudo-time until the iterative residuals of continuity and both momentum equations drop below tolerance (default `1e-10`). Studies cover Reynolds numbers from 10 to 1000, grids from 9×9 to 129×129, and non-square cavity geometries.

## Repository Layout

```
├── start-code/                    # Main solver code and all study notebooks
│   ├── main_solver.py             # Driver script: parameters, grid, main iteration loop
│   ├── req_functions.py           # Numerical methods library (see below)
│   ├── utils.py                   # Source terms, initialization, I/O helpers
│   ├── phase*_*.ipynb             # Study notebooks (see Notebook Guide)
│   └── Phase */                   # Output folders: plots, CSV summaries, cached cases
├── openfoam-cavity-Re100/         # OpenFOAM Re=100 case + post-processing notebook
├── literature-reference/          # Ghia et al., Erturk, Shankar reference papers
├── req_functions_DONT_EDIT.py     # Original assignment scaffold (unmodified)
└── MAE_4100_5100_course_projects.pdf  # Assignment specification
```

### Numerical methods library (`start-code/req_functions.py`)

The flow field is stored as a single array `u[i, j, k]` with `k = 0` pressure, `k = 1` x-velocity, `k = 2` y-velocity.

| Component | Functions |
|---|---|
| Local pseudo-time step (CFL-based, preconditioned) | `compute_time_step` |
| 4th-order artificial viscosity (pressure stabilization) | `Compute_Artificial_Viscosity` |
| Boundary conditions (cavity + MMS) | `set_boundary_conditions`, `bndry`, `bndrymms` |
| Coupled solver updates | `point_Jacobi`, `SGS_forward_sweep`, `SGS_backward_sweep` |
| Numba-accelerated SGS (wavefront-parallel) | `SGS_forward_sweep_acc`, `SGS_backward_sweep_acc` |
| Fractional-step method | `compute_intermediate_velocity`, `solve_PPE_Jacobi`, `solve_PPE_SGS`, `solve_PPE_SGS_acc`, `correct_velocity` |
| Convergence and MMS error norms | `check_iterative_convergence`, `discretization_error_norms` |

Most functions take a `vectorize` flag so the same operation can run as explicit Python loops (matching the assignment scaffold) or as NumPy array operations.

## Getting Started

```bash
git clone https://github.com/windy-schmieder/mae4100-courseproject2.git
cd mae4100-courseproject2
pip install -r requirements.txt
```

Run the baseline solver directly:

```bash
cd start-code
python main_solver.py
```

Key parameters at the top of `main_solver.py`:

| Parameter | Default | Meaning |
|---|---|---|
| `imax`, `jmax` | 9 | Grid nodes per direction (odd numbers) |
| `Re` | 10 | Reynolds number |
| `solver_method` | `'coupled'` | `'coupled'` or `'fractional_step'` |
| `isgs` | 1 | 1 = symmetric Gauss-Seidel, 0 = point Jacobi |
| `cfl` | 0.5 | CFL number for the local time step |
| `rkappa` | 0.5 | Artificial-compressibility preconditioning constant |
| `Cx`, `Cy` | 0.01 | 4th-order artificial viscosity coefficients |
| `imms` | 0 | 1 = manufactured-solution verification mode |
| `toler` | 1e-10 | Iterative residual convergence tolerance |

The study notebooks patch these parameters in memory and run cases programmatically, saving plots and summaries into descriptive `Phase */` folders. Long-running studies cache solved fields as `.npz` files in `case_cache/` subfolders so plots can be regenerated without re-solving.

## Project Phases

**Phase I — Baseline solver.** Serial coupled artificial-compressibility solver with SGS sweeps. Contours, residual histories, and centerline profiles (`phase1_plotting.ipynb`).

**Phase II — Vectorization.** NumPy-vectorized coupled solver, validated point-by-point and timed against the serial implementation (`phase2_plotting.ipynb`, `phase2_vectorized_validation.ipynb`).

**Phase III — Verification and studies.**
- *MMS code verification:* L1/L2/L∞ error norms across a 9→129 mesh sequence and observed order-of-accuracy estimation (`phase3_plotting.ipynb`).
- *Accelerated-solver validation:* Numba wavefront SGS checked against baseline SGS before use in long studies (`phase3_accelerated_sgs_validation.ipynb`).
- *Parameter studies:* preconditioning constant κ (`phase3_kappa*.ipynb`), artificial viscosity C4 (`phase3_c4_accelerated_17x17.ipynb`), mesh refinement at Re=10 (`phase3_mesh_comparison_Re10.ipynb`), cavity aspect ratio (`phase3_geometry_comparison_17x17.ipynb`).
- *Reynolds-number studies:* Re = 500 and 1000 with the coupled solver (`Phase3_reynolds_*.ipynb`) and a Re–mesh refinement map with the fractional-step solver (`Phase3_reynolds_comparison_mesh_refinement_accelerated.ipynb`).
- *Literature benchmark:* centerline profiles compared against Ghia, Ghia & Shin (1982) — Re=100 directly, and Re=500 bracketed by the published Re=400/1000 data (`Phase3_ghia_literature_comparison_Re500_bracket.ipynb`).

**Section 4 — Fractional-step method** (required for MAE 5100, bonus for MAE 4100). Pressure-based projection solver with Jacobi/SGS/accelerated PPE solvers (`phase4_fractional_step_plotting.ipynb`).

**OpenFOAM comparison.** An icoFoam-style lid-driven cavity at Re=100, post-processed into the same plot style as the Python solver for direct comparison (`openfoam-cavity-Re100/openfoam_results_plotting.ipynb`).

## Dependencies

Python 3 with NumPy, Matplotlib, Pandas, Numba, and Jupyter — see [requirements.txt](requirements.txt). The OpenFOAM notebook only post-processes an already-completed case; OpenFOAM itself is not required.

## References

1. Ghia, U., Ghia, K. N., & Shin, C. T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. *Journal of Computational Physics*, 48(3), 387–411.
2. Shankar, P. N., & Deshpande, M. D. (2000). Fluid mechanics in the driven cavity. *Annual Review of Fluid Mechanics*, 32, 93–136.
3. Erturk, E., Corke, T. C., & Gökçöl, C. (2005). Numerical solutions of 2-D steady incompressible driven cavity flow at high Reynolds numbers. *International Journal for Numerical Methods in Fluids*, 48, 747–774.

PDF copies are included in [literature-reference/](literature-reference/).

---

*A more detailed function-by-function and notebook-by-notebook guide is in [README_final.md](README_final.md).*
