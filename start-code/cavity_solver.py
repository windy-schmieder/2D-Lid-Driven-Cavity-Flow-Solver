"""Refactored, importable driver for the lid-driven cavity solver.

This module wraps the numerical kernels in ``req_functions.py`` / ``utils.py``
behind a single callable, ``run_cavity(CavityConfig(...))``, so cases can be
run programmatically (tests, parameter sweeps, surrogate-model data
generation) without patching ``main_solver.py`` source text in memory.

``main_solver.py`` is kept unchanged for backward compatibility with the
course notebooks, which string-replace settings in its source before exec'ing
it.

Command-line usage::

    python cavity_solver.py --imax 33 --re 100 --accelerated --outdir out_Re100
"""

import argparse
import io
import time
from dataclasses import dataclass, field

import numpy as np

from req_functions import (
    Compute_Artificial_Viscosity,
    SGS_backward_sweep,
    SGS_backward_sweep_acc,
    SGS_forward_sweep,
    SGS_forward_sweep_acc,
    check_iterative_convergence,
    compute_intermediate_velocity,
    compute_time_step,
    correct_velocity,
    discretization_error_norms,
    point_Jacobi,
    set_boundary_conditions,
    solve_PPE_Jacobi,
    solve_PPE_SGS,
    solve_PPE_SGS_acc,
)
from utils import compute_source_terms, initialize, pressure_rescaling

# Manufactured-solution constants (same values as main_solver.py)
PHI0 = [0.25, 0.3, 0.2]
PHIX = [0.5, 0.15, 1.0 / 6.0]
PHIY = [0.4, 0.2, 0.25]
PHIXY = [1.0 / 3.0, 0.25, 0.1]
APX = [0.5, 1.0 / 3.0, 7.0 / 17.0]
APY = [0.2, 0.25, 1.0 / 6.0]
APXY = [2.0 / 7.0, 0.4, 1.0 / 3.0]
FSINX = [0.0, 1.0, 0.0]
FSINY = [1.0, 0.0, 0.0]
FSINXY = [1.0, 1.0, 0.0]

NEQ = 3


@dataclass
class CavityConfig:
    """All user-settable solver parameters with the course-project defaults."""

    imax: int = 9                 # grid nodes in x (odd)
    jmax: int = 9                 # grid nodes in y (odd)
    xmin: float = 0.0
    xmax: float = 0.05            # cavity width (m)
    ymin: float = 0.0
    ymax: float = 0.05            # cavity height (m)

    re: float = 10.0              # Reynolds number
    uinf: float = 1.0             # lid velocity (m/s)
    rho: float = 1.0              # density (kg/m^3)
    pinf: float = 0.801333844662  # reference pressure (N/m^2)

    solver_method: str = "coupled"  # 'coupled' or 'fractional_step'
    isgs: int = 1                 # 1 = symmetric Gauss-Seidel, 0 = point Jacobi
    vectorize: bool = False       # NumPy-vectorized kernels where available
    accelerated: bool = False     # Numba wavefront SGS / PPE sweeps

    cfl: float = 0.5
    rkappa: float = 0.5           # artificial-compressibility preconditioning
    cx: float = 0.01              # 4th-order artificial viscosity in x
    cy: float = 0.01              # 4th-order artificial viscosity in y

    imms: int = 0                 # 1 = manufactured-solution verification mode
    nmax: int = 500_000           # maximum pseudo-time iterations
    toler: float = 1e-10          # iterative residual convergence tolerance
    fsmall: float = 1e-20

    # fractional-step pressure solver settings
    p_toler: float = 1e-3
    p_iterations: int = 1000

    verbose: bool = True


@dataclass
class CavityResult:
    """Solution fields and convergence history from one solver run."""

    u: np.ndarray                 # (imax, jmax, 3): [p, u, v]
    x: np.ndarray                 # (imax,) node coordinates
    y: np.ndarray                 # (jmax,)
    config: CavityConfig
    converged: bool
    iterations: int
    elapsed_sec: float
    res_history: np.ndarray       # (n_iter, 3) per-equation residuals
    conv_history: np.ndarray      # (n_iter,) min residual norm
    rL1norm: np.ndarray = field(default=None)   # MMS error norms (imms=1 only)
    rL2norm: np.ndarray = field(default=None)
    rLinfnorm: np.ndarray = field(default=None)
    ummsArray: np.ndarray = field(default=None)

    @property
    def pressure(self):
        return self.u[:, :, 0]

    @property
    def u_velocity(self):
        return self.u[:, :, 1]

    @property
    def v_velocity(self):
        return self.u[:, :, 2]

    def save(self, path):
        np.savez_compressed(
            path, u=self.u, x=self.x, y=self.y, converged=self.converged,
            iterations=self.iterations, elapsed_sec=self.elapsed_sec,
            res_history=self.res_history, conv_history=self.conv_history,
            re=self.config.re, imax=self.config.imax, jmax=self.config.jmax,
        )


def run_cavity(cfg: CavityConfig) -> CavityResult:
    """Run one cavity (or MMS) case to convergence and return the result.

    Unlike ``main_solver.py`` this writes no history/field/restart files;
    everything is returned in memory.
    """
    if cfg.imax % 2 == 0 or cfg.jmax % 2 == 0:
        raise ValueError("imax and jmax must be odd")
    if cfg.solver_method not in ("coupled", "fractional_step"):
        raise ValueError(f"unknown solver_method: {cfg.solver_method}")

    rhoinv = 1.0 / cfg.rho
    rlength = cfg.xmax - cfg.xmin
    rmu = cfg.rho * cfg.uinf * rlength / cfg.re
    vel2ref = cfg.uinf * cfg.uinf
    dx = (cfg.xmax - cfg.xmin) / (cfg.imax - 1)
    dy = (cfg.ymax - cfg.ymin) / (cfg.jmax - 1)

    u = np.full((cfg.imax, cfg.jmax, NEQ), -99.9)
    s = np.full((cfg.imax, cfg.jmax, NEQ), -99.9)
    ummsArray = np.full((cfg.imax, cfg.jmax, NEQ), -99.9)

    res = np.zeros(NEQ)
    ninit, rtime, resinit, ummsArray = initialize(
        0, -99.9, np.zeros(NEQ), 0, NEQ, cfg.uinf, cfg.pinf,
        cfg.xmax, cfg.xmin, cfg.ymax, cfg.ymin, u, s, ummsArray,
        rlength, PHI0, PHIX, PHIY, PHIXY, APX, APY, APXY, FSINX, FSINY, FSINXY)

    u = set_boundary_conditions(u, cfg.uinf, ummsArray, NEQ, cfg.imms, cfg.vectorize)

    s = compute_source_terms(
        s, cfg.imax, cfg.jmax, cfg.imms, cfg.xmax, cfg.xmin, cfg.ymax, cfg.ymin,
        cfg.rho, rmu, rlength, PHI0, PHIX, PHIY, PHIXY, APX, APY, APXY)

    # check_iterative_convergence logs residual lines to a file handle;
    # capture them in memory instead of ./history.txt
    log = io.StringIO()

    if cfg.accelerated:
        fwd_sweep, bwd_sweep = SGS_forward_sweep_acc, SGS_backward_sweep_acc
    else:
        fwd_sweep, bwd_sweep = SGS_forward_sweep, SGS_backward_sweep

    res_history, conv_history = [], []
    converged = False
    n = ninit
    starttime = time.time()

    for n in range(ninit, cfg.nmax):
        dt, dtmin = compute_time_step(
            u, vel2ref, rmu, cfg.rho, dx, dy, cfg.cfl, cfg.rkappa, cfg.vectorize)
        uold = np.copy(u)

        if cfg.solver_method == "coupled":
            if cfg.isgs == 1:
                artviscx, artviscy = Compute_Artificial_Viscosity(
                    u, dx, dy, cfg.cx, cfg.cy, vel2ref, cfg.rkappa, cfg.vectorize)
                u = fwd_sweep(u, uold, dt, s, cfg.rho, rhoinv, dx, dy,
                              cfg.rkappa, rmu, vel2ref, artviscx, artviscy)
                u = set_boundary_conditions(u, cfg.uinf, ummsArray, NEQ, cfg.imms, cfg.vectorize)
                artviscx, artviscy = Compute_Artificial_Viscosity(
                    u, dx, dy, cfg.cx, cfg.cy, vel2ref, cfg.rkappa, cfg.vectorize)
                u = bwd_sweep(u, uold, dt, s, cfg.rho, rhoinv, dx, dy,
                              cfg.rkappa, rmu, vel2ref, artviscx, artviscy)
                u = set_boundary_conditions(u, cfg.uinf, ummsArray, NEQ, cfg.imms, cfg.vectorize)
            else:
                artviscx, artviscy = Compute_Artificial_Viscosity(
                    u, dx, dy, cfg.cx, cfg.cy, vel2ref, cfg.rkappa, cfg.vectorize)
                u = point_Jacobi(u, uold, dt, s, cfg.rho, rhoinv, dx, dy,
                                 cfg.rkappa, rmu, vel2ref, artviscx, artviscy, cfg.vectorize)
                u = set_boundary_conditions(u, cfg.uinf, ummsArray, NEQ, cfg.imms, cfg.vectorize)
        else:  # fractional_step
            u = compute_intermediate_velocity(u, uold, dt, s, cfg.rho, dx, dy, rmu, cfg.vectorize)
            u = set_boundary_conditions(u, cfg.uinf, ummsArray, NEQ, cfg.imms, cfg.vectorize)
            if cfg.accelerated:
                u = solve_PPE_SGS_acc(u, dt, s, dx, dy, cfg.rho,
                                      p_toler=cfg.p_toler, iterations=cfg.p_iterations)
            elif cfg.isgs == 1:
                u = solve_PPE_SGS(u, dt, s, dx, dy, cfg.rho, p_toler=cfg.p_toler,
                                  iterations=cfg.p_iterations, vectorize=cfg.vectorize)
            else:
                u = solve_PPE_Jacobi(u, dt, s, dx, dy, cfg.rho, p_toler=cfg.p_toler,
                                     iterations=cfg.p_iterations, vectorize=cfg.vectorize)
            u = correct_velocity(u, dt, dx, dy, cfg.rho, cfg.vectorize)
            u = set_boundary_conditions(u, cfg.uinf, ummsArray, NEQ, cfg.imms, cfg.vectorize)

        u = pressure_rescaling(
            u, cfg.imms, cfg.xmax, cfg.xmin, cfg.ymax, cfg.ymin, cfg.pinf,
            cfg.imax, cfg.jmax, 0, rlength, PHI0, PHIX, PHIY, PHIXY,
            APX, APY, APXY, FSINX, FSINY, FSINXY)

        rtime += dtmin
        res, resinit, conv = check_iterative_convergence(
            n, res, resinit, ninit, rtime, dtmin, u, uold, dt,
            cfg.imax, cfg.jmax, NEQ, cfg.fsmall, log, cfg.vectorize)

        res_history.append(res.copy())
        conv_history.append(conv)

        if conv < cfg.toler:
            converged = True
            break

    elapsed = time.time() - starttime
    if cfg.verbose:
        state = "converged" if converged else "NOT converged"
        print(f"{state} after {n + 1} iterations in {elapsed:.2f} s "
              f"(final residual {conv_history[-1]:.3e})")

    result = CavityResult(
        u=u,
        x=np.linspace(cfg.xmin, cfg.xmax, cfg.imax),
        y=np.linspace(cfg.ymin, cfg.ymax, cfg.jmax),
        config=cfg,
        converged=converged,
        iterations=n + 1,
        elapsed_sec=elapsed,
        res_history=np.array(res_history),
        conv_history=np.array(conv_history),
        ummsArray=ummsArray if cfg.imms == 1 else None,
    )

    if cfg.imms == 1:
        result.rL1norm, result.rL2norm, result.rLinfnorm = discretization_error_norms(
            cfg.imax, cfg.jmax, NEQ, cfg.imms, u, ummsArray)

    return result


def save_standard_plots(result: CavityResult, outdir):
    """Save the standard contour and residual figures for a run."""
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    X, Y = np.meshgrid(result.x, result.y, indexing="ij")

    for k, (name, title) in enumerate(
            [("pcontour", "p (N/m$^2$)"), ("ucontour", "u (m/s)"), ("vcontour", "v (m/s)")]):
        plt.figure(figsize=(6, 5))
        c = plt.contourf(X, Y, result.u[:, :, k], 20)
        plt.colorbar(c)
        plt.title(title)
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.tight_layout()
        plt.savefig(outdir / f"{name}.png", dpi=200)
        plt.close()

    plt.figure(figsize=(6, 5))
    labels = ["continuity", "x-momentum", "y-momentum"]
    for k in range(NEQ):
        plt.semilogy(result.res_history[:, k], label=labels[k])
    plt.xlabel("iteration")
    plt.ylabel("normalized residual")
    plt.legend(frameon=False)
    plt.title(f"Re={result.config.re:g}, {result.config.imax}x{result.config.jmax}")
    plt.tight_layout()
    plt.savefig(outdir / "residuals.png", dpi=200)
    plt.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="2D lid-driven cavity solver")
    parser.add_argument("--imax", type=int, default=9, help="grid nodes per direction (odd)")
    parser.add_argument("--re", type=float, default=10.0, help="Reynolds number")
    parser.add_argument("--cfl", type=float, default=0.5)
    parser.add_argument("--rkappa", type=float, default=0.5)
    parser.add_argument("--c4", type=float, default=0.01, help="artificial viscosity Cx=Cy")
    parser.add_argument("--toler", type=float, default=1e-10)
    parser.add_argument("--nmax", type=int, default=500_000)
    parser.add_argument("--solver", choices=["coupled", "fractional_step"], default="coupled")
    parser.add_argument("--jacobi", action="store_true", help="point Jacobi instead of SGS")
    parser.add_argument("--vectorize", action="store_true")
    parser.add_argument("--accelerated", action="store_true", help="Numba wavefront sweeps")
    parser.add_argument("--mms", action="store_true", help="manufactured-solution mode")
    parser.add_argument("--outdir", default=None, help="save plots and cavity_result.npz here")
    args = parser.parse_args(argv)

    cfg = CavityConfig(
        imax=args.imax, jmax=args.imax, re=args.re, cfl=args.cfl,
        rkappa=args.rkappa, cx=args.c4, cy=args.c4, toler=args.toler,
        nmax=args.nmax, solver_method=args.solver, isgs=0 if args.jacobi else 1,
        vectorize=args.vectorize, accelerated=args.accelerated,
        imms=1 if args.mms else 0)

    result = run_cavity(cfg)

    if cfg.imms == 1:
        print("MMS L1 norms:  ", result.rL1norm)
        print("MMS L2 norms:  ", result.rL2norm)
        print("MMS Linf norms:", result.rLinfnorm)

    if args.outdir:
        from pathlib import Path
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        result.save(outdir / "cavity_result.npz")
        save_standard_plots(result, outdir)
        print(f"Saved results and plots to {outdir}/")

    return result


if __name__ == "__main__":
    main()
