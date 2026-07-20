"""Physics-informed neural network (PINN) for the steady lid-driven cavity.

A natural extension of the course project: instead of discretizing the
incompressible Navier-Stokes equations on a mesh, a small neural network is
trained so that its outputs satisfy the PDEs at random collocation points and
the cavity boundary conditions on the walls. The trained network is a
continuous, mesh-free surrogate of the flow field.

Formulation (nondimensional unit cavity, lid velocity U = 1):

- The network maps (x, y) -> (psi, p), with velocities recovered from the
  streamfunction, u = d(psi)/dy and v = -d(psi)/dx, so continuity is
  satisfied exactly by construction.
- The loss is the mean-squared x/y-momentum residual at interior collocation
  points plus the mean-squared velocity error on the boundaries.
- Training: Adam warm-up followed by L-BFGS refinement.

Validation compares the PINN centerline profiles against the classic
Ghia, Ghia & Shin (1982) benchmark data and, when available, against this
repository's own finite-difference solver (``start-code/cavity_solver.py``).

Usage::

    python pinn_cavity.py                       # Re=100 demo, saves to plots/
    python pinn_cavity.py --re 100 --adam-steps 5000 --lbfgs-steps 1000
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Ghia, Ghia & Shin (1982) centerline benchmark data, Re = 100
GHIA_Y = np.array([
    1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5000,
    0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000])
GHIA_U_RE100 = np.array([
    1.00000, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332, -0.13641,
    -0.20581, -0.21090, -0.15662, -0.10150, -0.06434, -0.04775, -0.04192,
    -0.03717, 0.00000])
GHIA_X = np.array([
    1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 0.5000,
    0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000])
GHIA_V_RE100 = np.array([
    0.00000, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.22445,
    -0.24533, 0.05454, 0.17527, 0.17507, 0.16077, 0.12317, 0.10890,
    0.10091, 0.09233, 0.00000])


class CavityPINN(nn.Module):
    """MLP mapping (x, y) -> (streamfunction psi, pressure p)."""

    def __init__(self, hidden=64, layers=4):
        super().__init__()
        sizes = [2] + [hidden] * layers + [2]
        mods = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            mods.append(nn.Linear(a, b))
            mods.append(nn.Tanh())
        mods.pop()  # no activation on the output layer
        self.net = nn.Sequential(*mods)

    def forward(self, xy):
        return self.net(xy)

    def velocity_pressure(self, xy):
        """Return (u, v, p) with u, v from streamfunction derivatives."""
        xy = xy.requires_grad_(True)
        out = self.net(xy)
        psi, p = out[:, 0:1], out[:, 1:2]
        dpsi = torch.autograd.grad(psi, xy, torch.ones_like(psi), create_graph=True)[0]
        u = dpsi[:, 1:2]         # d(psi)/dy
        v = -dpsi[:, 0:1]        # -d(psi)/dx
        return u, v, p


def _grad(f, xy):
    return torch.autograd.grad(f, xy, torch.ones_like(f), create_graph=True)[0]


def momentum_residuals(model, xy, re):
    """Steady incompressible NS momentum residuals at collocation points."""
    xy = xy.requires_grad_(True)
    out = model(xy)
    psi, p = out[:, 0:1], out[:, 1:2]

    dpsi = _grad(psi, xy)
    u, v = dpsi[:, 1:2], -dpsi[:, 0:1]

    du = _grad(u, xy)
    dv = _grad(v, xy)
    dp = _grad(p, xy)
    u_x, u_y = du[:, 0:1], du[:, 1:2]
    v_x, v_y = dv[:, 0:1], dv[:, 1:2]

    u_xx = _grad(u_x, xy)[:, 0:1]
    u_yy = _grad(u_y, xy)[:, 1:2]
    v_xx = _grad(v_x, xy)[:, 0:1]
    v_yy = _grad(v_y, xy)[:, 1:2]

    rx = u * u_x + v * u_y + dp[:, 0:1] - (u_xx + u_yy) / re
    ry = u * v_x + v * v_y + dp[:, 1:2] - (v_xx + v_yy) / re
    return rx, ry


def sample_collocation(n, device):
    """Uniform interior collocation points. (Excluding neighborhoods of the
    singular lid corners was tried and validated worse across seeds.)"""
    return torch.rand(n, 2, device=device)


def boundary_points(n_side, device):
    """Wall points with target (u, v). Lid corners are excluded from the lid
    segment so the no-slip walls own them (matches the FD solver convention)."""
    t = torch.linspace(0.0, 1.0, n_side, device=device).unsqueeze(1)
    zero, one = torch.zeros_like(t), torch.ones_like(t)
    interior = t[1:-1]
    z_in = torch.zeros_like(interior)

    lid = torch.cat([interior, torch.ones_like(interior)], dim=1)
    bottom = torch.cat([t, zero], dim=1)
    left = torch.cat([zero, t], dim=1)
    right = torch.cat([one, t], dim=1)

    xy = torch.cat([lid, bottom, left, right], dim=0)
    u_target = torch.cat([torch.ones_like(z_in), zero, zero, zero], dim=0)
    v_target = torch.zeros_like(u_target)
    return xy, u_target, v_target


def train(re=100.0, n_collocation=2500, n_side=101, hidden=64, layers=4,
          adam_steps=5000, lbfgs_steps=1000, bc_weight=10.0, seed=0,
          device="cpu", log_every=500, resample_every=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device)

    model = CavityPINN(hidden=hidden, layers=layers).to(device)
    xy_f = sample_collocation(n_collocation, device)
    xy_b, u_b, v_b = boundary_points(n_side, device)
    # anchor the pressure level at the cavity center
    xy_ref = torch.tensor([[0.5, 0.5]], device=device)

    history = []

    def loss_fn():
        rx, ry = momentum_residuals(model, xy_f, re)
        loss_pde = (rx ** 2).mean() + (ry ** 2).mean()
        u, v, p_ref = model.velocity_pressure(torch.cat([xy_b, xy_ref], dim=0))
        n_b = xy_b.shape[0]
        loss_bc = ((u[:n_b] - u_b) ** 2).mean() + ((v[:n_b] - v_b) ** 2).mean()
        loss_p = (p_ref[n_b:] ** 2).mean()
        return loss_pde + bc_weight * loss_bc + loss_p, loss_pde, loss_bc

    t0 = time.time()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=adam_steps)
    for step in range(adam_steps):
        if resample_every and step > 0 and step % resample_every == 0:
            xy_f = sample_collocation(n_collocation, device)
        opt.zero_grad()
        loss, loss_pde, loss_bc = loss_fn()
        loss.backward()
        opt.step()
        sched.step()
        history.append([step, loss.item(), loss_pde.item(), loss_bc.item()])
        if step % log_every == 0:
            print(f"Adam {step:5d}  total {loss.item():.3e}  "
                  f"pde {loss_pde.item():.3e}  bc {loss_bc.item():.3e}")

    if lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=lbfgs_steps,
                                  history_size=50, tolerance_grad=1e-12,
                                  tolerance_change=1e-14,
                                  line_search_fn="strong_wolfe")

        def closure():
            lbfgs.zero_grad()
            loss, loss_pde, loss_bc = loss_fn()
            loss.backward()
            history.append([len(history), loss.item(), loss_pde.item(), loss_bc.item()])
            return loss

        print("L-BFGS refinement...")
        lbfgs.step(closure)

    loss, loss_pde, loss_bc = loss_fn()
    print(f"Final: total {loss.item():.3e}  pde {loss_pde.item():.3e}  "
          f"bc {loss_bc.item():.3e}  ({time.time() - t0:.1f} s)")
    return model, np.array(history)


def evaluate_grid(model, n=101, device="cpu"):
    """Evaluate (u, v, p) on an n x n uniform grid; returns numpy arrays."""
    lin = torch.linspace(0.0, 1.0, n, device=device)
    X, Y = torch.meshgrid(lin, lin, indexing="ij")
    xy = torch.stack([X.flatten(), Y.flatten()], dim=1)
    u, v, p = model.velocity_pressure(xy)
    shape = (n, n)
    return (X.detach().cpu().numpy(), Y.detach().cpu().numpy(),
            u.detach().cpu().numpy().reshape(shape),
            v.detach().cpu().numpy().reshape(shape),
            p.detach().cpu().numpy().reshape(shape))


def load_solver_reference(path):
    """Load a cavity_solver.py result npz and nondimensionalize it."""
    d = np.load(path)
    u = d["u"]
    x = d["x"] / d["x"].max()
    y = d["y"] / d["y"].max()
    return x, y, u[:, :, 1], u[:, :, 2]


def make_plots(model, history, re, outdir, solver_npz=None, device="cpu"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    X, Y, U, V, P = evaluate_grid(model, n=129, device=device)
    n_mid = (U.shape[0] - 1) // 2

    for field, name, title in [(U, "pinn_ucontour", "PINN u/U"),
                               (V, "pinn_vcontour", "PINN v/U"),
                               (P, "pinn_pcontour", "PINN pressure")]:
        plt.figure(figsize=(5.5, 4.8))
        c = plt.contourf(X, Y, field, 20)
        plt.colorbar(c)
        plt.title(f"{title}, Re={re:g}")
        plt.xlabel("x/L")
        plt.ylabel("y/L")
        plt.tight_layout()
        plt.savefig(outdir / f"{name}.png", dpi=200)
        plt.close()

    solver = load_solver_reference(solver_npz) if solver_npz else None

    # vertical centerline u
    plt.figure(figsize=(5.8, 5.0))
    plt.plot(U[n_mid, :], Y[n_mid, :], "-k", lw=2, label="PINN")
    if solver is not None:
        sx, sy, su, sv = solver
        i_mid = (su.shape[0] - 1) // 2
        plt.plot(su[i_mid, :], sy, "--b", lw=2, label="FD solver 33x33")
    if abs(re - 100.0) < 1e-9:
        plt.scatter(GHIA_U_RE100, GHIA_Y, marker="^", s=40, color="tab:red",
                    zorder=3, label="Ghia et al. (1982)")
    plt.xlabel("u/U")
    plt.ylabel("y/L")
    plt.title(f"Vertical centerline u, Re={re:g}")
    plt.legend(frameon=False)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(outdir / "pinn_vertical_centerline_u.png", dpi=200)
    plt.close()

    # horizontal centerline v
    plt.figure(figsize=(5.8, 5.0))
    plt.plot(X[:, n_mid], V[:, n_mid], "-k", lw=2, label="PINN")
    if solver is not None:
        sx, sy, su, sv = solver
        j_mid = (sv.shape[1] - 1) // 2
        plt.plot(sx, sv[:, j_mid], "--b", lw=2, label="FD solver 33x33")
    if abs(re - 100.0) < 1e-9:
        plt.scatter(GHIA_X, GHIA_V_RE100, marker="^", s=40, color="tab:red",
                    zorder=3, label="Ghia et al. (1982)")
    plt.xlabel("x/L")
    plt.ylabel("v/U")
    plt.title(f"Horizontal centerline v, Re={re:g}")
    plt.legend(frameon=False)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(outdir / "pinn_horizontal_centerline_v.png", dpi=200)
    plt.close()

    # loss history
    plt.figure(figsize=(5.8, 4.4))
    plt.semilogy(history[:, 1], label="total")
    plt.semilogy(history[:, 2], label="momentum residual")
    plt.semilogy(history[:, 3], label="boundary")
    plt.xlabel("optimizer step")
    plt.ylabel("loss")
    plt.title("PINN training history")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outdir / "pinn_loss_history.png", dpi=200)
    plt.close()

    # accuracy metrics against Ghia benchmark
    metrics = {}
    if abs(re - 100.0) < 1e-9:
        lin = Y[n_mid, :]
        u_interp = np.interp(GHIA_Y, lin, U[n_mid, :])
        v_interp = np.interp(GHIA_X, X[:, n_mid], V[:, n_mid])
        metrics["rmse_u_vs_ghia"] = float(np.sqrt(np.mean((u_interp - GHIA_U_RE100) ** 2)))
        metrics["rmse_v_vs_ghia"] = float(np.sqrt(np.mean((v_interp - GHIA_V_RE100) ** 2)))
        print(f"Centerline RMSE vs Ghia Re=100:  u {metrics['rmse_u_vs_ghia']:.4f}  "
              f"v {metrics['rmse_v_vs_ghia']:.4f}")
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description="PINN surrogate for the lid-driven cavity")
    parser.add_argument("--re", type=float, default=100.0)
    parser.add_argument("--adam-steps", type=int, default=5000)
    parser.add_argument("--lbfgs-steps", type=int, default=1000)
    parser.add_argument("--n-collocation", type=int, default=2500)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bc-weight", type=float, default=10.0)
    parser.add_argument("--resample-every", type=int, default=0,
                        help="resample collocation points every N Adam steps (0 = fixed set)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--outdir", default="plots")
    parser.add_argument("--solver-npz",
                        default="solver_reference_Re100_33x33/cavity_result.npz",
                        help="optional cavity_solver.py npz for comparison")
    args = parser.parse_args(argv)

    model, history = train(
        re=args.re, n_collocation=args.n_collocation, hidden=args.hidden,
        layers=args.layers, adam_steps=args.adam_steps,
        lbfgs_steps=args.lbfgs_steps, bc_weight=args.bc_weight,
        resample_every=args.resample_every, seed=args.seed, device=args.device)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), outdir / f"pinn_cavity_Re{args.re:g}.pt")

    solver_npz = args.solver_npz if args.solver_npz and Path(args.solver_npz).exists() else None
    if args.solver_npz and solver_npz is None:
        print(f"note: solver reference {args.solver_npz} not found, skipping FD comparison")
    make_plots(model, history, args.re, outdir, solver_npz=solver_npz, device=args.device)
    print(f"Saved model and plots to {outdir}/")


if __name__ == "__main__":
    main()
