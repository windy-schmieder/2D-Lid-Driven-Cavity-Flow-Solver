"""Generate the explanatory diagrams used in the README theory sections.

Run from the repository root:  python docs/make_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#1b2430"
ACCENT = "#0b6fa4"
ACCENT2 = "#c1440e"
MUTED = "#7a8794"
FILL = "#eaf3f8"


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT / name}")


# ---------------------------------------------------------------- domain & BCs
def domain_bc():
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=FILL, edgecolor=INK, lw=2.5))

    # lid arrow
    for x in np.linspace(0.12, 0.88, 6):
        ax.add_patch(FancyArrowPatch((x - 0.05, 1.07), (x + 0.05, 1.07),
                                     arrowstyle="-|>", mutation_scale=13,
                                     color=ACCENT2, lw=2))
    ax.text(0.5, 1.15, r"lid:  $u = U_{\rm lid}$,  $v = 0$",
            ha="center", fontsize=12, color=ACCENT2)

    # recirculation sketch
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(0.5 + 0.27 * np.cos(th), 0.62 + 0.24 * np.sin(th),
            color=ACCENT, lw=1.4, alpha=0.75)
    ax.plot(0.5 + 0.15 * np.cos(th), 0.62 + 0.13 * np.sin(th),
            color=ACCENT, lw=1.2, alpha=0.55)
    ax.add_patch(FancyArrowPatch((0.77, 0.66), (0.77, 0.55), arrowstyle="-|>",
                                 mutation_scale=14, color=ACCENT, lw=1.6))
    ax.plot(0.5, 0.62, "o", color=ACCENT, ms=5)
    ax.text(0.53, 0.60, "primary\nvortex", fontsize=9.5, color=ACCENT)

    # corner eddies
    for cx, cy in [(0.07, 0.07), (0.93, 0.07)]:
        ax.plot(cx + 0.045 * np.cos(th), cy + 0.045 * np.sin(th),
                color=MUTED, lw=1.0)
    ax.text(0.5, 0.10, "corner eddies", fontsize=9, color=MUTED, ha="center")

    # wall labels
    ax.text(-0.04, 0.5, r"$u = v = 0$", rotation=90, va="center", ha="center", fontsize=11)
    ax.text(1.04, 0.5, r"$u = v = 0$", rotation=-90, va="center", ha="center", fontsize=11)
    ax.text(0.5, -0.07, r"$u = v = 0$", ha="center", fontsize=11)

    # singular corners
    for cx in (0.0, 1.0):
        ax.add_patch(Circle((cx, 1.0), 0.035, facecolor="none",
                            edgecolor=ACCENT2, lw=1.8, ls="--"))
    ax.text(0.5, 0.93, "singular corners", ha="center", fontsize=9,
            color=ACCENT2, style="italic")

    ax.annotate("", xy=(0.0, -0.16), xytext=(1.0, -0.16),
                arrowprops=dict(arrowstyle="<->", color=INK))
    ax.text(0.5, -0.20, r"$L$", ha="center", fontsize=12)

    ax.set_xlim(-0.22, 1.22)
    ax.set_ylim(-0.28, 1.26)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Lid-driven cavity: domain and boundary conditions",
                 fontsize=13, pad=14)
    save(fig, "domain_bc.png")


# ------------------------------------------------------------------- stencils
def stencils():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9))

    # --- 5-point central stencil
    ax = axes[0]
    for i in range(5):
        for j in range(5):
            ax.plot(i, j, "o", ms=5, color="#c9d4dd")
    pts = {(2, 2): r"$i,j$", (3, 2): r"$i{+}1,j$", (1, 2): r"$i{-}1,j$",
           (2, 3): r"$i,j{+}1$", (2, 1): r"$i,j{-}1$"}
    for (i, j), lab in pts.items():
        col = ACCENT2 if (i, j) == (2, 2) else ACCENT
        ax.plot(i, j, "o", ms=13, color=col)
        ax.text(i, j - 0.42, lab, ha="center", fontsize=9.5, color=INK)
    for a, b in [((2, 2), (3, 2)), ((2, 2), (1, 2)), ((2, 2), (2, 3)), ((2, 2), (2, 1))]:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=ACCENT, lw=1.6, zorder=0)
    ax.annotate("", xy=(3, 3.55), xytext=(2, 3.55),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.3))
    ax.text(2.5, 3.65, r"$\Delta x$", ha="center", fontsize=11, color=MUTED)
    ax.annotate("", xy=(3.55, 3), xytext=(3.55, 2),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.3))
    ax.text(3.68, 2.5, r"$\Delta y$", va="center", fontsize=11, color=MUTED)
    ax.set_title("2nd-order central stencil\n"
                 r"convection, pressure gradient, diffusion", fontsize=11.5)

    # --- 5-point 4th-difference stencil
    ax = axes[1]
    for i in range(7):
        for j in range(5):
            ax.plot(i, j, "o", ms=5, color="#c9d4dd")
    coeffs = {(1, 2): "+1", (2, 2): "-4", (3, 2): "+6", (4, 2): "-4", (5, 2): "+1"}
    for (i, j), c in coeffs.items():
        col = ACCENT2 if c == "+6" else ACCENT
        ax.plot(i, j, "o", ms=13, color=col)
        ax.text(i, j + 0.35, c, ha="center", fontsize=10.5, color=col, weight="bold")
    ax.plot([1, 5], [2, 2], color=ACCENT, lw=1.6, zorder=0)
    ax.text(3, 0.75, r"$\dfrac{\partial^4 p}{\partial x^4}\approx$"
                     r"$\dfrac{p_{i-2}-4p_{i-1}+6p_{i}-4p_{i+1}+p_{i+2}}{\Delta x^4}$",
            ha="center", fontsize=11)
    ax.set_title("4th-difference dissipation stencil\n"
                 "pressure smoothing (odd-even decoupling)", fontsize=11.5)

    for ax in axes:
        ax.set_aspect("equal")
        ax.axis("off")
    fig.tight_layout()
    save(fig, "stencils.png")


# ------------------------------------------------------------- SGS wavefronts
def wavefront():
    n = 9
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.9))

    ax = axes[0]
    order = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            order[i, j] = j * n + i
    im = ax.imshow(order.T, origin="lower", cmap="viridis")
    ax.set_title("Lexicographic SGS sweep\n(strictly sequential)", fontsize=11.5)
    fig.colorbar(im, ax=ax, fraction=0.046, label="update order")

    ax = axes[1]
    diag = np.add.outer(np.arange(n), np.arange(n)).astype(float)
    im = ax.imshow(diag.T, origin="lower", cmap="viridis")
    for d in range(0, 2 * n - 1, 2):
        xs = [i for i in range(n) if 0 <= d - i < n]
        ys = [d - i for i in xs]
        ax.plot(xs, ys, color="white", lw=1.1, alpha=0.75)
    ax.set_title("Diagonal wavefront ordering\n"
                 r"nodes with equal $i+j$ update in parallel", fontsize=11.5)
    fig.colorbar(im, ax=ax, fraction=0.046, label=r"wavefront index $i+j$")

    for ax in axes:
        ax.set_xlabel("i")
        ax.set_ylabel("j")
    fig.tight_layout()
    save(fig, "sgs_wavefront.png")


# ------------------------------------------------------------------ flowchart
def flowchart():
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 8.4))

    def box(ax, y, text, color=ACCENT, w=0.88, h=0.055, fs=9.6):
        ax.add_patch(FancyBboxPatch((0.5 - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.012",
                                    facecolor=FILL, edgecolor=color, lw=1.6))
        ax.text(0.5, y, text, ha="center", va="center", fontsize=fs, color=INK)

    def arrow(ax, y0, y1):
        ax.add_patch(FancyArrowPatch((0.5, y0), (0.5, y1), arrowstyle="-|>",
                                     mutation_scale=13, color=MUTED, lw=1.4))

    # coupled
    ax = axes[0]
    steps = [
        r"initialize $\mathbf{q}^0$, apply BCs",
        r"local time step  $\Delta t_{i,j}$  (CFL)",
        r"artificial viscosity  $\epsilon_x,\ \epsilon_y$",
        r"SGS forward sweep  (lower-left $\to$ upper-right)",
        "apply boundary conditions",
        "recompute artificial viscosity",
        r"SGS backward sweep  (reverse order)",
        "apply boundary conditions",
        "rescale pressure to reference point",
        r"residuals  $R_k = \max |\Delta q_k / \Delta t|$",
    ]
    ys = np.linspace(0.93, 0.14, len(steps))
    for y, s in zip(ys, steps):
        box(ax, y, s)
    for y0, y1 in zip(ys[:-1], ys[1:]):
        arrow(ax, y0 - 0.029, y1 + 0.029)
    ax.add_patch(FancyArrowPatch((0.06, ys[-1]), (0.06, ys[1]),
                                 connectionstyle="arc3,rad=0.32",
                                 arrowstyle="-|>", mutation_scale=13,
                                 color=ACCENT2, lw=1.5))
    ax.text(0.015, 0.55, r"until $\max_k R_k < $ tol", rotation=90,
            va="center", fontsize=9.5, color=ACCENT2)
    ax.set_title("Coupled artificial-compressibility solver", fontsize=12.5, pad=10)

    # fractional step
    ax = axes[1]
    steps = [
        r"initialize $\mathbf{u}^0$, apply BCs",
        r"local time step  $\Delta t_{i,j}$",
        r"predictor: $\mathbf{u}^* = \mathbf{u}^n + \Delta t\,(-\mathbf{u}\!\cdot\!\nabla\mathbf{u} + \nu\nabla^2\mathbf{u})$",
        r"apply BCs to $\mathbf{u}^*$",
        r"solve PPE  $\nabla^2 p = \frac{\rho}{\Delta t}\nabla\!\cdot\!\mathbf{u}^*$",
        r"(Jacobi / SGS / Numba wavefront)",
        r"corrector: $\mathbf{u}^{n+1} = \mathbf{u}^* - \frac{\Delta t}{\rho}\nabla p$",
        "apply boundary conditions",
        "rescale pressure",
        "residuals and convergence check",
    ]
    ys = np.linspace(0.93, 0.14, len(steps))
    for y, s in zip(ys, steps):
        box(ax, y, s, color=ACCENT2 if "PPE" in s else ACCENT, fs=9.0)
    for y0, y1 in zip(ys[:-1], ys[1:]):
        arrow(ax, y0 - 0.029, y1 + 0.029)
    ax.add_patch(FancyArrowPatch((0.06, ys[-1]), (0.06, ys[1]),
                                 connectionstyle="arc3,rad=0.32",
                                 arrowstyle="-|>", mutation_scale=13,
                                 color=ACCENT2, lw=1.5))
    ax.text(0.015, 0.55, "until converged", rotation=90,
            va="center", fontsize=9.5, color=ACCENT2)
    ax.set_title("Fractional-step (pressure projection) solver", fontsize=12.5, pad=10)

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0.05, 1.0)
        ax.axis("off")
    fig.tight_layout()
    save(fig, "algorithm_flowchart.png")


# --------------------------------------------------------------- MMS ordering
def mms_order():
    import csv
    path = (Path(__file__).parent.parent / "start-code" /
            "Phase III Code Verification Checkpoints" / "error_table_partial.csv")
    rows = list(csv.DictReader(path.open()))
    data = {}
    for r in rows:
        data.setdefault(r["equation"], []).append(
            (float(r["h"]), float(r["L1"]), float(r["L2"]), float(r["Linf"])))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    labels = {"p": "pressure", "u": "u-velocity", "v": "v-velocity"}
    colors = {"p": ACCENT, "u": ACCENT2, "v": "#3b7a57"}

    ax = axes[0]
    for eq, vals in data.items():
        vals.sort()
        h = np.array([v[0] for v in vals])
        l2 = np.array([v[2] for v in vals])
        ax.loglog(h, l2, "o-", color=colors[eq], lw=1.8, label=labels[eq])
    h_ref = np.array([4.0, 16.0])
    ax.loglog(h_ref, 3e-5 * (h_ref / 4.0) ** 2, "k--", lw=1.3,
              label=r"slope 2 (formal order)")
    ax.set_xlabel(r"normalized mesh spacing $h$")
    ax.set_ylabel(r"$L_2$ discretization error")
    ax.set_title("MMS grid convergence", fontsize=12)
    ax.legend(frameon=False, fontsize=9.5)
    ax.grid(alpha=0.25, which="both")

    ax = axes[1]
    order_path = (Path(__file__).parent.parent / "start-code" /
                  "Phase III Code Verification Checkpoints" /
                  "observed_order_L2_partial.csv")
    orows = list(csv.DictReader(order_path.open()))
    xs, ys, cs = [], [], []
    ticks = []
    for i, r in enumerate(orows):
        xs.append(i)
        ys.append(float(r["observed_order_L2"]))
        cs.append(colors[r["equation"]])
        ticks.append(f"{labels[r['equation']]}\n{int(float(r['coarse_h']))}→"
                     f"{int(float(r['fine_h']))}")
    ax.bar(xs, ys, color=cs, alpha=0.85)
    ax.axhline(2.0, color=INK, ls="--", lw=1.4)
    ax.text(len(xs) - 0.4, 2.06, "formal order 2", ha="right", fontsize=9.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(ticks, fontsize=8)
    ax.set_ylabel(r"observed order $\hat{p}$")
    ax.set_title("Observed order of accuracy", fontsize=12)
    ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    save(fig, "mms_order.png")


# ----------------------------------------------------------- PINN architecture
def pinn_arch():
    fig, ax = plt.subplots(figsize=(11.2, 5.0))

    layers = [2, 6, 6, 6, 2]
    xs = np.linspace(0.08, 0.52, len(layers))
    pos = []
    for x, n in zip(xs, layers):
        ys = np.linspace(0.5 - 0.055 * (n - 1), 0.5 + 0.055 * (n - 1), n)
        pos.append([(x, y) for y in ys])

    for a, b in zip(pos[:-1], pos[1:]):
        for p in a:
            for q in b:
                ax.plot([p[0], q[0]], [p[1], q[1]], color="#d5dee6", lw=0.5, zorder=0)
    for li, layer in enumerate(pos):
        col = ACCENT2 if li in (0, len(pos) - 1) else ACCENT
        for (x, y) in layer:
            ax.plot(x, y, "o", ms=11, color=col, zorder=3)

    ax.text(xs[0], 0.68, r"$(x,y)$", ha="center", fontsize=12, color=ACCENT2)
    ax.text(np.mean(xs[1:-1]), 0.855, r"4 hidden layers $\times$ 64 units, $\tanh$",
            ha="center", fontsize=11, color=ACCENT)
    ax.text(xs[-1], 0.68, r"$(\psi,\ p)$", ha="center", fontsize=12, color=ACCENT2)

    # derived quantities
    ax.add_patch(FancyArrowPatch((0.55, 0.5), (0.62, 0.5), arrowstyle="-|>",
                                 mutation_scale=15, color=MUTED, lw=1.6))
    ax.text(0.585, 0.545, "autograd", ha="center", fontsize=9, color=MUTED)

    boxes = [
        (0.80, r"$u = \dfrac{\partial \psi}{\partial y},\qquad v = -\dfrac{\partial \psi}{\partial x}$"
               "\n" r"$\Rightarrow\ \nabla\!\cdot\!\mathbf{u} \equiv 0$  (exact)", ACCENT2),
        (0.56, r"$\mathcal{L}_{\rm PDE} = \left\| u u_x + v u_y + p_x - \frac{1}{Re}\nabla^2 u \right\|^2$"
               "\n" r"$\qquad\ \ +\ \left\| u v_x + v v_y + p_y - \frac{1}{Re}\nabla^2 v \right\|^2$", ACCENT),
        (0.28, r"$\mathcal{L}_{\rm BC} = \left\| \mathbf{u} - \mathbf{u}_{\rm wall} \right\|^2_{\partial\Omega}$"
               "\n" r"$\mathcal{L} = \mathcal{L}_{\rm PDE} + \lambda\,\mathcal{L}_{\rm BC} + \mathcal{L}_{p_{\rm ref}}$", ACCENT),
    ]
    for y, txt, col in boxes:
        ax.add_patch(FancyBboxPatch((0.635, y - 0.085), 0.35, 0.17,
                                    boxstyle="round,pad=0.012",
                                    facecolor=FILL, edgecolor=col, lw=1.5))
        ax.text(0.81, y, txt, ha="center", va="center", fontsize=9.6)

    ax.set_xlim(0, 1)
    ax.set_ylim(0.12, 0.93)
    ax.axis("off")
    ax.set_title("PINN surrogate: streamfunction formulation and loss",
                 fontsize=12.5, pad=8)
    save(fig, "pinn_architecture.png")


if __name__ == "__main__":
    domain_bc()
    stencils()
    wavefront()
    flowchart()
    mms_order()
    pinn_arch()
