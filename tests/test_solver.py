"""Fast regression checks for the cavity solver.

Run from the repository root with:  python -m pytest tests/ -v
"""

import numpy as np
import pytest

from cavity_solver import CavityConfig, run_cavity


def test_serial_vs_vectorized_agree():
    """The NumPy-vectorized kernels must reproduce the serial loops."""
    common = dict(imax=9, jmax=9, re=10.0, isgs=0, nmax=200, toler=-1.0,
                  verbose=False)
    serial = run_cavity(CavityConfig(vectorize=False, **common))
    vectorized = run_cavity(CavityConfig(vectorize=True, **common))

    assert np.max(np.abs(serial.u - vectorized.u)) < 1e-10


def test_cavity_converges_with_physical_boundaries():
    """Baseline 9x9 Re=10 SGS case converges and satisfies the cavity BCs."""
    result = run_cavity(CavityConfig(imax=9, jmax=9, re=10.0, toler=1e-8,
                                     verbose=False))

    assert result.converged
    u = result.u
    # moving lid: u = uinf on the top boundary (interior of the lid)
    assert np.allclose(u[1:-1, -1, 1], result.config.uinf)
    # no-slip on bottom and side walls (top corner nodes belong to the lid)
    assert np.allclose(u[:, 0, 1], 0.0)
    assert np.allclose(u[0, :-1, 1], 0.0)
    assert np.allclose(u[-1, :-1, 1], 0.0)
    # v = 0 on all walls
    assert np.allclose(u[:, 0, 2], 0.0)
    assert np.allclose(u[:, -1, 2], 0.0)
    assert np.allclose(u[0, :, 2], 0.0)
    assert np.allclose(u[-1, :, 2], 0.0)
    # fields are finite everywhere
    assert np.all(np.isfinite(u))


def test_mms_observed_order_of_accuracy():
    """MMS L2 errors between 17x17 and 33x33 reproduce the recorded course
    verification data (Phase III Code Verification Checkpoints).

    The recorded observed orders there are p: 2.30, u: 3.08, v: 2.34 — the
    scheme is formally 2nd order and u superconverges on these meshes.
    """
    # L2 norms from error_table_partial.csv, columns (p, u, v)
    recorded_L2 = {
        17: np.array([3.6149136496873647e-04, 8.1239098904409960e-05, 3.0683304385104470e-05]),
        33: np.array([7.3627213147708080e-05, 9.5987300824144340e-06, 6.0660992732574690e-06]),
    }

    norms = {}
    for nodes in (17, 33):
        result = run_cavity(CavityConfig(imax=nodes, jmax=nodes, re=10.0,
                                         imms=1, toler=1e-8, accelerated=True,
                                         verbose=False))
        assert result.converged, f"{nodes}x{nodes} MMS case did not converge"
        norms[nodes] = result.rL2norm
        assert np.allclose(norms[nodes], recorded_L2[nodes], rtol=1e-2), (
            f"{nodes}x{nodes} L2 norms {norms[nodes]} differ from recorded "
            f"checkpoint values {recorded_L2[nodes]}")

    observed_order = np.log(norms[17] / norms[33]) / np.log(2.0)
    for k, name in enumerate(["pressure", "u", "v"]):
        assert 1.5 < observed_order[k] < 3.5, (
            f"{name}: observed order {observed_order[k]:.2f} outside 2nd-order range")


def test_fractional_step_runs():
    """The pressure-based fractional-step path advances without blowing up."""
    result = run_cavity(CavityConfig(imax=9, jmax=9, re=10.0,
                                     solver_method="fractional_step",
                                     vectorize=True, nmax=50, toler=-1.0,
                                     p_toler=1e-3, p_iterations=200,
                                     verbose=False))
    assert np.all(np.isfinite(result.u))
    assert np.allclose(result.u[1:-1, -1, 1], result.config.uinf)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
