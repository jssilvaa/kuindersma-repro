# scripts/test_costs_preview.py
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.lipm import discretize_double_integrator
from src.planning.segments import build_segments, p_ref_in_segment, segment_at_time
from src.control.constraints_preview import build_polygon_preview_constraints

from src.control.costs_preview import (
    build_zmp_affine_terms,
    build_cost_zmp_preview,
    build_cost_du_smoothing,
    combine_quadratic_costs,
)


def finite_diff_grad(fun, U, eps=1e-6):
    g = np.zeros_like(U)
    f0 = fun(U)
    for i in range(U.size):
        Up = U.copy()
        Up[i] += eps
        g[i] = (fun(Up) - f0) / eps
    return g


def main():
    np.random.seed(0)

    dt = 0.01
    zc = 0.9
    g = 9.81
    alpha = zc / g

    segs, T = build_segments(n_steps=2, ds=0.2, ss=0.2, step_length=0.08, step_width=0.24, x0=0.0)

    Ad, Bd = discretize_double_integrator(dt)
    N = 20
    x0 = np.array([0.02, 0.00, 0.0, 0.0])
    foot_half = np.array([0.09, 0.045])

    # Build polygon preview constraints just to get meta (R_prev_U and r_prev_aff)
    A_poly, l_poly, u_poly, meta = build_polygon_preview_constraints(
        segs=segs, t0=0.0, dt=dt, N=N,
        Ad=Ad, Bd=Bd, x0=x0, zc=zc, g=g,
        foot_half_sizes=foot_half,
        m_target=8,
    )

    # You stored these in meta:
    R_prev_U = meta["R_prev_U"] if "R_prev_U" in meta else None
    if R_prev_U is None:
        # If you didn't store it, reconstruct from stacks you said you added.
        # Preferred: store R_prev_U explicitly in meta.
        # For now we can recover from P_U and alpha if you stored P_U;
        # otherwise, raise.
        raise RuntimeError("Please store R_prev_U in meta['R_prev_U'] for this test.")

    r_prev_aff = meta["r_prev_aff"]

    # Build ZMP affine mapping: P(U) = p_aff + P_U U
    P_U, p_aff = build_zmp_affine_terms(R_prev_U=R_prev_U, r_prev_aff=r_prev_aff, alpha=alpha)

    # Build Pref stack: Pref = [pref(t1); ...; pref(tN)] where ti = i*dt
    Pref = np.zeros(2 * N)
    for i in range(1, N + 1):
        ti = float(i * dt)
        seg = segment_at_time(segs, ti)
        pref_i = p_ref_in_segment(seg, ti)  # (2,)
        Pref[2 * (i - 1): 2 * (i - 1) + 2] = pref_i

    Wp = np.eye(2) * 50.0
    Wdu = np.eye(2) * 0.1
    u_prev = np.array([0.1, -0.05])

    Pz, qz = build_cost_zmp_preview(P_U=P_U, p_aff=p_aff, P_ref=Pref, Wp=Wp)
    Pdu, qdu, D, d0 = build_cost_du_smoothing(N=N, Wdu=Wdu, u_prev=u_prev)

    P, q = combine_quadratic_costs((Pz, qz), (Pdu, qdu))

    # Basic checks
    assert P.shape == (2 * N, 2 * N)
    assert q.shape == (2 * N,)
    # Symmetry check (numerical)
    asym = (P - P.T).nnz
    assert asym == 0, "P should be symmetric after symmetrization"

    # Pick random U
    U = 0.2 * np.random.randn(2 * N)

    # Objective function
    def J(Uv):
        Uv = Uv.reshape(-1)
        # 0.5 U^T P U + q^T U
        return 0.5 * (Uv @ (P @ Uv)) + (q @ Uv)

    # Analytic gradient: P U + q
    grad_ana = (P @ U + q).reshape(-1)
    grad_num = finite_diff_grad(J, U, eps=1e-6)

    err = np.max(np.abs(grad_ana - grad_num))
    print("max |grad_ana - grad_num| =", err)
    assert err < 5e-5, f"Gradient check failed: {err}"

    print("OK: Block 3 cost matrices verified.")


if __name__ == "__main__":
    main()
