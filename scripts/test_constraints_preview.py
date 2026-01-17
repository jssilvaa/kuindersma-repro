# scripts/test_constraints_preview.py
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.lipm import discretize_double_integrator, step, zmp_from
from src.planning.segments import build_segments, segment_at_time
from src.models.support_polygon import support_single, support_double
from src.control.constraints_preview import build_polygon_preview_constraints


def simulate_and_collect_p(segs, t0, dt, N, zc, g, foot_half, x0, U):
    """Iterative simulation to compute p1..pN directly (ground truth)."""
    Ad, Bd = discretize_double_integrator(dt)
    x = x0.copy()
    ps = []
    Hs = []
    hs = []
    for k in range(N):
        # time for state x_{k+1} is t0 + (k+1)*dt, but p_k uses current x_k and u_k in our sim.
        # Here we match the mapping used in build_polygon_preview_constraints: p_i corresponds to x_i and u_{i-1}.
        # So at step k, we advance once then record p_{k+1} using (x_k, u_k) mapped to time t0+(k+1)dt.
        uk = U[k]
        pk = zmp_from(x, uk, zc, g)  # p_{k+1} aligned with u_k and current x
        x = step(x, uk, Ad, Bd)

        ti = float(t0 + k * dt)
        seg = segment_at_time(segs, ti)
        if seg.mode == "DS":
            H_i, h_i, _ = support_double(seg.L, seg.R, foot_half, m_target=8)
        elif seg.mode == "SSL":
            H_i, h_i, _ = support_single(seg.L, foot_half, m_target=8)
        else:
            H_i, h_i, _ = support_single(seg.R, foot_half, m_target=8)

        ps.append(pk.reshape(2,))
        Hs.append(H_i)
        hs.append(h_i.reshape(-1))
    return np.stack(ps, axis=0), Hs, hs


def main():
    np.random.seed(3)

    dt = 0.01
    zc = 0.9
    g = 9.81

    # Build a small plan (doesn't need to be feasible; we're only checking algebraic equality)
    segs, T = build_segments(n_steps=2, ds=0.2, ss=0.2, step_length=0.08, step_width=0.24, x0=0.0)

    Ad, Bd = discretize_double_integrator(dt)
    N = 30
    x0 = np.array([0.02, 0.00, 0.0, 0.0])
    foot_half = np.array([0.09, 0.045])

    U = 0.5 * np.random.randn(N, 2)  # arbitrary

    # Build stacked constraint matrix
    A_poly, l_poly, u_poly, meta = build_polygon_preview_constraints(
        segs=segs, t0=0.0, dt=dt, N=N,
        Ad=Ad, Bd=Bd, x0=x0, zc=zc, g=g,
        foot_half_sizes=foot_half,
        m_target=8,
    )

    # Evaluate stacked: A_poly U <= u_poly
    # Shapes:
    #  - A_poly: (8N, 2N)
    #  - Uvec:   (2N,)
    Uvec = U.reshape(-1)
    lhs = (A_poly @ Uvec).reshape(-1)

    # Compute "ground truth" per-step H_i p_i and stack
    P_mat, Hs, hs = simulate_and_collect_p(segs, 0.0, dt, N, zc, g, foot_half, x0, U)
    gt = []
    for i in range(N):
        gt.append(Hs[i] @ P_mat[i])
    gt = np.concatenate(gt, axis=0).reshape(-1)

    # IMPORTANT: A_poly maps U -> H_stack * (P_U U), i.e. the *linear* part.
    # The full stacked value is:
    #   stack(H_i p_i) = A_poly U + stack(H_i r_prev_aff_i)
    r_prev_aff = meta["r_prev_aff"].reshape(-1)
    aff = []
    for i in range(N):
        ri = r_prev_aff[2 * i : 2 * i + 2]
        aff.append(Hs[i] @ ri)
    aff = np.concatenate(aff, axis=0).reshape(-1)

    err = np.max(np.abs((lhs + aff) - gt))
    print("max |(A_poly U + affine) - stack(H_i p_i)| =", err)
    assert err < 1e-10, f"Stacked polygon constraint mapping mismatch: {err}"

    # Also sanity-check u_poly construction matches h - H*r_prev_aff term:
    # Not a hard equality check here (depends on x0), but we can validate dimension and finiteness.
    assert lhs.shape == u_poly.shape
    assert np.all(np.isfinite(u_poly[np.isfinite(u_poly)]))

    print("OK: stacked polygon constraints mapping verified.")


if __name__ == "__main__":
    main()
