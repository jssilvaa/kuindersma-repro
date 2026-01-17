# scripts/test_mpc_one_step.py
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.planning.segments import build_segments, segment_at_time, p_ref_in_segment
from src.control.zmp_mpc import ZmpMpcController


def build_Pref_stack(segs, t0, dt, N):
    Pref = np.zeros(2 * N, dtype=float)
    for i in range(1, N + 1):
        ti = float(t0 + i * dt)
        seg = segment_at_time(segs, ti)
        pref_i = p_ref_in_segment(seg, ti)
        Pref[2 * (i - 1): 2 * (i - 1) + 2] = pref_i
    return Pref


def main():
    dt = 0.01
    zc = 0.9
    g = 9.81

    segs, T = build_segments(n_steps=2, ds=0.2, ss=0.2, step_length=0.08, step_width=0.24, x0=0.0)

    N = 20
    foot_half = np.array([0.09, 0.045])
    x0 = np.array([0.02, 0.00, 0.0, 0.0])
    u_prev = np.array([0.0, 0.0])

    mpc = ZmpMpcController(
        N=N,
        m_target=8,
        dt=dt,
        zc=zc,
        g=g,
        Wp=np.eye(2) * 50.0,
        Wdu=np.eye(2) * 0.1,
        umin=np.array([-8.0, -8.0]),
        umax=np.array([ 8.0,  8.0]),
        verbose=False,
    )

    t0 = 0.0
    Pref = build_Pref_stack(segs, t0, dt, N)

    U_star, ok, status = mpc.solve(
        segs=segs,
        t0=t0,
        x0=x0,
        foot_half_sizes=foot_half,
        Pref=Pref,
        u_prev=u_prev,
    )

    print("ok:", ok, "status:", status)
    print("u0:", U_star[0:2])

    # Basic sanity: finite, within bounds
    assert np.all(np.isfinite(U_star))
    assert np.all(U_star[0:2] >= mpc.umin - 1e-6)
    assert np.all(U_star[0:2] <= mpc.umax + 1e-6)

    print("OK: MPC one-step solve produced a bounded control.")


if __name__ == "__main__":
    main()
