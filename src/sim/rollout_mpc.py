# src/sim/rollout_mpc.py
from __future__ import annotations

import numpy as np

from src.models.lipm import discretize_double_integrator, step, zmp_from
from src.models.support_polygon import support_single, support_double, margin_halfspaces
from src.planning.segments import segment_at_time, p_ref_in_segment


def build_Pref_stack(segs, t0: float, dt: float, N: int) -> np.ndarray:
    """
    Build stacked ZMP reference:
      Pref = [pref_1; ...; pref_N], each pref_i ∈ R^2
    where pref_i is evaluated at time t0 + (i-1)*dt, consistent with
    p_i computed from (x_{i-1}, u_{i-1}).
    """
    Pref = np.zeros(2 * N, dtype=float)
    for i in range(1, N + 1):
        ti = float(t0 + (i - 1) * dt)
        seg = segment_at_time(segs, ti)
        #pref_i = p_ref_in_segment(seg, ti)  # shape (2,)
        # Feasible-by-construction ZMP reference:
        #   DS: support midpoint
        #   SS: stance foot center
        if seg.mode == "DS":
            pref_i = 0.5 * (seg.L + seg.R)
        elif seg.mode == "SSL":
            pref_i = seg.L
        else:  # "SSR"
            pref_i = seg.R
        Pref[2 * (i - 1): 2 * (i - 1) + 2] = pref_i
    return Pref


def rollout_mpc(
    segs,
    mpc,
    dt: float,
    zc: float,
    g: float,
    foot_half_sizes: np.ndarray,
    x0: np.ndarray,
    T_end: float | None = None,
    u_prev0: np.ndarray | None = None,
):
    """
    Closed-loop rollout using MPC:
      - at each time t_k:
          Pref <- horizon reference
          U* <- solve MPC
          apply u_k = U*[0:2]
          propagate x_{k+1} = step(x_k, u_k)
          log support polygon at time t_k+dt (matches your convention for segment timing)

    Returns:
      logs dict compatible with your previous visualizers style.
    """
    Ad, Bd = discretize_double_integrator(dt)

    x = np.asarray(x0, dtype=float).reshape(-1)
    if x.shape != (4,):
        raise ValueError("x0 must be shape (4,)")

    if u_prev0 is None:
        u_prev = np.zeros(2, dtype=float)
    else:
        u_prev = np.asarray(u_prev0, dtype=float).reshape(2,)

    # Determine rollout horizon
    if T_end is None:
        T_end = float(segs[-1].t1)
    else:
        T_end = float(T_end)

    Nsteps = int(np.ceil(T_end / dt))
    alpha = float(zc / g)

    logs = {
        "t": np.zeros(Nsteps),
        "x": np.zeros((Nsteps, 4)),
        "u": np.zeros((Nsteps, 2)),
        "u_prev": np.zeros((Nsteps, 2)),
        "U_star": [None] * Nsteps,  # store full sequence sometimes useful
        "p": np.zeros((Nsteps, 2)),
        "p_ref_1": np.zeros((Nsteps, 2)),  # first reference in horizon (at t)
        "Pref": [None] * Nsteps,           # full stacked Pref for debugging
        "margin": np.zeros(Nsteps),
        "mode": np.empty(Nsteps, dtype=object),
        "L": np.zeros((Nsteps, 2)),
        "R": np.zeros((Nsteps, 2)),
        "verts": [None] * Nsteps,          # polygon vertices for viz
        "qp_ok": np.zeros(Nsteps, dtype=bool),
        "osqp_iters": np.full(Nsteps, np.nan),
        "osqp_run_time": np.full(Nsteps, np.nan),
        "osqp_status": np.empty(Nsteps, dtype=object),
        "slack_max": np.full(Nsteps, np.nan),
    }

    for k in range(Nsteps):
        t = float(k * dt)

        # Build horizon reference
        Pref = build_Pref_stack(segs, t0=t, dt=dt, N=mpc.N)
        p_ref_1 = Pref[0:2].copy()

        # Solve MPC (uses segment info internally for constraints)
        U_star, ok, status = mpc.solve(
            segs=segs,
            t0=t,
            x0=x,
            foot_half_sizes=foot_half_sizes,
            Pref=Pref,
            u_prev=u_prev,
        )
        u = U_star[0:2].copy()

        # Evaluate ZMP from current state x and applied u (your convention)
        p = zmp_from(x, u, zc, g)

        # For margin/support polygon, use segment at time t (align with p from x_k,u_k)
        seg = segment_at_time(segs, float(t))
        L = seg.L
        R = seg.R
        if seg.mode == "DS":
            H, h, verts = support_double(L, R, foot_half_sizes, m_target=8)
        elif seg.mode == "SSL":
            H, h, verts = support_single(L, foot_half_sizes, m_target=8)
        else:
            H, h, verts = support_single(R, foot_half_sizes, m_target=8)

        m = margin_halfspaces(H, h, p)

        # OSQP stats (if patch applied)
        iters = np.nan
        run_time = np.nan
        slack_max = np.nan
        if getattr(mpc, "last_info", None) is not None:
            info = mpc.last_info
            iters = float(getattr(info, "iter", np.nan))
            # osqp uses 'run_time' in seconds
            run_time = float(getattr(info, "run_time", np.nan))

        # Slack diagnostics (if MPC uses slack variables)
        if getattr(mpc, "last_result", None) is not None and getattr(mpc.last_result, "x", None) is not None:
            try:
                z = np.asarray(mpc.last_result.x, dtype=float).reshape(-1)
                nU = int(getattr(mpc, "nU", 0))
                if z.shape[0] > nU:
                    s = z[nU:]
                    if s.size > 0:
                        slack_max = float(np.max(s))
            except Exception:
                slack_max = np.nan

        # Log (all at time t, pre-propagation)
        logs["t"][k] = t
        logs["x"][k] = x
        logs["u"][k] = u
        logs["u_prev"][k] = u_prev
        logs["U_star"][k] = U_star
        logs["p"][k] = p
        logs["p_ref_1"][k] = p_ref_1
        logs["Pref"][k] = Pref
        logs["margin"][k] = m
        logs["mode"][k] = seg.mode
        logs["L"][k] = L
        logs["R"][k] = R
        logs["verts"][k] = verts
        logs["qp_ok"][k] = bool(ok)
        logs["osqp_iters"][k] = iters
        logs["osqp_run_time"][k] = run_time
        logs["osqp_status"][k] = status
        logs["slack_max"][k] = slack_max

        # Propagate dynamics and update u_prev
        x = step(x, u, Ad, Bd)
        u_prev = u

    logs["foot_half_sizes"] = np.asarray(foot_half_sizes, dtype=float).reshape(2,)
    logs["zc"] = float(zc)
    logs["g"] = float(g)
    logs["alpha"] = alpha
    return logs
