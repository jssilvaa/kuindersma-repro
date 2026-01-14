import numpy as np
from src.models.lipm import discretize_double_integrator, step, zmp_from
from src.models.support_polygon import support_single, support_double, margin_halfspaces
from src.control.u_des import pd_u_des
from src.planning.segments import segment_at_time, p_ref_in_segment

""" 
Simulate walking over a sequence of footstep segments using a ZMP QP controller.
    Given a planned schedule of foot placements and timings (segs),
    simulate the LIPM dynamics controlled by a ZMP QP controller.
Rolls forward the discrete-time dynamics step-by-step and logs states and variables involved.
"""

def rollout_walk(segs,
                 dt: float,
                 zc: float,
                 g: float,
                 foot_half_sizes: np.ndarray,
                 qp_ctrl,
                 x0: np.ndarray,
                 Kp: float,
                 Kd: float,
                 com_ref: str = "p_ref",
                 com_ref_tau: float | np.ndarray = 0.0,
                 ss_p_des_inset: float = 0.0,
                 ss_qp_weight_scales: tuple[float, float, float] | None = None,
                 ss_qp_ramp: float = 0.0,
                 v_ref_max: float | np.ndarray | None = None):
    """
    
    """

    # Discrete-time state space matrices
    Ad, Bd = discretize_double_integrator(dt)
    T = segs[-1].t1
    N = int(np.ceil(T / dt))

    x = x0.copy()
    u_prev = np.zeros(2)

    # Initial values
    r_ref = np.zeros(2)
    v_ref = np.zeros(2)
    r_ref_prev = r_ref.copy()

    # Optional smoothing for COM reference (useful for phase-aware references
    # that otherwise jump between support feet). 0.0 disables smoothing.
    tau = np.asarray(com_ref_tau, dtype=float)
    if tau.shape == ():
        tau = np.array([float(tau), float(tau)], dtype=float)
    tau = np.maximum(tau, 0.0)
    r_ref_filt = x[0:2].copy()

    alpha = zc / g

    # SS QP weight scaling
    if ss_qp_weight_scales is None:
        ss_W_scale, ss_Wp_scale, ss_Wdu_scale = 1.0, 1.0, 1.0
    else:
        ss_W_scale, ss_Wp_scale, ss_Wdu_scale = ss_qp_weight_scales

    # logs 
    logs = {
        "t": np.zeros(N),
        "x": np.zeros((N, 4)),
        "u": np.zeros((N, 2)),
        "p": np.zeros((N, 2)),
        "p_ref": np.zeros((N, 2)),
        "p_des": np.zeros((N, 2)),
        "u_des": np.zeros((N, 2)),
        "margin": np.zeros(N),
        "min_slack": np.zeros(N),
        "active_face": np.zeros(N, dtype=int),
        "u_sat": np.zeros(N, dtype=bool),
        "mode": np.empty(N, dtype=object),
        "L": np.zeros((N, 2)),
        "R": np.zeros((N, 2)),
        "verts": [None]*N,   # polygon vertices for viz
        "qp_ok": np.zeros(N, dtype=bool),
    }

    # iterate over time steps 
    #     during each step, find current segment and its foot placements
    #     build support polygon accordingly, compute references, solve QP, log data, propagate dynamics
    for k in range(N):
        t = k * dt # current tick 
        seg = segment_at_time(segs, t) # retrieve segment with: t0 <= t < t1

        L = seg.L # left foot center, np.ndarray (2,)
        R = seg.R # right foot center, np.ndarray (2,)
        pref = p_ref_in_segment(seg, t) # reference point in segment (COP/ZMP to be tracked)

        if com_ref == "p_ref": # COM reference = p_ref
            r_target = pref
        elif com_ref == "support_mid": # COM reference = midpoint between support feet, np.ndarray (2,) between L and R (also np.ndarray (2,))
            r_target = 0.5 * (L + R)
        elif com_ref in ("phase_aware", "phase"): # com reference changes depending on the phase of the kinematic chain 
            # Phase-aware COM reference:
            #  - DS: support midpoint
            #  - SS: stance foot center, this makes more sense (i think) because we shift our weight over the stance foot when walking
            if seg.mode == "DS":
                r_target = 0.5 * (L + R) # support mid when both feet on ground 
            elif seg.mode == "SSL":
                r_target = L # support left when right is in the air
            elif seg.mode == "SSR": 
                r_target = R # support right when left is in the air
        else:
            raise ValueError(
                f"Unknown com_ref='{com_ref}' (expected 'p_ref', 'support_mid' or 'phase_aware'/'phase')"
            )

        # Apply optional low-pass smoothing to COM reference.
        # This reduces abrupt u_des demands that otherwise force ZMP to the boundary.
        if np.any(tau > 0.0) and com_ref != "p_ref":
            # Per-axis exponential smoothing: r_filt <- r_filt + beta*(r_target - r_filt)
            beta = 1.0 - np.exp(-dt / np.maximum(tau, 1e-12))
            r_ref_filt = r_ref_filt + beta * (r_target - r_ref_filt)
            r_ref = r_ref_filt
        else:
            r_ref = r_target

        # Velocity reference via finite differences.
        # If r_ref changes discontinuously (e.g., DS midpoint -> SS stance foot),
        # raw finite differences can be unrealistically large and will inject
        # huge u_des spikes, often pinning the ZMP/COP to a constraint boundary.
        # Default behavior keeps v_ref = 0 for discontinuous references.
        v_ref_raw = (r_ref - r_ref_prev) / dt
        if com_ref == "p_ref":
            v_ref = v_ref_raw
        else:
            # Default behavior: keep v_ref = 0 for discontinuous references.
            # (Using finite-difference v_ref here can inject large spikes.)
            if v_ref_max is None:
                v_ref = np.zeros(2)
            else:
                vmax = np.asarray(v_ref_max, dtype=float)
                if vmax.shape == ():
                    vmax = np.array([float(vmax), float(vmax)], dtype=float)
                v_ref = np.clip(v_ref_raw, -vmax, vmax)
        r_ref_prev = r_ref.copy()

        # Build support polygon (padded to 8 halfspaces)
        if seg.mode == "DS":
            H, h, verts = support_double(L, R, foot_half_sizes, m_target=8)
        elif seg.mode == "SSL":
            H, h, verts = support_single(L, foot_half_sizes, m_target=8)
        else:  # "SSR"
            H, h, verts = support_single(R, foot_half_sizes, m_target=8)

        r = x[0:2]
        u_des = pd_u_des(x, r_ref, v_ref, Kp, Kd)

        # SS-only: keep the PD-implied ZMP strictly inside the stance foot.
        # This reduces the tendency for the QP to project onto the support boundary
        # simply to satisfy an aggressive u_des.
        if ss_p_des_inset > 0.0 and seg.mode in ("SSL", "SSR"):
            inset = float(max(ss_p_des_inset, 0.0))
            stance = L if seg.mode == "SSL" else R
            hx, hy = float(foot_half_sizes[0]), float(foot_half_sizes[1])
            # Ensure the inset doesn't invert the box.
            inset_x = min(inset, 0.99 * hx)
            inset_y = min(inset, 0.99 * hy)
            xmin, xmax = stance[0] - hx + inset_x, stance[0] + hx - inset_x
            ymin, ymax = stance[1] - hy + inset_y, stance[1] + hy - inset_y

            p_des_intent = r - alpha * u_des
            p_des_clipped = np.array([
                float(np.clip(p_des_intent[0], xmin, xmax)),
                float(np.clip(p_des_intent[1], ymin, ymax)),
            ])
            u_des = (r - p_des_clipped) / alpha

        if seg.mode in ("SSL", "SSR") and (ss_W_scale, ss_Wp_scale, ss_Wdu_scale) != (1.0, 1.0, 1.0):
            # Smoothly blend SS weights in/out near phase boundaries to avoid
            # objective discontinuities that create sharp control jumps.
            if ss_qp_ramp > 0.0:
                tau_in = (t - seg.t0) / ss_qp_ramp
                tau_out = (seg.t1 - t) / ss_qp_ramp
                f = float(np.clip(min(tau_in, tau_out), 0.0, 1.0))
                # Smoothstep blend (C1) for slightly smoother transitions.
                f = f * f * (3.0 - 2.0 * f)
            else:
                f = 1.0

            W_scale = 1.0 + f * (ss_W_scale - 1.0)
            Wp_scale = 1.0 + f * (ss_Wp_scale - 1.0)
            Wdu_scale = 1.0 + f * (ss_Wdu_scale - 1.0)

            W_ss = W_scale * qp_ctrl.W
            Wp_ss = Wp_scale * qp_ctrl.Wp
            Wdu_ss = Wdu_scale * qp_ctrl.Wdu
            u, ok, _ = qp_ctrl.solve(
                r=r, u_prev=u_prev, u_des=u_des, p_ref=pref, H=H, h=h,
                W=W_ss, Wp=Wp_ss, Wdu=Wdu_ss,
            )
        else:
            u, ok, _ = qp_ctrl.solve(r=r, u_prev=u_prev, u_des=u_des, p_ref=pref, H=H, h=h)

        p = zmp_from(x, u, zc, g)
        m = margin_halfspaces(H, h, p)

        # Diagnostics: implied ZMP from PD intent and which polygon face is active
        p_des = r - alpha * u_des
        slack = h - H @ p
        # Ignore padded halfspaces (h=+inf) when reporting the active face.
        finite = np.isfinite(h) & np.isfinite(slack)
        if np.any(finite):
            slack_f = slack[finite]
            idx_f = int(np.argmin(slack_f))
            active = int(np.flatnonzero(finite)[idx_f])
            min_slack = float(slack[active])
        else:
            active = 0
            min_slack = float("nan")
        u_sat = bool(np.any(np.isclose(u, qp_ctrl.umin, atol=1e-6) | np.isclose(u, qp_ctrl.umax, atol=1e-6)))

        # log (state/control at time t, before propagation)
        logs["t"][k] = t
        logs["x"][k] = x
        logs["u"][k] = u
        logs["p"][k] = p
        logs["p_ref"][k] = pref
        logs["p_des"][k] = p_des
        logs["u_des"][k] = u_des
        logs["margin"][k] = m
        logs["min_slack"][k] = min_slack
        logs["active_face"][k] = active
        logs["u_sat"][k] = u_sat
        logs["mode"][k] = seg.mode
        logs["L"][k] = L
        logs["R"][k] = R
        logs["verts"][k] = verts
        logs["qp_ok"][k] = ok

        # propagate
        x = step(x, u, Ad, Bd)
        u_prev = u

    logs["foot_half_sizes"] = foot_half_sizes
    return logs
