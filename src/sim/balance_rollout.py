import numpy as np

from src.models.lipm import step, zmp_from, discretize_double_integrator
from src.models.foot_rect import margin
from src.control.u_des import pd_u_des

def rollout_balance(T: float,
                    dt: float,
                    zc: float,
                    g: float,
                    foot_center: np.ndarray,
                    foot_half_sizes: np.ndarray,
                    qp_ctrl,
                    x0: np.ndarray,
                    Kp: float,
                    Kd: float):
    Ad, Bd = discretize_double_integrator(dt)

    N = int(T / dt)
    x = x0.copy()
    u_prev = np.zeros(2)

    r_ref = np.zeros(2)
    v_ref = np.zeros(2)
    p_ref = foot_center.copy()

    logs = {
        "t": np.zeros(N),
        "x": np.zeros((N, 4)),
        "u": np.zeros((N, 2)),
        "p": np.zeros((N, 2)),
        "margin": np.zeros(N),
        "qp_ok": np.zeros(N, dtype=bool),
    }

    cx, cy = float(foot_center[0]), float(foot_center[1])
    hx, hy = float(foot_half_sizes[0]), float(foot_half_sizes[1])

    from src.models.foot_rect import rectangle_Hh
    H_foot, h_foot = rectangle_Hh(cx, cy, hx, hy)

    for k in range(N): 
        t = k * dt
        r = x[0:2]
        v = x[2:4]
        p = zmp_from(x, np.zeros(2), zc, g)

        u_des = pd_u_des(x, r_ref, v_ref, Kp, Kd)

        u_opt, qp_ok, _ = qp_ctrl.solve(r=r, u_prev=u_prev, u_des=u_des, p_ref=p_ref, H=H_foot, h=h_foot)

        p = zmp_from(x, u_opt, zc, g)
        m = margin(H_foot, h_foot, p)

        x = step(x, u_opt, Ad, Bd)
        u_prev = u_opt

        logs["t"][k] = t
        logs["x"][k, :] = x
        logs["u"][k, :] = u_opt
        logs["p"][k, :] = p
        logs["margin"][k] = m
        logs["qp_ok"][k] = qp_ok
    
    logs["H"] = H_foot
    logs["h"] = h_foot
    logs["foot_center"] = foot_center
    logs["foot_half_sizes"] = foot_half_sizes
    return logs