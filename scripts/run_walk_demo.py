import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.control.qp_zmp import ZmpQpController
from src.planning.segments import build_segments
from src.sim.walk_rollout import rollout_walk
from src.viz.animate_walk import animate_walk
from src.viz.plot_walk_timeseries import plot_walk_timeseries

def main():
    dt = 0.01
    zc = 0.9
    g  = 9.81

    # QP weights
    W   = np.eye(2) * 1.0
    # For longer walks, a large global Wp tends to drive the COM backwards
    # (tracking p_ref too aggressively). Keep Wp modest and rely on SS-only
    # scaling to track p_ref tighter during SSL/SSR.
    Wp  = np.eye(2) * 2.0
    Wdu = np.eye(2) * 0.1
    umin = np.array([-8.0, -8.0])
    umax = np.array([ 8.0,  8.0])

    qp = ZmpQpController(zc=zc, g=g, W=W, Wp=Wp, Wdu=Wdu, umin=umin, umax=umax)

    # Longer walk: start at x=-0.4 and progress to ~x=+0.4.
    # With this segment builder, the support midpoint advances by ~step_length/2 per step
    # (only one foot moves each step), so we use more steps for a longer trajectory.
    segs, T = build_segments(
        n_steps=20,
        ds=0.2,
        ss=0.25,
        step_length=0.08,
        step_width=0.24,
        x0=-0.4
    )

    # Initial COM near the initial support region
    x0 = np.array([-0.38, 0.00, 0.0, 0.0])
    foot_half = np.array([0.09, 0.045])

    # SS-only QP weight scaling: track p_ref harder during single support.
    # (W, Wp, Wdu) -> (0.2*W, 15*Wp, 1.0*Wdu) in SSL/SSR.
    logs = rollout_walk(
        segs, dt, zc, g, foot_half, qp, x0,
        Kp=25.0, Kd=10.0,
        com_ref="phase_aware",
        ss_qp_weight_scales=(0.2, 15.0, 1.0),
        ss_qp_ramp=0.05,
    )

    # Animation downsampling: longer rollouts can be slow to encode as GIF.
    anim_stride = 3  # keep every Nth sample for the GIF
    logs_anim = dict(logs)
    for key in ("t", "x", "u", "p", "p_ref", "p_des", "u_des", "margin", "min_slack", "active_face", "u_sat", "mode", "L", "R", "qp_ok"):
        if key in logs_anim:
            logs_anim[key] = logs_anim[key][::anim_stride]
    if "verts" in logs_anim:
        logs_anim["verts"] = logs_anim["verts"][::anim_stride]

    os.makedirs("results/plots", exist_ok=True)
    plot_walk_timeseries(logs, "results/plots/walk_timeseries.png")
    animate_walk(logs_anim, "results/walk.gif", fps=15, trail=220)

    print("Done.")
    print("T:", T, "N:", len(logs["t"]))
    print("min margin:", float(np.min(logs["margin"])))
    print("qp ok ratio:", float(np.mean(logs["qp_ok"])))

if __name__ == "__main__":
    main()
