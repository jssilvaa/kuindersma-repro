# scripts/run_mpc_demo.py
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.planning.segments import build_segments
from src.control.zmp_mpc import ZmpMpcController
from src.sim.rollout_mpc import rollout_mpc


def plot_timeseries(logs, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    t = logs["t"]
    p = logs["p"]
    pref = logs["p_ref_1"]
    margin = logs["margin"]
    iters = logs["osqp_iters"]
    rt = logs["osqp_run_time"]
    ok = logs["qp_ok"]

    # ZMP tracking
    plt.figure()
    plt.plot(t, p[:, 0], label="p_x")
    plt.plot(t, pref[:, 0], label="p_ref_x", linestyle="--")
    plt.plot(t, p[:, 1], label="p_y")
    plt.plot(t, pref[:, 1], label="p_ref_y", linestyle="--")
    plt.xlabel("t [s]")
    plt.ylabel("ZMP [m]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "zmp_tracking.png", dpi=160)
    plt.close()

    # Margin
    plt.figure()
    plt.plot(t, margin, label="margin")
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel("t [s]")
    plt.ylabel("margin (>=0 feasible) [m-ish]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "margin.png", dpi=160)
    plt.close()

    # OSQP iters
    plt.figure()
    plt.plot(t, iters, label="osqp iters")
    plt.xlabel("t [s]")
    plt.ylabel("iterations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "osqp_iters.png", dpi=160)
    plt.close()

    # OSQP runtime
    plt.figure()
    plt.plot(t, rt * 1000.0, label="osqp run_time [ms]")
    plt.xlabel("t [s]")
    plt.ylabel("ms")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "osqp_runtime.png", dpi=160)
    plt.close()

    # Feasibility / ok ratio
    ok_ratio = float(np.mean(ok))
    print("qp ok ratio:", ok_ratio)

    # COM y drift diagnostic
    y_abs_max = float(np.max(np.abs(logs["x"][:, 1])))
    print("max |COM y|:", y_abs_max)
    x_abs_max = float(np.max(np.abs(logs["x"][:, 0])))
    print("max |COM x|:", x_abs_max)


def animate_topdown(logs, out_path: Path, stride: int = 2):
    """
    Simple top-down animation:
      - support polygon (verts)
      - ZMP point p
      - reference p_ref_1
      - COM position r (from x)
    Saves GIF if pillow is available.
    """
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except Exception as e:
        print("Animation skipped (missing matplotlib animation deps):", e)
        return

    t = logs["t"]
    x = logs["x"]
    p = logs["p"]
    pref = logs["p_ref_1"]
    verts = logs["verts"]

    # auto bounds
    xs = np.concatenate([p[:, 0], pref[:, 0], x[:, 0]])
    ys = np.concatenate([p[:, 1], pref[:, 1], x[:, 1]])
    pad = 0.15
    xmin, xmax = float(xs.min() - pad), float(xs.max() + pad)
    ymin, ymax = float(ys.min() - pad), float(ys.max() + pad)

    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    poly_line, = ax.plot([], [], linewidth=2)
    p_sc = ax.scatter([], [])
    pref_sc = ax.scatter([], [])
    com_sc = ax.scatter([], [])

    title = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    def init():
        poly_line.set_data([], [])
        title.set_text("")
        return poly_line, p_sc, pref_sc, com_sc, title

    def update(frame_idx):
        k = frame_idx * stride
        if k >= len(t):
            k = len(t) - 1

        vk = verts[k]
        if vk is not None and len(vk) > 0:
            V = np.asarray(vk, dtype=float)
            # close polygon for plotting
            Vc = np.vstack([V, V[0]])
            poly_line.set_data(Vc[:, 0], Vc[:, 1])
        else:
            poly_line.set_data([], [])

        # update scatter offsets
        p_sc.set_offsets(p[k].reshape(1, 2))
        pref_sc.set_offsets(pref[k].reshape(1, 2))
        com_sc.set_offsets(x[k, 0:2].reshape(1, 2))

        title.set_text(f"t={t[k]:.2f}s  margin={logs['margin'][k]:.3f}")
        return poly_line, p_sc, pref_sc, com_sc, title

    anim = FuncAnimation(fig, update, init_func=init, frames=int(np.ceil(len(t) / stride)), interval=30, blit=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        anim.save(out_path, writer=PillowWriter(fps=30))
        print("Saved GIF:", out_path)
    except Exception as e:
        print("Failed to save GIF (need pillow?):", e)

    plt.close(fig)


def main():
    # --- walking plan ---
    segs, T = build_segments(
        n_steps=20,
        ds=0.2,
        ss=0.2,
        step_length=0.08,
        step_width=0.24,
        x0=0.0,
    )

    # --- MPC setup ---
    dt = 0.01
    zc = 0.9
    g = 9.81
    N = 20
    foot_half = np.array([0.09, 0.045])
    x0 = np.array([0.02, 0.00, 0.0, 0.0])

    mpc = ZmpMpcController(
        N=N,
        m_target=8,
        dt=dt,
        zc=zc,
        g=g,
        Wp=np.eye(2) * 2.0,
        Wr=np.diag([80.0, 80.0]),
        # Track a small forward COM velocity so the feet don't walk away from the COM.
        v_ref=np.array([0.10, 0.0]),
        Wv=np.diag([60.0, 20.0]),
        Wu=np.diag([0.1, 0.2]),
        Wdu=np.eye(2) * 0.2,
        slack_weight=1.0e6,
        x_min=-0.10,
        x_max=2.0,
        vx_min=-0.50,
        vx_max=2.0,
        y_max=0.20,
        vy_max=1.00,
        umin=np.array([-8.0, -8.0]),
        umax=np.array([ 8.0,  8.0]),
        verbose=False,
    )

    logs = rollout_mpc(
        segs=segs,
        mpc=mpc,
        dt=dt,
        zc=zc,
        g=g,
        foot_half_sizes=foot_half,
        x0=x0,
    )

    statuses = logs["osqp_status"]
    print("status counts:", Counter(statuses))

    print("min margin:", float(np.min(logs["margin"])))
    print("qp ok ratio:", float(np.mean(logs["qp_ok"])))

    if "slack_max" in logs:
        print("max slack:", float(np.max(logs["slack_max"])))
        print("mean slack:", float(np.mean(logs["slack_max"])))

    bad = np.where(~logs["qp_ok"])[0]
    if len(bad) > 0:
        k0 = int(bad[0])
        print("first fail at t =", logs["t"][k0], "status =", logs["osqp_status"][k0],
            "margin =", logs["margin"][k0])

    out_dir = Path("out_mpc")
    plot_timeseries(logs, out_dir)
    animate_topdown(logs, out_dir / "mpc_demo.gif", stride=2)


if __name__ == "__main__":
    main()
