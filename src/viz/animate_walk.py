import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def _rect_patch(center, half_sizes, **kwargs):
    cx, cy = float(center[0]), float(center[1])
    hx, hy = float(half_sizes[0]), float(half_sizes[1])
    return plt.Rectangle((cx - hx, cy - hy), 2*hx, 2*hy, fill=False, **kwargs)

def animate_walk(logs, outpath_gif: str, fps: int = 30, trail: int = 200):
    os.makedirs(os.path.dirname(outpath_gif), exist_ok=True)

    t = logs["t"]
    x = logs["x"]          # (N,4)
    p = logs["p"]          # (N,2)
    pref = logs["p_ref"]   # (N,2)
    margin = logs["margin"]
    mode = logs["mode"]    # (N,)
    Ls = logs["L"]         # (N,2)
    Rs = logs["R"]         # (N,2)
    verts_list = logs["verts"]  # list of arrays (m,2)
    hs = logs["foot_half_sizes"]

    # Axis limits from all relevant points
    all_pts = np.vstack([Ls, Rs, x[:,0:2], p, pref])
    finite = np.isfinite(all_pts).all(axis=1)
    if np.any(finite):
        mn = all_pts[finite].min(axis=0)
        mx = all_pts[finite].max(axis=0)
    else:
        mn = np.array([-0.5,-0.5]); mx = np.array([0.5,0.5])

    pad = np.array([0.25, 0.25])
    mn -= pad; mx += pad

    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(mn[0], mx[0])
    ax.set_ylim(mn[1], mx[1])
    ax.set_title("Walk demo: COM + ZMP/COP + reference + support polygon")

    # feet patches
    footL = _rect_patch(Ls[0], hs, linewidth=2)
    footR = _rect_patch(Rs[0], hs, linewidth=2)
    ax.add_patch(footL)
    ax.add_patch(footR)

    # support polygon outline
    poly_line, = ax.plot([], [], linewidth=2, label="support polygon")

    # points
    com_pt, = ax.plot([], [], marker="o", markersize=6, linestyle="None", label="COM")
    zmp_pt, = ax.plot([], [], marker="x", markersize=6, linestyle="None", label="ZMP/COP")
    ref_pt, = ax.plot([], [], marker="+", markersize=9, linestyle="None", label="ZMP_ref")

    # trails
    com_tr, = ax.plot([], [], linewidth=1)
    zmp_tr, = ax.plot([], [], linewidth=1)
    ref_tr, = ax.plot([], [], linewidth=1)

    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")
    ax.legend(loc="lower right")

    def init():
        poly_line.set_data([], [])
        com_pt.set_data([], [])
        zmp_pt.set_data([], [])
        ref_pt.set_data([], [])
        com_tr.set_data([], [])
        zmp_tr.set_data([], [])
        ref_tr.set_data([], [])
        txt.set_text("")
        return (footL, footR, poly_line, com_pt, zmp_pt, ref_pt, com_tr, zmp_tr, ref_tr, txt)

    def update(k):
        # feet
        L = Ls[k]; R = Rs[k]
        hx, hy = float(hs[0]), float(hs[1])
        footL.set_xy((float(L[0]) - hx, float(L[1]) - hy))
        footR.set_xy((float(R[0]) - hx, float(R[1]) - hy))

        # support polygon outline (verts are CCW)
        verts = verts_list[k]
        if verts is not None and len(verts) >= 3:
            vv = np.vstack([verts, verts[0]])
            poly_line.set_data(vv[:,0], vv[:,1])
        else:
            poly_line.set_data([], [])

        # color polygon by feasibility
        if margin[k] < -1e-6:
            poly_line.set_linestyle("--")
        else:
            poly_line.set_linestyle("-")

        # points
        com = x[k, 0:2]
        z = p[k]
        pr = pref[k]
        com_pt.set_data([com[0]], [com[1]])
        zmp_pt.set_data([z[0]], [z[1]])
        ref_pt.set_data([pr[0]], [pr[1]])

        # trails (last `trail` samples)
        k0 = max(0, k - trail)
        com_tr.set_data(x[k0:k+1,0], x[k0:k+1,1])
        zmp_tr.set_data(p[k0:k+1,0], p[k0:k+1,1])
        ref_tr.set_data(pref[k0:k+1,0], pref[k0:k+1,1])

        txt.set_text(f"t={t[k]:.2f}s  mode={mode[k]}  margin={margin[k]:+.2e}")
        return (footL, footR, poly_line, com_pt, zmp_pt, ref_pt, com_tr, zmp_tr, ref_tr, txt)

    interval_ms = int(1000 / fps)
    ani = animation.FuncAnimation(fig, update, frames=len(t), init_func=init,
                                  interval=interval_ms, blit=False)

    ani.save(outpath_gif, writer="pillow", fps=fps)
    plt.close(fig)
