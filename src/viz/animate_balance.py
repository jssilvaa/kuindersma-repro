import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def animate_balance(logs, outpath_gif):
    os.makedirs(os.path.dirname(outpath_gif), exist_ok=True)

    t = logs["t"]
    x = logs["x"]
    p = logs["p"]
    margin = logs["margin"]

    c = logs["foot_center"]
    hs = logs["foot_half_sizes"]
    cx, cy = float(c[0]), float(c[1])
    hx, hy = float(hs[0]), float(hs[1])

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(cx - 0.25, cx + 0.25)
    ax.set_ylim(cy - 0.25, cy + 0.25)
    ax.set_title("Balance demo: COM + ZMP/COP")

    # foot rectangle
    rect = plt.Rectangle((cx - hx, cy - hy), 2*hx, 2*hy,
                         fill=False, linewidth=2)
    ax.add_patch(rect)

    com_pt, = ax.plot([], [], marker="o", markersize=7, linestyle="None", label="COM")
    zmp_pt, = ax.plot([], [], marker="x", markersize=7, linestyle="None", label="ZMP/COP")

    com_tr, = ax.plot([], [], linewidth=1)
    zmp_tr, = ax.plot([], [], linewidth=1)

    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                  va="top", ha="left")

    ax.legend(loc="lower right")

    def init():
        com_pt.set_data([], [])
        zmp_pt.set_data([], [])
        com_tr.set_data([], [])
        zmp_tr.set_data([], [])
        txt.set_text("")
        return com_pt, zmp_pt, com_tr, zmp_tr, txt

    def update(k):
        rx, ry = x[k,0], x[k,1]
        px, py = p[k,0], p[k,1]

        com_pt.set_data([rx], [ry])
        zmp_pt.set_data([px], [py])

        com_tr.set_data(x[:k+1,0], x[:k+1,1])
        zmp_tr.set_data(p[:k+1,0], p[:k+1,1])

        m = margin[k]
        txt.set_text(f"t={t[k]:.2f}s  margin={m:+.4f}")
        if m < 0:
            txt.set_color("red")
        else:
            txt.set_color("white")

        return com_pt, zmp_pt, com_tr, zmp_tr, txt

    ani = animation.FuncAnimation(
        fig, update, frames=len(t),
        init_func=init, interval=30, blit=True
    )

    ani.save(outpath_gif, writer="pillow", fps=30)
    plt.close(fig)
