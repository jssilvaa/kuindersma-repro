import os
import numpy as np
import matplotlib.pyplot as plt

def plot_walk_timeseries(logs, outpath: str):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    t = logs["t"]
    x = logs["x"]
    u = logs["u"]
    p = logs["p"]
    pref = logs["p_ref"]
    margin = logs["margin"]

    fig = plt.figure(figsize=(11, 10))

    ax1 = plt.subplot(5, 1, 1)
    ax1.plot(t, x[:,0], label="r_x")
    ax1.plot(t, x[:,1], label="r_y")
    ax1.legend()
    ax1.set_ylabel("COM pos [m]")

    ax2 = plt.subplot(5, 1, 2)
    ax2.plot(t, x[:,2], label="v_x")
    ax2.plot(t, x[:,3], label="v_y")
    ax2.legend()
    ax2.set_ylabel("COM vel [m/s]")

    ax3 = plt.subplot(5, 1, 3)
    ax3.plot(t, u[:,0], label="u_x")
    ax3.plot(t, u[:,1], label="u_y")
    ax3.legend()
    ax3.set_ylabel("COM accel [m/s^2]")

    ax4 = plt.subplot(5, 1, 4)
    ax4.plot(t, p[:,0], label="zmp_x")
    ax4.plot(t, pref[:,0], linestyle="--", label="zmp_ref_x")
    ax4.plot(t, p[:,1], label="zmp_y")
    ax4.plot(t, pref[:,1], linestyle="--", label="zmp_ref_y")
    ax4.legend(ncol=2)
    ax4.set_ylabel("ZMP/COP [m]")

    ax5 = plt.subplot(5, 1, 5)
    ax5.plot(t, margin, label="min margin")
    ax5.axhline(0.0, linestyle="--")
    ax5.legend()
    ax5.set_ylabel("margin [m]")
    ax5.set_xlabel("time [s]")

    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)
