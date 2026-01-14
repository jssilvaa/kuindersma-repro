import os
import numpy as np
import matplotlib.pyplot as plt

def plot_balance_timeseries(logs, outpath):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    t = logs["t"]
    x = logs["x"]
    u = logs["u"]
    p = logs["p"]
    m = logs["margin"]

    fig = plt.figure(figsize=(10, 9))

    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(t, x[:,0], label="r_x")
    ax1.plot(t, x[:,1], label="r_y")
    ax1.legend()
    ax1.set_ylabel("COM pos [m]")

    ax2 = plt.subplot(4, 1, 2)
    ax2.plot(t, x[:,2], label="v_x")
    ax2.plot(t, x[:,3], label="v_y")
    ax2.legend()
    ax2.set_ylabel("COM vel [m/s]")

    ax3 = plt.subplot(4, 1, 3)
    ax3.plot(t, u[:,0], label="u_x")
    ax3.plot(t, u[:,1], label="u_y")
    ax3.legend()
    ax3.set_ylabel("COM accel [m/s^2]")

    ax4 = plt.subplot(4, 1, 4)
    ax4.plot(t, m, label="min margin")
    ax4.axhline(0.0, linestyle="--")
    ax4.legend()
    ax4.set_ylabel("margin [m]")
    ax4.set_xlabel("time [s]")

    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)
