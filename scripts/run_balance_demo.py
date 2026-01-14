import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.control.qp_zmp import ZmpQpController
from src.sim.balance_rollout import rollout_balance
from src.viz.animate_balance import animate_balance
from src.viz.plot_timeseries import plot_balance_timeseries

def main():
    # ---------- params ----------
    dt = 0.01
    T  = 6.0
    g  = 9.81
    zc = 0.9

    foot_center = np.array([0.0, 0.0])
    foot_half_sizes = np.array([0.09, 0.045])  # ~18cm x 9cm

    # PD gains
    Kp = 25.0
    Kd = 10.0

    # QP weights
    W   = np.eye(2) * 1.0
    Wp  = np.eye(2) * 50.0
    Wdu = np.eye(2) * 0.1

    umin = np.array([-8.0, -8.0])
    umax = np.array([ 8.0,  8.0])

    # Initial state disturbance
    x0 = np.array([0.05, 0.02, 0.0, 0.0])  # 5cm x, 2cm y offset

    qp = ZmpQpController(
        zc=zc, g=g,
        W=W, Wp=Wp, Wdu=Wdu,
        umin=umin, umax=umax
    )

    logs = rollout_balance(
        T=T, dt=dt, zc=zc, g=g,
        foot_center=foot_center,
        foot_half_sizes=foot_half_sizes,
        qp_ctrl=qp,
        x0=x0,
        Kp=Kp, Kd=Kd
    )

    os.makedirs("results/plots", exist_ok=True)

    plot_balance_timeseries(logs, "results/plots/balance_timeseries.png")
    animate_balance(logs, "results/balance.gif")

    print("Done.")
    print("Min margin:", logs["margin"].min())

if __name__ == "__main__":
    main()