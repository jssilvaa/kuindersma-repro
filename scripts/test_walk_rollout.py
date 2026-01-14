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

dt=0.01; zc=0.9; g=9.81
W=np.eye(2)*1.0; Wp=np.eye(2)*50.0; Wdu=np.eye(2)*0.1
umin=np.array([-8,-8]); umax=np.array([8,8])
qp = ZmpQpController(zc,g,W,Wp,Wdu,umin,umax)

# Parameters must be feasible w.r.t. u bounds; long SS with large step length
# will drive the COM away until the ZMP constraints become infeasible.
segs, T = build_segments(n_steps=4, ds=0.2, ss=0.2, step_length=0.08, step_width=0.24, x0=0.0)

x0 = np.array([0.02, 0.00, 0.0, 0.0])
foot_half = np.array([0.09, 0.045])

logs = rollout_walk(segs, dt, zc, g, foot_half, qp, x0, Kp=25.0, Kd=10.0, com_ref="support_mid")

print("T:", T, "N:", len(logs["t"]))
print("min margin:", logs["margin"].min())
print("qp ok ratio:", logs["qp_ok"].mean())
