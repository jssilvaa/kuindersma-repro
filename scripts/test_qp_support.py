import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
    
from src.control.qp_zmp import ZmpQpController
from src.models.support_polygon import support_single, support_double
from src.models.lipm import zmp_from

dt = 0.01
g  = 9.81
zc = 0.9

W   = np.eye(2)*1.0
Wp  = np.eye(2)*50.0
Wdu = np.eye(2)*0.1
umin = np.array([-8.0,-8.0])
umax = np.array([ 8.0, 8.0])

qp = ZmpQpController(zc=zc, g=g, W=W, Wp=Wp, Wdu=Wdu, umin=umin, umax=umax)

half = np.array([0.09, 0.045])
L = np.array([-0.10, 0.06])
R = np.array([ 0.10,-0.06])

# fake "state"
x = np.array([0.05, 0.02, 0.0, 0.0])
r = x[0:2]
u_prev = np.zeros(2)
u_des  = np.array([-1.0, -0.5])
p_ref  = np.array([0.0, 0.0])

Hs, hs, _ = support_single(L, half)
Hd, hd, _ = support_double(L, R, half)

u_s, ok_s, st_s = qp.solve(r, u_prev, u_des, p_ref, Hs, hs)
u_d, ok_d, st_d = qp.solve(r, u_prev, u_des, p_ref, Hd, hd)

print("SS:", ok_s, st_s, "u=", u_s, "zmp=", zmp_from(x, u_s, zc, g))
print("DS:", ok_d, st_d, "u=", u_d, "zmp=", zmp_from(x, u_d, zc, g))
