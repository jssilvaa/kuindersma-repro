import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
    
from src.models.support_polygon import support_single, support_double, margin_halfspaces

half = np.array([0.09, 0.045])
L = np.array([-0.10, 0.06])
R = np.array([ 0.10,-0.06])

Hs, hs, _ = support_single(L, half)
Hd, hd, _ = support_double(L, R, half)

print("single H shape:", Hs.shape, "finite edges:", np.sum(np.isfinite(hs)))
print("double  H shape:", Hd.shape, "finite edges:", np.sum(np.isfinite(hd)))

# test margins at foot centers
print("margin single at L:", margin_halfspaces(Hs, hs, L))
print("margin double at L:", margin_halfspaces(Hd, hd, L))
print("margin double at R:", margin_halfspaces(Hd, hd, R))
