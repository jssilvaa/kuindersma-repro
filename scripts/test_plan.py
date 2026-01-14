import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
    
from src.planning.footsteps import build_walk_schedule
from src.planning.zmp_plan import phase_at_time, p_ref_from_phase

phases, T = build_walk_schedule(
    n_steps=4, ds=0.2, ss=0.6,
    step_length=0.18, step_width=0.20, x0=0.0
)

print("T_total:", T)
for t in np.unique([e for e in [0.0, 0.1, 0.25, 0.7, 0.85, 1.1, 1.8] if e <= T] + [T]):
    ph = phase_at_time(phases, t)
    pref = p_ref_from_phase(ph, t)
    print(f"t={t:4.2f} mode={ph.mode:3s} p_ref={pref}")
