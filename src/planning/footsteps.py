import numpy as np 
from dataclasses import dataclass

@dataclass
class Phase: 
    t0: float
    t1: float 
    mode: str # 'SSL' 'SSR' 'DS'
    p_start: np.ndarray 
    p_end: np.ndarray

def build_walk_schedule(
        n_steps: int,
        ds: float,
        ss: float,
        step_length: float,
        step_width: float,
        x0: float = 0.0
):
    """
    alternating walk: 
        start with both feet on ground (DS)
        first single support = left (right up)
        then alternate
    
    Feet are axis-aligned rectangles. we only move foot centers in x 
    y positions fixed at +/- step_width / 2.
    """
    yL = +step_width / 2.0
    yR = -step_width / 2.0

    L = np.array([x0, yL], dtype=float)
    R = np.array([x0, yR], dtype=float)

    phases = []
    t = 0.0 

    # initial ds: move COP from midpoint to first stance L 
    p_mid = 0.5 * (L + R)
    p_first = L.copy()

    phases.append(Phase(t, t + ds, 'DS', p_mid, p_first))
    t += ds

    stance = "L" # first ss stance
    for k in range(n_steps): 
        if stance == "L": 
            phases.append(Phase(t, t + ss, 'SSL', L.copy(), L.copy()))
            t += ss
            # place R foot at end of SS 
            R = R + np.array([step_length, 0.0], dtype=float)
            phases.append(Phase(t, t + ds, 'DS', L.copy(), R.copy()))
            t += ds
            stance = "R"
        else:
            phases.append(Phase(t, t + ss, 'SSR', R.copy(), R.copy()))
            t += ss
            # place L foot at end of SS 
            L = L + np.array([step_length, 0.0], dtype=float)
            phases.append(Phase(t, t + ds, 'DS', R.copy(), L.copy()))
            t += ds
            stance = "L"
        
        T_total = t
        return phases, T_total
