import numpy as np

def pd_u_des(x: np.ndarray, 
            r_ref: np.ndarray,
            v_ref: np.ndarray, 
            Kp: float, 
            Kd: float) -> np.ndarray:
    r = x[0:2]
    v = x[2:4]
    u = - Kp * (r - r_ref) - Kd * (v - v_ref)
    return u