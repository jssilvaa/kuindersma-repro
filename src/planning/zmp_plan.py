import numpy as np

def phase_at_time(phases, t: float): 
    # return the Phase that contains t (clamp to last)
    if t <= phases[0].t0: 
        return phases[0]
    for phase in phases: 
        if phase.t0 <= t <= phase.t1: 
            return phase
    return phases[-1]

def p_ref_from_phase(ph, t: float) -> np.ndarray:
    """ 
    linear interpolation of foot positions over phase 
    """
    # ss: p_start == p_end constant
    if ph.mode != 'DS': 
        return ph.p_start.copy()
    
    # ds: linear interpolation: p_start -> p_end 
    if ph.t1 <= ph.t0: 
        return ph.p_end.copy()  # avoid div by zero

    alpha = (t - ph.t0) / (ph.t1 - ph.t0)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    p_ref = (1.0 - alpha) * ph.p_start + alpha * ph.p_end
    return p_ref