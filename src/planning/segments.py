import numpy as np
from dataclasses import dataclass

@dataclass
class Segment:
    t0: float
    t1: float
    mode: str            # "DS", "SSL", "SSR"
    L: np.ndarray        # left foot center (2,)
    R: np.ndarray        # right foot center (2,)
    p_start: np.ndarray  # COP/ZMP ref start
    p_end: np.ndarray    # COP/ZMP ref end

def build_segments(n_steps: int, ds: float, ss: float, step_length: float, step_width: float, x0: float = 0.0):
    yL = +step_width/2.0
    yR = -step_width/2.0

    L = np.array([x0, yL], dtype=float)
    R = np.array([x0, yR], dtype=float)

    segs = []
    t = 0.0

    # initial DS: midpoint -> left
    p_mid = 0.5*(L+R)
    segs.append(Segment(t, t+ds, "DS", L.copy(), R.copy(), p_mid, L.copy()))
    t += ds

    stance = "L"
    for _ in range(n_steps):
        if stance == "L":
            # SS left
            segs.append(Segment(t, t+ss, "SSL", L.copy(), R.copy(), L.copy(), L.copy()))
            t += ss
            # place right forward
            R = R + np.array([step_length, 0.0], dtype=float)
            # DS: L -> new R
            segs.append(Segment(t, t+ds, "DS", L.copy(), R.copy(), L.copy(), R.copy()))
            t += ds
            stance = "R"
        else:
            segs.append(Segment(t, t+ss, "SSR", L.copy(), R.copy(), R.copy(), R.copy()))
            t += ss
            L = L + np.array([step_length, 0.0], dtype=float)
            segs.append(Segment(t, t+ds, "DS", L.copy(), R.copy(), R.copy(), L.copy()))
            t += ds
            stance = "L"

    return segs, t

def segment_at_time(segs, t: float):
    if t <= segs[0].t0:
        return segs[0]
    for s in segs:
        if s.t0 <= t < s.t1:
            return s
    return segs[-1]

def p_ref_in_segment(seg: Segment, t: float):
    if seg.mode != "DS":
        return seg.p_start.copy()
    s = (t - seg.t0) / (seg.t1 - seg.t0)
    s = float(np.clip(s, 0.0, 1.0))
    return (1.0 - s) * seg.p_start + s * seg.p_end
