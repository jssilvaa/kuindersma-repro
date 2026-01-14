import numpy as np

def rectangle_Hh(cx: float, cy: float, hx: float, hy:float):
    """
    Axis-aligned rectangle centered at (cx, cy) with half-lengths hx and hy.
    Returns H, h s.t. H.T @ [x, y] <= h
    """
    H = np.array([[1, 0],
                  [-1, 0],
                  [0, 1],
                  [0, -1]])
    h = np.array([cx + hx,
                  -cx + hx,
                  cy + hy,
                  -cy + hy])
    return H, h

def margin(H: np.ndarray, h: np.ndarray, p: np.ndarray) -> float:
    """
    Returns the minimum margin over all halfspaces: 
    margin = min_i h_i - H_i @ p
    Negative means p is outside the polytope defined by H, h.
    """
    margins = h - H @ p
    return np.min(margins)

