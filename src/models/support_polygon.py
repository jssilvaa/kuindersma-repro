import numpy as np
from scipy.spatial import ConvexHull

def rect_corners(center: np.ndarray, half_sizes: np.ndarray):
    cx, cy = float(center[0]), float(center[1])
    hx, hy = float(half_sizes[0]), float(half_sizes[1])
    return np.array([
        [cx - hx, cy - hy],
        [cx + hx, cy - hy],
        [cx + hx, cy + hy],
        [cx - hx, cy + hy],
    ], dtype=float)

def polygon_halfspaces_from_hull(points: np.ndarray):
    """
    Given a set of 2D points, compute convex hull and return:
      H (m x 2), h (m,) s.t. H p <= h describes the hull.
    Also return ordered hull vertices (counterclockwise).
    """
    hull = ConvexHull(points)
    verts = points[hull.vertices]  # ordered (CCW)

    # Hull equations: each row is [a, b, c] with a*x + b*y + c == 0 on the boundary
    # For points inside hull: a*x + b*y + c <= 0
    eq = hull.equations  # shape (m, 3)

    H = eq[:, 0:2].copy()
    h = (-eq[:, 2]).copy()  # because a*x + b*y + c <= 0  ->  a*x + b*y <= -c
    return H, h, verts

def pad_halfspaces(H: np.ndarray, h: np.ndarray, m_target: int = 8):
    """
    Pad halfspace constraints to fixed size:
      Hpad p <= hpad
    Padded rows are 0 <= +inf (never active).
    """
    m = H.shape[0]
    if m > m_target:
        # In practice hull of 8 rect corners should not exceed 8 edges,
        # but if it does, you'd rather know now.
        raise ValueError(f"Too many hull halfspaces: {m} > {m_target}")

    Hpad = np.zeros((m_target, 2), dtype=float)
    hpad = np.full((m_target,), np.inf, dtype=float)

    Hpad[:m, :] = H
    hpad[:m] = h
    return Hpad, hpad

def support_single(center: np.ndarray, half_sizes: np.ndarray, m_target: int = 8):
    """
    Single support polygon: the rectangle itself.
    Returns (H,h,verts) with H,h padded to m_target.
    """
    # Rectangle as 4 halfspaces (axis-aligned)
    cx, cy = float(center[0]), float(center[1])
    hx, hy = float(half_sizes[0]), float(half_sizes[1])

    H = np.array([
        [ 1.0, 0.0],
        [-1.0, 0.0],
        [ 0.0, 1.0],
        [ 0.0,-1.0],
    ], dtype=float)

    h = np.array([
        cx + hx,
        -(cx - hx),
        cy + hy,
        -(cy - hy),
    ], dtype=float)

    verts = rect_corners(center, half_sizes)
    Hpad, hpad = pad_halfspaces(H, h, m_target=m_target)
    return Hpad, hpad, verts

def support_double(center_L: np.ndarray, center_R: np.ndarray, half_sizes: np.ndarray, m_target: int = 8):
    """
    Double support polygon = convex hull of both foot rectangles.
    Returns (H,h,verts) with H,h padded to m_target.
    """
    pts = np.vstack([
        rect_corners(center_L, half_sizes),
        rect_corners(center_R, half_sizes),
    ])
    H, h, verts = polygon_halfspaces_from_hull(pts)
    Hpad, hpad = pad_halfspaces(H, h, m_target=m_target)
    return Hpad, hpad, verts

def margin_halfspaces(H: np.ndarray, h: np.ndarray, p: np.ndarray) -> float:
    """
    min_i (h_i - H_i p)
    Negative => violation.
    Works even with +inf padded h rows.
    """
    m = h - H @ p
    return float(np.min(m))
