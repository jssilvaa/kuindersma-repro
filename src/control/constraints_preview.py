# src/control/constraints_preview.py
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from src.planning.segments import segment_at_time, p_ref_in_segment
from src.models.support_polygon import support_single, support_double
from src.control.prediction import build_prediction_matrices, build_C_r


def build_polygon_preview_constraints(
    segs,
    t0: float,
    dt: float,
    N: int,
    Ad: np.ndarray,
    Bd: np.ndarray,
    x0: np.ndarray,
    zc: float,
    g: float,
    foot_half_sizes: np.ndarray,
    m_target: int = 8,
):
    """
    Build stacked polygon constraints over a horizon:
        For i=1..N:
            H_i p_i <= h_i
            p_i = r_{i-1} - alpha u_{i-1}
    with stacking convention:
        X = [x1; ...; xN]         shape (4N,)
        U = [u0; ...; u_{N-1}]    shape (2N,)

    Returns (OSQP form):
        A_poly : csc_matrix, shape (m_target*N, 2N)
        l_poly : ndarray,    shape (m_target*N,)  (=-inf)
        u_poly : ndarray,    shape (m_target*N,)  (= h_stack - H_stack * r_prev_aff)
        meta   : dict with keys 'H_list','h_list','mode_list','verts_list','t_list'

    Notes on array conversions / shapes:
    - OSQP expects 1D numpy arrays for l,u.
    - We keep A_poly sparse CSC.
    - We avoid `.A` and `.toarray()` in core; use `.reshape(-1)` explicitly.
    """
    x0 = np.asarray(x0).reshape(-1)
    if x0.shape[0] != Ad.shape[0]:
        raise ValueError("x0 dimension mismatch with Ad")

    alpha = float(zc / g)

    # --- Prediction matrices (sparse) ---
    Sx, Su = build_prediction_matrices(Ad, Bd, N)   # Sx: (4N,4), Su: (4N,2N)
    C_r = build_C_r(N)                              # (2N,4N)

    # R = C_r X = C_r(Sx x0 + Su U) = (C_r Sx)x0 + (C_r Su)U
    # r1_to_rN_aff is the affine term (depends on x0) for [r1;..;rN].
    # Important: (C_r @ Sx @ x0) returns a 2N vector. Keep it 1D.
    r1_to_rN_aff = (C_r @ (Sx @ x0)).reshape(-1)    # shape (2N,) = [r1;..;rN]
    R_U = (C_r @ Su).tocsc()                        # shape (2N, 2N) maps U -> [r1;..;rN]

    # We constrain p_i using the *previous* position r_{i-1} with u_{i-1}.
    # This matches the step-by-step simulation convention where p is computed
    # from the current state x_k and the input u_k.
    # So we need a stacked vector of positions:
    #   R_prev = [r0; r1; ...; r_{N-1}]  (2N vector)
    r0 = x0[0:2].reshape(-1)
    r_prev_aff = np.concatenate([r0, r1_to_rN_aff[: 2 * (N - 1)]], axis=0)

    # Build R_prev_U with a leading 2 rows of zeros (r0 does not depend on U)
    # followed by the first 2*(N-1) rows of R_U.
    Z2 = sp.csc_matrix((2, 2 * N))
    R_prev_U = sp.vstack([Z2, R_U[: 2 * (N - 1), :]], format="csc")

    # ZMP mapping: p_i = r_{i-1} - alpha u_{i-1}
    Dz = sp.eye(2 * N, format="csc") * (-alpha)     # maps U -> (-alpha U)

    # So: P = (R_prev_U + Dz)U + r_prev_aff
    P_U = (R_prev_U + Dz).tocsc()                   # shape (2N, 2N)

    # --- Build per-step polygons (H_i, h_i) and stack ---
    H_blocks = []
    h_blocks = []
    mode_list = []
    verts_list = []
    t_list = []

    for i in range(1, N + 1):
        # We constrain p_i computed from (r_{i-1}, u_{i-1}), which corresponds
        # to time t0 + (i-1)*dt in the rollout convention.
        ti = float(t0 + (i - 1) * dt)
        seg = segment_at_time(segs, ti)

        L = seg.L
        R = seg.R

        if seg.mode == "DS":
            H_i, h_i, verts = support_double(L, R, foot_half_sizes, m_target=m_target)
        elif seg.mode == "SSL":
            H_i, h_i, verts = support_single(L, foot_half_sizes, m_target=m_target)
        else:  # "SSR"
            H_i, h_i, verts = support_single(R, foot_half_sizes, m_target=m_target)

        # H_i: (m_target,2), h_i: (m_target,)
        H_blocks.append(sp.csc_matrix(H_i))
        h_blocks.append(np.asarray(h_i).reshape(-1))

        mode_list.append(seg.mode)
        verts_list.append(verts)
        t_list.append(ti)

    # Block-diagonal H_stack acting on stacked P = [p1;..;pN]
    # H_stack shape: (mN, 2N)
    H_stack = sp.block_diag(H_blocks, format="csc")

    # h_stack shape: (mN,)
    h_stack = np.concatenate(h_blocks, axis=0)

    # Constraint is: H_stack * P <= h_stack
    # Substitute: P = P_U * U + r_prev_aff
    # => (H_stack * P_U) U <= h_stack - H_stack * r_prev_aff
    A_poly = (H_stack @ P_U).tocsc()

    # Compute affine RHS term:
    # H_stack * r_prev_aff returns (mN,) vector.
    Hr0 = (H_stack @ r_prev_aff).reshape(-1)

    u_poly = (h_stack - Hr0).reshape(-1)
    l_poly = (-np.inf) * np.ones_like(u_poly)

    meta = {
        "H_list": H_blocks,
        "h_list": h_blocks,
        "mode_list": mode_list,
        "verts_list": verts_list,
        "t_list": np.array(t_list, dtype=float),
        "r_prev_aff": r_prev_aff,     # useful for debugging
        "r1_to_rN_aff": r1_to_rN_aff, # useful for debugging
        "H_stack": H_stack,
        "h_stack": h_stack,
        "R_prev_U": R_prev_U
    }
    
    return A_poly, l_poly, u_poly, meta
