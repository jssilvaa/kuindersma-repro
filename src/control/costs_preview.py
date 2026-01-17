# src/control/costs_preview.py
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def kron_I(n: int, A: sp.spmatrix) -> sp.csc_matrix:
    """Sparse Kronecker I_n ⊗ A."""
    return sp.kron(sp.eye(n, format="csc"), A, format="csc").tocsc()


def build_block_weight(N: int, W2: np.ndarray) -> sp.csc_matrix:
    """
    Build block-diagonal weight matrix: Wbar = I_N ⊗ W2.
    Shapes:
      - W2: (2,2)
      - Wbar: (2N, 2N)
    """
    W2 = np.asarray(W2, dtype=float)
    if W2.shape != (2, 2):
        raise ValueError("W2 must be 2x2")
    return kron_I(N, sp.csc_matrix(W2))


def build_diff_operator(N: int, dim: int = 2) -> sp.csc_matrix:
    r"""
    Build D such that ΔU = D U implements:
      Δu_0 = u_0
      Δu_k = u_k - u_{k-1}, k>=1

    Shapes:
      - U: (dim*N,)
      - D: (dim*N, dim*N)

    Structure:
      block row 0: [ I  0  0 ...]
      block row k: [ ... -I  I ...]
    """
    if N <= 0:
        raise ValueError("N must be positive")
    if dim <= 0:
        raise ValueError("dim must be positive")

    I = sp.eye(dim, format="csc")
    Z = sp.csc_matrix((dim, dim))

    blocks = [[Z for _ in range(N)] for __ in range(N)]
    for k in range(N):
        if k == 0:
            blocks[k][k] = I
        else:
            blocks[k][k] = I
            blocks[k][k - 1] = -I

    return sp.bmat(blocks, format="csc").tocsc()


def build_zmp_affine_terms(
    R_prev_U: sp.csc_matrix,
    r_prev_aff: np.ndarray,
    alpha: float,
) -> tuple[sp.csc_matrix, np.ndarray]:
    r"""
    Given:
      R_prev(U) = r_prev_aff + R_prev_U U    where R_prev stacks [r0; r1; ...; r_{N-1}] (2N,)
    ZMP mapping:
      P(U) = R_prev(U) - alpha U
    Returns:
      P_U   : (2N, 2N) sparse
      p_aff : (2N,)    dense
    """
    r_prev_aff = np.asarray(r_prev_aff, dtype=float).reshape(-1)
    if R_prev_U.shape[0] != r_prev_aff.shape[0]:
        raise ValueError("R_prev_U and r_prev_aff dimension mismatch")
    if R_prev_U.shape[0] != R_prev_U.shape[1]:
        raise ValueError("R_prev_U must be square (2N x 2N)")

    N2 = R_prev_U.shape[0]
    P_U = (R_prev_U + sp.eye(N2, format="csc") * (-float(alpha))).tocsc()
    p_aff = r_prev_aff.copy()
    return P_U, p_aff


def build_cost_zmp_preview(
    P_U: sp.csc_matrix,
    p_aff: np.ndarray,
    P_ref: np.ndarray,
    Wp: np.ndarray,
) -> tuple[sp.csc_matrix, np.ndarray]:
    r"""
    Build (P, q) for ZMP tracking cost:
      0.5 * (P(U) - Pref)^T Wp_bar (P(U) - Pref)
    where:
      P(U) = p_aff + P_U U
      Wp_bar = I_N ⊗ Wp

    Returns:
      P_qp : (2N,2N) sparse
      q_qp : (2N,)   dense
    """
    P_ref = np.asarray(P_ref, dtype=float).reshape(-1)
    p_aff = np.asarray(p_aff, dtype=float).reshape(-1)

    if P_U.shape[0] != P_ref.shape[0] or P_U.shape[0] != p_aff.shape[0]:
        raise ValueError("Dimension mismatch in ZMP cost terms")
    if P_U.shape[0] % 2 != 0:
        raise ValueError("Expected 2D stacking (length multiple of 2)")

    N = P_U.shape[0] // 2
    Wp_bar = build_block_weight(N, Wp)

    # Residual affine part: (p_aff - Pref)
    e_aff = (p_aff - P_ref).reshape(-1)

    # P = P_U^T Wp_bar P_U
    P_qp = (P_U.T @ (Wp_bar @ P_U)).tocsc()

    # q = P_U^T Wp_bar e_aff
    q_qp = (P_U.T @ (Wp_bar @ e_aff)).reshape(-1)

    return P_qp, q_qp


def build_cost_du_smoothing(
    N: int,
    Wdu: np.ndarray,
    u_prev: np.ndarray | None = None,
) -> tuple[sp.csc_matrix, np.ndarray, sp.csc_matrix, np.ndarray]:
    r"""
    Build Δu smoothing cost:
      If u_prev is None:
        0.5 * ||D U||_{Wdu_bar}^2
      Else:
        Δu_0 = u_0 - u_prev, Δu_k = u_k - u_{k-1}
        0.5 * ||D U - d0||_{Wdu_bar}^2
        where d0 = [u_prev; 0; ...; 0].

    Returns:
      P_qp : (2N,2N) sparse
      q_qp : (2N,)   dense
      D    : (2N,2N) sparse (for debugging)
      d0   : (2N,)   dense (for debugging)
    """
    D = build_diff_operator(N, dim=2)
    Wdu_bar = build_block_weight(N, Wdu)

    if u_prev is None:
        d0 = np.zeros(2 * N, dtype=float)
    else:
        u_prev = np.asarray(u_prev, dtype=float).reshape(-1)
        if u_prev.shape != (2,):
            raise ValueError("u_prev must be shape (2,)")
        d0 = np.zeros(2 * N, dtype=float)
        d0[0:2] = u_prev

    # Cost: 0.5 (D U - d0)^T W (D U - d0)
    # => 0.5 U^T (D^T W D) U  +  (- D^T W d0)^T U  + const
    P_qp = (D.T @ (Wdu_bar @ D)).tocsc()
    q_qp = (-(D.T @ (Wdu_bar @ d0))).reshape(-1)

    return P_qp, q_qp, D, d0


def combine_quadratic_costs(*terms: tuple[sp.csc_matrix, np.ndarray]) -> tuple[sp.csc_matrix, np.ndarray]:
    """
    Sum multiple (P,q) terms. Ensures consistent shapes.
    """
    if not terms:
        raise ValueError("No terms provided")

    P_sum = None
    q_sum = None
    for P, q in terms:
        P = P.tocsc()
        q = np.asarray(q, dtype=float).reshape(-1)
        if P_sum is None:
            P_sum = P.copy()
            q_sum = q.copy()
        else:
            if P.shape != P_sum.shape or q.shape != q_sum.shape:
                raise ValueError("Shape mismatch when combining costs")
            P_sum = (P_sum + P).tocsc()
            q_sum = (q_sum + q).reshape(-1)

    # Symmetrize numerically (good practice for OSQP)
    P_sum = (0.5 * (P_sum + P_sum.T)).tocsc()
    return P_sum, q_sum
