# src/control/prediction.py
from __future__ import annotations

import numpy as np 
import scipy.sparse as sp 

def kron_I(n: int, A: sp.spmatrix) -> sp.spmatrix: 
    """
    Kronecker product of I_n with sparse matrix A.
    """
    return sp.kron(sp.eye(n, format="csc"), A, format="csc")

def build_C_r(N: int) -> sp.csc_matrix:
    """
    Extracts position r=[x,y] from stacked states X=[x1;...;xN] where each 
    xi = [rx,ry,vx,vy]. Returns C_r such that R = C_r X, with R stacked as 
    [r1;r2;...;rN] (2N vector)
    Shape: (2N, 4N)
    """
    # for one state, select first two rows: [I_2; 0_{2x2}]
    C1 = sp.hstack([sp.eye(2, format="csc"), sp.csc_matrix((2,2))], format="csc")
    return kron_I(N, C1)

def build_prediction_matrices(Ad: np.ndarray, Bd: np.ndarray, N: int) -> tuple[sp.csc_matrix, sp.csc_matrix]: 
    r"""
    Build stacked prediction matrices for: 
        x_{k+1} = Ad x_k + Bd u_k 

    Define: 
        X = [x1; x2; ...; xN] \in R^{n_x N}
        U = [u0; u1; ...; u_{N-1}] \in R^{n_u N}
    
    Then: 
        X = Sx x0 + Su U 

    Returns: 
        Sx: (n_x N, n_x) sparse
        Su: (n_x N, n_u N) sparse block lower triangular Toeplitz    
    """
    Ad = np.asarray(Ad)
    Bd = np.asarray(Bd)
    nx = Ad.shape[0]
    nu = Bd.shape[1]

    if Ad.shape != (nx, nx):
        raise ValueError("Ad must be square")
    if Bd.shape != (nx, nu):
        raise ValueError("Bd must have compatible dimensions with Ad")
    if N <= 0:
        raise ValueError("N must be positive")
    
    # precompute powers of Ad
    Ad_pows = [np.eye(nx)]
    for _ in range(N):
        Ad_pows.append(Ad @ Ad_pows[-1])

    # Sx: stack of Ad^i for i=1..N
    Sx_blocks = [sp.csc_matrix(Ad_pows[i+1]) for i in range(N)]
    Sx = sp.vstack(Sx_blocks, format="csc")

    # Su: block lower triangular Toeplitz
    blocks = [[None for _ in range(N)] for _ in range(N)]
    Z = sp.csc_matrix((nx, nu))
    for i in range(N):
        for j in range(N):
            if j <= i:
                blocks[i][j] = Ad_pows[i - j] @ Bd  # keep as dense
            else:
                blocks[i][j] = Z
    Su = sp.bmat(blocks, format="csc")

    return Sx, Su

def predict_stacked(Sx: sp.csc_matrix, Su: sp.csc_matrix, x0: np.ndarray, U: np.ndarray) -> np.ndarray: 
    """ 
    Convenience: returns X = Sx x0 + Su U as a dense array.
    """
    x0 = np.asarray(x0).reshape(-1)
    U = np.asarray(U).reshape(-1)
    return (Sx @ x0 + Su @ U).toarray().ravel()

def build_zmp_mapping_blocks(N: int, alpha: float) -> sp.csc_matrix:
    r"""
    Returns Dz such that:
        P = R + Dz U
    where
        p_i = r_i - alpha u_i   (with u_i = u_{i-1} depending on indexing conventions)

    Here we assume the horizon is aligned as:
        X stacks x1..xN
        U stacks u0..u_{N-1}
    and we want P stacks p1..pN corresponding to each predicted state x1..xN using u0..u_{N-1}.
    So:
        p_{i} = r_{i} - alpha u_{i-1}, i=1..N

    Therefore Dz is block-diagonal (2N x 2N) with -alpha I2 on the diagonal, mapping U directly.
    """
    D1 = sp.csc_matrix(-alpha * np.eye(2))
    return kron_I(N, D1)