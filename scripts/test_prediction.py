# scripts/test_prediction.py
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.lipm import discretize_double_integrator, step
from src.control.prediction import build_prediction_matrices, build_C_r


def rollout_iterative(Ad, Bd, x0, U):
    x = x0.copy()
    xs = []
    for k in range(U.shape[0]):
        x = step(x, U[k], Ad, Bd)   # produces x_{k+1}
        xs.append(x.copy())
    return np.stack(xs, axis=0)     # (N, nx)


def main():
    np.random.seed(7)

    dt = 0.01
    Ad, Bd = discretize_double_integrator(dt)
    nx = Ad.shape[0]
    nu = Bd.shape[1]

    N = 50
    x0 = np.random.randn(nx)
    U = 0.5 * np.random.randn(N, nu)

    Sx, Su = build_prediction_matrices(Ad, Bd, N)

    # Stacked prediction
    X_stack = (Sx @ x0 + Su @ U.reshape(-1)).ravel()  # (nx*N,)
    X_mat = X_stack.reshape(N, nx)  # rows are x1..xN

    # Iterative rollout
    X_it = rollout_iterative(Ad, Bd, x0, U)

    err = np.max(np.abs(X_mat - X_it))
    print("max |X_stack - X_iter| =", err)
    assert err < 1e-10, f"Prediction matrices mismatch: {err}"

    # Test C_r extraction (r components)
    C_r = build_C_r(N)
    R_stack = (C_r @ X_stack).ravel()
    R_mat = R_stack.reshape(N, 2)

    err_r = np.max(np.abs(R_mat - X_it[:, 0:2]))
    print("max |R_stack - r_iter| =", err_r)
    assert err_r < 1e-10, f"C_r mismatch: {err_r}"

    print("OK: prediction matrices and C_r verified.")


if __name__ == "__main__":
    main()
