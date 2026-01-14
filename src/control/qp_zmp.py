import numpy as np
import scipy.sparse as sp 
import osqp

class ZmpQpController:
    def __init__(self,
                 zc: float,
                 g: float,
                 W: np.ndarray,
                 Wp: np.ndarray,
                 Wdu: np.ndarray,
                 umin: np.ndarray,
                 umax: np.ndarray): 
        self.zc = zc
        self.g = g 

        self.W = W 
        self.Wp = Wp 
        self.Wdu = Wdu 
        self.umin = umin
        self.umax = umax

        self.alpha = self.zc / self.g
        self.D = - self.alpha * np.eye(2)

        # fixed sparsity (8 polygon constraints + 2 bounds on each input) 
        self.m = 8 + 2 
        self.n = 2

        # OSQP requires that matrix updates keep the same sparsity pattern.
        # Pre-allocate *fixed* patterns for P (upper triangular) and A.
        # We intentionally store zero-valued entries so that the number of
        # non-zeros (nnz) stays constant across updates.
        self._P_pattern = sp.csc_matrix(np.triu(np.ones((self.n, self.n), dtype=float)))
        self._A_pattern = sp.csc_matrix(np.ones((self.m, self.n), dtype=float))

        # setup OSQP problem
        # P is provided as upper triangular; keep pattern fixed.
        P0 = self._P_pattern.copy()
        P0.data[:] = np.array([1.0, 0.0, 1.0], dtype=float)

        # A pattern is fully dense (m x n) with stored zeros.
        A0 = self._A_pattern.copy()
        A0.data[:] = 0.0
        q0 = np.zeros(2)
        l0 = -np.inf * np.ones(self.m)
        u0 = np.inf * np.ones(self.m)

        self.prob = osqp.OSQP()
        self.prob.setup(P=P0, q=q0, A=A0, l=l0, u=u0, warm_start=True, verbose=False)

    def solve(self, 
                r: np.ndarray,
                u_prev: np.ndarray,
                u_des: np.ndarray,
                p_ref: np.ndarray,
                H: np.ndarray,
                h: np.ndarray,
                W: np.ndarray | None = None,
                Wp: np.ndarray | None = None,
                Wdu: np.ndarray | None = None) -> np.ndarray:
        """
        Solve QP over u in R^2

        Objective: 
            min 1/2 (u - u_des).T @ W @ (u - u_des)
                + 1/2 (p - p_ref).T @ Wp @ (p - p_ref), p = r + D @ u 
                + 1/2 (u - u_prev).T @ Wdu @ (u - u_prev)

        Constraints: 
            H @ (r + D @ u) <= h 
            umin <= u <= umax
        """
        r = r.reshape(2,)
        u_prev = u_prev.reshape(2,)
        u_des = u_des.reshape(2,)
        p_ref = p_ref.reshape(2,)

        W = self.W if W is None else W
        Wp = self.Wp if Wp is None else Wp
        Wdu = self.Wdu if Wdu is None else Wdu

        # P = W + D.T @ Wp @ D + Wdu
        P = W + self.D.T @ Wp @ self.D + Wdu
        P = 0.5 * (P + P.T)  # ensure symmetry

        # q = - W @ u_des - D.T @ Wp @ (p_ref - r) - Wdu @ u_prev
        q = - W @ u_des - self.D.T @ Wp @ (p_ref - r) - Wdu @ u_prev

        # Constraints: A u in [l,u]
        # (H D) u <= h - H r
        A1 = H @ self.D
        u1 = h - H @ r 
        l1 = -np.inf * np.ones_like(u1)

        # umin <= I u <= umax
        A2 = np.eye(2)
        l2 = self.umin
        u2 = self.umax

        A = np.vstack((A1, A2))
        l = np.hstack((l1, l2))
        u = np.hstack((u1, u2))

        # update OSQP problem
        # Keep the same sparsity pattern as in setup.
        # P (upper triangular) data order for 2x2 CSC of triu(ones):
        #   [(0,0), (0,1), (1,1)]
        Px = np.array([P[0, 0], P[0, 1], P[1, 1]], dtype=float)

        # A data order for CSC of ones(m,n) is column-major:
        #   col0 rows 0..m-1, then col1 rows 0..m-1
        Ax = np.hstack((A[:, 0], A[:, 1])).astype(float, copy=False)

        self.prob.update(Px=Px, q=q, Ax=Ax, l=l, u=u)
        res = self.prob.solve()

        ok = (res.info.status_val in (1,2))  # solved or solved inaccurate
        if not ok:
            return np.zeros(2), False, res.info.status
        return np.array(res.x), True, res.info.status
