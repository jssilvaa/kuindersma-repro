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

        # fixed sparsity (4 polygon constraints + 2 bounds on each input) 
        self.m = 4 + 2 
        self.n = 2

        # setup OSQP problem
        P0 = sp.csc_matrix(np.eye(2))
        A0 = sp.csc_matrix(np.zeros((self.m, 2)))
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
                h: np.ndarray) -> np.ndarray:
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

        # P = W + D.T @ Wp @ D + Wdu
        P = self.W + self.D.T @ self.Wp @ self.D + self.Wdu
        P = 0.5 * (P + P.T)  # ensure symmetry

        # q = - W @ u_des - D.T @ Wp @ (p_ref - r) - Wdu @ u_prev
        q = - self.W @ u_des - self.D.T @ self.Wp @ (p_ref - r) - self.Wdu @ u_prev

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
        self.prob.update(Px=sp.csc_matrix(P).data, q=q, Ax=sp.csc_matrix(A).data, l=l, u=u)

        res = self.prob.solve()

        ok = (res.info.status_val in (1,2))  # solved or solved inaccurate
        if not ok:
            return u_prev.copy(), False, res.info.status
        return np.array(res.x), True, res.info.status
