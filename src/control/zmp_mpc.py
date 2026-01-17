# src/control/zmp_mpc.py
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import osqp

from src.models.lipm import discretize_double_integrator
from src.control.prediction import build_prediction_matrices, build_C_r
from src.control.costs_preview import (
    build_zmp_affine_terms,
    build_cost_zmp_preview,
    build_cost_du_smoothing,
    build_block_weight,
    combine_quadratic_costs,
)
from src.control.constraints_preview import build_polygon_preview_constraints
from src.planning.segments import segment_at_time


def _csc_data_view(M: sp.csc_matrix) -> np.ndarray:
    """Return a writable view of CSC data; ensures CSC format."""
    if not sp.isspmatrix_csc(M):
        M = M.tocsc()
    return M.data


def _upper_triangular_csc_pattern(n: int) -> sp.csc_matrix:
    """
    Build an upper-triangular sparsity pattern for a dense n×n symmetric matrix,
    stored as CSC with only upper-tri entries present.
    """
    # Create a dense boolean mask and convert to sparse -> upper triangular nnz pattern.
    # This is fine for n up to a few hundred (we're at 2N ~ 40..200 typically).
    mask = np.triu(np.ones((n, n), dtype=float))
    Ppat = sp.csc_matrix(mask)
    # Store zeros but keep nnz structure
    Ppat.data[:] = 0.0
    return Ppat


class ZmpMpcController:
    """
    ZMP-preview MPC QP solver (OSQP) with fixed sparsity patterns.

    Decision:
      U = [u0; u1; ...; u_{N-1}] ∈ R^(2N)

    Constraints (from Block 2):
      A_poly U <= u_poly   (polygon halfspaces stacked)
    Plus box:
      umin <= u_k <= umax  for all k

    Objective (from Block 3):
      0.5 U^T P U + q^T U
      with ZMP tracking + Δu smoothing (optionally uses u_prev in Δu cost)

    Notes on array conversions:
    - OSQP expects CSC sparse for P and A.
    - Updates must preserve nnz pattern; we only mutate `.data`.
    - Vectors q,l,u are 1D np.ndarray.
    """

    def __init__(
        self,
        N: int,
        m_target: int = 8,
        dt: float = 0.01,
        zc: float = 0.9,
        g: float = 9.81,
        Wp: np.ndarray | None = None,
        Wr: np.ndarray | None = None,
        Wv: np.ndarray | None = None,
        v_ref: np.ndarray | None = None,
        Wu: np.ndarray | None = None,
        Wdu: np.ndarray | None = None,
        slack_weight: float | None = None,
        x_min: float | None = None,
        x_max: float | None = None,
        vx_min: float | None = None,
        vx_max: float | None = None,
        y_min: float | None = None,
        y_max: float | None = None,
        vy_min: float | None = None,
        vy_max: float | None = None,
        umin: np.ndarray | None = None,
        umax: np.ndarray | None = None,
        verbose: bool = False,
        last_result: osqp.Result | None = None,
        last_info: osqp.Info | None = None,
    ):
        self.N = int(N)
        self.m_target = int(m_target)
        self.dt = float(dt)
        self.zc = float(zc)
        self.g = float(g)
        self.alpha = self.zc / self.g

        self.Wp = np.eye(2) * 50.0 if Wp is None else np.asarray(Wp, dtype=float)
        self.Wr = np.eye(2) * 0.0 if Wr is None else np.asarray(Wr, dtype=float)
        self.Wv = np.eye(2) * 0.0 if Wv is None else np.asarray(Wv, dtype=float)
        self.v_ref = np.zeros(2, dtype=float) if v_ref is None else np.asarray(v_ref, dtype=float).reshape(2,)
        self.Wu = np.eye(2) * 0.0 if Wu is None else np.asarray(Wu, dtype=float)
        self.Wdu = np.eye(2) * 0.1 if Wdu is None else np.asarray(Wdu, dtype=float)

        # Soft constraints: add nonnegative slack s for polygon inequalities and penalize it.
        # Set slack_weight to a large value (e.g. 1e5..1e7) to keep violations tiny.
        self.slack_weight = 0.0 if slack_weight is None else float(slack_weight)
        self.use_slack = self.slack_weight > 0.0

        # Optional hard bounds on predicted COM state to prevent runaway.
        self.x_min = None if x_min is None else float(x_min)
        self.x_max = None if x_max is None else float(x_max)
        self.vx_min = None if vx_min is None else float(vx_min)
        self.vx_max = None if vx_max is None else float(vx_max)
        self.y_min = None if y_min is None else float(y_min)
        self.y_max = None if y_max is None else float(y_max)
        self.vy_min = None if vy_min is None else float(vy_min)
        self.vy_max = None if vy_max is None else float(vy_max)
        self.use_state_bounds = any(
            v is not None
            for v in (
                self.x_min,
                self.x_max,
                self.vx_min,
                self.vx_max,
                self.y_min,
                self.y_max,
                self.vy_min,
                self.vy_max,
            )
        )

        self.umin = np.array([-8.0, -8.0]) if umin is None else np.asarray(umin, dtype=float).reshape(2,)
        self.umax = np.array([ 8.0,  8.0]) if umax is None else np.asarray(umax, dtype=float).reshape(2,)

        self.verbose = bool(verbose)

        # System matrices (fixed for a given dt)
        self.Ad, self.Bd = discretize_double_integrator(self.dt)

        # Dimensions
        self.nU = 2 * self.N
        self.nx = self.Ad.shape[0]  # 4

        # Constraint rows:
        # polygon: m_target * N
        self.m_poly = self.m_target * self.N

        # Slack variables (one per polygon inequality)
        self.nS = self.m_poly if self.use_slack else 0

        # Total decision: z = [U; s]
        self.nZ = self.nU + self.nS

        # box: 2 * N (since u in R^2 per step)
        self.m_box = self.nU
        # state bounds rows (y and/or vy over horizon)
        self.m_state = 0
        if (self.x_min is not None) or (self.x_max is not None):
            self.m_state += self.N
        if (self.vx_min is not None) or (self.vx_max is not None):
            self.m_state += self.N
        if (self.y_min is not None) or (self.y_max is not None):
            self.m_state += self.N
        if (self.vy_min is not None) or (self.vy_max is not None):
            self.m_state += self.N
        # slack nonnegativity constraints
        self.m_slack = self.nS
        self.m_total = self.m_poly + self.m_state + self.m_box + self.m_slack

        # Prebuild fixed structures used each solve
        self.Sx, self.Su = build_prediction_matrices(self.Ad, self.Bd, self.N)
        self.Cr = build_C_r(self.N)

        # Precompute state-bound mapping (depends only on Sx/Su structure)
        self._C_state = None
        self._A_state_U = None

        # OSQP structures (allocated in setup)
        self._P_pat = None
        self._A_pat = None
        self._P = None
        self._A = None
        self._prob = None

        # For warm-start shifting
        self._U_prev = np.zeros(self.nU, dtype=float)
        self._z_prev = np.zeros(self.nZ, dtype=float)

        self.last_result = last_result
        self.last_info = last_info

    def _setup_osqp(self):
        # --- P pattern: upper triangular dense
        self._P_pat = _upper_triangular_csc_pattern(self.nZ)
        P0 = self._P_pat.copy()  # data all zeros
        q0 = np.zeros(self.nZ, dtype=float)

        # --- A pattern:
        # top block: A_poly pattern depends on H_stack and prediction; BUT its sparsity is stable:
        # - H_stack is block diagonal with dense 2 cols per step.
        # - P_U is banded-ish / generally dense lower-tri via R_prev_U; but in practice (2N×2N) dense-ish.
        # Easiest CLEAN approach: allocate A_poly as dense CSC pattern (all entries present) but still structured.
        # This avoids “pattern changes” when H padding switches (inf rows etc.).
        # Poly block pattern (dense in U columns). If using slack: add -I in slack columns.
        A_poly_U_pat = sp.csc_matrix(np.ones((self.m_poly, self.nU), dtype=float))
        A_poly_U_pat.data[:] = 0.0
        if self.use_slack:
            A_poly_s = -sp.eye(self.nS, format="csc")
            A_poly_pat = sp.hstack([A_poly_U_pat, A_poly_s], format="csc")
        else:
            A_poly_pat = A_poly_U_pat

        # Optional state-bound block pattern (dense in U columns; zeros in slack columns)
        if self.m_state > 0:
            A_state_U_pat = sp.csc_matrix(np.ones((self.m_state, self.nU), dtype=float))
            A_state_U_pat.data[:] = 0.0
            if self.use_slack:
                A_state_pat = sp.hstack([A_state_U_pat, sp.csc_matrix((self.m_state, self.nS))], format="csc")
            else:
                A_state_pat = A_state_U_pat
        else:
            A_state_pat = None

        # Box constraints on U
        if self.use_slack:
            A_box = sp.hstack([sp.eye(self.nU, format="csc"), sp.csc_matrix((self.nU, self.nS))], format="csc")
        else:
            A_box = sp.eye(self.nU, format="csc")

        # Slack nonnegativity
        if self.use_slack:
            A_slack = sp.hstack([sp.csc_matrix((self.nS, self.nU)), sp.eye(self.nS, format="csc")], format="csc")
            blocks = [A_poly_pat]
            if A_state_pat is not None:
                blocks.append(A_state_pat)
            blocks.extend([A_box, A_slack])
            self._A_pat = sp.vstack(blocks, format="csc")
        else:
            blocks = [A_poly_pat]
            if A_state_pat is not None:
                blocks.append(A_state_pat)
            blocks.append(A_box)
            self._A_pat = sp.vstack(blocks, format="csc")
        A0 = self._A_pat.copy()  # zeros in poly rows + identity in box rows

        # bounds
        # Polygon rows will be updated in solve.
        # State-bound rows will be updated in solve.
        if self.use_slack:
            l_parts = [(-np.inf) * np.ones(self.m_poly)]
            u_parts = [( np.inf) * np.ones(self.m_poly)]
            if self.m_state > 0:
                l_parts.append((-np.inf) * np.ones(self.m_state))
                u_parts.append(( np.inf) * np.ones(self.m_state))
            l_parts.append(self.umin.repeat(self.N))
            u_parts.append(self.umax.repeat(self.N))
            l_parts.append(np.zeros(self.nS, dtype=float))
            u_parts.append(( np.inf) * np.ones(self.nS))
            l0 = np.hstack(l_parts).astype(float)
            u0 = np.hstack(u_parts).astype(float)
        else:
            l_parts = [(-np.inf) * np.ones(self.m_poly)]
            u_parts = [( np.inf) * np.ones(self.m_poly)]
            if self.m_state > 0:
                l_parts.append((-np.inf) * np.ones(self.m_state))
                u_parts.append(( np.inf) * np.ones(self.m_state))
            l_parts.append(self.umin.repeat(self.N))
            u_parts.append(self.umax.repeat(self.N))
            l0 = np.hstack(l_parts).astype(float)
            u0 = np.hstack(u_parts).astype(float)

        self._prob = osqp.OSQP()
        self._prob.setup(
            P=P0,
            q=q0,
            A=A0,
            l=l0,
            u=u0,
            warm_start=True,
            verbose=self.verbose,
            max_iter=10000,
            polish=True,
        )

        # Keep references for updating data arrays
        self._P = P0
        self._A = A0
        self._z_prev = np.zeros(self.nZ, dtype=float)

    def _shift_warm_start(self, U_star: np.ndarray) -> np.ndarray:
        """
        Shift optimal sequence forward by one step for warm start:
          [u0,u1,...,u_{N-1}] -> [u1,u2,...,u_{N-1},u_{N-1}]
        """
        U_star = np.asarray(U_star, dtype=float).reshape(-1)
        if U_star.shape[0] != self.nU:
            raise ValueError("U_star has wrong size")
        U_next = np.empty_like(U_star)
        U_next[0:-2] = U_star[2:]
        U_next[-2:] = U_star[-2:]
        return U_next

    def solve(
        self,
        segs,
        t0: float,
        x0: np.ndarray,
        foot_half_sizes: np.ndarray,
        Pref: np.ndarray,
        u_prev: np.ndarray | None = None,
    ) -> tuple[np.ndarray, bool, str]:
        """
        Solve MPC QP at current time t0 with state x0.
        Inputs:
          - segs: walking segments
          - t0: current time
          - x0: current state (4,)
          - foot_half_sizes: (2,)
          - Pref: stacked ZMP reference for horizon, shape (2N,) = [pref1;..;prefN]
          - u_prev: previous applied control (2,), for Δu_0 = u0 - u_prev (optional)

        Returns:
          - U*: optimal sequence (2N,)
          - ok: bool
          - status: OSQP status string
        """
        # Lazy OSQP setup (required before first call to solve).
        if self._prob is None or self._P is None or self._A is None:
            self._setup_osqp()

        x0 = np.asarray(x0, dtype=float).reshape(-1)
        if x0.shape != (self.nx,):
            raise ValueError("x0 must be shape (4,)")

        Pref = np.asarray(Pref, dtype=float).reshape(-1)
        if Pref.shape != (self.nU,):
            raise ValueError(f"Pref must be shape ({self.nU},)")

        # --- Constraints (Block 2) ---
        # A_poly U <= u_poly, with l=-inf
        A_poly, l_poly, u_poly, meta = build_polygon_preview_constraints(
            segs=segs,
            t0=t0,
            dt=self.dt,
            N=self.N,
            Ad=self.Ad,
            Bd=self.Bd,
            x0=x0,
            zc=self.zc,
            g=self.g,
            foot_half_sizes=foot_half_sizes,
            m_target=self.m_target,
        )

        # Optional state bounds over the horizon (COM x/y and/or vx/vy):
        #   x_i  = [1 0 0 0] x_i,   vx_i = [0 0 1 0] x_i
        #   y_i  = [0 1 0 0] x_i,   vy_i = [0 0 0 1] x_i  for i=1..N
        # X = Sx x0 + Su U  =>  C_state X in [l_state, u_state]
        A_state_U = None
        l_state = None
        u_state = None
        if self.m_state > 0:
            rows = []
            cols = []
            data = []
            row_cursor = 0
            for i in range(self.N):
                base = 4 * i
                if (self.x_min is not None) or (self.x_max is not None):
                    rows.append(row_cursor)
                    cols.append(base + 0)
                    data.append(1.0)
                    row_cursor += 1
                if (self.vx_min is not None) or (self.vx_max is not None):
                    rows.append(row_cursor)
                    cols.append(base + 2)
                    data.append(1.0)
                    row_cursor += 1
                if (self.y_min is not None) or (self.y_max is not None):
                    rows.append(row_cursor)
                    cols.append(base + 1)
                    data.append(1.0)
                    row_cursor += 1
                if (self.vy_min is not None) or (self.vy_max is not None):
                    rows.append(row_cursor)
                    cols.append(base + 3)
                    data.append(1.0)
                    row_cursor += 1

            C_state = sp.csc_matrix((data, (rows, cols)), shape=(self.m_state, 4 * self.N))
            X_aff = (self.Sx @ x0).reshape(-1)
            b_state = (C_state @ X_aff).reshape(-1)
            A_state_U = (C_state @ self.Su).tocsc()

            l_parts = []
            u_parts = []
            row_cursor = 0
            for i in range(self.N):
                if (self.x_min is not None) or (self.x_max is not None):
                    if self.x_min is None and self.x_max is not None:
                        lo = -self.x_max
                        hi = +self.x_max
                    else:
                        lo = -np.inf if self.x_min is None else self.x_min
                        hi = +np.inf if self.x_max is None else self.x_max
                    l_parts.append(lo)
                    u_parts.append(hi)
                    row_cursor += 1
                if (self.vx_min is not None) or (self.vx_max is not None):
                    if self.vx_min is None and self.vx_max is not None:
                        lo = -self.vx_max
                        hi = +self.vx_max
                    else:
                        lo = -np.inf if self.vx_min is None else self.vx_min
                        hi = +np.inf if self.vx_max is None else self.vx_max
                    l_parts.append(lo)
                    u_parts.append(hi)
                    row_cursor += 1
                if (self.y_min is not None) or (self.y_max is not None):
                    if self.y_min is None and self.y_max is not None:
                        lo = -self.y_max
                        hi = +self.y_max
                    else:
                        lo = -np.inf if self.y_min is None else self.y_min
                        hi = +np.inf if self.y_max is None else self.y_max
                    l_parts.append(lo)
                    u_parts.append(hi)
                    row_cursor += 1
                if (self.vy_min is not None) or (self.vy_max is not None):
                    if self.vy_min is None and self.vy_max is not None:
                        lo = -self.vy_max
                        hi = +self.vy_max
                    else:
                        lo = -np.inf if self.vy_min is None else self.vy_min
                        hi = +np.inf if self.vy_max is None else self.vy_max
                    l_parts.append(lo)
                    u_parts.append(hi)
                    row_cursor += 1
            l_state = (np.asarray(l_parts, dtype=float) - b_state).reshape(-1)
            u_state = (np.asarray(u_parts, dtype=float) - b_state).reshape(-1)

        # --- Cost (Block 3) ---
        # Need R_prev_U and r_prev_aff (stored in meta by you)
        R_prev_U = meta.get("R_prev_U", None)
        r_prev_aff = meta.get("r_prev_aff", None)
        if R_prev_U is None or r_prev_aff is None:
            raise RuntimeError("Expected meta['R_prev_U'] and meta['r_prev_aff'] from constraints builder.")

        P_U, p_aff = build_zmp_affine_terms(R_prev_U=R_prev_U, r_prev_aff=r_prev_aff, alpha=self.alpha)

        Pz, qz = build_cost_zmp_preview(P_U=P_U, p_aff=p_aff, P_ref=Pref, Wp=self.Wp)
        Pdu, qdu, _, _ = build_cost_du_smoothing(N=self.N, Wdu=self.Wdu, u_prev=u_prev)

        # Optional COM position tracking over the preview horizon to prevent drift
        # that can make future SS polygon constraints infeasible.
        Pr = None
        qr = None
        if np.any(self.Wr != 0.0):
            # R stacks r1..rN corresponding to x1..xN (times t0+dt .. t0+N*dt)
            R_aff = (self.Cr @ (self.Sx @ x0)).reshape(-1)
            R_U = (self.Cr @ self.Su).tocsc()

            # Build a COM reference stack.
            # Key choice: use the *midpoint between the (previewed) feet* from the segment
            # rather than snapping to the stance foot in SS.
            # Also apply an x-offset so the reference matches the current COM position at t0,
            # preventing an artificial initial pull-back when x0 != midfoot_x(t0).
            R_ref = np.zeros(2 * self.N, dtype=float)

            seg0 = segment_at_time(segs, float(t0))
            mid0 = 0.5 * (np.asarray(seg0.L, dtype=float).reshape(2,) + np.asarray(seg0.R, dtype=float).reshape(2,))
            x_offset = float(x0[0] - mid0[0])

            for i in range(1, self.N + 1):
                ti = float(t0 + i * self.dt)
                seg = segment_at_time(segs, ti)
                rref_i = 0.5 * (seg.L + seg.R)
                rref_i = np.asarray(rref_i, dtype=float).reshape(2,)
                rref_i[0] += x_offset
                R_ref[2 * (i - 1): 2 * (i - 1) + 2] = rref_i

            Wr_bar = build_block_weight(self.N, self.Wr)
            e_aff = (R_aff - R_ref).reshape(-1)
            Pr = (R_U.T @ (Wr_bar @ R_U)).tocsc()
            qr = (R_U.T @ (Wr_bar @ e_aff)).reshape(-1)

        # Optional COM velocity damping over the preview horizon (helps prevent p_y/r_y blow-up)
        Pv = None
        qv = None
        if np.any(self.Wv != 0.0):
            # Build selector C_v such that V = C_v X, V stacks [v1;..;vN]
            Z = sp.csc_matrix((2, 2))
            Cv1 = sp.hstack([Z, sp.eye(2, format="csc")], format="csc")  # [0  I]
            Cv = sp.kron(sp.eye(self.N, format="csc"), Cv1, format="csc")

            V_aff = (Cv @ (self.Sx @ x0)).reshape(-1)
            V_U = (Cv @ self.Su).tocsc()

            V_ref = np.tile(self.v_ref.reshape(2,), self.N).astype(float)
            Wv_bar = build_block_weight(self.N, self.Wv)
            e_v = (V_aff - V_ref).reshape(-1)
            Pv = (V_U.T @ (Wv_bar @ V_U)).tocsc()
            qv = (V_U.T @ (Wv_bar @ e_v)).reshape(-1)

        # Optional absolute control effort penalty (helps avoid bang-bang u that destabilizes r)
        Pu = None
        qu = None
        if np.any(self.Wu != 0.0):
            Wu_bar = build_block_weight(self.N, self.Wu)
            Pu = Wu_bar.tocsc()
            qu = np.zeros(self.nU, dtype=float)

        terms = [(Pz, qz), (Pdu, qdu)]
        if Pr is not None:
            terms.append((Pr, qr))
        if Pv is not None:
            terms.append((Pv, qv))
        if Pu is not None:
            terms.append((Pu, qu))
        P, q = combine_quadratic_costs(*terms)

        # If enabled, add slack penalty: (slack_weight/2)*||s||^2
        if self.use_slack:
            P = sp.block_diag([P, sp.eye(self.nS, format="csc") * self.slack_weight], format="csc")
            q = np.hstack([q, np.zeros(self.nS, dtype=float)]).astype(float)

        # --- Assemble OSQP A, l, u ---
        # Our A pattern has:
        #   top m_poly rows: full dense CSC (stored zeros allowed)
        #   bottom nU rows: identity for box bounds
        #
        # We'll update:
        #   - P (upper triangular) values according to our fixed P pattern order
        #   - A_poly values into the first m_poly rows of A
        #   - q, l, u

        # Bounds:
        l_parts = [l_poly]
        u_parts = [u_poly]
        if self.m_state > 0:
            if l_state is None or u_state is None:
                raise RuntimeError("State bounds enabled but l_state/u_state not built")
            l_parts.append(l_state)
            u_parts.append(u_state)
        l_parts.append(self.umin.repeat(self.N))
        u_parts.append(self.umax.repeat(self.N))
        if self.use_slack:
            l_parts.append(np.zeros(self.nS, dtype=float))
            u_parts.append((np.inf) * np.ones(self.nS))
        l = np.hstack(l_parts).astype(float)
        u = np.hstack(u_parts).astype(float)

        # Update P:
        # We have P as full dense symmetric in sparse upper-tri pattern.
        # Need to write P's upper-tri elements into self._P.data in the same ordering.
        # Ordering is CSC column-major over the stored upper-tri pattern.
        P_dense = P.toarray()
        # Fill according to pattern indices
        # self._P_pat has indices in .indices and .indptr; we mirror that layout.
        data = self._P.data
        indptr = self._P.indptr
        indices = self._P.indices
        k = 0
        for j in range(self.nZ):
            for idx in range(indptr[j], indptr[j + 1]):
                i = indices[idx]
                # stored pattern is i <= j (upper-tri)
                data[idx] = P_dense[i, j]
                k += 1

        # Update A:
        # Our A stores poly rows as dense pattern (all entries), then identity.
        # The poly block in A must become A_poly (dense in general).
        # Assemble dense prefix for A updates (poly rows + optional state rows)
        A_poly_U = A_poly.toarray()
        if self.use_slack:
            A_poly_dense = np.zeros((self.m_poly, self.nZ), dtype=float)
            A_poly_dense[:, : self.nU] = A_poly_U
            # Soft constraint: A_poly U - s <= u_poly
            A_poly_dense[:, self.nU :] = -np.eye(self.nS, dtype=float)
        else:
            A_poly_dense = A_poly_U

        if self.m_state > 0:
            if A_state_U is None:
                raise RuntimeError("State bounds enabled but A_state_U not built")
            A_state_dense_U = A_state_U.toarray()
        else:
            A_state_dense_U = None
        Adata = self._A.data
        Aindptr = self._A.indptr
        Aindices = self._A.indices

        # We write column by column into the first m_poly rows; bottom identity rows remain unchanged.
        # For U-columns (0..nU-1), the poly block stores ALL rows 0..m_poly-1.
        # For slack columns (nU..nZ-1), the poly block is sparse (-I), so we must not
        # assume there are m_poly entries to overwrite.
        # Fill dense prefix rows in U-columns: [poly; state] blocks
        dense_prefix_rows = self.m_poly + self.m_state
        if self.use_slack:
            for j in range(self.nU):
                col_start = Aindptr[j]
                # poly part
                for krow in range(self.m_poly):
                    Adata[col_start + krow] = A_poly_dense[krow, j]
                # state part
                if self.m_state > 0:
                    for krow in range(self.m_state):
                        Adata[col_start + self.m_poly + krow] = A_state_dense_U[krow, j]
        else:
            for j in range(self.nU):
                col_start = Aindptr[j]
                for krow in range(self.m_poly):
                    Adata[col_start + krow] = A_poly_dense[krow, j]
                if self.m_state > 0:
                    for krow in range(self.m_state):
                        Adata[col_start + self.m_poly + krow] = A_state_dense_U[krow, j]

        # Update OSQP
        self._prob.update(q=q, l=l, u=u, Px=self._P.data, Ax=self._A.data)

        # Warm start with shifted previous sequence
        # Warm start full decision vector z = [U; s]
        if self.use_slack:
            if self._z_prev.shape != (self.nZ,):
                self._z_prev = np.zeros(self.nZ, dtype=float)
            self._z_prev[: self.nU] = self._U_prev
        else:
            self._z_prev = self._U_prev.copy()
        self._prob.warm_start(x=self._z_prev)

        res = self._prob.solve()
        self.last_result = res
        self.last_info = res.info
        status = res.info.status
        ok = res.info.status_val in (1, 2)  # solved / solved inaccurate

        if ok and res.x is not None:
            z_star = np.asarray(res.x, dtype=float).reshape(-1)
            U_star = z_star[: self.nU]
            # prepare warm start for next call
            self._U_prev = self._shift_warm_start(U_star)
            if self.use_slack:
                self._z_prev = z_star
            else:
                self._z_prev = self._U_prev.copy()
            return U_star, True, status

        # Failure: return previous warm start (or zeros) so caller can decide
        # Failure: fall back to a safe control sequence.
        return np.zeros(self.nU, dtype=float), False, status
