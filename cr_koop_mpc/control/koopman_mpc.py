"""Koopman MPC (eqs. 16-18).

Parametric quadratic program in the lifted space:

    min  Σ_i  ||z_i − z^ref_i||²_Q  + ||u_i||²_R + ||Δu_i||²_S  + terminal
    s.t. z_{i+1} = A z_i + B u_i
         u_i ∈ U,  Δu_i ∈ D

Notes for the paper:

1. The physical-space weight ``Q`` is zero-padded into the lifted space. The
   first ``n_state`` coordinates of ``z`` are the identity (both EDMD and
   Deep liftings here prepend ``x``), so this is exact.
2. The reference in lifted space is computed by lifting the physical
   reference at solve time (Korda & Mezić, 2018, §4).
3. ``P`` is a terminal weight ``terminal_scale · Q`` for practical recursive
   feasibility; swap in the DARE solution if the paper claims invariance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from ..models import BaseLifting, KoopmanPredictor


@dataclass
class MPCSolution:
    U: np.ndarray         # optimal input sequence (N, m)
    Z: np.ndarray         # predicted lifted trajectory (N+1, n_lift)
    solve_time: float
    status: str
    cost: float


class KoopmanMPC:
    """Parametric Koopman MPC. Build once, solve many times."""

    def __init__(
        self,
        lifting: BaseLifting,
        koopman: KoopmanPredictor,
        N: int,
        Q_diag: list[float],
        R_diag: list[float],
        S_diag: list[float],
        u_bounds: np.ndarray,
        du_bounds: np.ndarray,
        terminal_scale: float = 5.0,
        solver: str = "OSQP",
    ) -> None:
        self.lifting = lifting
        self.koopman = koopman
        self.N = N
        self.n_lift = koopman.n_lift
        self.m = koopman.n_input
        self.solver = solver

        n_state = len(Q_diag)
        Q_full = np.zeros(self.n_lift)
        Q_full[:n_state] = np.asarray(Q_diag)
        self.Q = np.diag(Q_full)
        self.R = np.diag(R_diag)
        self.S = np.diag(S_diag)
        self.P = terminal_scale * self.Q

        self.u_bounds = np.asarray(u_bounds, dtype=float)
        self.du_bounds = np.asarray(du_bounds, dtype=float)

        self._build_problem()

    def _build_problem(self) -> None:
        N, n, m = self.N, self.n_lift, self.m

        self._z0 = cp.Parameter(n)
        self._z_ref = cp.Parameter((N + 1, n))
        self._u_prev = cp.Parameter(m)
        self._A_par = cp.Parameter((n, n))
        self._B_par = cp.Parameter((n, m))

        self._Z = cp.Variable((N + 1, n))
        self._U = cp.Variable((N, m))

        cost = 0
        constraints = [self._Z[0] == self._z0]
        for k in range(N):
            cost += cp.quad_form(self._Z[k] - self._z_ref[k], cp.psd_wrap(self.Q))
            cost += cp.quad_form(self._U[k], cp.psd_wrap(self.R))
            du = self._U[k] - (self._U[k - 1] if k > 0 else self._u_prev)
            cost += cp.quad_form(du, cp.psd_wrap(self.S))
            constraints += [
                self._Z[k + 1] == self._A_par @ self._Z[k] + self._B_par @ self._U[k],
                self._U[k] >= self.u_bounds[:, 0],
                self._U[k] <= self.u_bounds[:, 1],
                du >= self.du_bounds[:, 0],
                du <= self.du_bounds[:, 1],
            ]
        cost += cp.quad_form(self._Z[N] - self._z_ref[N], cp.psd_wrap(self.P))
        self._problem = cp.Problem(cp.Minimize(cost), constraints)

    def solve(
        self, x0: np.ndarray, x_ref: np.ndarray, u_prev: np.ndarray
    ) -> MPCSolution:
        """Solve one MPC instance."""
        z0 = self.lifting.lift(x0)
        Z_ref = self.lifting.lift(x_ref)

        self._z0.value = z0
        self._z_ref.value = Z_ref
        self._u_prev.value = u_prev
        self._A_par.value = self.koopman.A
        self._B_par.value = self.koopman.B

        t0 = time.perf_counter()
        try:
            # Tightened OSQP tolerances to suppress "Solution may be
            # inaccurate" warnings on the parametric MPC QP. Other CVXPY
            # solvers (ECOS, SCS) ignore these unknown kwargs.
            solver_opts: dict = {}
            if self.solver.upper() == "OSQP":
                solver_opts = dict(
                    eps_abs=1e-6,
                    eps_rel=1e-6,
                    max_iter=20000,
                    polish=True,
                )
            self._problem.solve(
                solver=self.solver,
                warm_start=True,
                verbose=False,
                **solver_opts,
            )
        except cp.error.SolverError as exc:
            return MPCSolution(
                U=np.tile(u_prev, (self.N, 1)),
                Z=np.tile(z0, (self.N + 1, 1)),
                solve_time=time.perf_counter() - t0,
                status=f"solver_error: {exc}",
                cost=float("inf"),
            )
        solve_time = time.perf_counter() - t0

        if self._U.value is None:
            return MPCSolution(
                U=np.tile(u_prev, (self.N, 1)),
                Z=np.tile(z0, (self.N + 1, 1)),
                solve_time=solve_time,
                status=self._problem.status or "infeasible",
                cost=float("inf"),
            )

        return MPCSolution(
            U=np.asarray(self._U.value).copy(),
            Z=np.asarray(self._Z.value).copy(),
            solve_time=solve_time,
            status=self._problem.status,
            cost=float(self._problem.value),
        )
