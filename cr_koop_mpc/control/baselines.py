"""Baseline controllers for the paper's comparison study.

Two controllers live here:

* :class:`NominalNMPC` — SLSQP-based nonlinear MPC on the dynamic bicycle.
  This is the "expensive but accurate" baseline. We do not optimise it for
  speed; we want it to be representative of textbook NMPC.

* :class:`TubeMPC` — Koopman MPC with a constant disturbance bound used to
  tighten constraints once and for all. This is the natural competitor to
  the adaptive conformal bound; the paper's key empirical claim is that
  conformal beats tube under distribution shift.

The remaining baselines (``periodic``, ``static_et``, ``no_filter``,
``cbf_qp``, ``ours``) are realised by toggling features inside
:class:`~cr_koop_mpc.control.controller.ResidualDrivenController` — they
share the Koopman MPC engine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .koopman_mpc import KoopmanMPC, MPCSolution


# ---------------------------------------------------------------------------
# Nonlinear MPC
# ---------------------------------------------------------------------------


@dataclass
class NMPCResult:
    U: np.ndarray
    solve_time: float
    status: str
    cost: float


class NominalNMPC:
    """Single-shooting NMPC with SLSQP.

    Uses the bicycle dynamics from the env directly (a callable
    ``f_dyn(x, u) → x_next``). Single-shooting keeps the optimisation
    variables small at the cost of some numerical stiffness — good enough
    for a baseline.
    """

    def __init__(
        self,
        f_dyn,
        N: int,
        Q_diag: list[float],
        R_diag: list[float],
        S_diag: list[float],
        u_bounds: np.ndarray,
        du_bounds: np.ndarray,
        terminal_scale: float = 5.0,
    ) -> None:
        self.f = f_dyn
        self.N = N
        self.Q = np.diag(Q_diag)
        self.R = np.diag(R_diag)
        self.S = np.diag(S_diag)
        self.P = terminal_scale * self.Q
        self.m = u_bounds.shape[0]
        self.u_bounds = u_bounds
        self.du_bounds = du_bounds

    def _rollout(self, x0: np.ndarray, U_flat: np.ndarray) -> np.ndarray:
        U = U_flat.reshape(self.N, self.m)
        X = np.zeros((self.N + 1, x0.shape[0]))
        X[0] = x0
        for k in range(self.N):
            X[k + 1] = self.f(X[k], U[k])
        return X

    def _cost(self, U_flat, x0, x_ref, u_prev):
        U = U_flat.reshape(self.N, self.m)
        X = self._rollout(x0, U_flat)
        J = 0.0
        for k in range(self.N):
            dx = X[k] - x_ref[k]
            J += dx @ self.Q @ dx + U[k] @ self.R @ U[k]
            du = U[k] - (U[k - 1] if k > 0 else u_prev)
            J += du @ self.S @ du
        dxN = X[self.N] - x_ref[self.N]
        J += dxN @ self.P @ dxN
        return J

    def solve(self, x0: np.ndarray, x_ref: np.ndarray, u_prev: np.ndarray) -> NMPCResult:
        U0 = np.tile(u_prev, self.N).astype(float)
        lb = np.tile(self.u_bounds[:, 0], self.N)
        ub = np.tile(self.u_bounds[:, 1], self.N)

        t0 = time.perf_counter()
        res = minimize(
            self._cost,
            U0,
            args=(x0, x_ref, u_prev),
            method="SLSQP",
            bounds=list(zip(lb, ub)),
            options={"maxiter": 50, "ftol": 1e-4},
        )
        return NMPCResult(
            U=res.x.reshape(self.N, self.m),
            solve_time=time.perf_counter() - t0,
            status="ok" if res.success else f"slsqp: {res.message}",
            cost=float(res.fun),
        )


# ---------------------------------------------------------------------------
# Tube Koopman MPC
# ---------------------------------------------------------------------------


class TubeMPC:
    """Koopman MPC with a constant disturbance bound for constraint tightening.

    Wraps :class:`KoopmanMPC` but tightens input bounds by a constant tube
    radius (proxy for input-disturbance margin). The natural competitor to
    the adaptive conformal bound; deliberately *non-adaptive* so the paper
    can demonstrate the advantage of online ACI under distribution shift.
    """

    def __init__(self, mpc: KoopmanMPC, tube_radius: float = 0.02) -> None:
        self.mpc = mpc
        self.tube_radius = tube_radius
        # Tighten the underlying MPC's input bounds in-place
        u_b = self.mpc.u_bounds.copy()
        u_b[:, 0] += tube_radius
        u_b[:, 1] -= tube_radius
        # Rebuild with tightened bounds
        self.mpc.u_bounds = u_b
        self.mpc._build_problem()

    def solve(
        self, x0: np.ndarray, x_ref: np.ndarray, u_prev: np.ndarray
    ) -> MPCSolution:
        return self.mpc.solve(x0, x_ref, u_prev)
