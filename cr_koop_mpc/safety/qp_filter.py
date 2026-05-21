"""Conservative discrete-time CBF safety filter as a parametric QP.

Implements eq. (23) of the paper:

    u^SF_k = argmin  ||u − u^MPC_k||²_H + λ_s s²
              s.t.  a_k^⊤ u  ≥  b_k + ρ_k − s,
                    u ∈ U,  u − u_{k−1} ∈ D,  s ≥ 0.

The linearized D-CBF constraint is built from a finite-difference Jacobian
``G`` of the (Koopman) dynamics at ``u^MPC_k`` and the gradient of ``h_θ`` at
the predicted next state. The QP is parametric so we build it once and
solve every sampling instant.

If the slack ``s`` exceeds an emergency threshold the outer loop falls back
to controlled braking — handled in :mod:`control.controller`, not here.
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp

from .neural_cbf import NeuralCBF


class SafetyFilterQP:
    """Parametric QP filter projecting ``u_MPC`` onto the robust safe set."""

    def __init__(
        self,
        cbf: NeuralCBF,
        u_bounds: np.ndarray,        # (m, 2)
        du_bounds: np.ndarray,       # (m, 2)
        H_diag: np.ndarray,
        lambda_slack: float,
    ) -> None:
        self.cbf = cbf
        self.u_bounds = np.asarray(u_bounds, dtype=float)
        self.du_bounds = np.asarray(du_bounds, dtype=float)
        self.H = np.diag(H_diag)
        self.lambda_slack = lambda_slack
        self.m = self.u_bounds.shape[0]

        self._u_var = cp.Variable(self.m)
        self._s_var = cp.Variable(nonneg=True)
        self._u_mpc_par = cp.Parameter(self.m)
        self._u_prev_par = cp.Parameter(self.m)
        self._a_par = cp.Parameter(self.m)
        self._b_par = cp.Parameter()
        self._rho_par = cp.Parameter(nonneg=True)

        cost = cp.quad_form(
            self._u_var - self._u_mpc_par, cp.psd_wrap(self.H)
        ) + lambda_slack * self._s_var ** 2

        constraints = [
            self._a_par @ self._u_var
            >= self._b_par + self._rho_par - self._s_var,
            self._u_var >= self.u_bounds[:, 0],
            self._u_var <= self.u_bounds[:, 1],
            self._u_var - self._u_prev_par >= self.du_bounds[:, 0],
            self._u_var - self._u_prev_par <= self.du_bounds[:, 1],
        ]
        self._prob = cp.Problem(cp.Minimize(cost), constraints)

    # ---------------- linearization helper ----------------

    def _affine_dynamics(
        self, x: np.ndarray, u_nom: np.ndarray, f_dyn
    ) -> tuple[np.ndarray, np.ndarray]:
        """Local affine approximation ``x_{k+1} ≈ x̄_{k+1} + G_k (u − ū)``.

        ``f_dyn(x, u)`` returns the one-step next state (e.g. Koopman
        reconstruction). Jacobian computed by forward differences.
        """
        eps = 1e-3
        x_bar_next = f_dyn(x, u_nom)
        G = np.zeros((x.shape[0], self.m))
        for j in range(self.m):
            u_pert = u_nom.copy()
            u_pert[j] += eps
            G[:, j] = (f_dyn(x, u_pert) - x_bar_next) / eps
        return x_bar_next, G

    # ---------------- solve ----------------

    def solve(
        self,
        x: np.ndarray,
        u_mpc: np.ndarray,
        u_prev: np.ndarray,
        f_dyn,
        rho: float,
        gamma: float,
    ) -> tuple[np.ndarray, float, dict]:
        """Run one filter QP."""
        x_bar_next, G = self._affine_dynamics(x, u_mpc, f_dyn)
        grad_h = self.cbf.grad_h(x_bar_next)
        h_now = self.cbf.value(x)
        h_pred = self.cbf.value(x_bar_next)

        # Linearized D-CBF:
        #   h(x_next) ≈ h(x̄_next) + ∇h·G·(u − ū)   ≥   (1−γ) h(x) + ρ
        a = G.T @ grad_h
        b = (1.0 - gamma) * h_now - h_pred + a @ u_mpc

        # Defensive: if the linearisation produced a degenerate constraint
        # (NaN/Inf from upstream numerical issues), pass the MPC input
        # through unchanged with maximal slack reported so the controller
        # can decide whether to fire emergency braking.
        if (
            not np.all(np.isfinite(a))
            or not np.isfinite(b)
            or not np.isfinite(h_now)
            or not np.isfinite(h_pred)
        ):
            return u_mpc.copy(), float("inf"), {"status": "non_finite_input"}

        self._u_mpc_par.value = u_mpc
        self._u_prev_par.value = u_prev
        self._a_par.value = a
        self._b_par.value = float(b)
        self._rho_par.value = float(max(rho, 0.0))

        try:
            # Tightened OSQP settings: lower eps + more iterations than the
            # default reduce "Solution may be inaccurate" warnings on
            # borderline-feasible filter problems without measurable cost
            # on warm-started solves.
            self._prob.solve(
                solver=cp.OSQP,
                warm_start=True,
                verbose=False,
                eps_abs=1e-6,
                eps_rel=1e-6,
                max_iter=20000,
                polish=True,
            )
        except cp.error.SolverError:
            return u_mpc.copy(), float("inf"), {"status": "solver_error"}

        status = str(self._prob.status)
        if self._u_var.value is None:
            return u_mpc.copy(), float("inf"), {"status": "infeasible", "h_now": h_now, "h_pred": h_pred}

        # Do not trust partially iterated OSQP values as if they were a valid
        # safety projection.  The outer controller will switch to the CBF
        # recovery guard when the margin is low.
        if status not in {"optimal", "optimal_inaccurate"}:
            return u_mpc.copy(), float("inf"), {"status": status, "h_now": h_now, "h_pred": h_pred}

        return (
            np.asarray(self._u_var.value).copy(),
            float(self._s_var.value if self._s_var.value is not None else 0.0),
            {
                "status": status,
                "h_now": h_now,
                "h_pred": h_pred,
            },
        )
