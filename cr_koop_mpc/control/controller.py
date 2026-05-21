"""Top-level residual-driven controller.

Wires together the four building blocks (Koopman MPC, adaptive conformal
residual, neural-CBF safety filter, four-condition trigger) into the online
loop described in §IX.B of the paper.

Ablation flags on :class:`ControllerConfig` let the same code reproduce all
six baselines plus the proposed method:

    * ``use_event_trigger=False``  → periodic Koopman MPC (resolve every step)
    * ``use_residual_margin=False``→ static event-triggered Koopman MPC
    * ``use_safety_filter=False``  → unfiltered Koopman MPC
    * ``use_mpc=False``            → CBF-QP only (no MPC)
    * ``baseline="tube"``          → tube Koopman MPC
    * ``baseline="nmpc"``          → nonlinear MPC
    * all flags on, baseline="ours" → the proposed method
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..models import ConformalResidual, KoopmanPredictor, BaseLifting
from ..safety import NeuralCBF, SafetyFilterQP
from ..triggering import ResidualDrivenTrigger, TriggerDecision
from .koopman_mpc import KoopmanMPC, MPCSolution
from .baselines import NominalNMPC, TubeMPC


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ControllerConfig:
    """Ablation flags + run-time parameters for the controller."""

    use_event_trigger: bool = True
    use_residual_margin: bool = True
    use_safety_filter: bool = True
    use_mpc: bool = True

    # Robust-margin coefficients (eq. 14)
    L_x: float = 1.0       # reconstruction Lipschitz (1.0 when lifting prepends identity)
    eps_rec: float = 0.0
    eps_sens: float = 0.05
    rho_lin: float = 0.05
    fallback_Lh: float = 5.0
    use_local_lipschitz: bool = True
    rho_cap: float | None = None
    lane_half_width: float = 1.75

    gamma_cbf: float = 0.2
    emergency_slack: float = 0.5
    # Emergency fallback is lane-recovery steering plus controlled braking.
    # It is used only if the CBF-QP needs excessive slack.
    emergency_brake: float = -3.0
    recovery_k_py: float = 0.35
    recovery_k_psi: float = 0.80
    recovery_k_vy: float = 0.12
    safety_guard_margin: float = 0.15
    lane_guard_buffer: float = 0.20
    N_h: int = 5  # lookahead horizon for M_h

    # Physical metric used for conformal scores and E_z(k).  These indices
    # are applied to the physical state residual, not the lifted residual.
    residual_indices: list[int] | None = None
    trigger_indices: list[int] | None = None

    # Online Koopman re-id (paper extension)
    online_koopman: bool = False
    online_trigger_ratio: float = 1.5
    online_consecutive: int = 3

    extras: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Step log
# ---------------------------------------------------------------------------


@dataclass
class StepLog:
    """Per-step quantities for analysis & paper figures."""
    x: np.ndarray
    u: np.ndarray
    w_bar: float
    rho: float
    h_now: float
    trigger: TriggerDecision | None
    mpc_solved: bool
    mpc_solve_time: float
    slack: float
    residual_norm: float
    coverage: float
    alpha: float
    emergency: bool
    conformal_covered: float = float("nan")
    conformal_miscovered: float = float("nan")
    w_bar_before_update: float = float("nan")
    u_mpc: np.ndarray | None = None
    filter_correction: float = 0.0
    err_radius: float = 0.0
    L_h_eff: float = 0.0
    h_minus_rho: float = 0.0
    mpc_status: str = "not_solved"
    mpc_cost: float = 0.0
    safety_status: str = "not_run"
    h_pred: float = 0.0
    slack_triggered_emergency: bool = False
    safety_recovery: bool = False


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class ResidualDrivenController:
    """The end-to-end controller from the paper (with ablation flags)."""

    def __init__(
        self,
        lifting: BaseLifting,
        koopman: KoopmanPredictor,
        mpc: KoopmanMPC | TubeMPC | NominalNMPC,
        cbf: NeuralCBF,
        conformal: ConformalResidual,
        trigger: ResidualDrivenTrigger,
        safety_filter: SafetyFilterQP,
        cfg: ControllerConfig,
    ) -> None:
        self.lifting = lifting
        self.koopman = koopman
        self.mpc = mpc
        self.cbf = cbf
        self.conformal = conformal
        self.trigger = trigger
        self.safety_filter = safety_filter
        self.cfg = cfg

        self._u_prev = np.zeros(self.koopman.n_input)
        self._U_plan: np.ndarray | None = None
        self._plan_age: int = 0
        self._x_prev: np.ndarray | None = None
        self._online_streak = 0
        # State at the most recent replanning instant t_i. Used to roll the
        # held plan forward to obtain ẑ_{k|t_i} for the E_z diagnostic (paper
        # eq. 17). Reset to the current state every time MPC re-solves.
        self._x_at_replan: np.ndarray | None = None
        # Magnitude of the safety-filter correction at the *previous* step.
        # Used as the I_u(k) diagnostic in paper eq. (20): at trigger time the
        # current step's filter has not yet run, so we use the last step's
        # correction as a one-step-delayed proxy.
        self._last_filter_correction: float = 0.0

        self._L_h = (
            self.cbf.lipschitz_constant()
            if self.cbf.spectral_norm
            else self.cfg.fallback_Lh
        )
        self._residual_indices = (
            None if self.cfg.residual_indices is None
            else np.asarray(self.cfg.residual_indices, dtype=int)
        )
        self._trigger_indices = (
            self._residual_indices if self.cfg.trigger_indices is None
            else np.asarray(self.cfg.trigger_indices, dtype=int)
        )

    # ---------------- public API ----------------

    def reset(self, x0: np.ndarray) -> None:
        self._u_prev = np.zeros(self.koopman.n_input)
        self._U_plan = None
        self._plan_age = 0
        self._x_prev = x0.copy()
        self._online_streak = 0
        self._x_at_replan = None
        self._last_filter_correction = 0.0
        self._last_conformal_covered = float("nan")
        self._last_conformal_miscovered = float("nan")
        self._last_w_bar_before_update = float("nan")
        self.trigger.reset()

    def step(self, x: np.ndarray, x_ref: np.ndarray) -> StepLog:
        """One control step: update residual, evaluate trigger, solve MPC if
        needed, run safety filter, return the applied input + logs."""

        # 1) Update conformal residual from the previous transition (if any).
        residual_norm = self._update_residual(x)

        # 2) Read current bound and form the residual-aware margin ρ_k.
        #
        # Important: do NOT use the global spectral Lipschitz bound of the
        # prior-CBF during normal driving.  That bound is intentionally very
        # conservative (~8 in v7); multiplying it by the adaptive residual made
        # ρ exceed h(0,0), so the safety QP was infeasible even at lane centre.
        #
        # Instead use the local CBF gradient at the current physical state:
        #       ρ_k = ||∇h(x_k)|| · (Lx*w̄_k + eps_rec + eps_sens) + ρ_lin
        # with the same conformal score coordinates as the CBF input [py, vy].
        w_bar_raw = self.conformal.bound() if self.conformal.is_calibrated else 0.0
        # Tube / fixed-margin baselines see no online bound (constant disturbance).
        w_bar = w_bar_raw if self.cfg.use_residual_margin else 0.0
        err_radius = 0.0
        L_h_eff = 0.0
        if self.cfg.use_residual_margin:
            err_radius = self.cfg.L_x * w_bar + self.cfg.eps_rec + self.cfg.eps_sens
            if self.cfg.use_local_lipschitz:
                try:
                    grad_h = np.asarray(self.cbf.grad_h(x), dtype=float)
                    L_h_eff = float(np.linalg.norm(grad_h))
                    if not np.isfinite(L_h_eff):
                        L_h_eff = self._L_h
                except Exception:
                    L_h_eff = self._L_h
            else:
                L_h_eff = self._L_h
            rho = L_h_eff * err_radius + self.cfg.rho_lin
            if self.cfg.rho_cap is not None:
                rho = min(float(rho), float(self.cfg.rho_cap))
        else:
            rho = self.cfg.rho_lin  # constant fallback
            err_radius = self.cfg.eps_rec + self.cfg.eps_sens
            L_h_eff = 0.0

        # 3) Decide whether to re-solve the MPC.
        mpc_solved = False
        mpc_solve_time = 0.0
        mpc_status = "not_solved"
        mpc_cost = 0.0
        trigger_decision: TriggerDecision | None = None

        # Predicted state ẑ_{k|t_i} (paper eq. 17): roll the held plan forward
        # from the state at the most recent replanning instant t_i to the
        # current step k. plan_age = k - t_i is the number of plan steps
        # already applied since the last re-solve.
        z_now = self.lifting.lift(x)
        if self._U_plan is not None and self._x_at_replan is not None:
            z_pred = self.lifting.lift(self._x_at_replan)
            for k in range(min(self._plan_age, self._U_plan.shape[0])):
                z_pred = self.koopman.A @ z_pred + self.koopman.B @ self._U_plan[k]
        else:
            z_pred = z_now

        # Trigger model-error metric in physical units.  Do not use the full
        # lifted error here: polynomial/RBF coordinates have artificial units
        # and caused the trigger to fire every step in v5.
        x_pred = self.lifting.reconstruct(z_pred)
        if self._trigger_indices is None:
            metric_now = x
            metric_pred = x_pred
        else:
            metric_now = x[self._trigger_indices]
            metric_pred = x_pred[self._trigger_indices]

        min_safety_margin = self._min_safety_margin_along_plan(x, rho)

        # I_u(k) per paper eq. (20): magnitude of safety-filter correction.
        # The current-step correction is not yet known at trigger time, so we
        # use the *previous* step's correction as a one-step-delayed proxy.
        last_input_change = self._last_filter_correction
        if self.cfg.use_event_trigger:
            trigger_decision = self.trigger.step(
                z_now=metric_now,
                z_pred=metric_pred,
                min_safety_margin=min_safety_margin,
                last_input_change=last_input_change,
                w_bar=w_bar,
            )
            fire = trigger_decision.fire or self._U_plan is None
        else:
            fire = True  # periodic baseline

        if fire and self.cfg.use_mpc:
            sol = self.mpc.solve(x, x_ref, self._u_prev)
            self._U_plan = sol.U
            self._plan_age = 0
            self._x_at_replan = x.copy()  # record state at t_i for E_z
            mpc_solved = True
            mpc_solve_time = (
                sol.solve_time if isinstance(sol, MPCSolution) else getattr(sol, "solve_time", 0.0)
            )
            mpc_status = str(getattr(sol, "status", "unknown"))
            try:
                mpc_cost = float(getattr(sol, "cost", 0.0))
            except Exception:
                mpc_cost = float("nan")
        elif self._U_plan is None:
            # First step before any plan; idle input
            self._U_plan = np.zeros((self.mpc.N, self.koopman.n_input))
            self._x_at_replan = x.copy()

        # Recompute the look-ahead safety margin after a possible replan.
        # v10 made the trigger diagnostic before the new plan, then executed a
        # freshly solved plan even if the remaining trajectory was already too
        # close to h-rho=0.  Keep both notions: trigger_decision.M_h is the
        # trigger-side diagnostic; ``min_safety_margin_exec`` guards the input
        # that is actually about to be executed.
        min_safety_margin_exec = self._min_safety_margin_along_plan(x, rho)

        # 4) Pull nominal input from the (possibly held) plan
        idx = min(self._plan_age, self._U_plan.shape[0] - 1)
        u_mpc = self._U_plan[idx]

        # 5) Safety filter
        emergency = False
        slack = 0.0
        safety_status = "not_run"
        h_pred = 0.0
        slack_triggered_emergency = False
        safety_recovery = False
        filter_correction = 0.0
        if self.cfg.use_safety_filter:
            u_safe, slack, info = self.safety_filter.solve(
                x=x,
                u_mpc=u_mpc,
                u_prev=self._u_prev,
                f_dyn=self._f_lifted,
                rho=rho,
                gamma=self.cfg.gamma_cbf,
            )
            safety_status = str(info.get("status", "unknown")) if isinstance(info, dict) else "unknown"
            try:
                h_pred = float(info.get("h_pred", 0.0)) if isinstance(info, dict) else 0.0
            except Exception:
                h_pred = float("nan")
            # Slack by itself is only a feasibility diagnostic.  In v7, a
            # conservative ρ made slack large while the vehicle was still well
            # inside the lane, so this branch fired for 357/400 steps and
            # collapsed vx to ~1 m/s.  Emergency braking is now reserved for
            # genuinely unsafe physical states or clearly negative CBF values.
            h_now_for_emergency = self.cbf.value(x)
            physically_outside_lane = (
                len(x) > 1 and abs(float(x[1])) > self.cfg.lane_half_width
            )
            near_lane_boundary = (
                len(x) > 1
                and abs(float(x[1])) >= self.cfg.lane_half_width - self.cfg.lane_guard_buffer
            )
            cbf_clearly_unsafe = h_now_for_emergency < -0.25
            robust_margin_low = (self.cbf.value(x) - rho) <= self.cfg.safety_guard_margin
            lookahead_margin_low = min_safety_margin_exec <= self.cfg.safety_guard_margin
            bad_filter_status = safety_status not in {"optimal", "optimal_inaccurate"}
            slack_too_large = bool(np.isfinite(slack) and slack > self.cfg.emergency_slack)

            # The CBF-QP is soft-constrained by design; slack is useful for
            # feasibility, but it must not silently execute a plan whose robust
            # safety margin is already near/below zero.  v10 failed here: the
            # QP used slack around the boundary and the vehicle left the lane.
            # v11 keeps the neural CBF and QP, but adds a pre-violation
            # recovery guard when h-rho/M_h is low, when the physical lane
            # boundary is close, or when OSQP reports a non-optimal status.
            needs_recovery = (
                physically_outside_lane
                or near_lane_boundary
                or cbf_clearly_unsafe
                or robust_margin_low
                or lookahead_margin_low
                or (bad_filter_status and (robust_margin_low or lookahead_margin_low or near_lane_boundary))
                or (slack_too_large and (robust_margin_low or lookahead_margin_low or near_lane_boundary))
            )
            if needs_recovery:
                emergency = True
                safety_recovery = True
                slack_triggered_emergency = slack_too_large
                u_safe = self._emergency_brake(x, u_safe)
            # Record this step's filter correction for the *next* step's I_u(k).
            filter_correction = float(np.linalg.norm(u_safe - u_mpc))
            self._last_filter_correction = filter_correction
        else:
            u_safe = u_mpc.copy()
            filter_correction = 0.0
            self._last_filter_correction = 0.0

        # 6) Optional online Koopman update
        if self.cfg.online_koopman and self._x_prev is not None:
            if residual_norm > self.cfg.online_trigger_ratio * max(w_bar, 1e-6):
                self._online_streak += 1
            else:
                self._online_streak = 0
            if self._online_streak >= self.cfg.online_consecutive:
                self.koopman.online_update(self._x_prev, self._u_prev, x)
                self._online_streak = 0

        # 7) Bookkeeping
        self._u_prev = u_safe.copy()
        self._x_prev = x.copy()
        self._plan_age += 1

        return StepLog(
            x=x.copy(),
            u=u_safe.copy(),
            w_bar=w_bar,
            rho=rho,
            h_now=self.cbf.value(x),
            trigger=trigger_decision,
            mpc_solved=mpc_solved,
            mpc_solve_time=mpc_solve_time,
            slack=slack,
            residual_norm=residual_norm,
            coverage=self.conformal.empirical_coverage(),
            alpha=self.conformal.alpha,
            emergency=emergency,
            conformal_covered=float(self._last_conformal_covered),
            conformal_miscovered=float(self._last_conformal_miscovered),
            w_bar_before_update=float(self._last_w_bar_before_update),
            u_mpc=u_mpc.copy(),
            filter_correction=filter_correction,
            err_radius=float(err_radius),
            L_h_eff=float(L_h_eff),
            h_minus_rho=float(self.cbf.value(x) - rho),
            mpc_status=mpc_status,
            mpc_cost=mpc_cost,
            safety_status=safety_status,
            h_pred=float(h_pred),
            slack_triggered_emergency=slack_triggered_emergency,
            safety_recovery=safety_recovery,
        )

    # ---------------- helpers ----------------

    def _update_residual(self, x_now: np.ndarray) -> float:
        """Compute ``||w_k||`` from (x_prev, u_prev, x_now) and push to ACI.

        Skips the very first call after ``reset`` (where ``_x_prev`` equals
        ``x_now`` because no step has actually been taken yet). Pushing that
        spurious "self-residual" into ACI biases the bound upward for many
        subsequent steps.
        """
        self._last_conformal_covered = float("nan")
        self._last_conformal_miscovered = float("nan")
        self._last_w_bar_before_update = float("nan")
        if self._x_prev is None or not self.conformal.is_calibrated:
            return 0.0
        # First step after reset: _x_prev was set to x0 by reset(), and x_now
        # is also x0 because no integration has happened. There is no real
        # transition to score yet — skip.
        if self._plan_age == 0 and self._U_plan is None:
            return 0.0
        score = self.koopman.residual_score(
            self._x_prev, self._u_prev, x_now, indices=self._residual_indices
        )
        self.conformal.update(score)
        self._last_conformal_covered = float(self.conformal.last_covered)
        self._last_conformal_miscovered = float(self.conformal.last_miscovered)
        self._last_w_bar_before_update = float(self.conformal.last_bound_before_update)
        return score

    def _min_safety_margin_along_plan(self, x: np.ndarray, rho: float) -> float:
        """Compute ``M_h = min_j (h(x̂_{k+j}) − ρ)`` along the current plan."""
        if self._U_plan is None:
            return self.cbf.value(x) - rho
        # Roll lifted prediction forward, reconstruct, evaluate h.
        z = self.lifting.lift(x)
        m = self.cbf.value(x) - rho
        remaining = max(0, self._U_plan.shape[0] - int(self._plan_age))
        n_steps = min(self.cfg.N_h, remaining)
        for k in range(n_steps):
            u_idx = min(int(self._plan_age) + k, self._U_plan.shape[0] - 1)
            z = self.koopman.A @ z + self.koopman.B @ self._U_plan[u_idx]
            x_hat = self.lifting.reconstruct(z)
            m = min(m, self.cbf.value(x_hat) - rho)
        return float(m)

    def _last_planned(self) -> np.ndarray:
        if self._U_plan is None or self._plan_age >= self._U_plan.shape[0]:
            return self._u_prev
        return self._U_plan[min(self._plan_age, self._U_plan.shape[0] - 1)]

    def _f_lifted(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """One-step prediction through Koopman, reconstructed to physical space.

        Used by the safety filter's finite-difference Jacobian. The CBF is
        evaluated in physical space so we project the lifted prediction back.
        """
        z = self.lifting.lift(x)
        z_next = self.koopman.A @ z + self.koopman.B @ u
        return self.lifting.reconstruct(z_next)

    def _emergency_brake(
        self, x: np.ndarray | None = None, u_proposed: np.ndarray | None = None
    ) -> np.ndarray:
        """Fallback when the filter slack is too high.

        Earlier versions applied straight braking, which made the vehicle stop
        while still drifting out of lane.  The fallback now combines controlled
        braking with a simple lane-recovery steering law.  This is still a last
        resort; a healthy nominal run should not rely on it.
        """
        u = np.zeros(self.koopman.n_input)
        # Controlled braking, not necessarily max brake.  Max braking at every
        # emergency step caused vx collapse and residual explosions.
        brake = float(self.cfg.emergency_brake)
        u[1] = float(
            np.clip(
                brake,
                self.safety_filter.u_bounds[1, 0],
                self.safety_filter.u_bounds[1, 1],
            )
        )

        steer_cmd = 0.0
        if x is not None and len(x) >= 5:
            py = float(x[1])
            psi = float(x[2])
            vy = float(x[4])
            steer_cmd = (
                -self.cfg.recovery_k_py * py
                -self.cfg.recovery_k_psi * psi
                -self.cfg.recovery_k_vy * vy
            )

        if u_proposed is not None and np.isfinite(u_proposed[0]):
            # Use the QP steering only if it is corrective in the same direction
            # as the lane-recovery law and has larger magnitude.  Do not let an
            # infeasible/degenerate QP override recovery with opposite steering.
            q = float(u_proposed[0])
            if (abs(steer_cmd) < 1e-9 and abs(q) > 0.0) or (q * steer_cmd > 0.0 and abs(q) > abs(steer_cmd)):
                steer_cmd = q

        u[0] = float(
            np.clip(
                steer_cmd,
                self.safety_filter.u_bounds[0, 0],
                self.safety_filter.u_bounds[0, 1],
            )
        )
        return u
