"""Kinematic + lateral-dynamics bicycle backend.

Used for unit tests, theory experiments, and CI. No external sim
dependencies. Implements the same :class:`EnvBackend` interface as MetaDrive
and CARLA so the controller code does not change.

Dynamic bicycle at moderate-to-high speed, kinematic fall-back below
``VX_MIN`` to avoid the 1/vx singularity. Friction scaled by ``mu`` so the
low-friction scenario is a single parameter change.
"""

from __future__ import annotations

import numpy as np

from .base import EnvBackend, EnvObservation, Reference


class BicycleEnv(EnvBackend):
    """Discrete-time dynamic-bicycle simulator with switchable scenarios."""

    VX_MIN = 0.5

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        b = cfg.env.bicycle
        self.m = b.mass
        self.Iz = b.Iz
        self.lf = b.lf
        self.lr = b.lr
        self.Cf = b.Cf
        self.Cr = b.Cr
        self.mu = b.mu
        self.L = self.lf + self.lr
        self.scenario = "lane_keeping"
        self.t = 0.0
        self.x = np.zeros(6)

    # ---------------- interface ----------------

    def reset(self, scenario: str | None = None) -> EnvObservation:
        self.scenario = scenario or self.cfg.env.reference
        self.t = 0.0
        self.x = np.array([0.0, 0.0, 0.0, 15.0, 0.0, 0.0])

        # Deterministic stress-test initial conditions.  They keep the same
        # reference trajectory but start the ego vehicle away from it, which is
        # essential for checking whether event-triggering is genuinely reactive
        # rather than only a periodic watchdog.
        if self.scenario in ("lane_offset_pos", "offset_pos", "py_plus_1"):
            self.x[1] = 1.0
        elif self.scenario in ("lane_offset_neg", "offset_neg", "py_minus_1"):
            self.x[1] = -1.0
        elif self.scenario in ("near_boundary_pos", "boundary_pos"):
            self.x[1] = 1.45
        elif self.scenario in ("near_boundary_neg", "boundary_neg"):
            self.x[1] = -1.45
        elif self.scenario in ("heading_error_pos", "heading_pos"):
            self.x[2] = 0.12
        elif self.scenario in ("heading_error_neg", "heading_neg"):
            self.x[2] = -0.12
        elif self.scenario in ("speed_low", "vx_low"):
            self.x[3] = 11.0
        elif self.scenario in ("speed_high", "vx_high"):
            self.x[3] = 18.0

        self.mu = 0.4 if self.scenario == "low_friction" else self.cfg.env.bicycle.mu
        return self._obs()

    def step(self, u: np.ndarray) -> EnvObservation:
        u = self._clip_input(u)
        self.x = self._integrate(self.x, u, self.dt)
        # Clip longitudinal velocity at zero: a real braking vehicle stops,
        # it doesn't drive backward. Without this, an aggressive
        # emergency-brake fallback rides vx all the way to -100 m/s, which
        # is meaningless and makes residual-based diagnostics explode.
        if self.x[3] < 0.0:
            self.x[3] = 0.0
            self.x[4] = 0.0   # also kill vy when stopped
            self.x[5] = 0.0   # and yaw rate
        self.t += self.dt
        return self._obs()

    def get_reference(self, horizon: int) -> Reference:
        return Reference(x_ref=self._reference_horizon(horizon))

    def scenario_done(self, obs: EnvObservation) -> bool:
        return obs.t >= self.cfg.env.horizon_seconds

    def close(self) -> None:
        pass

    def safety_features(self, obs: EnvObservation) -> np.ndarray:
        py = obs.x[1]
        lane_half_width = 1.75
        return np.array(
            [lane_half_width - abs(py), self._synthetic_lead_distance(obs.t)]
        )

    # ---------------- dynamics ----------------

    def _integrate(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """RK4 step on the bicycle ODE."""
        k1 = self._f(x, u)
        k2 = self._f(x + 0.5 * dt * k1, u)
        k3 = self._f(x + 0.5 * dt * k2, u)
        k4 = self._f(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        px, py, psi, vx, vy, r = x
        delta, ax = u
        if vx < self.VX_MIN:
            beta = np.arctan(self.lr / self.L * np.tan(delta))
            return np.array(
                [
                    vx * np.cos(psi + beta),
                    vx * np.sin(psi + beta),
                    vx / self.L * np.tan(delta) * np.cos(beta),
                    ax,
                    0.0,
                    0.0,
                ]
            )
        Cf = self.mu * self.Cf
        Cr = self.mu * self.Cr
        alpha_f = delta - (vy + self.lf * r) / vx
        alpha_r = -(vy - self.lr * r) / vx
        Fyf = Cf * alpha_f
        Fyr = Cr * alpha_r
        dpx = vx * np.cos(psi) - vy * np.sin(psi)
        dpy = vx * np.sin(psi) + vy * np.cos(psi)
        dpsi = r
        dvx = ax - (Fyf * np.sin(delta)) / self.m + vy * r
        dvy = (Fyf * np.cos(delta) + Fyr) / self.m - vx * r
        dr = (self.lf * Fyf * np.cos(delta) - self.lr * Fyr) / self.Iz
        return np.array([dpx, dpy, dpsi, dvx, dvy, dr])

    def _clip_input(self, u: np.ndarray) -> np.ndarray:
        sb = self.cfg.control.bounds.steer
        ab = self.cfg.control.bounds.accel
        return np.array([np.clip(u[0], sb[0], sb[1]), np.clip(u[1], ab[0], ab[1])])

    # ---------------- references / scenarios ----------------

    def _reference_horizon(self, horizon: int) -> np.ndarray:
        ref = np.zeros((horizon + 1, 6))
        v_des = 15.0
        for k in range(horizon + 1):
            tk = self.t + k * self.dt
            py_des = self._lateral_reference(tk)
            ref[k] = [self.x[0] + v_des * k * self.dt, py_des, 0.0, v_des, 0.0, 0.0]
        return ref

    def _lateral_reference(self, t: float) -> float:
        if self.scenario in (
            "lane_keeping", "low_friction", "sensor_noise",
            "lane_offset_pos", "offset_pos", "py_plus_1",
            "lane_offset_neg", "offset_neg", "py_minus_1",
            "near_boundary_pos", "boundary_pos",
            "near_boundary_neg", "boundary_neg",
            "heading_error_pos", "heading_pos",
            "heading_error_neg", "heading_neg",
            "speed_low", "vx_low", "speed_high", "vx_high",
        ):
            return 0.0
        if self.scenario == "dlc":
            if 2.0 <= t < 4.0:
                return 1.0
            if 4.0 <= t < 6.0:
                return -1.0
            return 0.0
        if self.scenario == "cut_in":
            return 0.0
        if self.scenario == "ood":
            return 1.5 * np.sin(0.8 * t)
        return 0.0

    def _synthetic_lead_distance(self, t: float) -> float:
        if self.scenario != "cut_in":
            return 50.0
        if t < 3.0:
            return 30.0
        return max(2.0, 8.0 - 1.0 * (t - 3.0))

    def _obs(self) -> EnvObservation:
        x_obs = self.x.copy()
        if self.scenario == "sensor_noise":
            # Noise scale calibrated to sit just above eps_sens = 0.05.
            sigma = np.array([0.03, 0.03, 0.005, 0.05, 0.02, 0.01])
            x_obs = x_obs + np.random.normal(0.0, sigma)
        return EnvObservation(
            x=x_obs, t=self.t, extras={"scenario": self.scenario}
        )
