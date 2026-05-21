"""MetaDrive backend (v3).

Wraps `metadrive-simulator <https://github.com/metadriverse/metadrive>`_ so
the controller runs against richer scenarios than the pure bicycle while
remaining cheaper than CARLA. ``metadrive`` is imported lazily inside
``__init__`` so the rest of the codebase works without it installed.

v3 improvements over v2:

1. **Real lane-centerline reference.** ``get_reference`` queries the ego
   vehicle's current lane through ``agent.lane.local_coordinates`` to
   build a curve-following reference. Falls back to straight-ahead if the
   lane handle is unavailable (e.g. during a reset frame).

2. **Scenario-specific configuration.** ``reset(scenario=...)`` picks
   appropriate MetaDrive flags per scenario (traffic density, map type).

3. **Working safety features.** ``safety_features`` returns
   ``[lane_clearance, lead_distance]`` queried from MetaDrive's lane
   geometry and forward-cone vehicle scan instead of constants.

4. **Sensor-noise scenario.** Observation is perturbed by the same
   per-coordinate noise std used in the bicycle env, so the controller
   sees identical noise statistics across backends.
"""

from __future__ import annotations

import numpy as np

from .base import EnvBackend, EnvObservation, Reference


# Per-state-component noise std for the sensor_noise scenario (mirrors bicycle).
SENSOR_NOISE_STD = np.array([0.05, 0.05, 0.01, 0.10, 0.05, 0.02])


class MetaDriveEnv(EnvBackend):
    """MetaDrive backend producing the codebase-standard 6-state observation."""

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        try:
            from metadrive.envs.metadrive_env import MetaDriveEnv as _MDEnv
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "MetaDrive backend requested but `metadrive-simulator` is not installed. "
                "Run: pip install metadrive-simulator"
            ) from exc

        md_cfg = cfg.env.metadrive
        self._md_cls = _MDEnv
        self._base_md_config = {
            "map": md_cfg.map,
            "traffic_density": md_cfg.traffic_density,
            "use_render": md_cfg.use_render,
            "physics_world_step_size": cfg.env.dt,
            "decision_repeat": 1,
            "horizon": int(cfg.env.horizon_seconds / cfg.env.dt) + 10,
        }
        self._env = None
        self._prev_x: np.ndarray | None = None
        self.t = 0.0
        self.scenario = "lane_keeping"
        self._rng = np.random.default_rng(cfg.seed)

    # ---------------- interface ----------------

    def reset(self, scenario: str | None = None) -> EnvObservation:
        self.scenario = scenario or self.cfg.env.reference
        md_config = self._scenario_md_config(self.scenario)
        # Recreate the env if the per-scenario config changed; MetaDrive's
        # map / traffic_density are set at construction time on most versions.
        if self._env is not None:
            self._env.close()
            self._env = None
        self._env = self._md_cls(md_config)
        self._env.reset()
        self._prev_x = None
        self.t = 0.0
        return self._read_state()

    def step(self, u: np.ndarray) -> EnvObservation:
        # MetaDrive action: [steer in [-1,1], throttle/brake in [-1,1]]
        steer_max = self.cfg.control.bounds.steer[1]
        accel_max = self.cfg.control.bounds.accel[1]
        steer = float(np.clip(u[0] / steer_max, -1.0, 1.0))
        accel = float(np.clip(u[1] / accel_max, -1.0, 1.0))
        self._env.step(np.array([steer, accel]))
        self.t += self.dt
        return self._read_state()

    def get_reference(self, horizon: int) -> Reference:
        """Lane-centerline reference projected ``horizon`` steps ahead.

        Queries the ego vehicle's current lane, advances by
        ``v_des * k * dt`` of longitudinal arclength, reads the world-frame
        coordinate of the lane centerline at that point, and packages it as
        the 6-state reference. Yaw reference is the lane heading at the
        same point. Falls back to the straight-ahead reference if the lane
        handle is unavailable.
        """
        x_obs = self._read_state().x
        v_des = max(5.0, x_obs[3])
        ref = np.zeros((horizon + 1, 6))

        lane = self._current_lane()
        if lane is None:
            for k in range(horizon + 1):
                ref[k] = [x_obs[0] + v_des * k * self.dt, x_obs[1], x_obs[2], v_des, 0.0, 0.0]
            return Reference(x_ref=ref)

        try:
            long_now, _lat_now = lane.local_coordinates(self._env.agent.position)
        except Exception:
            long_now = 0.0

        last_valid = [x_obs[0], x_obs[1], x_obs[2]]
        for k in range(horizon + 1):
            s = long_now + v_des * k * self.dt
            try:
                pos = lane.position(s, 0.0)
                heading = lane.heading_theta_at(s)
                last_valid = [float(pos[0]), float(pos[1]), float(heading)]
            except Exception:
                # Past the end of the lane: hold the last valid point.
                pass
            ref[k] = [last_valid[0], last_valid[1], last_valid[2], v_des, 0.0, 0.0]

        return Reference(x_ref=ref)

    def scenario_done(self, obs: EnvObservation) -> bool:
        return obs.t >= self.cfg.env.horizon_seconds

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def safety_features(self, obs: EnvObservation) -> np.ndarray:
        """Return ``[lane_clearance, lead_distance]`` for the safety filter."""
        lane = self._current_lane()
        lane_clearance = 1.75
        if lane is not None:
            try:
                _, lat = lane.local_coordinates(self._env.agent.position)
                half_width = lane.width / 2.0
                lane_clearance = float(half_width - abs(lat))
            except Exception:
                pass

        lead_distance = 50.0
        try:
            ego_pos = np.asarray(self._env.agent.position)
            ego_heading = float(self._env.agent.heading_theta)
            cone = np.array([np.cos(ego_heading), np.sin(ego_heading)])
            for other in self._other_vehicles():
                other_pos = np.asarray(other.position)
                rel = other_pos - ego_pos
                dist = float(np.linalg.norm(rel))
                if dist < 1e-3:
                    continue
                forward = float(np.dot(rel / dist, cone))
                if forward > 0.5 and dist < lead_distance:
                    lead_distance = dist
        except Exception:
            pass

        return np.array([lane_clearance, lead_distance])

    # ---------------- internals ----------------

    def _scenario_md_config(self, scenario: str) -> dict:
        """Build a MetaDrive config dict customised for the requested scenario."""
        cfg = dict(self._base_md_config)
        if scenario == "low_friction":
            cfg["traffic_density"] = 0.0
        elif scenario == "cut_in":
            cfg["traffic_density"] = max(0.3, cfg.get("traffic_density", 0.1))
        elif scenario == "dlc":
            cfg["map"] = "S"
            cfg["traffic_density"] = 0.0
        elif scenario == "ood":
            cfg["map"] = "SC"  # curve map
            cfg["traffic_density"] = 0.2
        elif scenario in ("lane_keeping", "sensor_noise"):
            cfg["traffic_density"] = 0.0
        return cfg

    def _current_lane(self):
        """Return the ego vehicle's current lane object, or None."""
        try:
            return self._env.agent.lane
        except Exception:
            return None

    def _other_vehicles(self):
        """Iterate over non-ego vehicles in the engine."""
        try:
            engine = self._env.engine
        except Exception:
            return []
        try:
            ego_id = self._env.agent.id
        except Exception:
            ego_id = None
        out = []
        try:
            for v in engine.get_objects().values():
                if hasattr(v, "position") and hasattr(v, "heading_theta"):
                    if ego_id is not None and getattr(v, "id", None) == ego_id:
                        continue
                    out.append(v)
        except Exception:
            pass
        return out

    def _read_state(self) -> EnvObservation:
        v = self._env.agent
        px, py = v.position
        psi = float(v.heading_theta)
        vx_w, vy_w = v.velocity
        c, s = np.cos(psi), np.sin(psi)
        vx = c * vx_w + s * vy_w
        vy = -s * vx_w + c * vy_w
        x = np.array([px, py, psi, vx, vy, 0.0])
        if self._prev_x is not None:
            dpsi = self._wrap_angle(psi - self._prev_x[2])
            x[5] = dpsi / self.dt
        self._prev_x = x.copy()

        # Sensor-noise scenario: corrupt the observation only; the true
        # state propagates noise-free in MetaDrive's physics.
        if self.scenario == "sensor_noise":
            x = x + self._rng.normal(0.0, SENSOR_NOISE_STD)

        return EnvObservation(x=x, t=self.t, extras={"scenario": self.scenario})

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return (a + np.pi) % (2 * np.pi) - np.pi
