"""CARLA backend.

CARLA's Python client must be installed manually (``carla`` egg matching the
server version). This skeleton lays out the same interface as the other
backends. Replace ``_spawn_vehicle`` and ``_read_state`` with production
code on the lab machine that runs the actual simulator.
"""

from __future__ import annotations

import numpy as np

from .base import EnvBackend, EnvObservation, Reference


class CarlaEnv(EnvBackend):
    """CARLA backend (production stub)."""

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        try:
            import carla
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "CARLA backend requested but `carla` python client is not installed. "
                "Install the egg matching your CARLA server version."
            ) from exc

        c = cfg.env.carla
        self._carla = carla
        self._client = carla.Client(c.host, c.port)
        self._client.set_timeout(10.0)
        self._world = self._client.load_world(c.town)
        self._configure_sync(sync=c.sync_mode)
        self._vehicle = None
        self._prev_x: np.ndarray | None = None
        self.t = 0.0
        self.scenario = "lane_keeping"

    def reset(self, scenario: str | None = None) -> EnvObservation:
        self.scenario = scenario or self.cfg.env.reference
        if self._vehicle is not None:
            self._vehicle.destroy()
        self._vehicle = self._spawn_vehicle()
        self._prev_x = None
        self.t = 0.0
        self._world.tick()
        return self._read_state()

    def step(self, u: np.ndarray) -> EnvObservation:
        delta, ax = u
        steer_max = self.cfg.control.bounds.steer[1]
        steer = float(np.clip(delta / steer_max, -1.0, 1.0))
        if ax >= 0:
            throttle = float(np.clip(ax / self.cfg.control.bounds.accel[1], 0.0, 1.0))
            brake = 0.0
        else:
            throttle = 0.0
            brake = float(np.clip(-ax / abs(self.cfg.control.bounds.accel[0]), 0.0, 1.0))
        ctrl = self._carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        self._vehicle.apply_control(ctrl)
        self._world.tick()
        self.t += self.dt
        return self._read_state()

    def get_reference(self, horizon: int) -> Reference:
        x = self._read_state().x
        v_des = max(5.0, x[3])
        ref = np.zeros((horizon + 1, 6))
        for k in range(horizon + 1):
            ref[k] = [x[0] + v_des * k * self.dt, 0.0, 0.0, v_des, 0.0, 0.0]
        return Reference(x_ref=ref)

    def scenario_done(self, obs: EnvObservation) -> bool:
        return obs.t >= self.cfg.env.horizon_seconds

    def close(self) -> None:
        if self._vehicle is not None:
            self._vehicle.destroy()
            self._vehicle = None
        self._configure_sync(sync=False)

    # ---------------- internals (replace at lab time) ----------------

    def _configure_sync(self, sync: bool) -> None:
        settings = self._world.get_settings()
        settings.synchronous_mode = sync
        settings.fixed_delta_seconds = self.dt if sync else 0.0
        self._world.apply_settings(settings)

    def _spawn_vehicle(self):
        blueprint = self._world.get_blueprint_library().find(self.cfg.env.carla.vehicle)
        spawn_point = self._world.get_map().get_spawn_points()[0]
        return self._world.spawn_actor(blueprint, spawn_point)

    def _read_state(self) -> EnvObservation:
        tf = self._vehicle.get_transform()
        v = self._vehicle.get_velocity()
        psi = np.deg2rad(tf.rotation.yaw)
        c, s = np.cos(psi), np.sin(psi)
        vx = c * v.x + s * v.y
        vy = -s * v.x + c * v.y
        x = np.array([tf.location.x, tf.location.y, psi, vx, vy, 0.0])
        if self._prev_x is not None:
            dpsi = (psi - self._prev_x[2] + np.pi) % (2 * np.pi) - np.pi
            x[5] = dpsi / self.dt
        self._prev_x = x.copy()
        return EnvObservation(x=x, t=self.t, extras={"scenario": self.scenario})
