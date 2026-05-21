"""Simulator backends.

All implement :class:`EnvBackend` so the controller stack never knows which
simulator is underneath. Switch with one line in the config:
``env.backend: bicycle | metadrive | carla``.
"""

from .base import EnvBackend, EnvObservation, Reference
from .bicycle import BicycleEnv


def make_env(cfg) -> EnvBackend:
    """Factory dispatching on ``cfg.env.backend``.

    Heavy backends are imported lazily so the rest of the codebase runs in CI
    without them installed.
    """
    backend = cfg.env.backend.lower()
    if backend == "bicycle":
        return BicycleEnv(cfg)
    if backend == "metadrive":
        from .metadrive_env import MetaDriveEnv
        return MetaDriveEnv(cfg)
    if backend == "carla":
        from .carla_env import CarlaEnv
        return CarlaEnv(cfg)
    raise ValueError(f"Unknown env backend: {backend}")


__all__ = ["EnvBackend", "EnvObservation", "Reference", "BicycleEnv", "make_env"]
