"""Simulator backend interface.

Every backend (bicycle / MetaDrive / CARLA) implements this six-method
contract. The controller stack programs against the abstract base class only.

State convention everywhere in the codebase:

    x = [px, py, psi, vx, vy, r]
    u = [delta, accel]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class EnvObservation:
    """One time-step observation from the simulator."""
    x: np.ndarray         # physical state, shape (6,)
    t: float              # simulation time [s]
    extras: dict          # backend-specific (scenario tag, obstacle list, …)


@dataclass
class Reference:
    """Reference trajectory for the MPC over the lookahead horizon.

    ``x_ref`` has shape ``(N+1, n)`` with ``n`` the state dim.
    """
    x_ref: np.ndarray


class EnvBackend(ABC):
    """All simulator backends must satisfy this six-method contract."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.dt = cfg.env.dt

    @abstractmethod
    def reset(self, scenario: str | None = None) -> EnvObservation:
        """Reset to start of episode, optionally selecting a named scenario."""

    @abstractmethod
    def step(self, u: np.ndarray) -> EnvObservation:
        """Advance one ``dt`` step with control ``u = [delta, accel]``."""

    @abstractmethod
    def get_reference(self, horizon: int) -> Reference:
        """Return reference trajectory over the next ``horizon`` steps."""

    @abstractmethod
    def scenario_done(self, obs: EnvObservation) -> bool:
        """True if the scenario reached its goal or hit a terminal state."""

    @abstractmethod
    def close(self) -> None:
        """Release simulator resources (CARLA actors, MetaDrive engine, …)."""

    # ------------------------------------------------------------------
    # Optional hooks. Backends may override; defaults are sensible no-ops.
    # ------------------------------------------------------------------

    def safety_features(self, obs: EnvObservation) -> np.ndarray:
        """Auxiliary features for the CBF (lane offset, lead-vehicle gap …)."""
        return np.zeros(0)

    def render(self) -> None:  # pragma: no cover
        """Optional rendering. Default no-op."""

    def seed(self, seed: int) -> None:
        """Optional deterministic seeding."""
        np.random.seed(seed)
