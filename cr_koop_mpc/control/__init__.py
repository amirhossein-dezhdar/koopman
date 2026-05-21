"""Control layer (lazy-imported).

The MPC engines depend on ``cvxpy`` and the controller depends on the safety
layer (``torch``). Importing ``cr_koop_mpc.control`` does not trigger those
imports; the symbols load on first attribute access.
"""

from __future__ import annotations

__all__ = [
    "KoopmanMPC",
    "MPCSolution",
    "NominalNMPC",
    "TubeMPC",
    "ResidualDrivenController",
    "ControllerConfig",
]


def __getattr__(name: str):
    if name in {"KoopmanMPC", "MPCSolution"}:
        from .koopman_mpc import KoopmanMPC, MPCSolution
        return {"KoopmanMPC": KoopmanMPC, "MPCSolution": MPCSolution}[name]
    if name in {"NominalNMPC", "TubeMPC"}:
        from .baselines import NominalNMPC, TubeMPC
        return {"NominalNMPC": NominalNMPC, "TubeMPC": TubeMPC}[name]
    if name in {"ResidualDrivenController", "ControllerConfig"}:
        from .controller import ResidualDrivenController, ControllerConfig
        return {
            "ResidualDrivenController": ResidualDrivenController,
            "ControllerConfig": ControllerConfig,
        }[name]
    raise AttributeError(name)
