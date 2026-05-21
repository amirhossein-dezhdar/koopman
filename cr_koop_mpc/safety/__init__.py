"""Safety layer (lazy-imported).

Both :class:`NeuralCBF` and :class:`SafetyFilterQP` depend on heavy optional
packages (``torch`` and ``cvxpy``). Importing ``cr_koop_mpc.safety`` does
not trigger those imports; the symbols are loaded on first access.
"""

from __future__ import annotations

__all__ = ["NeuralCBF", "SpectralLinear", "train_neural_cbf", "SafetyFilterQP"]


def __getattr__(name: str):
    if name in {"NeuralCBF", "SpectralLinear", "train_neural_cbf"}:
        from .neural_cbf import NeuralCBF, SpectralLinear, train_neural_cbf
        return {
            "NeuralCBF": NeuralCBF,
            "SpectralLinear": SpectralLinear,
            "train_neural_cbf": train_neural_cbf,
        }[name]
    if name == "SafetyFilterQP":
        from .qp_filter import SafetyFilterQP
        return SafetyFilterQP
    raise AttributeError(name)
