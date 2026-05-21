"""Koopman lifting maps.

The EDMD lifting keeps the full physical state as the first coordinates,
so ``reconstruct(lift(x)) == x`` and therefore ``L_x = 1`` for the physical
projection.  Nonlinear dictionary terms, however, should not be applied to
translation coordinates such as absolute ``px``.  Those coordinates grow
monotonically during a normal drive and their squared/RBF features can
artificially dominate residuals even when the physical tracking is good.

By default this implementation uses identity features for all states but
uses polynomial/RBF features only on the dynamics-relevant coordinates
configured by ``koopman.edmd.nonlinear_state_indices``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import only for type-checkers
    import torch
    from torch import nn


class BaseLifting(ABC):
    """Maps physical state ``x`` to lifted state ``z`` and back."""

    n_state: int
    n_lift: int

    @abstractmethod
    def lift(self, x: np.ndarray) -> np.ndarray:
        """Apply lifting. Accepts ``(n,)`` or ``(N, n)``; returns matching shape."""

    @abstractmethod
    def reconstruct(self, z: np.ndarray) -> np.ndarray:
        """Approximate inverse map."""

    @property
    def Lx(self) -> float:
        """Lipschitz constant of the physical reconstruction map."""
        return 1.0


class EDMDLifting(BaseLifting):
    """EDMD lifting with position-safe nonlinear features.

    The raw dictionary is

        z = [x, x_nl^2, RBF_1(x_nl), ..., RBF_M(x_nl)]

    padded/truncated to ``n_lift``.  ``x`` is the full physical state, while
    ``x_nl`` is the subset of states selected by ``feature_indices``.  For the
    bicycle model the safe default is ``[psi, vx, vy, r]``; absolute positions
    are identity-only.
    """

    def __init__(
        self,
        n_state: int,
        n_lift: int,
        n_rbf: int,
        rbf_width: float,
        seed: int = 0,
        feature_indices: Sequence[int] | None = None,
    ) -> None:
        self.n_state = int(n_state)
        self.n_lift = int(n_lift)
        self.n_rbf = int(n_rbf)
        self.rbf_width = float(rbf_width)
        if feature_indices is None:
            feature_indices = list(range(self.n_state))
        self.feature_indices = np.asarray(feature_indices, dtype=int)
        if self.feature_indices.ndim != 1 or len(self.feature_indices) == 0:
            raise ValueError("feature_indices must be a non-empty 1D list")
        if self.feature_indices.min() < 0 or self.feature_indices.max() >= self.n_state:
            raise ValueError("feature_indices contains an out-of-range state index")

        raw_dim = self.n_state + len(self.feature_indices) + self.n_rbf
        if self.n_lift < self.n_state:
            raise ValueError("n_lift must be at least n_state so reconstruction is identity")
        if self.n_lift < raw_dim:
            # Truncation is allowed for backward compatibility, but it should
            # not remove the identity block.  The check above guarantees that.
            pass

        rng = np.random.default_rng(seed)
        self.rbf_centres = rng.uniform(-1.0, 1.0, size=(self.n_rbf, len(self.feature_indices)))
        self.x_scale = np.ones(self.n_state)
        self.feature_scale = np.ones(len(self.feature_indices))

    def fit_scaling(self, X: np.ndarray) -> None:
        """Set per-coordinate scales for dimensionally sane nonlinear features."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if X.shape[1] != self.n_state:
            raise ValueError(f"Expected X with {self.n_state} states, got {X.shape[1]}")
        self.x_scale = np.clip(np.std(X, axis=0), 1e-3, None)
        self.feature_scale = self.x_scale[self.feature_indices]

    def lift(self, x: np.ndarray) -> np.ndarray:
        single = x.ndim == 1
        X = np.atleast_2d(x).astype(np.float64)
        if X.shape[1] != self.n_state:
            raise ValueError(f"Expected state dimension {self.n_state}, got {X.shape[1]}")
        Xf = X[:, self.feature_indices]
        Xfs = Xf / self.feature_scale

        # Identity for the full physical state; nonlinear features only for
        # the selected dynamics-relevant coordinates.
        poly = np.concatenate([X, Xf ** 2], axis=1)
        diffs = Xfs[:, None, :] - self.rbf_centres[None, :, :]
        d2 = np.sum(diffs ** 2, axis=-1)
        rbf = np.exp(-d2 / (2.0 * self.rbf_width ** 2))
        Z = np.concatenate([poly, rbf], axis=1)
        Z = self._resize(Z)
        return Z[0] if single else Z

    def reconstruct(self, z: np.ndarray) -> np.ndarray:
        single = z.ndim == 1
        Z = np.atleast_2d(z)
        X = Z[:, : self.n_state]
        return X[0] if single else X

    def _resize(self, Z: np.ndarray) -> np.ndarray:
        N, raw = Z.shape
        if raw == self.n_lift:
            return Z
        if raw > self.n_lift:
            return Z[:, : self.n_lift]
        pad = np.zeros((N, self.n_lift - raw))
        return np.concatenate([Z, pad], axis=1)


def _make_deep_lifting_classes():
    """Lazy factory: build the torch-dependent lifting classes on demand."""
    import torch
    from torch import nn

    class _EncoderMLP(nn.Module):
        def __init__(self, n_in: int, hidden: list[int], n_out: int):
            super().__init__()
            layers: list[nn.Module] = []
            last = n_in
            for h in hidden:
                layers += [nn.Linear(last, h), nn.GELU()]
                last = h
            layers.append(nn.Linear(last, n_out - n_in))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.cat([x, self.net(x)], dim=-1)

    class _DeepLifting(BaseLifting):
        """Encoder-based lifting: ``z = [x, φ_θ(x)]``."""

        def __init__(
            self,
            n_state: int,
            n_lift: int,
            hidden: list[int],
            device: str = "cpu",
        ) -> None:
            if n_lift <= n_state:
                raise ValueError("n_lift must exceed n_state for DeepLifting")
            self.n_state = n_state
            self.n_lift = n_lift
            self.device = device
            self.encoder = _EncoderMLP(n_state, hidden, n_lift).to(device)
            self.encoder.eval()

        def lift(self, x: np.ndarray) -> np.ndarray:
            single = x.ndim == 1
            X = np.atleast_2d(x).astype(np.float32)
            with torch.no_grad():
                Z = self.encoder(torch.from_numpy(X).to(self.device)).cpu().numpy()
            return Z[0] if single else Z

        def reconstruct(self, z: np.ndarray) -> np.ndarray:
            single = z.ndim == 1
            Z = np.atleast_2d(z)
            X = Z[:, : self.n_state]
            return X[0] if single else X

    return _DeepLifting


# Exposed under the public name for backward compat / type hinting.
def DeepLifting(*args, **kwargs):  # noqa: N802 - kept as the public class name
    cls = _make_deep_lifting_classes()
    return cls(*args, **kwargs)


def _cfg_get(obj, name: str, default):
    return obj[name] if isinstance(obj, dict) and name in obj else getattr(obj, name, default)


def make_lifting(cfg) -> BaseLifting:
    """Factory selecting EDMD vs deep from config."""
    if cfg.koopman.type == "edmd":
        e = cfg.koopman.edmd
        feature_indices = _cfg_get(e, "nonlinear_state_indices", list(range(cfg.state.dim)))
        return EDMDLifting(
            n_state=cfg.state.dim,
            n_lift=cfg.koopman.lifting_dim,
            n_rbf=e.n_rbf,
            rbf_width=e.rbf_width,
            seed=cfg.seed,
            feature_indices=feature_indices,
        )
    if cfg.koopman.type == "deep":
        return DeepLifting(
            n_state=cfg.state.dim,
            n_lift=cfg.koopman.lifting_dim,
            hidden=list(cfg.koopman.deep.encoder_hidden),
        )
    raise ValueError(f"Unknown koopman type: {cfg.koopman.type}")
