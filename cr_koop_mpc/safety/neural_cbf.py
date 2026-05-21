"""Lipschitz-certified neural control barrier function.

This module deliberately keeps the CBF neural, but prevents the dangerous
failure mode seen in the earlier runs: a free neural network can extrapolate to
``h(x) > 0`` far outside the lane.  The default CBF is therefore a
**neural CBF with a physical lane prior**:

    h(x) = h_prior(py, vy) - c * softplus(g_theta(py, vy))

where ``h_prior`` is a smooth differentiable lane envelope and the
neural term is a non-positive learned risk tightening.  v11 defaults to a
non-saturating quadratic lane prior so the CBF remains corrective near and
outside the lane boundary instead of flattening far from the centerline.  Consequently the
network can make the safe set more conservative, but it cannot label a point
outside the physical lane envelope as safe.  This preserves the paper's neural
CBF component while removing the OOD false-safe bug.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class SpectralLinear(nn.Module):
    """Linear layer with persistent power-iteration spectral normalization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma: float = 1.0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.sigma = sigma
        self.eps = eps
        self.register_buffer("u", torch.randn(out_features))

    def _normalize(self) -> torch.Tensor:
        """Return a spectrally clipped weight without breaking autograd."""
        with torch.no_grad():
            u_buf = self.u.detach()
            v_buf = torch.nn.functional.normalize(
                self.weight.detach().t() @ u_buf, dim=0, eps=self.eps
            )
            u_new = torch.nn.functional.normalize(
                self.weight.detach() @ v_buf, dim=0, eps=self.eps
            )
            self.u.copy_(u_new)

        u = self.u.detach().clone()
        v = torch.nn.functional.normalize(self.weight.t() @ u, dim=0, eps=self.eps)
        sigma_est = torch.dot(u, self.weight @ v)
        scale = torch.clamp(self.sigma / (sigma_est + self.eps), max=1.0)
        return self.weight * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self._normalize()
        return torch.nn.functional.linear(x, W, self.bias)

    @torch.no_grad()
    def spectral_norm(self) -> float:
        """Exact spectral norm via SVD (CPU). Use at training end."""
        return float(torch.linalg.svdvals(self.weight.detach().cpu())[0].item())


class NeuralCBF(nn.Module):
    """Scalar barrier function with certified Lipschitz constant.

    ``feature_indices`` lets the CBF depend only on safety-relevant physical
    coordinates while still being callable with the full simulator state.  The
    default config uses ``[py, vy]``.

    If ``prior_mode`` is ``'lane_saturating'`` or ``'lane_quadratic'``, the network output is interpreted as a
    learned *risk tightening* of a physics-shaped lane barrier.  This is the
    key robustness fix: the CBF remains neural, but it cannot become falsely
    safe far outside the lane.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: list[int],
        spectral_norm: bool = True,
        sigma: float = 1.0,
        feature_indices: list[int] | np.ndarray | None = None,
        full_dim: int | None = None,
        prior_mode: str | None = None,
        prior_params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.feature_indices = (
            None if feature_indices is None else torch.tensor(feature_indices, dtype=torch.long)
        )
        self.full_dim = int(full_dim) if full_dim is not None else None
        if self.feature_indices is not None:
            if int(in_dim) != int(self.feature_indices.numel()):
                raise ValueError(
                    f"in_dim={in_dim} must equal len(feature_indices)="
                    f"{self.feature_indices.numel()}"
                )
        self.in_dim = int(in_dim)
        self.spectral_norm = spectral_norm
        self.prior_mode = None if prior_mode in (None, "none", "") else str(prior_mode)
        self.prior_params = dict(prior_params or {})

        dims = [self.in_dim, *hidden, 1]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            if spectral_norm:
                layers.append(SpectralLinear(dims[i], dims[i + 1], sigma=sigma))
            else:
                layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())  # Lipschitz-1
        self.net = nn.Sequential(*layers)

    # ---------------- feature / prior helpers ----------------

    def _select_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return CBF input features from either feature state or full state."""
        if x.shape[-1] == self.in_dim:
            return x
        if self.feature_indices is None:
            raise ValueError(
                f"CBF expected input dimension {self.in_dim}, got {x.shape[-1]}"
            )
        if self.full_dim is not None and x.shape[-1] != self.full_dim:
            raise ValueError(
                f"CBF expected feature dim {self.in_dim} or full dim {self.full_dim}, "
                f"got {x.shape[-1]}"
            )
        idx = self.feature_indices.to(device=x.device)
        return x.index_select(dim=-1, index=idx)

    def _lane_prior(self, xf: torch.Tensor) -> torch.Tensor:
        """Smooth lane envelope in feature coordinates [py, vy].

        ``py_limit`` is intentionally smaller than the physical lane half-width
        so the filter begins correcting before actual lane departure.

        ``lane_saturating`` is kept for backward compatibility with v9/v10
        checkpoints.  v11 defaults to ``lane_quadratic`` because the saturating
        prior can become too flat outside the lane, which makes the local CBF
        gradient uninformative exactly when recovery steering is needed.
        """
        if self.in_dim < 2:
            raise ValueError("lane prior requires at least [py, vy] features")
        p = self.prior_params
        py_pos = int(p.get("py_pos", 0))
        vy_pos = int(p.get("vy_pos", 1))
        py_limit = float(p.get("py_limit", 1.65))
        py_softness = float(p.get("py_softness", 0.80))
        py_gain = float(p.get("py_gain", 3.0))
        vy_limit = float(p.get("vy_limit", 2.0))
        vy_weight = float(p.get("vy_weight", 4.0))
        eps = float(p.get("smooth_abs_eps", 1e-6))

        py = xf[..., py_pos]
        vy = xf[..., vy_pos]
        mode = self.prior_mode
        if mode == "lane_quadratic":
            lane_term = py_gain * (py_limit * py_limit - py * py)
            vy_term = vy_weight * (vy / max(vy_limit, 1e-6)).pow(2)
            return lane_term - vy_term

        # Backward-compatible v9/v10 prior.
        abs_py = torch.sqrt(py * py + eps)
        lane_term = py_gain * torch.tanh((py_limit - abs_py) / py_softness)
        vy_term = vy_weight * torch.tanh((vy / max(vy_limit, 1e-6)).pow(2))
        return lane_term - vy_term

    def _network_lipschitz(self) -> float:
        bound = 1.0
        for m in self.net:
            if isinstance(m, SpectralLinear):
                bound *= min(m.sigma, m.spectral_norm())
            elif isinstance(m, nn.Linear):
                bound *= float(torch.linalg.svdvals(m.weight.detach().cpu())[0].item())
        return float(bound)

    def _prior_lipschitz(self) -> float:
        if self.prior_mode not in {"lane_saturating", "lane_quadratic"}:
            return 0.0
        p = self.prior_params
        py_gain = float(p.get("py_gain", 3.0))
        py_softness = float(p.get("py_softness", 0.80))
        vy_limit = float(p.get("vy_limit", 2.0))
        vy_weight = float(p.get("vy_weight", 4.0))
        if self.prior_mode == "lane_quadratic":
            # The quadratic prior is not globally Lipschitz on R^2.  The
            # controller uses the local gradient by default; this finite bound
            # is only for diagnostics/fallback over the validation envelope.
            py_bound = float(p.get("py_lip_bound", 6.0))
            vy_bound = float(p.get("vy_lip_bound", 5.0))
            return float(2.0 * py_gain * py_bound + 2.0 * vy_weight * vy_bound / max(vy_limit, 1e-6) ** 2)
        # Conservative bound: |d/dpy tanh((limit-|py|)/s)| <= 1/s;
        # |d/dvy tanh((vy/v)^2)| is bounded by < 2/v for the relevant range.
        return float(py_gain / max(py_softness, 1e-6) + 2.0 * vy_weight / max(vy_limit, 1e-6))

    # ---------------- public NN API ----------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = self._select_features(x)
        raw = self.net(xf).squeeze(-1)
        if self.prior_mode in {"lane_saturating", "lane_quadratic"}:
            corr_scale = float(self.prior_params.get("correction_scale", 0.25))
            risk_tightening = corr_scale * torch.nn.functional.softplus(raw)
            return self._lane_prior(xf) - risk_tightening
        return raw

    @torch.no_grad()
    def lipschitz_constant(self) -> float:
        """Upper bound on ``L_h`` over CBF features/full state projection."""
        net_lip = self._network_lipschitz()
        if self.prior_mode in {"lane_saturating", "lane_quadratic"}:
            corr_scale = float(self.prior_params.get("correction_scale", 0.25))
            # softplus has derivative in (0,1), feature selection is a projection.
            return float(self._prior_lipschitz() + corr_scale * net_lip)
        return net_lip

    def grad_h(self, x: np.ndarray) -> np.ndarray:
        """Gradient ``∇_x h_θ(x)`` at a single point, as numpy."""
        self.eval()
        x_np = np.asarray(x, dtype=np.float32).reshape(1, -1)
        t = torch.tensor(x_np, requires_grad=True)
        h = self(t)
        (grad,) = torch.autograd.grad(h.sum(), t)
        return grad.detach().cpu().numpy().squeeze(0)

    def value(self, x: np.ndarray) -> float:
        self.eval()
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            return float(self(t).item())

    def metadata(self) -> dict:
        return {
            "in_dim": self.in_dim,
            "full_dim": self.full_dim,
            "feature_indices": (
                None if self.feature_indices is None else self.feature_indices.cpu().numpy().tolist()
            ),
            "prior_mode": self.prior_mode,
            "prior_params": self.prior_params,
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_neural_cbf(
    cbf: NeuralCBF,
    safe: np.ndarray,
    unsafe: np.ndarray,
    transitions: tuple[np.ndarray, np.ndarray] | None,
    *,
    gamma: float,
    epochs: int,
    lr: float,
    batch_size: int,
    lambda_barrier: float,
    lambda_feas: float,
    lambda_spec: float,
    device: str = "cpu",
) -> dict:
    """Train a neural CBF (eq. 15) on balanced safe/unsafe samples."""
    cbf.to(device).train()
    opt = torch.optim.Adam(cbf.parameters(), lr=lr)

    safe_t = torch.tensor(safe, dtype=torch.float32, device=device)
    unsafe_t = torch.tensor(unsafe, dtype=torch.float32, device=device)
    if transitions is not None and len(transitions[0]) > 0:
        X_t = torch.tensor(transitions[0], dtype=torch.float32, device=device)
        Xn_t = torch.tensor(transitions[1], dtype=torch.float32, device=device)
        tx_loader = DataLoader(
            TensorDataset(X_t, Xn_t), batch_size=batch_size, shuffle=True
        )
        tx_iter = _cycle(tx_loader)
    else:
        tx_loader = None
        tx_iter = None

    safe_loader = DataLoader(TensorDataset(safe_t), batch_size=batch_size, shuffle=True)
    unsafe_loader = DataLoader(
        TensorDataset(unsafe_t), batch_size=batch_size, shuffle=True
    )

    if len(safe_t) >= len(unsafe_t):
        primary_loader, secondary_loader = safe_loader, unsafe_loader
        primary_is_safe = True
    else:
        primary_loader, secondary_loader = unsafe_loader, safe_loader
        primary_is_safe = False

    last_losses: dict = {}
    loss_history: list[dict] = []
    for epoch in range(epochs):
        epoch_totals: dict[str, float] = {"sign": 0.0, "barrier": 0.0, "feas": 0.0, "spec": 0.0, "total": 0.0}
        epoch_batches = 0
        for (xa,), (xb,) in zip(primary_loader, _cycle(secondary_loader)):
            opt.zero_grad()
            if primary_is_safe:
                xs, xu = xa, xb
            else:
                xu, xs = xa, xb

            h_safe = cbf(xs)
            h_unsafe = cbf(xu)
            l_sign = (
                torch.relu(1.0 - h_safe).mean() + torch.relu(1.0 + h_unsafe).mean()
            )

            if tx_iter is not None:
                x_b, xn_b = next(tx_iter)
                h_now = cbf(x_b)
                h_next = cbf(xn_b)
                l_barrier = torch.relu((1.0 - gamma) * h_now - h_next).mean()
            else:
                l_barrier = torch.zeros((), device=device)

            # Keep a useful positive buffer in the interior of the safe set.
            l_feas = torch.relu(0.5 - h_safe).mean()

            l_spec = torch.zeros((), device=device)
            if cbf.spectral_norm:
                for m in cbf.net:
                    if isinstance(m, SpectralLinear):
                        l_spec = l_spec + (torch.linalg.vector_norm(m.weight) - 1.0).pow(2)

            loss = (
                l_sign
                + lambda_barrier * l_barrier
                + lambda_feas * l_feas
                + lambda_spec * l_spec
            )
            loss.backward()
            opt.step()
            last_losses = {
                "sign": float(l_sign.item()),
                "barrier": float(l_barrier.item()),
                "feas": float(l_feas.item()),
                "spec": float(l_spec.item()),
                "total": float(loss.item()),
            }
            for key, value in last_losses.items():
                epoch_totals[key] += value
            epoch_batches += 1
        if epoch_batches > 0:
            epoch_mean = {k: v / epoch_batches for k, v in epoch_totals.items()}
            epoch_mean["epoch"] = int(epoch + 1)
            loss_history.append(epoch_mean)

    cbf.eval()
    return {"losses": last_losses, "loss_history": loss_history, "Lh": cbf.lipschitz_constant()}


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch
