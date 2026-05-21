"""CBF-specific regression tests for the lane-prior neural barrier."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cr_koop_mpc.safety import NeuralCBF
from cr_koop_mpc.utils import load_config


def test_lane_prior_cbf_is_px_invariant_and_not_false_safe_outside_lane():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    cbf = NeuralCBF(
        in_dim=len(cfg.cbf.state_indices),
        hidden=[8, 8],
        spectral_norm=True,
        feature_indices=list(cfg.cbf.state_indices),
        full_dim=cfg.state.dim,
        prior_mode=cfg.cbf.prior.mode,
        prior_params=dict(cfg.cbf.prior.params),
    )

    # Absolute position must not affect h.
    X = np.zeros((4, 6), dtype=np.float32)
    X[:, 0] = [0.0, 25.0, 100.0, 300.0]
    X[:, 1] = 0.0
    X[:, 3] = 15.0
    with torch.no_grad():
        h_px = cbf(torch.tensor(X)).numpy()
    assert float(np.max(h_px) - np.min(h_px)) < 1e-6

    # Physical lane departures must be unsafe even before training.  The neural
    # term only subtracts risk, so training cannot make these false-safe.
    Y = np.zeros((6, 6), dtype=np.float32)
    Y[:, 1] = [-6.0, -2.0, -1.76, 1.76, 2.0, 6.0]
    Y[:, 3] = 15.0
    with torch.no_grad():
        h_out = cbf(torch.tensor(Y)).numpy()
    assert np.all(h_out < 0.0), h_out
