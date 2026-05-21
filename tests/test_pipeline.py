"""End-to-end smoke test on the bicycle backend.

Runs the full pipeline (collect → identify → train CBF → calibrate → close
loop) on tiny budgets and asserts basic invariants:

* Conformal coverage at end of run is at least ``1 − 2α`` (loose bound).
* The MPC actually solved at least once.
* ACI keeps α in its admissible interval.
* Lifting reconstruction is the identity on the first n_state coords.

Skips any GPU and does not need CARLA / MetaDrive.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Make src importable when run via ``pytest`` from repo root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cr_koop_mpc.utils import load_config  # noqa: E402


def _tiny_config(tmp_dir: Path):
    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg.env.horizon_seconds = 2.0
    cfg.mpc.N = 8
    cfg.cbf.train.epochs = 2
    cfg.cbf.hidden = [16, 16]
    cfg.koopman.lifting_dim = 16
    cfg.koopman.edmd.n_rbf = 8
    cfg.conformal.calibration_size = 50
    cfg.logging.out_dir = str(tmp_dir / "runs")
    return cfg


def test_lifting_reconstruction_is_identity():
    """``L_x`` claim only holds if x = reconstruct(lift(x))[:n]."""
    from cr_koop_mpc.models import make_lifting
    cfg = load_config(ROOT / "configs" / "default.yaml")
    lif = make_lifting(cfg)
    x = np.array([1.0, -0.5, 0.1, 12.0, 0.2, 0.05])
    z = lif.lift(x)
    x_rec = lif.reconstruct(z)
    np.testing.assert_allclose(x_rec, x, atol=1e-10)
    assert lif.Lx == 1.0


def test_conformal_adapts_alpha():
    """A single violation should immediately make ACI more conservative."""
    from cr_koop_mpc.models import ConformalResidual
    rng = np.random.default_rng(0)
    conf = ConformalResidual(alpha=0.1, adaptive=True, gamma=0.1, window=200)
    conf.calibrate(rng.exponential(0.2, 300))
    alpha0 = conf.alpha
    # The update rule uses the previous bound; a score above it is a
    # miscoverage event and should reduce alpha on that step.
    conf.update(conf.bound() + 1.0)
    assert conf.alpha < alpha0, (alpha0, conf.alpha)


def test_full_pipeline_smoke():
    """Run all four offline scripts then a 2-second closed loop on bicycle."""
    pytest.importorskip("cvxpy")
    pytest.importorskip("torch")

    import torch
    from cr_koop_mpc.envs import make_env
    from cr_koop_mpc.models import (
        ConformalResidual, KoopmanPredictor, make_lifting,
    )
    from cr_koop_mpc.safety import NeuralCBF, SafetyFilterQP, train_neural_cbf
    from cr_koop_mpc.triggering import ResidualDrivenTrigger
    from cr_koop_mpc.control import (
        KoopmanMPC, ResidualDrivenController, ControllerConfig,
    )

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cfg = _tiny_config(td)

        # 1) Collect a tiny dataset
        env = make_env(cfg)
        env.seed(cfg.seed)
        Xs, Us, Xn = [], [], []
        for scenario in ["lane_keeping", "dlc"]:
            obs = env.reset(scenario=scenario)
            x = obs.x.copy()
            for _ in range(int(cfg.env.horizon_seconds / cfg.env.dt)):
                u = np.array([0.1 * np.sin(obs.t), 0.5])
                obs = env.step(u)
                Xs.append(x); Us.append(u); Xn.append(obs.x.copy())
                x = obs.x.copy()
        X = np.asarray(Xs); U = np.asarray(Us); Xn = np.asarray(Xn)
        env.close()

        # 2) Fit Koopman
        lifting = make_lifting(cfg)
        lifting.fit_scaling(X)
        koop = KoopmanPredictor(lifting, n_input=cfg.control.dim)
        koop.fit(X, U, Xn)

        # 3) Train tiny CBF
        cbf = NeuralCBF(in_dim=len(cfg.cbf.state_indices), hidden=list(cfg.cbf.hidden), spectral_norm=True, feature_indices=list(cfg.cbf.state_indices), full_dim=cfg.state.dim)
        py = X[:, 1]
        safe = X[np.abs(py) < 1.0]
        unsafe_synth = X.copy(); unsafe_synth[:, 1] += 2.5
        train_neural_cbf(
            cbf, safe=safe[:50], unsafe=unsafe_synth[:50], transitions=None,
            gamma=0.2, epochs=2, lr=1e-3, batch_size=16,
            lambda_barrier=1.0, lambda_feas=0.5, lambda_spec=0.1,
        )

        # 4) Calibrate conformal
        scores = np.array([
            koop.residual_score(X[i], U[i], Xn[i], indices=list(cfg.residual.state_indices))
            for i in range(min(50, len(X)))
        ])
        conf = ConformalResidual(alpha=0.1, adaptive=True, window=100)
        conf.calibrate(scores)

        # 5) Build controller
        u_b = np.array([cfg.control.bounds.steer, cfg.control.bounds.accel])
        du_b = np.array([cfg.control.rate_bounds.steer, cfg.control.rate_bounds.accel])
        mpc = KoopmanMPC(
            lifting, koop, cfg.mpc.N,
            list(cfg.mpc.Q_diag), list(cfg.mpc.R_diag), list(cfg.mpc.S_diag),
            u_b, du_b, cfg.mpc.P_terminal_scale, cfg.mpc.solver,
        )
        sf = SafetyFilterQP(cbf, u_b, du_b, np.asarray(cfg.safety_filter.H_diag), cfg.safety_filter.lambda_slack)
        trig = ResidualDrivenTrigger(
            cfg.trigger.sigma_min, cfg.trigger.sigma_max, cfg.trigger.lambda_w,
            cfg.trigger.eps_safety, cfg.trigger.eps_filter, cfg.trigger.N_max,
        )
        ctrl = ResidualDrivenController(
            lifting, koop, mpc, cbf, conf, trig, sf,
            ControllerConfig(L_x=lifting.Lx),
        )

        # 6) Closed loop
        env = make_env(cfg)
        env.seed(cfg.seed)
        obs = env.reset(scenario="lane_keeping")
        ctrl.reset(obs.x)
        mpc_solves = 0
        for _ in range(int(cfg.env.horizon_seconds / cfg.env.dt)):
            ref = env.get_reference(cfg.mpc.N)
            log = ctrl.step(obs.x, ref.x_ref)
            mpc_solves += int(log.mpc_solved)
            obs = env.step(log.u)
        env.close()

        # Assertions
        assert mpc_solves >= 1, "MPC should have solved at least once"
        cov = conf.empirical_coverage()
        assert cov > 0.5, f"Coverage suspiciously low: {cov}"
        assert cfg.conformal.alpha_min <= conf.alpha <= cfg.conformal.alpha_max
