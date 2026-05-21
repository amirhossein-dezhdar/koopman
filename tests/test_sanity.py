"""Lightweight sanity test (numpy only).

Verifies the deps-free portion of the codebase: lifting, Koopman LS
identification, conformal residual, trigger logic, and bicycle dynamics.

Does NOT exercise the safety filter (cvxpy) or neural CBF (torch).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cr_koop_mpc.envs import make_env
from cr_koop_mpc.models import ConformalResidual, KoopmanPredictor, make_lifting
from cr_koop_mpc.triggering import ResidualDrivenTrigger, TriggerReason
from cr_koop_mpc.utils import load_config


def test_lifting_reconstruction_identity():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    lif = make_lifting(cfg)
    x = np.array([1.0, -0.5, 0.1, 12.0, 0.2, 0.05])
    z = lif.lift(x)
    np.testing.assert_allclose(lif.reconstruct(z), x, atol=1e-10)
    assert lif.Lx == 1.0
    print("[OK] lifting reconstruction is identity, L_x = 1")


def test_bicycle_runs():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg.env.horizon_seconds = 1.0
    env = make_env(cfg)
    obs = env.reset(scenario="lane_keeping")
    n_steps = 0
    while not env.scenario_done(obs):
        obs = env.step(np.array([0.05, 0.5]))
        n_steps += 1
    env.close()
    assert n_steps == 20, n_steps
    print(f"[OK] bicycle 1s @ 20Hz produced {n_steps} steps")


def test_koopman_ls_identification():
    """Identify A, B on bicycle data and check residual norm decreases vs prior."""
    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg.env.horizon_seconds = 5.0
    env = make_env(cfg)

    X, U, Xn = [], [], []
    for sc in ["lane_keeping", "dlc"]:
        obs = env.reset(scenario=sc)
        x = obs.x.copy()
        while not env.scenario_done(obs):
            u = np.array([0.1 * np.sin(obs.t), 0.5])
            obs = env.step(u)
            X.append(x); U.append(u); Xn.append(obs.x.copy())
            x = obs.x.copy()
    env.close()
    X = np.asarray(X); U = np.asarray(U); Xn = np.asarray(Xn)

    lifting = make_lifting(cfg)
    lifting.fit_scaling(X)
    koop = KoopmanPredictor(lifting, n_input=cfg.control.dim)
    koop.fit(X, U, Xn)

    # Mean residual on training data should be small (well below the lifted
    # state magnitude). This is a sanity check, not a tight bound.
    res = []
    for i in range(min(200, len(X))):
        res.append(koop.residual_score(X[i], U[i], Xn[i], indices=list(cfg.residual.state_indices)))
    mean_z = np.mean(np.linalg.norm(lifting.lift(X), axis=1))
    mean_res = np.mean(res)
    assert mean_res < mean_z, (mean_res, mean_z)
    print(f"[OK] Koopman LS fit: mean residual {mean_res:.4f} << mean z {mean_z:.2f}")


def test_conformal_bound_and_aci():
    """ACI's actual guarantee: long-run empirical miscoverage is controlled.

    We feed a stationary heavy-tailed residual sequence. The paper's formal
    claim is one-sided: the long-run miscoverage frequency should not exceed
    the target beyond a small finite-sample slack.
    """
    rng = np.random.default_rng(0)
    target = 0.1
    conf = ConformalResidual(alpha=target, adaptive=True, gamma=0.05, window=300)
    conf.calibrate(rng.exponential(0.3, 300))

    # Stream a stationary heavy-tailed sequence: ACI should settle on an α
    # such that ~10% of scores exceed the running bound.
    n_total, n_violations = 0, 0
    for _ in range(2000):
        s = rng.exponential(0.5)
        prev_w = conf.bound()
        if s > prev_w:
            n_violations += 1
        n_total += 1
        conf.update(s)

    # Paper Theorem 1: lim sup of miscoverage frequency <= alpha_target.
    rate = n_violations / n_total
    assert rate <= target + 0.05, (
        f"miscoverage rate {rate:.3f} exceeds target {target} + slack"
    )
    assert conf.alpha_min <= conf.alpha <= conf.alpha_max
    print(f"[OK] ACI long-run miscoverage = {rate:.3f} (target {target}); alpha = {conf.alpha:.3f}")


def test_trigger_residual_aware_threshold():
    """Trigger σ_k should shrink as w̄_k grows, firing more easily."""
    trig = ResidualDrivenTrigger(
        sigma_min=0.05, sigma_max=0.5, lambda_w=2.0,
        eps_safety=0.1, eps_filter=0.05, N_max=20,
    )
    z_now = np.zeros(8); z_pred = np.array([0.2] + [0.0] * 7)  # E_z = 0.2

    # Small w_bar → σ ≈ 0.5 → 0.2 < 0.5, model condition does NOT fire
    d1 = trig.step(z_now, z_pred, min_safety_margin=1.0, last_input_change=0.0, w_bar=0.01)
    assert not (d1.reason & TriggerReason.MODEL)

    trig.reset()
    # Large w_bar → σ ≈ 0.05 → 0.2 >> 0.05, model condition DOES fire
    d2 = trig.step(z_now, z_pred, min_safety_margin=1.0, last_input_change=0.0, w_bar=0.5)
    assert d2.reason & TriggerReason.MODEL
    print(f"[OK] Trigger residual-aware: σ {d1.sigma_k:.2f} → {d2.sigma_k:.2f}")


def test_trigger_staleness_fires():
    trig = ResidualDrivenTrigger(0.05, 0.5, 2.0, 0.1, 0.05, N_max=3)
    z = np.zeros(8)
    fired_at = None
    for k in range(10):
        d = trig.step(z, z, min_safety_margin=1.0, last_input_change=0.0, w_bar=0.0)
        if d.reason & TriggerReason.STALE:
            fired_at = k
            break
    assert fired_at == 2, fired_at  # zero-indexed: N_max=3 means fires at step 2 (the 3rd)
    print(f"[OK] Trigger staleness fires at step {fired_at} (N_max=3)")


if __name__ == "__main__":
    test_lifting_reconstruction_identity()
    test_bicycle_runs()
    test_koopman_ls_identification()
    test_conformal_bound_and_aci()
    test_trigger_residual_aware_threshold()
    test_trigger_staleness_fires()
    print("\nAll smoke tests passed.")
