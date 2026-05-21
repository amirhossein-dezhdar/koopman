"""Closed-loop experiment driver.

Loads everything trained by the four offline scripts, builds the controller
with the requested baseline configuration, and rolls a closed-loop episode in
the chosen simulator. Logs per-step quantities to NPZ for plotting.

Baselines (paper §X):

    nmpc       — Nominal NMPC (no Koopman, no triggering, no filter)
    periodic   — Koopman MPC, re-solve every step
    static_et  — Koopman MPC, static event-trigger threshold (residual-aware disabled)
    no_filter  — Koopman MPC, event-triggered, but no safety filter
    cbf_qp     — Safety filter only, no MPC
    tube       — Tube Koopman MPC with constant disturbance bound
    ours       — Proposed method (everything on)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
DEFAULT_CONFIG = ROOT_DIR / "configs" / "default.yaml"
DEFAULT_DATA = ROOT_DIR / "artifacts" / "data" / "bicycle_dataset.npz"
DEFAULT_KOOPMAN = ROOT_DIR / "artifacts" / "data" / "koopman.npz"
DEFAULT_CBF = ROOT_DIR / "artifacts" / "data" / "cbf.pt"
DEFAULT_CONFORMAL = ROOT_DIR / "artifacts" / "data" / "conformal.npz"


def resolve_project_path(path: str | Path) -> Path:
    """Resolve relative paths against the repository root, not the shell CWD."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT_DIR / p


def _cfg_get(obj, name: str, default):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)

import numpy as np
import torch

from cr_koop_mpc.control import (
    ControllerConfig,
    KoopmanMPC,
    NominalNMPC,
    ResidualDrivenController,
    TubeMPC,
)
from cr_koop_mpc.envs import make_env
from cr_koop_mpc.models import ConformalResidual, KoopmanPredictor, make_lifting
from cr_koop_mpc.safety import NeuralCBF, SafetyFilterQP
from cr_koop_mpc.triggering import ResidualDrivenTrigger
from cr_koop_mpc.utils import RunLogger, load_config, save_json, stage_header


BASELINE_FLAGS = {
    "ours":      dict(use_event_trigger=True,  use_residual_margin=True,  use_safety_filter=True,  use_mpc=True),
    "periodic":  dict(use_event_trigger=False, use_residual_margin=True,  use_safety_filter=True,  use_mpc=True),
    "static_et": dict(use_event_trigger=True,  use_residual_margin=False, use_safety_filter=True,  use_mpc=True),
    "no_filter": dict(use_event_trigger=True,  use_residual_margin=True,  use_safety_filter=False, use_mpc=True),
    "cbf_qp":    dict(use_event_trigger=False, use_residual_margin=True,  use_safety_filter=True,  use_mpc=False),
    "tube":      dict(use_event_trigger=True,  use_residual_margin=False, use_safety_filter=True,  use_mpc=True),
    "nmpc":      dict(use_event_trigger=False, use_residual_margin=False, use_safety_filter=True,  use_mpc=True),
}


def build_controller(
    cfg,
    baseline: str,
    data_path: str | Path = DEFAULT_DATA,
    koopman_path: str | Path = DEFAULT_KOOPMAN,
    cbf_path: str | Path = DEFAULT_CBF,
    conformal_path: str | Path = DEFAULT_CONFORMAL,
):
    # --- Lifting & Koopman ---
    data = np.load(data_path)
    X = data["X"]
    min_trans = int(_cfg_get(getattr(cfg, "diagnostics", None), "min_dataset_transitions", 30000))
    if len(X) < min_trans and bool(_cfg_get(getattr(cfg, "diagnostics", None), "strict_artifact_checks", True)):
        raise RuntimeError(
            f"Closed-loop run is using a small dataset artifact ({len(X)} transitions < {min_trans}). "
            "Regenerate artifacts with scripts/collect_data.py before trusting results."
        )
    lifting = make_lifting(cfg)
    k = np.load(koopman_path)
    if "trained_on_n_transitions" in k.files:
        trained_n = int(np.asarray(k["trained_on_n_transitions"]).reshape(-1)[0])
        if trained_n != len(X):
            raise RuntimeError(
                f"Koopman artifact was trained on {trained_n} transitions but dataset has {len(X)}. "
                "This usually means stale artifacts. Rerun train_koopman.py and calibrate.py."
            )
    # Prefer the saved x_scale from training so the controller uses the
    # *identical* lifting that produced (A, B). Falling back to fit_scaling
    # on the full dataset (the v3/v4 behaviour) silently broke residual
    # calibration: training used X[id_idx], run-time used X, so the lifted
    # coordinates didn't match (A, B) and the first online residual exploded.
    if hasattr(lifting, "x_scale") and "x_scale" in k.files:
        lifting.x_scale = np.asarray(k["x_scale"])
    if hasattr(lifting, "feature_scale") and "feature_scale" in k.files:
        lifting.feature_scale = np.asarray(k["feature_scale"])
    elif hasattr(lifting, "fit_scaling"):
        # Backward compatibility: if scales were not saved (older Koopman
        # checkpoint), refit on the identification split so we at least match
        # train_koopman.py's behaviour.  Fresh v6 artifacts save both scales.
        id_idx = k["id_idx"] if "id_idx" in k.files else slice(None)
        lifting.fit_scaling(X[id_idx])
    koop = KoopmanPredictor(lifting, n_input=cfg.control.dim)
    koop.A = k["A"]
    koop.B = k["B"]
    if cfg.koopman.online_update.enabled:
        koop.enable_online_update(cfg.koopman.online_update.lambda_forget)

    # --- CBF ---
    # The trained CBF should depend only on safety-relevant coordinates
    # (default [py, vy]), but NeuralCBF.value/grad_h can still receive the
    # full six-state vector and will embed zero gradients for ignored states.
    cbf_blob = torch.load(cbf_path, map_location="cpu", weights_only=True)
    metadata = cbf_blob.get("metadata", {}) if isinstance(cbf_blob, dict) else {}
    cbf_indices = metadata.get(
        "feature_indices", list(getattr(cfg.cbf, "state_indices", range(cfg.state.dim)))
    )
    prior_mode = metadata.get("prior_mode", None)
    prior_params = metadata.get("prior_params", {})
    cbf = NeuralCBF(
        in_dim=len(cbf_indices),
        hidden=list(cfg.cbf.hidden),
        spectral_norm=cfg.cbf.spectral_norm,
        feature_indices=cbf_indices,
        full_dim=metadata.get("full_dim", cfg.state.dim),
        prior_mode=prior_mode,
        prior_params=prior_params,
    )
    try:
        cbf.load_state_dict(cbf_blob["state_dict"])
    except RuntimeError as exc:
        raise RuntimeError(
            "The saved CBF artifact is incompatible with the current reduced-input "
            "CBF architecture. Delete artifacts/data/cbf.pt and rerun "
            "scripts/train_cbf.py with the current config."
        ) from exc

    # --- Conformal ---
    conf = ConformalResidual(
        alpha=cfg.conformal.alpha,
        adaptive=cfg.conformal.adaptive,
        gamma=cfg.conformal.gamma_step,
        window=cfg.conformal.window,
        alpha_min=cfg.conformal.alpha_min,
        alpha_max=cfg.conformal.alpha_max,
    )
    calib = np.load(conformal_path)
    conf.calibrate(calib["scores"])
    # v8: use the current config as the source of truth. Older conformal files
    # may contain stale residual_indices=[py,psi,vx,vy,r], which over-tightened
    # the CBF margin. The conformal scores are regenerated below/instructions
    # with [py,vy], but this guard avoids silent old-artifact reuse.
    residual_indices = list(getattr(cfg.residual, "state_indices", range(cfg.state.dim)))
    trigger_indices = list(getattr(cfg.residual, "trigger_indices", residual_indices))

    # --- Bounds ---
    u_bounds = np.array([cfg.control.bounds.steer, cfg.control.bounds.accel])
    du_bounds = np.array([cfg.control.rate_bounds.steer, cfg.control.rate_bounds.accel])

    # --- MPC (Koopman, tube, or NMPC) ---
    mpc_engine: KoopmanMPC | TubeMPC | NominalNMPC
    if baseline == "nmpc":
        if cfg.env.backend.lower() != "bicycle":
            raise ValueError(
                "NMPC baseline currently requires the bicycle backend "
                "(needs differentiable dynamics)."
            )
        # NMPC uses an internal bicycle dynamics object for rollout. Do not close
        # this env here: f_dyn captures it and SLSQP calls it after construction.
        nmpc_env = make_env(cfg)
        nmpc_env.reset()

        def f_dyn(x, u):
            return nmpc_env._integrate(x, u, cfg.env.dt)

        mpc_engine = NominalNMPC(
            f_dyn=f_dyn,
            N=cfg.mpc.N,
            Q_diag=list(cfg.mpc.Q_diag),
            R_diag=list(cfg.mpc.R_diag),
            S_diag=list(cfg.mpc.S_diag),
            u_bounds=u_bounds,
            du_bounds=du_bounds,
        )
    else:
        mpc_engine = KoopmanMPC(
            lifting=lifting,
            koopman=koop,
            N=cfg.mpc.N,
            Q_diag=list(cfg.mpc.Q_diag),
            R_diag=list(cfg.mpc.R_diag),
            S_diag=list(cfg.mpc.S_diag),
            u_bounds=u_bounds,
            du_bounds=du_bounds,
            terminal_scale=cfg.mpc.P_terminal_scale,
            solver=cfg.mpc.solver,
        )
        if baseline == "tube":
            mpc_engine = TubeMPC(mpc_engine, tube_radius=0.02)

    # --- Safety filter ---
    safety_filter = SafetyFilterQP(
        cbf=cbf,
        u_bounds=u_bounds,
        du_bounds=du_bounds,
        H_diag=np.asarray(cfg.safety_filter.H_diag),
        lambda_slack=cfg.safety_filter.lambda_slack,
    )

    # --- Trigger ---
    trigger = ResidualDrivenTrigger(
        sigma_min=cfg.trigger.sigma_min,
        sigma_max=cfg.trigger.sigma_max,
        lambda_w=cfg.trigger.lambda_w,
        eps_safety=cfg.trigger.eps_safety,
        eps_filter=cfg.trigger.eps_filter,
        N_max=cfg.trigger.N_max,
    )

    # --- Controller config (ablation flags) ---
    flags = BASELINE_FLAGS[baseline]
    ctrl_cfg = ControllerConfig(
        use_event_trigger=flags["use_event_trigger"],
        use_residual_margin=flags["use_residual_margin"],
        use_safety_filter=flags["use_safety_filter"],
        use_mpc=flags["use_mpc"],
        L_x=lifting.Lx,
        eps_rec=cfg.margin.eps_rec,
        eps_sens=cfg.margin.eps_sens,
        rho_lin=cfg.margin.rho_lin,
        fallback_Lh=cfg.cbf.fallback_Lh,
        use_local_lipschitz=getattr(cfg.margin, "use_local_lipschitz", True),
        rho_cap=getattr(cfg.margin, "rho_cap", None),
        lane_half_width=getattr(cfg.lane, "half_width", 1.75),
        gamma_cbf=cfg.cbf.gamma,
        emergency_slack=cfg.safety_filter.emergency_slack,
        emergency_brake=getattr(cfg.safety_filter, "emergency_brake", -3.0),
        recovery_k_py=getattr(cfg.safety_filter, "recovery_k_py", 0.35),
        recovery_k_psi=getattr(cfg.safety_filter, "recovery_k_psi", 0.80),
        recovery_k_vy=getattr(cfg.safety_filter, "recovery_k_vy", 0.12),
        safety_guard_margin=getattr(cfg.safety_filter, "safety_guard_margin", 0.15),
        lane_guard_buffer=getattr(cfg.safety_filter, "lane_guard_buffer", 0.20),
        N_h=cfg.trigger.N_h,
        residual_indices=residual_indices,
        trigger_indices=trigger_indices,
        online_koopman=cfg.koopman.online_update.enabled,
        online_trigger_ratio=cfg.koopman.online_update.residual_trigger_ratio,
        online_consecutive=getattr(cfg.koopman.online_update, "consecutive_steps", 3),
    )

    return ResidualDrivenController(
        lifting=lifting,
        koopman=koop,
        mpc=mpc_engine,
        cbf=cbf,
        conformal=conf,
        trigger=trigger,
        safety_filter=safety_filter,
        cfg=ctrl_cfg,
    )


def run_episode(
    cfg,
    baseline: str,
    scenario: str,
    run_name: str | None = None,
    data_path: str | Path = DEFAULT_DATA,
    koopman_path: str | Path = DEFAULT_KOOPMAN,
    cbf_path: str | Path = DEFAULT_CBF,
    conformal_path: str | Path = DEFAULT_CONFORMAL,
) -> Path:
    env = make_env(cfg)
    env.seed(cfg.seed)

    controller = build_controller(
        cfg,
        baseline=baseline,
        data_path=data_path,
        koopman_path=koopman_path,
        cbf_path=cbf_path,
        conformal_path=conformal_path,
    )
    logger = RunLogger(
        out_dir=resolve_project_path(cfg.logging.out_dir),
        run_name=run_name or f"{baseline}_{scenario}",
    )
    logger.set_meta(
        baseline=baseline,
        scenario=scenario,
        backend=cfg.env.backend,
        data_path=str(data_path),
        koopman_path=str(koopman_path),
        cbf_path=str(cbf_path),
        conformal_path=str(conformal_path),
        dt=float(cfg.env.dt),
        horizon_seconds=float(cfg.env.horizon_seconds),
    )

    obs = env.reset(scenario=scenario)
    controller.reset(obs.x)
    steps = int(cfg.env.horizon_seconds / cfg.env.dt)

    for k in range(steps):
        ref = env.get_reference(cfg.mpc.N)
        log = controller.step(obs.x, ref.x_ref)
        obs = env.step(log.u)
        lane_half_width = float(getattr(getattr(cfg, "lane", {}), "half_width", 1.75))
        py = float(log.x[1]) if len(log.x) > 1 else 0.0
        target_now = ref.x_ref[0] if hasattr(ref, "x_ref") else np.zeros_like(log.x)
        logger.log(
            t=obs.t,
            x=log.x,
            u=log.u,
            u_mpc=np.zeros_like(log.u) if log.u_mpc is None else log.u_mpc,
            x_ref=target_now,
            target_px=float(target_now[0]) if len(target_now) > 0 else 0.0,
            target_py=float(target_now[1]) if len(target_now) > 1 else 0.0,
            target_vx=float(target_now[3]) if len(target_now) > 3 else 0.0,
            tracking_py_error=float(log.x[1] - target_now[1]) if len(log.x) > 1 and len(target_now) > 1 else 0.0,
            tracking_vx_error=float(log.x[3] - target_now[3]) if len(log.x) > 3 and len(target_now) > 3 else 0.0,
            abs_py=abs(py),
            lane_margin=lane_half_width - abs(py),
            lane_violation=int(abs(py) > lane_half_width),
            w_bar=log.w_bar,
            rho=log.rho,
            err_radius=log.err_radius,
            L_h_eff=log.L_h_eff,
            h=log.h_now,
            h_pred=log.h_pred,
            h_minus_rho=log.h_minus_rho,
            mpc_solved=int(log.mpc_solved),
            solve_time=log.mpc_solve_time,
            mpc_cost=log.mpc_cost,
            mpc_status=log.mpc_status,
            safety_status=log.safety_status,
            slack=log.slack,
            filter_correction=log.filter_correction,
            residual=log.residual_norm,
            coverage=log.coverage,
            conformal_covered=log.conformal_covered,
            conformal_miscovered=log.conformal_miscovered,
            w_bar_before_update=log.w_bar_before_update,
            target_coverage=float(1.0 - cfg.conformal.alpha),
            alpha=log.alpha,
            emergency=int(log.emergency),
            safety_recovery=int(getattr(log, "safety_recovery", False)),
            slack_triggered_emergency=int(log.slack_triggered_emergency),
            E_z=log.trigger.E_z if log.trigger else 0.0,
            M_h=log.trigger.M_h if log.trigger else 0.0,
            I_u=log.trigger.I_u if log.trigger else 0.0,
            sigma=log.trigger.sigma_k if log.trigger else 0.0,
            trigger_fire=int(log.trigger.fire) if log.trigger else 0,
            trigger_model=int(log.trigger.model) if log.trigger else 0,
            trigger_safety=int(log.trigger.safety) if log.trigger else 0,
            trigger_input=int(log.trigger.input) if log.trigger else 0,
            trigger_stale=int(log.trigger.stale) if log.trigger else 0,
            trigger_nonstale=int(log.trigger.nonstale) if log.trigger else 0,
            trigger_pure_stale=int(log.trigger.pure_stale) if log.trigger else 0,
            trigger_age=int(log.trigger.age) if log.trigger else 0,
            reason=int(log.trigger.reason) if log.trigger else 0,
        )
        if env.scenario_done(obs):
            break

    env.close()
    path = logger.flush()
    try:
        dat = np.load(path, allow_pickle=True)
        x = dat["x"] if "x" in dat.files else np.zeros((0, 6))
        u = dat["u"] if "u" in dat.files else np.zeros((0, 2))
        lane_violation = dat["lane_violation"] if "lane_violation" in dat.files else np.zeros(0)
        emergency = dat["emergency"] if "emergency" in dat.files else np.zeros(0)
        safety_recovery = dat["safety_recovery"] if "safety_recovery" in dat.files else np.zeros(0)
        summary = stage_header("closed_loop_run")
        trigger_nonstale = dat["trigger_nonstale"] if "trigger_nonstale" in dat.files else np.zeros(0)
        trigger_pure_stale = dat["trigger_pure_stale"] if "trigger_pure_stale" in dat.files else np.zeros(0)
        conformal_covered = dat["conformal_covered"] if "conformal_covered" in dat.files else np.zeros(0)
        summary.update({
            "run_npz": str(path),
            "steps": int(len(x)),
            "baseline": baseline,
            "scenario": scenario,
            "max_abs_y": float(np.max(np.abs(x[:, 1]))) if len(x) else None,
            "mean_speed": float(np.mean(x[:, 3])) if len(x) else None,
            "lane_violations": int(np.sum(lane_violation)) if len(lane_violation) else 0,
            "lane_safe_rate": float(1.0 - np.mean(lane_violation)) if len(lane_violation) else None,
            "emergency_count": int(np.sum(emergency)) if len(emergency) else 0,
            "safety_recovery_count": int(np.sum(safety_recovery)) if len(safety_recovery) else 0,
            "nonstale_event_count": int(np.nansum(trigger_nonstale)) if len(trigger_nonstale) else 0,
            "pure_watchdog_replan_count": int(np.nansum(trigger_pure_stale)) if len(trigger_pure_stale) else 0,
            "actual_conformal_coverage": float(np.nanmean(conformal_covered)) if len(conformal_covered) and np.isfinite(conformal_covered).any() else None,
            "steering_rms": float(np.sqrt(np.mean(u[:, 0] ** 2))) if len(u) else None,
            "accel_rms": float(np.sqrt(np.mean(u[:, 1] ** 2))) if len(u) else None,
        })
        save_json(path.with_suffix(".summary.json"), summary)
    except Exception as exc:
        print(f"WARNING: could not write run summary JSON: {exc}")
    print(f"Saved run to {path}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config YAML")
    ap.add_argument("--baseline", choices=list(BASELINE_FLAGS), default="ours")
    ap.add_argument("--scenario", default="lane_keeping")
    ap.add_argument("--name", default=None)
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="Path to collected dataset NPZ")
    ap.add_argument("--koopman", default=str(DEFAULT_KOOPMAN), help="Path to Koopman model NPZ")
    ap.add_argument("--cbf", default=str(DEFAULT_CBF), help="Path to trained CBF model")
    ap.add_argument("--conformal", default=str(DEFAULT_CONFORMAL), help="Path to conformal calibration NPZ")
    args = ap.parse_args()

    cfg = load_config(args.config)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    run_episode(
        cfg,
        baseline=args.baseline,
        scenario=args.scenario,
        run_name=args.name,
        data_path=args.data,
        koopman_path=args.koopman,
        cbf_path=args.cbf,
        conformal_path=args.conformal,
    )


if __name__ == "__main__":
    main()
