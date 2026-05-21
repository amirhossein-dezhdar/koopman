"""Offline pipeline step 1 — data collection with hard diagnostics.

This script is deliberately more conservative than the old quick demo.  The
bicycle backend is very light, so 30k-40k transitions may still be collected in
seconds; sufficiency is judged by the saved JSON report, not wall-clock time.

No controller/paper formula is changed here.  We only improve the offline
identification dataset and record enough information to debug whether later
failures are caused by weak data coverage.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
DEFAULT_CONFIG = ROOT_DIR / "configs" / "default.yaml"
DEFAULT_OUT = ROOT_DIR / "artifacts" / "data" / "bicycle_dataset.npz"

import numpy as np
from tqdm import tqdm

from cr_koop_mpc.envs import make_env
from cr_koop_mpc.utils import dataset_diagnostics, load_config, save_json, stage_header


DEFAULT_SCENARIOS = ["lane_keeping", "dlc", "cut_in", "low_friction", "sensor_noise", "ood"]
DEFAULT_MODES = ["center_probe", "sine_sweep", "boundary_recovery", "speed_sweep"]


def _cfg_get(obj: Any, name: str, default: Any) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def resolve_project_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT_DIR / p


def exploration_policy(t: float, x: np.ndarray, cfg, rng: np.random.Generator, mode: str, ep: int) -> np.ndarray:
    """Bounded exploration policy with several modes.

    The modes intentionally cover different parts of the operating envelope:
    center probing, sine steering sweeps, boundary recovery, and speed sweeps.
    This improves EDMD/CBF data coverage without changing the control law used
    in closed-loop experiments.
    """
    steer_b = np.asarray(cfg.control.bounds.steer, dtype=float)
    accel_b = np.asarray(cfg.control.bounds.accel, dtype=float)
    py, psi, vx, vy, r = float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])

    # Shared stabilizing term: enough to keep rollouts useful, weak enough to
    # still explore nonzero py/psi/vy/r.
    feedback = -0.32 * psi - 0.065 * py - 0.018 * vy - 0.050 * r

    if mode == "center_probe":
        probe = 0.22 * np.sin(0.55 * t + 0.37 * ep) + 0.08 * rng.normal()
        accel = 0.45 * (15.0 - vx) + 0.35 * rng.normal()
    elif mode == "sine_sweep":
        freq = 0.35 + 0.10 * (ep % 4)
        probe = 0.36 * np.sin(freq * t + rng.uniform(-0.4, 0.4))
        accel = 0.25 * np.sin(0.23 * t + ep) + 0.30 * rng.normal()
    elif mode == "boundary_recovery":
        # Push toward a boundary, then allow feedback to pull back.  This gives
        # the CBF/Koopman data near |py|≈lane boundary, which the old dataset
        # lacked and which made failures hard to diagnose.
        target = 1.25 * (1.0 if (ep % 2 == 0) else -1.0)
        boundary_push = 0.09 * (target - py)
        probe = boundary_push + 0.18 * np.sin(0.8 * t + ep)
        accel = 0.35 * (13.0 - vx) + 0.20 * rng.normal()
    elif mode == "speed_sweep":
        v_ref = 10.0 + 7.0 * (0.5 + 0.5 * np.sin(0.18 * t + 0.3 * ep))
        probe = 0.18 * np.sin(0.65 * t + ep) + 0.05 * rng.normal()
        accel = 0.70 * (v_ref - vx) + 0.25 * rng.normal()
    else:
        probe = 0.25 * np.sin(0.7 * t + rng.uniform(-0.2, 0.2))
        accel = 0.40 * (15.0 - vx) + 0.25 * rng.normal()

    steer = float(np.clip(probe + feedback, steer_b[0], steer_b[1]))
    accel = float(np.clip(accel, accel_b[0], accel_b[1]))
    return np.array([steer, accel], dtype=float)


def collect(cfg, n_episodes: int | None = None, *, py_max: float | None = None, psi_max: float | None = None) -> tuple[dict, dict]:
    dc = getattr(cfg, "data_collection", None)
    episodes = int(n_episodes if n_episodes is not None else _cfg_get(dc, "episodes", 80))
    py_max = float(py_max if py_max is not None else _cfg_get(dc, "py_max", 3.0))
    psi_max = float(psi_max if psi_max is not None else _cfg_get(dc, "psi_max", 0.9))
    scenarios = list(_cfg_get(dc, "scenario_schedule", DEFAULT_SCENARIOS))
    modes = list(_cfg_get(dc, "policy_modes", DEFAULT_MODES))

    env = make_env(cfg)
    env.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    Xs, Us, Xn = [], [], []
    scenario_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    drift_resets = 0
    normal_episode_finishes = 0
    transitions_by_scenario: defaultdict[str, int] = defaultdict(int)
    transitions_by_mode: defaultdict[str, int] = defaultdict(int)
    episode_lengths: list[int] = []

    steps_per_episode = int(cfg.env.horizon_seconds / cfg.env.dt)
    pbar = tqdm(range(episodes), desc="collect episodes")
    for ep in pbar:
        scenario = scenarios[ep % len(scenarios)]
        mode = modes[(ep // len(scenarios)) % len(modes)]
        scenario_counts[scenario] += 1
        mode_counts[mode] += 1
        obs = env.reset(scenario=scenario)
        x = obs.x.copy()
        ep_len = 0
        for _ in range(steps_per_episode):
            u = exploration_policy(obs.t, x, cfg, rng, mode=mode, ep=ep)
            obs = env.step(u)
            Xs.append(x.copy())
            Us.append(u.copy())
            Xn.append(obs.x.copy())
            ep_len += 1
            transitions_by_scenario[scenario] += 1
            transitions_by_mode[mode] += 1
            x = obs.x.copy()
            if env.scenario_done(obs):
                normal_episode_finishes += 1
                break
            if abs(float(x[1])) > py_max or abs(float(x[2])) > psi_max:
                drift_resets += 1
                obs = env.reset(scenario=scenario)
                x = obs.x.copy()
        episode_lengths.append(ep_len)
        pbar.set_postfix(n=len(Xs), resets=drift_resets)
    env.close()

    arrays = {"X": np.asarray(Xs), "U": np.asarray(Us), "X_next": np.asarray(Xn)}
    meta = {
        "episodes": episodes,
        "steps_per_episode": steps_per_episode,
        "scenario_counts": dict(scenario_counts),
        "mode_counts": dict(mode_counts),
        "transitions_by_scenario": dict(transitions_by_scenario),
        "transitions_by_mode": dict(transitions_by_mode),
        "drift_resets": int(drift_resets),
        "normal_episode_finishes": int(normal_episode_finishes),
        "episode_lengths": {
            "min": int(np.min(episode_lengths)) if episode_lengths else 0,
            "max": int(np.max(episode_lengths)) if episode_lengths else 0,
            "mean": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        },
        "py_reset_threshold": py_max,
        "psi_reset_threshold": psi_max,
    }
    return arrays, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config YAML")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="Output path for collected dataset NPZ")
    ap.add_argument("--episodes", type=int, default=None, help="Override data_collection.episodes")
    ap.add_argument("--min_transitions", type=int, default=None, help="Minimum acceptable transition count")
    ap.add_argument("--py_max", type=float, default=None, help="Override drift reset |py| threshold")
    ap.add_argument("--psi_max", type=float, default=None, help="Override drift reset |psi| threshold")
    args = ap.parse_args()

    cfg = load_config(args.config)
    np.random.seed(cfg.seed)
    t0 = time.perf_counter()

    out, meta = collect(cfg, n_episodes=args.episodes, py_max=args.py_max, psi_max=args.psi_max)
    elapsed = time.perf_counter() - t0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)

    lane_half = float(_cfg_get(getattr(cfg, "lane", None), "half_width", 1.75))
    diag = stage_header("collect_data")
    diag.update(meta)
    diag["elapsed_seconds"] = float(elapsed)
    diag["dataset"] = dataset_diagnostics(out["X"], out["U"], out["X_next"], lane_half_width=lane_half)
    diag["output_npz"] = str(out_path)

    min_trans = int(args.min_transitions if args.min_transitions is not None else _cfg_get(getattr(cfg, "data_collection", None), "min_transitions", 30000))
    diag["min_transitions_required"] = min_trans
    diag["passed_min_transitions"] = bool(len(out["X"]) >= min_trans)

    diag_dir = resolve_project_path(_cfg_get(getattr(cfg, "diagnostics", None), "out_dir", "artifacts/diagnostics"))
    save_json(diag_dir / "collect_data_summary.json", diag)

    X = out["X"]
    print(f"Saved {len(X)} transitions to {out_path}")
    print(f"Elapsed: {elapsed:.2f}s  episodes={meta['episodes']}  drift_resets={meta['drift_resets']}")
    print(f"Diagnostics: {diag_dir / 'collect_data_summary.json'}")
    print("State envelope:")
    for i, n in enumerate(["px", "py", "psi", "vx", "vy", "r"]):
        print(f"  {n}: min={X[:, i].min():+.3f}, mean={X[:, i].mean():+.3f}, std={X[:, i].std():.3f}, max={X[:, i].max():+.3f}")
    if len(X) < min_trans:
        raise RuntimeError(
            f"Dataset too small: {len(X)} transitions < required {min_trans}. "
            "Increase --episodes or lower data_collection.min_transitions only for quick debugging."
        )


if __name__ == "__main__":
    main()
