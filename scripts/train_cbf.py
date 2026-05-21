"""Offline pipeline step 3 of 4 — neural CBF training.

The CBF is **not removed**.  This version makes it usable for lane keeping by
combining a trainable neural risk tightening with a smooth physical lane prior.
The prior prevents the dangerous OOD failure where a free neural net returns
``h > 0`` tens of metres outside the lane; the neural part is still trained and
can only shrink/tighten the safe set.

Key changes versus the earlier fixed zip:

1. The CBF still sees only safety features ``[py, vy]`` so it cannot use
   absolute ``px`` as a shortcut.
2. Training data is augmented on a symmetric py/vy grid, including far-OOD lane
   departures and boundary bands on both sides of the lane.
3. A strict grid validator is run after training.  The script raises an error if
   the CBF labels any physical lane-departure grid point as safe.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
DEFAULT_CONFIG = ROOT_DIR / "configs" / "default.yaml"
DEFAULT_DATA = ROOT_DIR / "artifacts" / "data" / "bicycle_dataset.npz"
DEFAULT_OUT = ROOT_DIR / "artifacts" / "data" / "cbf.pt"

import numpy as np
import torch

# Small spectral-normalized MLPs are much faster with one CPU thread; on many
# Windows/Anaconda installs PyTorch's thread-pool overhead dominates batches.
torch.set_num_threads(1)

from cr_koop_mpc.safety import NeuralCBF, train_neural_cbf
from cr_koop_mpc.utils import load_config, save_json, stage_header


# Physical / training envelopes.  The controller should correct before the
# physical lane limit; therefore the positive labelled safe set is smaller than
# the physical lane half-width.
PHYSICAL_LANE_HALF_WIDTH = 1.75
SAFE_PY_MAX = 1.25
UNSAFE_PY_MIN = 1.75
SAFE_VY_MAX = 1.00
UNSAFE_VY_MIN = 2.50
GRID_PY_MAX = 6.00
GRID_VY_MAX = 5.00


def _cfg_get(obj: Any, name: str, default: Any) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def resolve_project_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT_DIR / p


def _copy_background_states(
    X: np.ndarray,
    rng: np.random.Generator,
    n: int,
    prefer_safe: bool = True,
) -> np.ndarray:
    """Draw full six-state backgrounds; py/vy are overwritten later."""
    if len(X) == 0:
        return np.zeros((n, 6), dtype=float)
    src = X
    if prefer_safe:
        mask = (np.abs(X[:, 1]) <= SAFE_PY_MAX) & (np.abs(X[:, 4]) <= SAFE_VY_MAX)
        if mask.any():
            src = X[mask]
    idx = rng.choice(len(src), size=n, replace=True)
    return src[idx].copy()


def _synth_safe(X: np.ndarray, rng: np.random.Generator, n: int) -> np.ndarray:
    S = _copy_background_states(X, rng, n, prefer_safe=True)
    # Bias toward the useful interior but still cover both signs.
    S[:, 1] = rng.uniform(-SAFE_PY_MAX, SAFE_PY_MAX, size=n)
    S[:, 4] = rng.uniform(-SAFE_VY_MAX, SAFE_VY_MAX, size=n)
    return S


def _synth_unsafe(X: np.ndarray, rng: np.random.Generator, n: int) -> np.ndarray:
    """Balanced unsafe samples: left/right lane departures + high |vy|."""
    U = _copy_background_states(X, rng, n, prefer_safe=False)
    thirds = np.array_split(np.arange(n), 3)

    # Far and near-boundary lane departures on both sides.
    idx = thirds[0]
    signs = rng.choice([-1.0, 1.0], size=len(idx))
    U[idx, 1] = signs * rng.uniform(UNSAFE_PY_MIN, GRID_PY_MAX, size=len(idx))
    U[idx, 4] = rng.uniform(-1.5, 1.5, size=len(idx))

    # Boundary band: the part the old CBF extrapolated badly.
    idx = thirds[1]
    signs = rng.choice([-1.0, 1.0], size=len(idx))
    U[idx, 1] = signs * rng.uniform(UNSAFE_PY_MIN, UNSAFE_PY_MIN + 0.75, size=len(idx))
    U[idx, 4] = rng.uniform(-2.0, 2.0, size=len(idx))

    # High lateral velocity even if py is still near the lane center.
    idx = thirds[2]
    U[idx, 1] = rng.uniform(-SAFE_PY_MAX, SAFE_PY_MAX, size=len(idx))
    signs = rng.choice([-1.0, 1.0], size=len(idx))
    U[idx, 4] = signs * rng.uniform(UNSAFE_VY_MIN, GRID_VY_MAX, size=len(idx))
    return U


def label_safe_unsafe(
    X: np.ndarray,
    X_next: np.ndarray,
    rng: np.random.Generator,
    n_synth_safe: int = 6000,
    n_synth_unsafe: int = 9000,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray], dict]:
    """Safety labelling with symmetric grid augmentation.

    Real-data labels are kept, but synthetic grid samples dominate the CBF
    geometry. This is intentional: the old dataset did not cover the negative
    side / OOD lane departures sufficiently, so the network became falsely safe.
    """
    py, vy = X[:, 1], X[:, 4]
    safe_mask = (np.abs(py) <= SAFE_PY_MAX) & (np.abs(vy) <= SAFE_VY_MAX)
    unsafe_mask = (np.abs(py) >= UNSAFE_PY_MIN) | (np.abs(vy) >= UNSAFE_VY_MIN)

    safe_real = X[safe_mask]
    unsafe_real = X[unsafe_mask]
    safe = np.concatenate([safe_real, _synth_safe(X, rng, n_synth_safe)], axis=0)
    unsafe = np.concatenate([unsafe_real, _synth_unsafe(X, rng, n_synth_unsafe)], axis=0)

    # Barrier-transition term: only use transitions that start and end in the
    # interior safe set.  Unsafe/random transitions are not guaranteed to be
    # controlled-invariant and made the barrier loss fight the sign loss.
    py_n, vy_n = X_next[:, 1], X_next[:, 4]
    trans_mask = (
        safe_mask
        & (np.abs(py_n) <= SAFE_PY_MAX)
        & (np.abs(vy_n) <= SAFE_VY_MAX)
    )
    stats = {
        "real_safe": int(safe_mask.sum()),
        "real_unsafe": int(unsafe_mask.sum()),
        "synth_safe": int(n_synth_safe),
        "synth_unsafe": int(n_synth_unsafe),
        "transitions": int(trans_mask.sum()),
    }
    return safe, unsafe, (X[trans_mask], X_next[trans_mask]), stats


def diagnose_cbf(cbf: NeuralCBF, safe: np.ndarray, unsafe: np.ndarray) -> dict:
    """Evaluate the trained barrier on labelled samples and px invariance."""
    with torch.no_grad():
        h_s = cbf(torch.tensor(safe, dtype=torch.float32)).cpu().numpy()
        h_u = cbf(torch.tensor(unsafe, dtype=torch.float32)).cpu().numpy()

        probe = np.zeros((4, safe.shape[1]), dtype=np.float32)
        probe[:, 0] = np.array([0.0, 25.0, 100.0, 300.0], dtype=np.float32)
        probe[:, 1] = 0.0   # py
        probe[:, 3] = 15.0  # vx
        h_px = cbf(torch.tensor(probe, dtype=torch.float32)).cpu().numpy()

    return {
        "h_safe_mean": float(h_s.mean()),
        "h_safe_p10": float(np.percentile(h_s, 10)),
        "h_unsafe_mean": float(h_u.mean()),
        "h_unsafe_p90": float(np.percentile(h_u, 90)),
        "separation": float(np.percentile(h_s, 10) - np.percentile(h_u, 90)),
        "px_invariance_span": float(np.max(h_px) - np.min(h_px)),
        "h_px_probe": h_px.tolist(),
    }


def validate_cbf_grid(cbf: NeuralCBF, full_dim: int = 6, n_py: int = 181, n_vy: int = 161) -> dict:
    """Strict physical grid test over py/vy.

    A CBF used for lane safety must never say ``h>0`` for points outside the
    physical lane limit on this grid.  This catches exactly the false-safe bug
    seen in ``analysis_outputs1.zip``.
    """
    py_grid = np.linspace(-GRID_PY_MAX, GRID_PY_MAX, n_py)
    vy_grid = np.linspace(-GRID_VY_MAX, GRID_VY_MAX, n_vy)
    PY, VY = np.meshgrid(py_grid, vy_grid, indexing="ij")
    Xg = np.zeros((PY.size, full_dim), dtype=np.float32)
    Xg[:, 1] = PY.reshape(-1)
    Xg[:, 3] = 15.0
    Xg[:, 4] = VY.reshape(-1)
    with torch.no_grad():
        h = cbf(torch.tensor(Xg, dtype=torch.float32)).cpu().numpy().reshape(PY.shape)

    outside_lane = np.abs(PY) >= PHYSICAL_LANE_HALF_WIDTH
    high_vy = np.abs(VY) >= UNSAFE_VY_MIN
    unsafe = outside_lane | high_vy
    interior = (np.abs(PY) <= SAFE_PY_MAX) & (np.abs(VY) <= SAFE_VY_MAX)

    false_safe_unsafe = unsafe & (h > 0.0)
    false_unsafe_interior = interior & (h < 0.0)
    lane_false_safe = outside_lane & (h > 0.0)

    return {
        "grid_shape": [int(n_py), int(n_vy)],
        "h_min": float(np.min(h)),
        "h_max": float(np.max(h)),
        "h_center": float(h[np.argmin(np.abs(py_grid)), np.argmin(np.abs(vy_grid))]),
        "max_h_outside_lane": float(np.max(h[outside_lane])),
        "max_h_unsafe": float(np.max(h[unsafe])),
        "min_h_interior": float(np.min(h[interior])),
        "false_safe_unsafe_count": int(false_safe_unsafe.sum()),
        "false_safe_unsafe_rate": float(false_safe_unsafe.mean()),
        "false_safe_lane_count": int(lane_false_safe.sum()),
        "false_safe_lane_rate": float(lane_false_safe.mean()),
        "false_unsafe_interior_count": int(false_unsafe_interior.sum()),
        "false_unsafe_interior_rate": float(false_unsafe_interior.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--n_synth_safe", type=int, default=5000)
    ap.add_argument("--n_synth_unsafe", type=int, default=7500)
    ap.add_argument("--max_train_per_class", type=int, default=2500)
    ap.add_argument("--max_transitions", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=None, help="Override cbf.train.epochs")
    ap.add_argument(
        "--no_strict_grid",
        action="store_true",
        help="Do not raise if post-training grid validation finds false-safe lane points",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    t0 = time.perf_counter()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    data = np.load(args.data)
    X, Xn = data["X"], data["X_next"]
    safe, unsafe, transitions, label_stats = label_safe_unsafe(
        X, Xn, rng, n_synth_safe=args.n_synth_safe, n_synth_unsafe=args.n_synth_unsafe
    )
    print("Label stats:", label_stats)
    print(f"raw safe={len(safe)}  raw unsafe={len(unsafe)}  raw transitions={len(transitions[0])}")

    def _subsample(A: np.ndarray, max_n: int) -> np.ndarray:
        if max_n <= 0 or len(A) <= max_n:
            return A
        return A[rng.choice(len(A), size=max_n, replace=False)]

    safe = _subsample(safe, args.max_train_per_class)
    unsafe = _subsample(unsafe, args.max_train_per_class)
    if args.max_transitions > 0 and len(transitions[0]) > args.max_transitions:
        ti = rng.choice(len(transitions[0]), size=args.max_transitions, replace=False)
        transitions = (transitions[0][ti], transitions[1][ti])
    print(f"train safe={len(safe)}  train unsafe={len(unsafe)}  train transitions={len(transitions[0])}")

    cbf_indices = list(getattr(cfg.cbf, "state_indices", range(cfg.state.dim)))
    prior_cfg = getattr(cfg.cbf, "prior", {})
    prior_enabled = bool(_cfg_get(prior_cfg, "enabled", True))
    prior_mode = _cfg_get(prior_cfg, "mode", "lane_saturating") if prior_enabled else None
    prior_params = dict(_cfg_get(prior_cfg, "params", {})) if prior_enabled else {}
    cbf = NeuralCBF(
        in_dim=len(cbf_indices),
        hidden=list(cfg.cbf.hidden),
        spectral_norm=cfg.cbf.spectral_norm,
        feature_indices=cbf_indices,
        full_dim=cfg.state.dim,
        prior_mode=prior_mode,
        prior_params=prior_params,
    )
    print(f"CBF feature indices: {cbf_indices}")
    print(f"CBF prior: mode={prior_mode}, params={prior_params}")

    info = train_neural_cbf(
        cbf,
        safe=safe,
        unsafe=unsafe,
        transitions=transitions,
        gamma=cfg.cbf.gamma,
        epochs=int(args.epochs if args.epochs is not None else cfg.cbf.train.epochs),
        lr=cfg.cbf.train.lr,
        batch_size=cfg.cbf.train.batch_size,
        lambda_barrier=cfg.cbf.train.lambda_barrier,
        lambda_feas=cfg.cbf.train.lambda_feas,
        lambda_spec=cfg.cbf.train.lambda_spec,
    )
    print(f"Final losses: {info['losses']}")
    print(f"Certified/upper-bound Lipschitz constant L_h = {info['Lh']:.3f}")

    diag = diagnose_cbf(cbf, safe, unsafe)
    grid = validate_cbf_grid(cbf, full_dim=cfg.state.dim)
    print("Barrier diagnostics:")
    print(f"  h(safe)   mean={diag['h_safe_mean']:+.3f}  p10={diag['h_safe_p10']:+.3f}")
    print(f"  h(unsafe) mean={diag['h_unsafe_mean']:+.3f}  p90={diag['h_unsafe_p90']:+.3f}")
    print(f"  separation (safe.p10 - unsafe.p90) = {diag['separation']:+.3f}")
    print(f"  px-invariance span at lane center = {diag['px_invariance_span']:.6f}")
    print("Grid validation:")
    for k, v in grid.items():
        print(f"  {k}: {v}")

    if diag["px_invariance_span"] > 1e-5:
        raise RuntimeError("CBF is not px-invariant; check state_indices/feature selection.")
    if not args.no_strict_grid and grid["false_safe_lane_count"] > 0:
        raise RuntimeError(
            "CBF grid validation failed: some physical lane-departure points have h>0. "
            "Do not use this CBF for closed-loop runs."
        )
    if grid["h_center"] <= 1.0:
        raise RuntimeError("CBF center margin is too small; increase prior py_gain or train longer.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": cbf.state_dict(),
            "Lh": info["Lh"],
            "diagnostics": diag,
            "grid_validation": grid,
            "label_stats": label_stats,
            "loss_history": info.get("loss_history", []),
            "metadata": cbf.metadata(),
        },
        out_path,
    )

    report = stage_header("train_cbf")
    report.update({
        "elapsed_seconds": float(time.perf_counter() - t0),
        "input_data": str(args.data),
        "output_pt": str(out_path),
        "cbf_feature_indices": cbf_indices,
        "prior_mode": prior_mode,
        "prior_params": prior_params,
        "label_stats": label_stats,
        "train_counts": {
            "safe": int(len(safe)),
            "unsafe": int(len(unsafe)),
            "transitions": int(len(transitions[0])),
            "n_synth_safe_requested": int(args.n_synth_safe),
            "n_synth_unsafe_requested": int(args.n_synth_unsafe),
            "max_train_per_class": int(args.max_train_per_class),
            "max_transitions": int(args.max_transitions),
            "epochs": int(args.epochs if args.epochs is not None else cfg.cbf.train.epochs),
        },
        "final_losses": info.get("losses", {}),
        "loss_history": info.get("loss_history", []),
        "Lh": float(info["Lh"]),
        "diagnostics": diag,
        "grid_validation": grid,
    })
    diag_dir = resolve_project_path(_cfg_get(getattr(cfg, "diagnostics", None), "out_dir", "artifacts/diagnostics"))
    save_json(diag_dir / "train_cbf_summary.json", report)

    print(f"Saved CBF to {out_path}")
    print(f"Diagnostics: {diag_dir / 'train_cbf_summary.json'}")


if __name__ == "__main__":
    main()
