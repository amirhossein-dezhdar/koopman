"""Offline pipeline step 4 — conformal calibration with diagnostics.

Formula is unchanged: we calibrate a scalar physical residual score using the
held-out split.  The script now refuses tiny calibration sets unless explicitly
allowed and writes score quantiles to JSON.
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
DEFAULT_KOOPMAN = ROOT_DIR / "artifacts" / "data" / "koopman.npz"
DEFAULT_OUT = ROOT_DIR / "artifacts" / "data" / "conformal.npz"

import numpy as np

from cr_koop_mpc.models import ConformalResidual, KoopmanPredictor, make_lifting
from cr_koop_mpc.utils import load_config, numeric_stats, save_json, stage_header


def _cfg_get(obj: Any, name: str, default: Any) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def resolve_project_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT_DIR / p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config YAML")
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="Path to collected dataset NPZ")
    ap.add_argument("--koopman", default=str(DEFAULT_KOOPMAN), help="Path to Koopman model NPZ")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="Output path for conformal calibration NPZ")
    ap.add_argument("--allow_small_calibration", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    np.random.seed(cfg.seed)
    t0 = time.perf_counter()

    data = np.load(args.data)
    X, U, Xn = data["X"], data["U"], data["X_next"]
    k = np.load(args.koopman)
    cal_idx = np.asarray(k["cal_idx"], dtype=int)

    lifting = make_lifting(cfg)
    if hasattr(lifting, "x_scale") and "x_scale" in k.files:
        lifting.x_scale = np.asarray(k["x_scale"])
    if hasattr(lifting, "feature_scale") and "feature_scale" in k.files:
        lifting.feature_scale = np.asarray(k["feature_scale"])
    elif hasattr(lifting, "fit_scaling"):
        lifting.fit_scaling(X[k["id_idx"]])

    koop = KoopmanPredictor(lifting, n_input=cfg.control.dim)
    koop.A = k["A"]
    koop.B = k["B"]

    residual_indices = np.asarray(getattr(cfg.residual, "state_indices", range(cfg.state.dim)), dtype=int)
    requested = int(cfg.conformal.calibration_size)
    n_cal = min(requested, len(cal_idx))
    min_cal = int(_cfg_get(getattr(cfg, "diagnostics", None), "min_calibration_size", 3000))
    if n_cal < min_cal and not args.allow_small_calibration:
        raise RuntimeError(
            f"Calibration set too small: using {n_cal}, required {min_cal}. "
            "Collect more data or pass --allow_small_calibration only for quick debugging."
        )

    idx = cal_idx[:n_cal]
    scores = np.array([koop.residual_score(X[j], U[j], Xn[j], indices=residual_indices) for j in idx])
    trigger_indices = np.asarray(getattr(cfg.residual, "trigger_indices", residual_indices), dtype=int)
    trigger_scores = np.array([koop.residual_score(X[j], U[j], Xn[j], indices=trigger_indices) for j in idx])

    conf = ConformalResidual(
        alpha=cfg.conformal.alpha,
        adaptive=cfg.conformal.adaptive,
        gamma=cfg.conformal.gamma_step,
        window=cfg.conformal.window,
        alpha_min=cfg.conformal.alpha_min,
        alpha_max=cfg.conformal.alpha_max,
    )
    w_bar = conf.calibrate(scores)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        scores=scores,
        trigger_scores=trigger_scores,
        w_bar0=w_bar,
        alpha=cfg.conformal.alpha,
        residual_indices=residual_indices,
        trigger_indices=trigger_indices,
        calibration_size=np.asarray([n_cal], dtype=int),
    )

    diag = stage_header("calibrate")
    diag.update({
        "elapsed_seconds": float(time.perf_counter() - t0),
        "input_data": str(args.data),
        "koopman_npz": str(args.koopman),
        "output_npz": str(out_path),
        "requested_calibration_size": requested,
        "available_calibration_split": int(len(cal_idx)),
        "used_calibration_size": int(n_cal),
        "alpha": float(cfg.conformal.alpha),
        "w_bar0": float(w_bar),
        "residual_indices": residual_indices.tolist(),
        "trigger_indices": trigger_indices.tolist(),
        "safety_score_stats": numeric_stats(scores),
        "trigger_score_stats": numeric_stats(trigger_scores),
    })
    diag_dir = resolve_project_path(_cfg_get(getattr(cfg, "diagnostics", None), "out_dir", "artifacts/diagnostics"))
    save_json(diag_dir / "calibrate_summary.json", diag)

    print(f"Calibration set size: {n_cal} / available {len(cal_idx)}")
    print(f"Residual state indices: {residual_indices.tolist()}")
    print(f"Mean safety residual score: {scores.mean():.5f}")
    print(f"Max  safety residual score: {scores.max():.5f}")
    print(f"Initial w_bar (1-α quantile, α={cfg.conformal.alpha}): {w_bar:.5f}")
    print(f"Saved calibration to {out_path}")
    print(f"Diagnostics: {diag_dir / 'calibrate_summary.json'}")


if __name__ == "__main__":
    main()
