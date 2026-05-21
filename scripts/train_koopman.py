"""Offline pipeline step 2 — Koopman identification with diagnostics.

The mathematical identification remains EDMD ridge least squares.  This version
only adds stronger data/artifact checks and writes a JSON report so a run cannot
silently continue with a tiny or stale dataset.
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
DEFAULT_OUT = ROOT_DIR / "artifacts" / "data" / "koopman.npz"

import numpy as np

from cr_koop_mpc.models import KoopmanPredictor, make_lifting
from cr_koop_mpc.utils import dataset_diagnostics, load_config, matrix_diagnostics, numeric_stats, save_json, stage_header


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
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="Output path for Koopman model NPZ")
    ap.add_argument("--allow_small_dataset", action="store_true", help="Do not fail if dataset is below diagnostics.min_dataset_transitions")
    args = ap.parse_args()

    cfg = load_config(args.config)
    np.random.seed(cfg.seed)
    t0 = time.perf_counter()

    data = np.load(args.data)
    X, U, Xn = data["X"], data["U"], data["X_next"]
    min_trans = int(_cfg_get(getattr(cfg, "diagnostics", None), "min_dataset_transitions", 30000))
    if len(X) < min_trans and not args.allow_small_dataset:
        raise RuntimeError(
            f"Dataset has only {len(X)} transitions; required {min_trans}. "
            "Run scripts/collect_data.py with the v10 defaults or pass --allow_small_dataset for a quick debug run."
        )

    N = X.shape[0]
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(N)
    n_id = int(0.8 * N)
    id_idx, cal_idx = perm[:n_id], perm[n_id:]

    lifting = make_lifting(cfg)
    if hasattr(lifting, "fit_scaling"):
        lifting.fit_scaling(X[id_idx])

    koop = KoopmanPredictor(lifting, n_input=cfg.control.dim)
    koop.fit(X[id_idx], U[id_idx], Xn[id_idx])

    residual_indices = list(getattr(cfg.residual, "state_indices", range(cfg.state.dim)))
    trigger_indices = list(getattr(cfg.residual, "trigger_indices", residual_indices))

    # Residual diagnostics on both ID and calibration splits.  Scores are
    # intentionally physical-state scores, not full lifted residuals.
    def residual_stats(indices: np.ndarray, max_eval: int = 5000) -> dict[str, Any]:
        if len(indices) == 0:
            return {}
        eval_idx = indices[: min(max_eval, len(indices))]
        lifted = np.array([np.linalg.norm(koop.residual(X[j], U[j], Xn[j])) for j in eval_idx])
        safety_scores = np.array([koop.residual_score(X[j], U[j], Xn[j], indices=residual_indices) for j in eval_idx])
        trigger_scores = np.array([koop.residual_score(X[j], U[j], Xn[j], indices=trigger_indices) for j in eval_idx])
        return {
            "evaluated": int(len(eval_idx)),
            "lifted_residual_norm": numeric_stats(lifted),
            "safety_physical_score": numeric_stats(safety_scores),
            "trigger_physical_score": numeric_stats(trigger_scores),
        }

    id_stats = residual_stats(id_idx)
    cal_stats = residual_stats(cal_idx)

    A_diag = matrix_diagnostics(koop.A, "Koopman A")
    B_diag = matrix_diagnostics(koop.B, "Koopman B")

    print(f"Dataset transitions: {N}  identification={len(id_idx)}  calibration={len(cal_idx)}")
    print(f"Mean safety residual on ID split:  {id_stats['safety_physical_score']['mean']:.5f}")
    print(f"Max  safety residual on ID split:  {id_stats['safety_physical_score']['max']:.5f}")
    print(f"Mean trigger residual on ID split: {id_stats['trigger_physical_score']['mean']:.5f}")
    print(f"A spectral radius: {A_diag.get('spectral_radius', float('nan')):.4f}  ({A_diag.get('eig_abs_gt_1p01', 0)} eigenvalues with |λ|>1.01)")
    print(f"A Frobenius norm:  {A_diag['fro_norm']:.2f}")
    print(f"A max |entry|:     {A_diag['max_abs_entry']:.2f}")
    print(f"B max |entry|:     {B_diag['max_abs_entry']:.4f}")

    warnings: list[str] = []
    if A_diag.get("spectral_radius", 0.0) > 1.10:
        warnings.append("A spectral radius > 1.10; predictor may diverge in closed loop.")
    if A_diag.get("max_abs_entry", 0.0) > 100.0:
        warnings.append("A has very large entries; check feature scaling/ridge/data coverage.")
    if cal_stats.get("safety_physical_score", {}).get("mean", 0.0) > 0.2:
        warnings.append("Mean calibration-like safety residual is high; Koopman may be weak in CBF coordinates.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x_scale = getattr(lifting, "x_scale", None)
    feature_scale = getattr(lifting, "feature_scale", None)
    feature_indices = getattr(lifting, "feature_indices", None)
    save_kwargs = dict(
        A=koop.A,
        B=koop.B,
        cal_idx=cal_idx,
        id_idx=id_idx,
        residual_indices=np.asarray(residual_indices, dtype=int),
        trigger_indices=np.asarray(trigger_indices, dtype=int),
        trained_on_n_transitions=np.asarray([N], dtype=int),
    )
    if x_scale is not None:
        save_kwargs["x_scale"] = np.asarray(x_scale)
    if feature_scale is not None:
        save_kwargs["feature_scale"] = np.asarray(feature_scale)
    if feature_indices is not None:
        save_kwargs["feature_indices"] = np.asarray(feature_indices, dtype=int)
    np.savez_compressed(out_path, **save_kwargs)

    elapsed = time.perf_counter() - t0
    lane_half = float(_cfg_get(getattr(cfg, "lane", None), "half_width", 1.75))
    diag = stage_header("train_koopman")
    diag.update({
        "elapsed_seconds": elapsed,
        "input_data": str(args.data),
        "output_npz": str(out_path),
        "n_transitions": int(N),
        "n_identification": int(len(id_idx)),
        "n_calibration": int(len(cal_idx)),
        "residual_indices": residual_indices,
        "trigger_indices": trigger_indices,
        "dataset": dataset_diagnostics(X, U, Xn, lane_half_width=lane_half),
        "id_residuals": id_stats,
        "cal_like_residuals": cal_stats,
        "A": A_diag,
        "B": B_diag,
        "warnings": warnings,
    })
    diag_dir = resolve_project_path(_cfg_get(getattr(cfg, "diagnostics", None), "out_dir", "artifacts/diagnostics"))
    save_json(diag_dir / "train_koopman_summary.json", diag)
    print(f"Saved Koopman matrices to {out_path}")
    print(f"Diagnostics: {diag_dir / 'train_koopman_summary.json'}")
    for w in warnings:
        print(f"WARNING: {w}")


if __name__ == "__main__":
    main()
