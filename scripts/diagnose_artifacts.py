"""Diagnose data/model artifacts before running closed-loop experiments.

This script does not change any artifact. It reads the dataset, Koopman model,
conformal scores, and CBF checkpoint, then writes one JSON report explaining
whether the offline pipeline is healthy enough to trust a closed-loop run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
DEFAULT_CONFIG = ROOT_DIR / "configs" / "default.yaml"
DEFAULT_DATA = ROOT_DIR / "artifacts" / "data" / "bicycle_dataset.npz"
DEFAULT_KOOPMAN = ROOT_DIR / "artifacts" / "data" / "koopman.npz"
DEFAULT_CBF = ROOT_DIR / "artifacts" / "data" / "cbf.pt"
DEFAULT_CONFORMAL = ROOT_DIR / "artifacts" / "data" / "conformal.npz"

import numpy as np

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
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--koopman", default=str(DEFAULT_KOOPMAN))
    ap.add_argument("--cbf", default=str(DEFAULT_CBF))
    ap.add_argument("--conformal", default=str(DEFAULT_CONFORMAL))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    lane_half = float(_cfg_get(getattr(cfg, "lane", None), "half_width", 1.75))
    report = stage_header("diagnose_artifacts")
    report["inputs"] = {"data": args.data, "koopman": args.koopman, "cbf": args.cbf, "conformal": args.conformal}
    issues: list[str] = []

    data = np.load(args.data)
    X, U, Xn = data["X"], data["U"], data["X_next"]
    report["dataset"] = dataset_diagnostics(X, U, Xn, lane_half_width=lane_half)
    min_trans = int(_cfg_get(getattr(cfg, "diagnostics", None), "min_dataset_transitions", 30000))
    if len(X) < min_trans:
        issues.append(f"Dataset too small: {len(X)} transitions < {min_trans}.")

    k = np.load(args.koopman)
    report["koopman"] = {"keys": list(k.files)}
    if "A" in k.files:
        report["koopman"]["A"] = matrix_diagnostics(k["A"], "A")
        if report["koopman"]["A"].get("spectral_radius", 0.0) > 1.10:
            issues.append("Koopman A spectral radius > 1.10; model may be unstable.")
    if "B" in k.files:
        report["koopman"]["B"] = matrix_diagnostics(k["B"], "B")
    if "trained_on_n_transitions" in k.files:
        trained_n = int(np.asarray(k["trained_on_n_transitions"]).reshape(-1)[0])
        report["koopman"]["trained_on_n_transitions"] = trained_n
        if trained_n != len(X):
            issues.append(f"Stale Koopman artifact: trained_on_n_transitions={trained_n}, dataset={len(X)}.")
    else:
        issues.append("Koopman artifact has no trained_on_n_transitions metadata; regenerate with v10.")

    conf = np.load(args.conformal)
    report["conformal"] = {"keys": list(conf.files)}
    if "scores" in conf.files:
        report["conformal"]["score_stats"] = numeric_stats(conf["scores"])
        report["conformal"]["calibration_size"] = int(len(conf["scores"]))
        min_cal = int(_cfg_get(getattr(cfg, "diagnostics", None), "min_calibration_size", 3000))
        if len(conf["scores"]) < min_cal:
            issues.append(f"Calibration set too small: {len(conf['scores'])} < {min_cal}.")
    if "residual_indices" in conf.files:
        ri = np.asarray(conf["residual_indices"], dtype=int).tolist()
        cfg_ri = list(getattr(cfg.residual, "state_indices", range(cfg.state.dim)))
        report["conformal"]["residual_indices"] = ri
        if ri != cfg_ri:
            issues.append(f"Conformal residual_indices {ri} do not match config {cfg_ri}.")

    try:
        import torch
        blob = torch.load(args.cbf, map_location="cpu", weights_only=False)
        report["cbf"] = {"keys": list(blob.keys()) if isinstance(blob, dict) else [], "type": str(type(blob))}
        if isinstance(blob, dict):
            report["cbf"]["metadata"] = blob.get("metadata", {})
            report["cbf"]["diagnostics"] = blob.get("diagnostics", {})
            report["cbf"]["grid_validation"] = blob.get("grid_validation", {})
            grid = blob.get("grid_validation", {})
            if int(grid.get("false_safe_lane_count", 1)) > int(_cfg_get(getattr(cfg, "diagnostics", None), "max_allowed_cbf_false_safe_lane", 0)):
                issues.append("CBF grid validation has false-safe lane points; retrain CBF.")
            meta = blob.get("metadata", {})
            if meta.get("feature_indices") != list(getattr(cfg.cbf, "state_indices", [])):
                issues.append(f"CBF feature_indices {meta.get('feature_indices')} do not match config {list(getattr(cfg.cbf, 'state_indices', []))}.")
    except Exception as exc:
        issues.append(f"Could not inspect CBF checkpoint: {exc}")

    report["issues"] = issues
    report["passed"] = len(issues) == 0
    out = Path(args.out) if args.out else resolve_project_path(_cfg_get(getattr(cfg, "diagnostics", None), "out_dir", "artifacts/diagnostics")) / "artifact_diagnosis.json"
    save_json(out, report)
    print(f"Saved artifact diagnosis to {out}")
    if issues:
        print("Issues found:")
        for i in issues:
            print(f"  - {i}")
        raise SystemExit(2)
    print("Artifact diagnosis passed.")


if __name__ == "__main__":
    main()
