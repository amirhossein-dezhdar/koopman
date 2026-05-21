"""Diagnostics helpers for reproducible CR-Koopman-MPC experiments.

These helpers intentionally avoid importing project-specific model/control
classes so they can be used from every offline script.  They produce JSON
summaries that make it obvious whether a stage ran with enough data, whether
artifacts are stale, and whether closed-loop failures are caused by data,
Koopman residuals, CBF geometry, margins, solver issues, or emergency fallback.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _to_builtin(x: Any) -> Any:
    """Convert numpy/Path/scalar objects to JSON-safe Python values."""
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return _to_builtin(x.item())
        return [_to_builtin(v) for v in x.tolist()]
    if isinstance(x, dict):
        return {str(k): _to_builtin(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_builtin(v) for v in x]
    return x


def save_json(path: str | Path, obj: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = _to_builtin(obj)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def numeric_stats(a: Any, quantiles: Iterable[float] = (0.0, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 1.0)) -> dict[str, Any]:
    arr = np.asarray(a, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    out: dict[str, Any] = {
        "count": int(arr.size),
        "finite_count": int(finite.size),
        "nan_count": int(np.isnan(arr).sum()),
        "inf_count": int(np.isinf(arr).sum()),
    }
    if finite.size == 0:
        return out
    out.update({
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    })
    out["quantiles"] = {f"q{int(round(q*100)):02d}": float(np.quantile(finite, q)) for q in quantiles}
    return out


def matrix_diagnostics(A: np.ndarray, name: str = "matrix") -> dict[str, Any]:
    A = np.asarray(A, dtype=float)
    out: dict[str, Any] = {
        "name": name,
        "shape": list(A.shape),
        "finite": bool(np.isfinite(A).all()),
        "fro_norm": float(np.linalg.norm(A)) if A.size else 0.0,
        "max_abs_entry": float(np.max(np.abs(A))) if A.size else 0.0,
    }
    try:
        if A.ndim == 2:
            svals = np.linalg.svd(A, compute_uv=False)
            out["singular_values"] = numeric_stats(svals)
            if svals.size and np.min(svals) > 0:
                out["condition_number"] = float(np.max(svals) / np.min(svals))
            if A.shape[0] == A.shape[1]:
                eig = np.linalg.eigvals(A)
                out["spectral_radius"] = float(np.max(np.abs(eig)))
                out["eig_abs_gt_1p01"] = int(np.sum(np.abs(eig) > 1.01))
                out["eig_abs_gt_1p10"] = int(np.sum(np.abs(eig) > 1.10))
    except Exception as exc:  # pragma: no cover - diagnostic fallback only
        out["diagnostic_error"] = str(exc)
    return out


def dataset_diagnostics(X: np.ndarray, U: np.ndarray | None = None, X_next: np.ndarray | None = None, *, lane_half_width: float = 1.75) -> dict[str, Any]:
    X = np.asarray(X, dtype=float)
    out: dict[str, Any] = {
        "n_transitions": int(X.shape[0]),
        "state_shape": list(X.shape),
        "lane_half_width": float(lane_half_width),
        "state_stats": {},
    }
    names = ["px", "py", "psi", "vx", "vy", "r"]
    for j in range(min(X.shape[1], len(names))):
        out["state_stats"][names[j]] = numeric_stats(X[:, j])
    if X.shape[1] >= 5:
        py = X[:, 1]
        vy = X[:, 4]
        out["lane_coverage"] = {
            "max_abs_py": float(np.max(np.abs(py))),
            "rms_py": float(np.sqrt(np.mean(py**2))),
            "inside_physical_lane_rate": float(np.mean(np.abs(py) <= lane_half_width)),
            "outside_physical_lane_count": int(np.sum(np.abs(py) > lane_half_width)),
            "positive_py_rate": float(np.mean(py > 0.0)),
            "negative_py_rate": float(np.mean(py < 0.0)),
            "near_boundary_count_abs_py_1p25_to_1p75": int(np.sum((np.abs(py) >= 1.25) & (np.abs(py) <= lane_half_width))),
            "far_ood_count_abs_py_gt_2p5": int(np.sum(np.abs(py) > 2.5)),
            "high_abs_vy_count_gt_2p5": int(np.sum(np.abs(vy) > 2.5)),
        }
    if U is not None:
        U = np.asarray(U, dtype=float)
        out["control_shape"] = list(U.shape)
        out["control_stats"] = {}
        unames = ["steer", "accel"]
        for j in range(min(U.shape[1], len(unames))):
            out["control_stats"][unames[j]] = numeric_stats(U[:, j])
    if X_next is not None:
        Xn = np.asarray(X_next, dtype=float)
        if Xn.shape == X.shape:
            dX = Xn - X
            out["transition_delta_norm"] = numeric_stats(np.linalg.norm(dX, axis=1))
            out["transition_delta_stats"] = {names[j]: numeric_stats(dX[:, j]) for j in range(min(X.shape[1], len(names)))}
    return out


def stage_header(stage: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "created_unix_time": float(time.time()),
    }
