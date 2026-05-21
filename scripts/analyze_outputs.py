#!/usr/bin/env python3
"""
analyze_outputs.py

Universal output analyzer for the CR-Koopman-MPC project.

What it does:
  1) Scans artifacts/data/*.npz and runs/*.npz
  2) Prints and saves key/shape/stat summaries
  3) Generates plots for datasets, Koopman/conformal files, and closed-loop runs
  4) Creates cross-run comparison metrics and figures
  5) Optionally inspects .pt files such as cbf.pt if torch is installed

Usage examples from the project root:
  python scripts/analyze_outputs.py
  python scripts/analyze_outputs.py --root .
  python scripts/analyze_outputs.py --data-dir artifacts/data --runs-dir runs --out-dir analysis_outputs

By default, paths are resolved relative to the project root, not the current
working directory. This keeps reports in <project>/analysis_outputs and scans
<project>/artifacts/data plus <project>/runs even when the script is launched
from inside scripts/ or from another directory.

The script does not require importing cr_koop_mpc, so it can analyze outputs even if the package import is broken.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


STATE_NAMES_6D = ["px [m]", "py lateral error [m]", "psi heading [rad]", "vx [m/s]", "vy [m/s]", "yaw rate r [rad/s]"]
CONTROL_NAMES_2D = ["steering delta [rad]", "acceleration [m/s^2]"]


@dataclass
class ArraySummary:
    file: str
    key: str
    shape: str
    dtype: str
    finite_count: int
    nan_count: int
    inf_count: int
    min: Optional[float]
    max: Optional[float]
    mean: Optional[float]
    std: Optional[float]


@dataclass
class RunMetrics:
    file: str
    baseline: Optional[str]
    scenario: Optional[str]
    backend: Optional[str]
    steps: int
    duration: Optional[float]
    final_x: Optional[float]
    final_y: Optional[float]
    max_abs_y: Optional[float]
    rms_y: Optional[float]
    max_abs_yaw: Optional[float]
    mean_speed: Optional[float]
    min_h: Optional[float]
    safety_violations_h_lt_0: Optional[int]
    physical_lane_violations_abs_py_gt_1p75: Optional[int]
    lane_safe_rate_abs_py_le_1p75: Optional[float]
    max_lane_excess_m: Optional[float]
    min_lane_margin_m: Optional[float]
    mean_residual: Optional[float]
    max_residual: Optional[float]
    mean_w_bar: Optional[float]
    mean_rho: Optional[float]
    empirical_coverage_mean: Optional[float]  # rolling in-window coverage from ConformalResidual
    actual_conformal_coverage: Optional[float] # mean(score_k <= wbar_{k-1})
    target_coverage: Optional[float]
    mean_alpha: Optional[float]
    solve_time_mean_ms: Optional[float]       # mean over all steps, including held-plan zero solves
    solve_time_p95_ms: Optional[float]
    solve_time_on_solved_mean_ms: Optional[float]
    solve_time_on_solved_p95_ms: Optional[float]
    mpc_solved_rate: Optional[float]
    trigger_model_count: Optional[int]
    trigger_safety_count: Optional[int]
    trigger_input_count: Optional[int]
    trigger_stale_count: Optional[int]
    trigger_nonstale_count: Optional[int]
    trigger_pure_stale_count: Optional[int]
    nonstale_event_rate: Optional[float]
    watchdog_fraction_of_replans: Optional[float]
    mean_inter_event_steps: Optional[float]
    std_inter_event_steps: Optional[float]
    emergency_count: Optional[int]
    safety_recovery_count: Optional[int]
    steering_rms: Optional[float]
    accel_rms: Optional[float]
    control_effort_mean: Optional[float]
    trigger_rate: Optional[float]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(s: str) -> str:
    s = str(s)
    s = s.replace(os.sep, "__")
    s = re.sub(r"[^A-Za-z0-9_.\-]+", "_", s)
    return s.strip("_") or "unnamed"


def is_numeric_array(a: np.ndarray) -> bool:
    return np.issubdtype(a.dtype, np.number) or np.issubdtype(a.dtype, np.bool_)


def to_numeric_array(a: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(a)
        if not is_numeric_array(arr):
            return None
        arr = arr.astype(float, copy=False)
        return arr
    except Exception:
        return None


def flatten_numeric(a: Any) -> Optional[np.ndarray]:
    arr = to_numeric_array(a)
    if arr is None:
        return None
    return arr.reshape(-1)


def finite_stats(a: Any) -> Tuple[int, int, int, Optional[float], Optional[float], Optional[float], Optional[float]]:
    arr = flatten_numeric(a)
    if arr is None or arr.size == 0:
        return 0, 0, 0, None, None, None, None
    finite = np.isfinite(arr)
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    finite_vals = arr[finite]
    if finite_vals.size == 0:
        return 0, nan_count, inf_count, None, None, None, None
    return (
        int(finite_vals.size),
        nan_count,
        inf_count,
        float(np.min(finite_vals)),
        float(np.max(finite_vals)),
        float(np.mean(finite_vals)),
        float(np.std(finite_vals)),
    )


def arr_summary(file: Path, key: str, a: Any) -> ArraySummary:
    arr = np.asarray(a)
    finite_count, nan_count, inf_count, mn, mx, mean, std = finite_stats(arr)
    return ArraySummary(
        file=str(file),
        key=key,
        shape=str(tuple(arr.shape)),
        dtype=str(arr.dtype),
        finite_count=finite_count,
        nan_count=nan_count,
        inf_count=inf_count,
        min=mn,
        max=mx,
        mean=mean,
        std=std,
    )


def save_table(rows: List[dict], csv_path: Path, json_path: Optional[Path] = None) -> None:
    ensure_dir(csv_path.parent)
    if pd is not None:
        pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    else:
        import csv
        if rows:
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        else:
            csv_path.write_text("", encoding="utf-8")
    if json_path is not None:
        json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def line_plot(
    y: np.ndarray,
    out: Path,
    title: str,
    xlabel: str = "step/sample",
    ylabel: str = "value",
    labels: Optional[List[str]] = None,
    x: Optional[np.ndarray] = None,
    max_points: int = 20000,
) -> None:
    arr = to_numeric_array(y)
    if arr is None or arr.size == 0:
        return
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return
    if arr.ndim == 1:
        arr2 = arr.reshape(-1, 1)
    elif arr.ndim == 2:
        arr2 = arr
    else:
        # Collapse all but first axis into channels; useful for small tensors.
        arr2 = arr.reshape(arr.shape[0], -1)
        if arr2.shape[1] > 12:
            arr2 = arr2[:, :12]
    n = arr2.shape[0]
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points).astype(int)
        arr2 = arr2[idx]
        xvals = idx if x is None else np.asarray(x).reshape(-1)[idx]
    else:
        xvals = np.arange(n) if x is None else np.asarray(x).reshape(-1)[:n]

    plt.figure(figsize=(10, 5))
    channels = min(arr2.shape[1], 12)
    for j in range(channels):
        lab = labels[j] if labels and j < len(labels) else f"dim {j}"
        plt.plot(xvals, arr2[:, j], label=lab, linewidth=1.2)
    plt.title(title, fontsize=12)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if channels > 1:
        plt.legend(loc="best", fontsize=9, frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def scatter_plot(x: np.ndarray, y: np.ndarray, out: Path, title: str, xlabel: str, ylabel: str, s: float = 4.0) -> None:
    xx = flatten_numeric(x)
    yy = flatten_numeric(y)
    if xx is None or yy is None or xx.size == 0 or yy.size == 0:
        return
    n = min(xx.size, yy.size)
    xx, yy = xx[:n], yy[:n]
    finite = np.isfinite(xx) & np.isfinite(yy)
    if finite.sum() == 0:
        return
    plt.figure(figsize=(7, 6))
    plt.scatter(xx[finite], yy[finite], s=s, alpha=0.6)
    plt.title(title, fontsize=12)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def hist_plot(a: np.ndarray, out: Path, title: str, xlabel: str = "value", bins: int = 60) -> None:
    arr = flatten_numeric(a)
    if arr is None or arr.size == 0:
        return
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(arr, bins=bins)
    plt.title(title, fontsize=12)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def bar_plot(names: Sequence[str], values: Sequence[float], out: Path, title: str, ylabel: str) -> None:
    vals = np.asarray(values, dtype=float)
    if len(names) == 0 or vals.size == 0:
        return
    plt.figure(figsize=(max(8, 0.5 * len(names)), 5))
    plt.bar(range(len(names)), vals)
    plt.xticks(range(len(names)), names, rotation=45, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def heatmap(a: np.ndarray, out: Path, title: str, xlabel: str = "columns", ylabel: str = "rows") -> None:
    arr = to_numeric_array(a)
    if arr is None or arr.ndim != 2 or arr.size == 0:
        return
    plt.figure(figsize=(8, 6))
    plt.imshow(arr, aspect="auto")
    plt.colorbar()
    plt.title(title, fontsize=12)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def correlation_heatmap(a: np.ndarray, out: Path, title: str, labels: Optional[List[str]] = None) -> None:
    arr = to_numeric_array(a)
    if arr is None:
        return
    if arr.ndim == 1:
        return
    arr = arr.reshape(arr.shape[0], -1)
    if arr.shape[0] < 3 or arr.shape[1] < 2:
        return
    if arr.shape[1] > 30:
        arr = arr[:, :30]
    with np.errstate(all="ignore"):
        corr = np.corrcoef(arr, rowvar=False)
    if not np.isfinite(corr).any():
        return
    plt.figure(figsize=(7, 6))
    plt.imshow(corr, vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(label="corr")
    plt.title(title)
    if labels:
        labs = labels[: corr.shape[0]]
        plt.xticks(range(len(labs)), labs, rotation=45, ha="right", fontsize=8)
        plt.yticks(range(len(labs)), labs, fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def get_key(data: Dict[str, Any], *names: str) -> Optional[np.ndarray]:
    for n in names:
        if n in data:
            return np.asarray(data[n])
    lower = {k.lower(): k for k in data.keys()}
    for n in names:
        if n.lower() in lower:
            return np.asarray(data[lower[n.lower()]])
    return None


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


def load_meta_for_npz(npz_path: Path) -> Dict[str, Any]:
    json_path = npz_path.with_suffix(".json")
    if not json_path.exists():
        return {}
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def analyze_dataset_npz(path: Path, data: Dict[str, np.ndarray], outdir: Path) -> List[str]:
    notes: List[str] = []
    stem = safe_name(path.stem)
    od = ensure_dir(outdir / "datasets" / stem)
    X = get_key(data, "X", "states", "state", "obs", "observations")
    U = get_key(data, "U", "actions", "action", "controls", "control")
    Xn = get_key(data, "X_next", "next_states", "next_obs", "Xn")

    if X is not None:
        labels = STATE_NAMES_6D if X.ndim == 2 and X.shape[1] == 6 else None
        line_plot(X, od / "states_timeseries.png", f"{path.name}: states", labels=labels)
        correlation_heatmap(X, od / "states_correlation.png", f"{path.name}: state correlation", labels=labels)
        if X.ndim >= 2:
            for j in range(min(X.reshape(X.shape[0], -1).shape[1], 12)):
                flat = X.reshape(X.shape[0], -1)
                lab = labels[j] if labels and j < len(labels) else f"state_{j}"
                hist_plot(flat[:, j], od / f"hist_{safe_name(lab)}.png", f"{path.name}: {lab} histogram", lab)
            if X.reshape(X.shape[0], -1).shape[1] >= 2:
                flat = X.reshape(X.shape[0], -1)
                scatter_plot(flat[:, 0], flat[:, 1], od / "xy_or_state0_state1_scatter.png", f"{path.name}: x-y / state0-state1", "state 0", "state 1")
            if X.reshape(X.shape[0], -1).shape[1] >= 5:
                flat = X.reshape(X.shape[0], -1)
                scatter_plot(flat[:, 1], flat[:, 4], od / "py_vy_phase.png", f"{path.name}: py-vy phase", "py / lateral position", "vy / lateral velocity")
    if U is not None:
        labels = CONTROL_NAMES_2D if U.ndim == 2 and U.shape[1] == 2 else None
        line_plot(U, od / "controls_timeseries.png", f"{path.name}: controls/actions", labels=labels)
        correlation_heatmap(U, od / "controls_correlation.png", f"{path.name}: control correlation", labels=labels)
        if U.ndim >= 2:
            flat = U.reshape(U.shape[0], -1)
            for j in range(min(flat.shape[1], 8)):
                lab = labels[j] if labels and j < len(labels) else f"u_{j}"
                hist_plot(flat[:, j], od / f"hist_{safe_name(lab)}.png", f"{path.name}: {lab} histogram", lab)
    if X is not None and Xn is not None:
        try:
            X2 = to_numeric_array(X).reshape(X.shape[0], -1)
            Xn2 = to_numeric_array(Xn).reshape(Xn.shape[0], -1)
            m = min(X2.shape[0], Xn2.shape[0])
            c = min(X2.shape[1], Xn2.shape[1])
            dX = Xn2[:m, :c] - X2[:m, :c]
            labels = [f"Δ{n}" for n in STATE_NAMES_6D] if c == 6 else None
            line_plot(dX, od / "state_transitions_delta.png", f"{path.name}: X_next - X", labels=labels)
            hist_plot(np.linalg.norm(dX, axis=1), od / "transition_norm_hist.png", f"{path.name}: ||X_next-X|| histogram", "transition norm")
        except Exception as e:
            notes.append(f"Could not plot transition deltas for {path}: {e}")
    return notes


def analyze_koopman_npz(path: Path, data: Dict[str, np.ndarray], outdir: Path, dataset_paths: List[Path]) -> List[str]:
    notes: List[str] = []
    stem = safe_name(path.stem)
    od = ensure_dir(outdir / "koopman" / stem)
    A = get_key(data, "A")
    B = get_key(data, "B")
    if A is not None and A.ndim == 2:
        heatmap(A, od / "A_matrix_heatmap.png", f"{path.name}: Koopman A matrix")
        try:
            eig = np.linalg.eigvals(A)
            plt.figure(figsize=(6, 6))
            th = np.linspace(0, 2 * np.pi, 400)
            plt.plot(np.cos(th), np.sin(th), linestyle="--", linewidth=1)
            plt.scatter(eig.real, eig.imag, s=18)
            plt.axhline(0, linewidth=0.8)
            plt.axvline(0, linewidth=0.8)
            plt.title(f"{path.name}: eigenvalues of A")
            plt.xlabel("real")
            plt.ylabel("imag")
            plt.grid(True, alpha=0.3)
            plt.axis("equal")
            plt.tight_layout()
            plt.savefig(od / "A_eigenvalues.png", dpi=160)
            plt.close()
            hist_plot(np.abs(eig), od / "A_eigenvalue_magnitudes_hist.png", f"{path.name}: |eig(A)|", "|eigenvalue|")
        except Exception as e:
            notes.append(f"Could not compute eigenvalues for {path}: {e}")
        try:
            svals = np.linalg.svd(A, compute_uv=False)
            line_plot(svals, od / "A_singular_values.png", f"{path.name}: singular values of A", xlabel="index", ylabel="singular value")
        except Exception:
            pass
    if B is not None and B.ndim == 2:
        heatmap(B, od / "B_matrix_heatmap.png", f"{path.name}: Koopman B matrix")
        try:
            svals = np.linalg.svd(B, compute_uv=False)
            line_plot(svals, od / "B_singular_values.png", f"{path.name}: singular values of B", xlabel="index", ylabel="singular value")
        except Exception:
            pass
    # Full Koopman prediction residual requires lifting metadata, which is not saved here.
    # Still, we report matrix properties because they are independent and useful.
    return notes


def analyze_conformal_npz(path: Path, data: Dict[str, np.ndarray], outdir: Path) -> List[str]:
    notes: List[str] = []
    stem = safe_name(path.stem)
    od = ensure_dir(outdir / "conformal" / stem)
    scores = get_key(data, "scores", "residual_scores")
    if scores is not None:
        hist_plot(scores, od / "scores_hist.png", f"{path.name}: conformal residual scores", "score")
        line_plot(np.sort(flatten_numeric(scores)), od / "scores_sorted.png", f"{path.name}: sorted conformal scores", xlabel="rank", ylabel="score")
        alpha = float(np.asarray(data.get("alpha", 0.1)).reshape(-1)[0]) if "alpha" in data else 0.1
        q = float(np.quantile(flatten_numeric(scores), 1 - alpha))
        lines = [f"alpha={alpha}", f"empirical_quantile_1_minus_alpha={q:.6g}"]
        if "w_bar0" in data:
            lines.append(f"saved_w_bar0={float(np.asarray(data['w_bar0']).reshape(-1)[0]):.6g}")
        (od / "conformal_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    return notes


def as_2d_time_series(a: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if a is None:
        return None
    arr = to_numeric_array(a)
    if arr is None:
        return None
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return None
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr.reshape(arr.shape[0], -1)


def scalar_series(data: Dict[str, np.ndarray], *keys: str) -> Optional[np.ndarray]:
    a = get_key(data, *keys)
    arr2 = as_2d_time_series(a)
    if arr2 is None:
        return None
    return arr2[:, 0]


def compute_run_metrics(path: Path, data: Dict[str, np.ndarray], meta: Dict[str, Any]) -> RunMetrics:
    t = scalar_series(data, "t", "time")
    x = as_2d_time_series(get_key(data, "x", "X", "state", "states"))
    u = as_2d_time_series(get_key(data, "u", "U", "action", "actions", "control"))
    h = scalar_series(data, "h", "barrier")
    residual = scalar_series(data, "residual", "residual_norm")
    w_bar = scalar_series(data, "w_bar", "wbar")
    rho = scalar_series(data, "rho")
    coverage = scalar_series(data, "coverage")
    conformal_covered = scalar_series(data, "conformal_covered")
    target_coverage = scalar_series(data, "target_coverage")
    alpha = scalar_series(data, "alpha")
    solve_time = scalar_series(data, "solve_time", "mpc_solve_time")
    mpc_solved = scalar_series(data, "mpc_solved")
    emergency = scalar_series(data, "emergency")
    safety_recovery = scalar_series(data, "safety_recovery")
    trig_model = scalar_series(data, "trigger_model")
    trig_safety = scalar_series(data, "trigger_safety")
    trig_input = scalar_series(data, "trigger_input")
    trig_stale = scalar_series(data, "trigger_stale")
    trig_nonstale = scalar_series(data, "trigger_nonstale")
    trig_pure_stale = scalar_series(data, "trigger_pure_stale")
    trig_age = scalar_series(data, "trigger_age")

    steps = int(max([len(v) for v in [t, x, u, h, residual, w_bar, rho, coverage, conformal_covered, solve_time, mpc_solved, emergency] if v is not None] or [0]))
    duration = None
    if t is not None and len(t) > 0:
        duration = float(np.nanmax(t) - np.nanmin(t))

    def finite_mean(a: Optional[np.ndarray]) -> Optional[float]:
        if a is None:
            return None
        aa = np.asarray(a, dtype=float).reshape(-1)
        aa = aa[np.isfinite(aa)]
        return float(aa.mean()) if aa.size else None

    def finite_max_abs(a: Optional[np.ndarray]) -> Optional[float]:
        if a is None:
            return None
        aa = np.asarray(a, dtype=float).reshape(-1)
        aa = aa[np.isfinite(aa)]
        return float(np.max(np.abs(aa))) if aa.size else None

    def finite_min(a: Optional[np.ndarray]) -> Optional[float]:
        if a is None:
            return None
        aa = np.asarray(a, dtype=float).reshape(-1)
        aa = aa[np.isfinite(aa)]
        return float(np.min(aa)) if aa.size else None

    final_x = final_y = max_abs_y = rms_y = max_abs_yaw = mean_speed = None
    physical_lane_violations = lane_safe_rate = max_lane_excess = min_lane_margin = None
    lane_half_width = 1.75
    if x is not None and x.shape[0] > 0:
        if x.shape[1] > 0:
            final_x = float(x[-1, 0])
        if x.shape[1] > 1:
            final_y = float(x[-1, 1])
            max_abs_y = finite_max_abs(x[:, 1])
            yy = x[:, 1][np.isfinite(x[:, 1])]
            rms_y = float(np.sqrt(np.mean(yy ** 2))) if yy.size else None
            if yy.size:
                abs_y = np.abs(yy)
                margins = lane_half_width - abs_y
                physical_lane_violations = int(np.sum(abs_y > lane_half_width))
                lane_safe_rate = float(np.mean(abs_y <= lane_half_width))
                max_lane_excess = float(max(0.0, np.max(abs_y - lane_half_width)))
                min_lane_margin = float(np.min(margins))
        if x.shape[1] > 2:
            max_abs_yaw = finite_max_abs(x[:, 2])
        if x.shape[1] > 3:
            mean_speed = finite_mean(x[:, 3])

    min_h = finite_min(h)
    safety_violations = None
    if h is not None:
        hh = np.asarray(h, dtype=float)
        safety_violations = int(np.sum(np.isfinite(hh) & (hh < 0)))

    solve_mean = solve_p95 = solve_on_mean = solve_on_p95 = None
    if solve_time is not None:
        st_all = np.asarray(solve_time, dtype=float).reshape(-1)
        st = st_all[np.isfinite(st_all)]
        if st.size:
            # Most code logs seconds. Convert to ms for report.
            solve_mean = float(np.mean(st) * 1000.0)
            solve_p95 = float(np.percentile(st, 95) * 1000.0)
        if mpc_solved is not None:
            flags = np.asarray(mpc_solved, dtype=float).reshape(-1)
            m = min(len(st_all), len(flags))
            st_on = st_all[:m][np.isfinite(st_all[:m]) & (flags[:m] > 0.5)]
            if st_on.size:
                solve_on_mean = float(np.mean(st_on) * 1000.0)
                solve_on_p95 = float(np.percentile(st_on, 95) * 1000.0)

    steering_rms = accel_rms = effort = None
    if u is not None and u.shape[0] > 0:
        uu = np.asarray(u, dtype=float)
        finite_rows = np.isfinite(uu).all(axis=1)
        if finite_rows.any():
            effort = float(np.mean(np.linalg.norm(uu[finite_rows], axis=1)))
        if u.shape[1] > 0:
            v = uu[:, 0]
            v = v[np.isfinite(v)]
            steering_rms = float(np.sqrt(np.mean(v ** 2))) if v.size else None
        if u.shape[1] > 1:
            v = uu[:, 1]
            v = v[np.isfinite(v)]
            accel_rms = float(np.sqrt(np.mean(v ** 2))) if v.size else None

    # trigger_rate is defined here as fraction of steps where MPC solved.
    trigger_rate = finite_mean(mpc_solved)

    def count_flag(a: Optional[np.ndarray]) -> Optional[int]:
        if a is None:
            return None
        aa = np.asarray(a, dtype=float).reshape(-1)
        aa = aa[np.isfinite(aa)]
        return int(np.sum(aa > 0.5)) if aa.size else 0

    n_model = count_flag(trig_model)
    n_safety = count_flag(trig_safety)
    n_input = count_flag(trig_input)
    n_stale = count_flag(trig_stale)
    n_nonstale = count_flag(trig_nonstale)
    n_pure_stale = count_flag(trig_pure_stale)
    nonstale_rate = None if steps == 0 or n_nonstale is None else float(n_nonstale / steps)
    watchdog_frac = None
    if mpc_solved is not None and n_pure_stale is not None:
        n_solved = count_flag(mpc_solved) or 0
        watchdog_frac = float(n_pure_stale / n_solved) if n_solved > 0 else None

    mean_ie = std_ie = None
    if mpc_solved is not None:
        flags = np.asarray(mpc_solved, dtype=float).reshape(-1)
        idx = np.flatnonzero(np.isfinite(flags) & (flags > 0.5))
        if len(idx) >= 2:
            gaps = np.diff(idx)
            mean_ie = float(np.mean(gaps))
            std_ie = float(np.std(gaps))

    actual_cov = finite_mean(conformal_covered)
    targ_cov = finite_mean(target_coverage)
    mean_alpha = finite_mean(alpha)

    return RunMetrics(
        file=str(path),
        baseline=meta.get("baseline"),
        scenario=meta.get("scenario"),
        backend=meta.get("backend"),
        steps=steps,
        duration=duration,
        final_x=final_x,
        final_y=final_y,
        max_abs_y=max_abs_y,
        rms_y=rms_y,
        max_abs_yaw=max_abs_yaw,
        mean_speed=mean_speed,
        min_h=min_h,
        safety_violations_h_lt_0=safety_violations,
        physical_lane_violations_abs_py_gt_1p75=physical_lane_violations,
        lane_safe_rate_abs_py_le_1p75=lane_safe_rate,
        max_lane_excess_m=max_lane_excess,
        min_lane_margin_m=min_lane_margin,
        mean_residual=finite_mean(residual),
        max_residual=finite_max_abs(residual),
        mean_w_bar=finite_mean(w_bar),
        mean_rho=finite_mean(rho),
        empirical_coverage_mean=finite_mean(coverage),
        actual_conformal_coverage=actual_cov,
        target_coverage=targ_cov,
        mean_alpha=mean_alpha,
        solve_time_mean_ms=solve_mean,
        solve_time_p95_ms=solve_p95,
        solve_time_on_solved_mean_ms=solve_on_mean,
        solve_time_on_solved_p95_ms=solve_on_p95,
        mpc_solved_rate=finite_mean(mpc_solved),
        trigger_model_count=n_model,
        trigger_safety_count=n_safety,
        trigger_input_count=n_input,
        trigger_stale_count=n_stale,
        trigger_nonstale_count=n_nonstale,
        trigger_pure_stale_count=n_pure_stale,
        nonstale_event_rate=nonstale_rate,
        watchdog_fraction_of_replans=watchdog_frac,
        mean_inter_event_steps=mean_ie,
        std_inter_event_steps=std_ie,
        emergency_count=int(np.nansum(emergency)) if emergency is not None else None,
        safety_recovery_count=int(np.nansum(safety_recovery)) if safety_recovery is not None else None,
        steering_rms=steering_rms,
        accel_rms=accel_rms,
        control_effort_mean=effort,
        trigger_rate=trigger_rate,
    )


def analyze_run_npz(path: Path, data: Dict[str, np.ndarray], outdir: Path) -> RunMetrics:
    stem = safe_name(path.stem)
    od = ensure_dir(outdir / "runs" / stem)
    meta = load_meta_for_npz(path)
    t = scalar_series(data, "t", "time")
    x = as_2d_time_series(get_key(data, "x", "X", "state", "states"))
    u = as_2d_time_series(get_key(data, "u", "U", "action", "actions", "control"))

    scenario_name = meta.get("scenario", "unknown") if meta else "unknown"
    baseline_name = meta.get("baseline", path.stem) if meta else path.stem
    title_prefix = f"{baseline_name} / {scenario_name}"

    if x is not None:
        labels = STATE_NAMES_6D if x.shape[1] == 6 else None
        line_plot(x, od / "states_timeseries.png", f"{title_prefix}: states", labels=labels, x=t, xlabel="time" if t is not None else "step")
        if x.shape[1] >= 2:
            xref = as_2d_time_series(get_key(data, "x_ref", "ref", "reference"))
            lane_half_width = 1.75
            plt.figure(figsize=(8, 6))
            plt.plot(x[:, 0], x[:, 1], label="Robot trajectory", linewidth=1.8)
            if xref is not None and xref.shape[0] >= 1 and xref.shape[1] >= 2:
                m = min(len(x), len(xref))
                plt.plot(xref[:m, 0], xref[:m, 1], linestyle="--", label="Target / reference centerline", linewidth=1.4)
            else:
                plt.plot(x[:, 0], np.zeros_like(x[:, 0]), linestyle="--", label="Target / lane center", linewidth=1.4)
            plt.plot(x[:, 0], np.full_like(x[:, 0], lane_half_width), linestyle=":", label="Upper safety boundary")
            plt.plot(x[:, 0], np.full_like(x[:, 0], -lane_half_width), linestyle=":", label="Lower safety boundary")
            plt.title(f"{title_prefix}: trajectory tracking and lane boundaries", fontsize=12)
            plt.xlabel("longitudinal position px [m]")
            plt.ylabel("lateral position py [m]")
            plt.grid(True, alpha=0.3)
            plt.legend(loc="best", fontsize=9, frameon=True)
            plt.tight_layout()
            plt.savefig(od / "trajectory_xy.png", dpi=160)
            plt.close()
            lane_mat = np.column_stack([
                np.abs(x[:, 1]),
                lane_half_width - np.abs(x[:, 1]),
                (np.abs(x[:, 1]) > lane_half_width).astype(float),
            ])
            line_plot(
                lane_mat,
                od / "physical_lane_safety.png",
                f"{title_prefix}: physical lane safety",
                labels=["abs_py", "lane_margin", "lane_violation"],
                x=t,
                xlabel="time" if t is not None else "step",
            )
    if u is not None:
        labels = CONTROL_NAMES_2D if u.shape[1] == 2 else None
        line_plot(u, od / "controls_timeseries.png", f"{title_prefix}: controls", labels=labels, x=t, xlabel="time" if t is not None else "step")

    # Known scalar logs from run_closed_loop.py.
    # Paper-facing plot set: keep only interpretable diagnostics.  Earlier
    # versions generated dozens of histograms/auxiliary plots, which made the
    # report noisy and easy to misread.
    scalar_groups = [
        (["residual", "w_bar", "w_bar_before_update"], "residual_conformal_bound.png", "Conformal residual score and adaptive bound"),
        (["h", "rho", "h_minus_rho", "M_h", "lane_margin"], "safety_margin_diagnostics.png", "CBF safety margin: h, rho, h-rho, M_h, and physical lane margin"),
        (["E_z", "sigma", "M_h", "I_u"], "event_trigger_components.png", "Event-trigger components: model, safety, and input-change tests"),
        (["mpc_solved", "trigger_model", "trigger_safety", "trigger_input", "trigger_stale", "safety_recovery", "emergency"], "events_and_recovery.png", "Replans, event causes, and safety recovery"),
        (["conformal_covered", "target_coverage", "alpha"], "conformal_coverage.png", "Actual conformal hits, target coverage, and adaptive alpha"),
        (["solve_time"], "mpc_solve_time.png", "MPC solve time per step"),
    ]
    for keys, fname, ttl in scalar_groups:
        series = []
        labels = []
        for k in keys:
            s = scalar_series(data, k)
            if s is not None:
                series.append(s.reshape(-1, 1))
                labels.append(k)
        if series:
            m = min(len(s) for s in series)
            mat = np.concatenate([s[:m] for s in series], axis=1)
            line_plot(mat, od / fname, f"{title_prefix}: {ttl}", labels=labels, x=t[:m] if t is not None and len(t) >= m else None, xlabel="time" if t is not None else "step")

    # reason counts if available.
    reason = scalar_series(data, "reason")
    if reason is not None:
        rr = reason[np.isfinite(reason)].astype(int)
        if rr.size:
            vals, counts = np.unique(rr, return_counts=True)
            bar_plot([str(v) for v in vals], counts, od / "trigger_reason_counts.png", f"{title_prefix}: trigger reason counts", "count")

    # Explicit event-type decomposition.  This prevents pure watchdog replans
    # from being mistaken for residual-driven events in the paper.
    event_keys = ["trigger_model", "trigger_safety", "trigger_input", "trigger_stale", "trigger_nonstale", "trigger_pure_stale"]
    event_counts = []
    event_names = []
    for k in event_keys:
        sflag = scalar_series(data, k)
        if sflag is not None:
            event_names.append(k.replace("trigger_", ""))
            event_counts.append(int(np.nansum(np.asarray(sflag, dtype=float) > 0.5)))
    if event_counts:
        bar_plot(event_names, event_counts, od / "trigger_decomposition_counts.png", f"{title_prefix}: trigger decomposition", "count")

    solved = scalar_series(data, "mpc_solved")
    if solved is not None:
        sf = np.asarray(solved, dtype=float).reshape(-1)
        idx = np.flatnonzero(np.isfinite(sf) & (sf > 0.5))
        if len(idx) >= 2:
            gaps = np.diff(idx)
            line_plot(gaps, od / "inter_event_intervals.png", f"{title_prefix}: inter-event intervals", xlabel="event index", ylabel="steps")
            hist_plot(gaps, od / "inter_event_intervals_hist.png", f"{title_prefix}: inter-event interval histogram", "steps")

    metrics = compute_run_metrics(path, data, meta)
    (od / "run_metrics.json").write_text(json.dumps(asdict(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def inspect_pt_file(path: Path, outdir: Path) -> List[str]:
    notes: List[str] = []
    od = ensure_dir(outdir / "pt_files")
    try:
        import torch
        blob = torch.load(path, map_location="cpu", weights_only=False)
        summary: Dict[str, Any] = {"file": str(path), "type": str(type(blob))}
        if isinstance(blob, dict):
            summary["keys"] = list(blob.keys())
            for k, v in blob.items():
                if isinstance(v, dict):
                    summary[k] = {kk: str(type(vv)) for kk, vv in v.items()}
                elif hasattr(v, "shape"):
                    summary[k] = {"shape": tuple(v.shape), "dtype": str(getattr(v, "dtype", ""))}
                else:
                    try:
                        json.dumps(v)
                        summary[k] = v
                    except Exception:
                        summary[k] = str(type(v))
        (od / f"{safe_name(path.stem)}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        # If saved diagnostics exist in cbf.pt, write readable text.
        if isinstance(blob, dict) and "diagnostics" in blob:
            (od / f"{safe_name(path.stem)}_diagnostics.txt").write_text(json.dumps(blob["diagnostics"], ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        if isinstance(blob, dict) and "grid_validation" in blob:
            (od / f"{safe_name(path.stem)}_grid_validation.txt").write_text(json.dumps(blob["grid_validation"], ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    except Exception as e:
        notes.append(f"Could not inspect PT file {path}: {e}")
    return notes


def compare_runs(metrics: List[RunMetrics], run_paths: List[Path], outdir: Path) -> None:
    if not metrics:
        return
    od = ensure_dir(outdir / "comparisons")
    rows = [asdict(m) for m in metrics]
    save_table(rows, od / "run_metrics.csv", od / "run_metrics.json")

    def label(m: RunMetrics) -> str:
        base = m.baseline or Path(m.file).stem
        scen = m.scenario
        return f"{base}-{scen}" if scen else base

    metric_names = [
        "max_abs_y",
        "physical_lane_violations_abs_py_gt_1p75",
        "lane_safe_rate_abs_py_le_1p75",
        "actual_conformal_coverage",
        "mpc_solved_rate",
        "trigger_nonstale_count",
        "trigger_pure_stale_count",
        "emergency_count",
        "safety_recovery_count",
        "solve_time_on_solved_mean_ms",
    ]
    for mn in metric_names:
        vals = []
        names = []
        for m in metrics:
            v = getattr(m, mn)
            if v is not None and np.isfinite(float(v)):
                vals.append(float(v))
                names.append(label(m))
        if vals:
            bar_plot(names, vals, od / f"compare_{mn}.png", f"Run comparison: {mn}", mn)

    # Combined trajectories.
    plt.figure(figsize=(8, 6))
    plotted = 0
    for rp, m in zip(run_paths, metrics):
        try:
            data = load_npz(rp)
            x = as_2d_time_series(get_key(data, "x", "X", "state", "states"))
            if x is not None and x.shape[1] >= 2:
                plt.plot(x[:, 0], x[:, 1], label=label(m), linewidth=1.4)
                plotted += 1
        except Exception:
            pass
    if plotted:
        plt.title("Trajectory comparison across runs", fontsize=12)
        plt.xlabel("longitudinal position px [m]")
        plt.ylabel("lateral position py [m]")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(od / "combined_trajectories_xy.png", dpi=160)
    plt.close()


def markdown_report(
    outdir: Path,
    array_rows: List[ArraySummary],
    metrics: List[RunMetrics],
    notes: List[str],
    data_files: List[Path],
    run_files: List[Path],
) -> None:
    lines: List[str] = []
    lines.append("# CR-Koopman-MPC Output Analysis Report")
    lines.append("")
    lines.append(f"Generated output directory: `{outdir}`")
    lines.append("")
    lines.append("## Files scanned")
    lines.append(f"- Data NPZ files: {len(data_files)}")
    for p in data_files:
        lines.append(f"  - `{p}`")
    lines.append(f"- Run NPZ files: {len(run_files)}")
    for p in run_files:
        lines.append(f"  - `{p}`")
    lines.append("")
    lines.append("## Basic array health checks")
    def _looks_numeric_dtype(dtype: str) -> bool:
        return dtype.startswith(("float", "int", "uint", "bool"))
    bad = [r for r in array_rows if r.nan_count or r.inf_count or (_looks_numeric_dtype(r.dtype) and r.finite_count == 0)]
    if bad:
        lines.append("Potential issues found:")
        for r in bad[:50]:
            lines.append(f"- `{r.file}` key `{r.key}` shape {r.shape}: nan={r.nan_count}, inf={r.inf_count}, finite={r.finite_count}")
        if len(bad) > 50:
            lines.append(f"- ... and {len(bad)-50} more")
    else:
        lines.append("No NaN/Inf problems found in numeric arrays.")
    lines.append("")

    if metrics:
        lines.append("## Closed-loop run metrics")
        header = ["run", "baseline", "scenario", "steps", "max_abs_y", "min_h", "phys_viol", "mean_residual", "actual_cov", "target_cov", "solve_ms_on", "mpc_rate", "nonstale", "pure_stale", "emergency", "recovery"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for m in metrics:
            vals = [
                Path(m.file).name,
                m.baseline or "",
                m.scenario or "",
                m.steps,
                m.max_abs_y,
                m.min_h,
                m.physical_lane_violations_abs_py_gt_1p75,
                m.mean_residual,
                m.actual_conformal_coverage,
                m.target_coverage,
                m.solve_time_on_solved_mean_ms,
                m.mpc_solved_rate,
                m.trigger_nonstale_count,
                m.trigger_pure_stale_count,
                m.emergency_count,
                m.safety_recovery_count,
            ]
            fmt = []
            for v in vals:
                if isinstance(v, float):
                    fmt.append(f"{v:.4g}")
                else:
                    fmt.append(str(v))
            lines.append("| " + " | ".join(fmt) + " |")
        lines.append("")
        lines.append("Full table: `comparisons/run_metrics.csv`")
        lines.append("")
    else:
        lines.append("## Closed-loop run metrics")
        lines.append("No run `.npz` files were found/analyzed. Run `scripts/run_closed_loop.py` first, then re-run this analyzer.")
        lines.append("")

    if notes:
        lines.append("## Notes / warnings")
        for n in notes:
            lines.append(f"- {n}")
        lines.append("")

    lines.append("## Where to look")
    lines.append("- Dataset plots: `datasets/<file_name>/`")
    lines.append("- Closed-loop plots: `runs/<run_name>/`")
    lines.append("- Multi-run comparison plots: `comparisons/`")
    lines.append("- Koopman matrix plots: `koopman/<file_name>/`")
    lines.append("- Conformal calibration plots: `conformal/<file_name>/`")
    lines.append("")
    (outdir / "summary_report.md").write_text("\n".join(lines), encoding="utf-8")


def discover_files(root: Path, data_dir: Optional[Path], runs_dir: Optional[Path]) -> Tuple[List[Path], List[Path], List[Path]]:
    if data_dir is None:
        data_dir = root / "artifacts" / "data"
    elif not data_dir.is_absolute():
        data_dir = root / data_dir
    if runs_dir is None:
        runs_dir = root / "runs"
    elif not runs_dir.is_absolute():
        runs_dir = root / runs_dir

    data_npz = sorted(data_dir.rglob("*.npz")) if data_dir.exists() else []
    run_npz = sorted(runs_dir.rglob("*.npz")) if runs_dir.exists() else []
    pt_files = sorted(data_dir.rglob("*.pt")) if data_dir.exists() else []
    return data_npz, run_npz, pt_files


def default_project_root() -> Path:
    """Return the repository root when this file lives in scripts/."""
    return Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze CR-Koopman-MPC .npz/.pt outputs and generate plots/reports.")
    ap.add_argument("--root", default=None, help="Project root. Default: auto-detected from scripts/analyze_outputs.py")
    ap.add_argument("--data-dir", default=None, help="Data directory, default: <root>/artifacts/data")
    ap.add_argument("--runs-dir", default=None, help="Runs directory, default: <root>/runs")
    ap.add_argument("--out-dir", default="analysis_outputs", help="Output directory for plots/report")
    ap.add_argument("--only", choices=["all", "data", "runs"], default="all", help="Analyze only data files or run files")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve() if args.root else default_project_root()
    outdir_arg = Path(args.out_dir).expanduser()
    outdir = ensure_dir(outdir_arg if outdir_arg.is_absolute() else root / outdir_arg)
    data_dir = Path(args.data_dir) if args.data_dir else None
    runs_dir = Path(args.runs_dir) if args.runs_dir else None

    data_files, run_files, pt_files = discover_files(root, data_dir, runs_dir)
    if args.only == "data":
        run_files = []
    elif args.only == "runs":
        data_files = []
        pt_files = []

    print(f"Root: {root}")
    print(f"Output directory: {outdir}")
    print(f"Found data NPZ: {len(data_files)}")
    print(f"Found run NPZ:  {len(run_files)}")
    print(f"Found PT files: {len(pt_files)}")

    notes: List[str] = []
    array_summaries: List[ArraySummary] = []
    run_metrics: List[RunMetrics] = []

    for p in data_files:
        print(f"Analyzing data: {p}")
        try:
            data = load_npz(p)
            for k, v in data.items():
                array_summaries.append(arr_summary(p, k, v))
            lname = p.name.lower()
            if "koopman" in lname or ("A" in data and "B" in data):
                notes.extend(analyze_koopman_npz(p, data, outdir, data_files))
            elif "conformal" in lname or "scores" in data:
                notes.extend(analyze_conformal_npz(p, data, outdir))
            else:
                notes.extend(analyze_dataset_npz(p, data, outdir))
        except Exception as e:
            notes.append(f"Failed to analyze data file {p}: {e}\n{traceback.format_exc()}")

    for p in run_files:
        print(f"Analyzing run: {p}")
        try:
            data = load_npz(p)
            for k, v in data.items():
                array_summaries.append(arr_summary(p, k, v))
            run_metrics.append(analyze_run_npz(p, data, outdir))
        except Exception as e:
            notes.append(f"Failed to analyze run file {p}: {e}\n{traceback.format_exc()}")

    if pt_files and args.only != "runs":
        for p in pt_files:
            print(f"Inspecting PT: {p}")
            notes.extend(inspect_pt_file(p, outdir))

    # Save array summaries.
    save_table([asdict(r) for r in array_summaries], outdir / "array_summaries.csv", outdir / "array_summaries.json")
    compare_runs(run_metrics, run_files, outdir)
    markdown_report(outdir, array_summaries, run_metrics, notes, data_files, run_files)

    print("\nDone.")
    print(f"Report: {outdir / 'summary_report.md'}")
    print(f"Array summary: {outdir / 'array_summaries.csv'}")
    if run_metrics:
        print(f"Run metrics: {outdir / 'comparisons' / 'run_metrics.csv'}")
    if notes:
        print("Warnings/notes were written to summary_report.md")


if __name__ == "__main__":
    main()
