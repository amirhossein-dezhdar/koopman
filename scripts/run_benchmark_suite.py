#!/usr/bin/env python3
"""Run a small but meaningful closed-loop benchmark suite.

This script is intentionally separate from run_full_pipeline.py.  After the
artifacts are trained once, it runs multiple scenarios so the paper is not
judged from an easy lane-center trajectory only.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIOS = [
    "lane_keeping",
    "lane_offset_pos",
    "lane_offset_neg",
    "near_boundary_pos",
    "near_boundary_neg",
    "heading_error_pos",
    "heading_error_neg",
    "low_friction",
    "sensor_noise",
    "dlc",
    "ood",
]


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--baseline", default="ours")
    ap.add_argument("--scenarios", nargs="*", default=DEFAULT_SCENARIOS)
    ap.add_argument("--prefix", default="v10")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--out-dir", default="analysis_outputs_v10_suite")
    args = ap.parse_args()

    py = sys.executable
    for scen in args.scenarios:
        name = f"{args.prefix}_{args.baseline}_{scen}"
        run([
            py, "scripts/run_closed_loop.py",
            "--config", args.config,
            "--baseline", args.baseline,
            "--scenario", scen,
            "--name", name,
        ])

    if args.analyze:
        run([
            py, "scripts/analyze_outputs.py",
            "--data-dir", "artifacts/data",
            "--runs-dir", "runs",
            "--out-dir", args.out_dir,
        ])


if __name__ == "__main__":
    main()
