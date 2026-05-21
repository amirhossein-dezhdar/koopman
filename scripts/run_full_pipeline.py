"""Run the offline pipeline in the correct order with diagnostics.

This is a convenience wrapper.  It runs data collection, Koopman training, CBF
training, conformal calibration, artifact diagnosis, and then optionally a
closed-loop rollout.  The closed-loop stage requires cvxpy/OSQP.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT_DIR / "configs" / "default.yaml"


def run(cmd: list[str]) -> None:
    print("\n>>> " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--skip_closed_loop", action="store_true")
    ap.add_argument("--baseline", default="ours")
    ap.add_argument("--scenario", default="lane_keeping")
    ap.add_argument("--name", default="ours_lane_keeping_v10")
    ap.add_argument("--skip_paper_alignment_check", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    collect = [py, "scripts/collect_data.py", "--config", args.config]
    if args.episodes is not None:
        collect += ["--episodes", str(args.episodes)]
    run(collect)
    run([py, "scripts/train_koopman.py", "--config", args.config])
    run([py, "scripts/train_cbf.py", "--config", args.config])
    run([py, "scripts/calibrate.py", "--config", args.config])
    run([py, "scripts/diagnose_artifacts.py", "--config", args.config])
    if not args.skip_paper_alignment_check:
        run([py, "scripts/check_paper_alignment.py", "--config", args.config])
    if not args.skip_closed_loop:
        run([py, "scripts/run_closed_loop.py", "--config", args.config, "--baseline", args.baseline, "--scenario", args.scenario, "--name", args.name])
        run([py, "scripts/analyze_outputs.py", "--data-dir", "artifacts/data", "--runs-dir", "runs", "--out-dir", "analysis_outputs_v10"])


if __name__ == "__main__":
    main()
