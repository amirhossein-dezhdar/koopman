#!/usr/bin/env python3
"""Check implementation-vs-paper alignment assumptions.

If a paper text extracted from the PDF is provided, this script also flags old
phrases that are no longer implementation-accurate (e.g., full lifted residual
scores or lifted-state event errors).  It does not edit the PDF; it writes a
machine-readable JSON and a Markdown report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cr_koop_mpc.utils import load_config


def cfg_list(x: Any) -> list[int]:
    return [int(v) for v in list(x)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT_DIR / "configs" / "default.yaml"))
    ap.add_argument("--paper-text", default=None, help="Optional plain-text extraction of the paper PDF")
    ap.add_argument("--out", default=str(ROOT_DIR / "docs" / "paper_code_alignment_report.json"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    report: dict[str, Any] = {
        "config": str(args.config),
        "status": "checked",
        "implementation_claims": {},
        "warnings": [],
        "required_paper_edits": [],
    }

    nonlinear = cfg_list(cfg.koopman.edmd.nonlinear_state_indices)
    residual = cfg_list(cfg.residual.state_indices)
    trigger = cfg_list(cfg.residual.trigger_indices)
    cbf_indices = cfg_list(cfg.cbf.state_indices)
    use_local = bool(cfg.margin.use_local_lipschitz)
    prior_enabled = bool(cfg.cbf.prior.enabled)
    nmax = int(cfg.trigger.N_max)

    report["implementation_claims"] = {
        "state": "[px, py, psi, vx, vy, r]",
        "nonlinear_lifting_state_indices": nonlinear,
        "safety_conformal_residual_indices": residual,
        "trigger_model_error_indices": trigger,
        "cbf_input_indices": cbf_indices,
        "prior_anchored_neural_cbf": prior_enabled,
        "local_lipschitz_margin": use_local,
        "watchdog_N_max_steps": nmax,
        "watchdog_N_max_seconds_at_dt": float(nmax * cfg.env.dt),
    }

    if 0 in nonlinear or 1 in nonlinear:
        report["warnings"].append("Nonlinear lifting includes absolute/lateral position; this can poison EDMD features.")
    if residual != cbf_indices:
        report["warnings"].append("Safety conformal residual indices must match CBF input indices for rho_k.")
    if nmax <= 20:
        report["warnings"].append("N_max is small; easy scenarios may look periodically triggered by watchdog replans.")

    report["required_paper_edits"] = [
        "Eq. (7): replace full lifted score ||z_{k+1}-Az_k-Bu_k|| with projected physical score ||P_safety Psi(z_{k+1}-Az_k-Bu_k)|| for safety conformal calibration; mention a separate P_trigger metric for E_x.",
        "Eq. (12): define rho_k with L_{h,k}=||grad h(x_k)|| when local_lipschitz is enabled, and state that global L_h remains an optional conservative bound.",
        "Eq. (17): replace lifted-state E_z with physical projected E_x over [py, psi, vx, vy, r].",
        "Table I / Eq. (21): identify k-t_i>=N_max as watchdog/staleness, and report pure watchdog replans separately from residual/safety/input events.",
        "NCBF section: state that the implementation uses a prior-anchored neural CBF h=h_prior-softplus(g_theta)*scale, so the network can tighten but not falsely enlarge the lane safe set.",
        "Evaluation section: report physical lane violations |py|>1.75 and actual conformal coverage mean(score_k<=wbar_{k-1}), not only h<0 or rolling-window coverage.",
    ]

    if args.paper_text:
        txt = Path(args.paper_text).read_text(encoding="utf-8", errors="ignore")
        old_patterns = {
            "full_lifted_score_eq7": "sk = ∥zk+1 −Azk −Buk∥2",
            "lifted_Ez_eq17": "where the implementation uses P = I, so this is the Euclidean\nlifted-state error",
            "global_margin_eq12": "ρk = Lh (Lx ¯wk + ϵrec + ϵsens) + ρlin",
        }
        for name, pat in old_patterns.items():
            if pat in txt:
                report["warnings"].append(f"Paper text still contains old implementation-inaccurate phrase: {name}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    md = out.with_suffix(".md")
    lines = ["# Paper-Code Alignment Report", "", f"Config: `{args.config}`", ""]
    lines.append("## Implementation claims")
    for k, v in report["implementation_claims"].items():
        lines.append(f"- **{k}**: `{v}`")
    lines.append("")
    lines.append("## Required paper edits")
    for e in report["required_paper_edits"]:
        lines.append(f"- {e}")
    lines.append("")
    if report["warnings"]:
        lines.append("## Warnings")
        for w in report["warnings"]:
            lines.append(f"- {w}")
    else:
        lines.append("## Warnings")
        lines.append("No config-level warnings.")
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {out}")
    print(f"Saved {md}")


if __name__ == "__main__":
    main()
