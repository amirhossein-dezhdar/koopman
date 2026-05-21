# v10 changelog

## Main fixes

- Separates true residual/safety/input-triggered events from pure watchdog
  replans.
- Fixes trigger `age` logging: age is now the pre-reset inter-event age.
- Changes default `N_max` from 15 steps to 120 steps so easy lane keeping no
  longer looks like periodic MPC at 0.75 s intervals.
- Adds `configs/event_pure.yaml` with effectively disabled watchdog replans.
- Logs `trigger_model`, `trigger_safety`, `trigger_input`, `trigger_stale`,
  `trigger_nonstale`, `trigger_pure_stale`, and `trigger_age`.
- Logs actual conformal coverage: `conformal_covered = 1{s_k <= wbar_{k-1}}`.
- Adds solve-time metrics on solved steps only, not just mean over all steps.
- Adds stress-test initial-condition scenarios in the bicycle backend:
  `lane_offset_pos`, `lane_offset_neg`, `near_boundary_pos`, `near_boundary_neg`,
  `heading_error_pos`, `heading_error_neg`, `speed_low`, `speed_high`.
- Adds `scripts/run_benchmark_suite.py` for multi-scenario evaluation.
- Adds `scripts/check_paper_alignment.py` and paper patch notes.

## What was not changed

- The core CBF-QP / Koopman-MPC equations were not re-derived or replaced.
- CBF is still neural and prior-anchored; it was not removed.
- The v9 trained artifacts are kept, but the default event watchdog is changed.
  Retraining is recommended for final paper runs.
