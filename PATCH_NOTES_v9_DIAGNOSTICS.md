# v9 diagnostics/data-strength patch

This version does **not** change the controller formulas relative to v8.  It
focuses on making the pipeline harder to misuse and much easier to debug.

## What changed

1. **Data collection is no longer a tiny demo.**
   - Default episodes: 80
   - Expected transitions: 32,000 for the bicycle backend at 20 s / 20 Hz
   - Minimum required transitions: 30,000
   - Multiple exploration modes: center probing, sine sweep, boundary recovery,
     speed sweep
   - Multiple scenarios cycled by default

2. **Every offline stage writes a JSON diagnostic report.**
   - `artifacts/diagnostics/collect_data_summary.json`
   - `artifacts/diagnostics/train_koopman_summary.json`
   - `artifacts/diagnostics/train_cbf_summary.json`
   - `artifacts/diagnostics/calibrate_summary.json`
   - `artifacts/diagnostics/artifact_diagnosis.json`

3. **Stale-artifact checks were added.**
   `run_closed_loop.py` refuses to run if the dataset size does not match the
   Koopman artifact metadata, because stale artifacts were a major source of
   misleading outputs in earlier runs.

4. **Closed-loop logging is much richer.**
   New logs include target/reference, tracking errors, raw MPC input, safety
   filter correction, margin components, `h-rho`, local Lipschitz value,
   MPC status/cost, safety-filter status, predicted barrier, and emergency
   reason diagnostics.

5. **Trajectory plot now shows robot + target + lane boundaries.**
   The file is still:
   `analysis_outputs_v9/runs/<run_name>/trajectory_xy.png`

6. **A standalone artifact diagnosis script was added.**
   Run:
   `python scripts/diagnose_artifacts.py --config configs/default.yaml`

7. **A full pipeline wrapper was added.**
   Run offline only:
   `python scripts/run_full_pipeline.py --skip_closed_loop`

## Results produced in this environment

Because this environment does not have `cvxpy`, closed-loop MPC could not be
run here.  The offline pipeline was run and passed artifact diagnosis:

- Dataset transitions: 32,000
- Koopman A spectral radius: 1.0021
- Koopman A Frobenius norm: 4.20
- Safety residual mean on ID split: 0.03955
- Calibration size used: 4,000
- Initial conformal bound `w_bar0`: 0.0752
- CBF false-safe lane grid count: 0
- Artifact diagnosis: passed

## Important formulation notes to review before Q1 submission

These are not changed in v9, but they must be reflected honestly in the paper:

1. v8/v9 use a local CBF-gradient margin by default (`margin.use_local_lipschitz=true`).
   If the theorem in the paper states a global Lipschitz bound, either the
   proof must be updated to justify local/on-trajectory tightening, or the
   implementation must be switched back to the global bound with a less
   conservative CBF/Lipschitz estimate.  v9 does not change this formula.

2. The CBF is not a free neural network. It is a neural tightening on top of a
   smooth physical lane prior.  The paper should describe it as a composite
   prior-anchored neural CBF, not as an unconstrained MLP CBF.

3. The conformal safety residual is in `[py, vy]`, while the trigger residual is
   broader `[py, psi, vx, vy, r]`. This is intentional, but the paper must state
   that the safety margin uses the CBF-coordinate projection while the trigger
   uses a broader model-trust metric.

4. Fast training is not by itself a bug. EDMD is a closed-form ridge least
   squares solve and the CBF input is only two-dimensional `[py, vy]`. The JSON
   diagnostics, not runtime, are the validity check.
