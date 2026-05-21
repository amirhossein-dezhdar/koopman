# v6_fixed patch notes

This zip fixes the logical bugs that caused the v5 run to produce emergency braking at every step.

## Main changes

1. **Conformal residual is now physical, not full lifted-space.**
   - Added `KoopmanPredictor.physical_residual()` and `KoopmanPredictor.residual_score()`.
   - `scripts/calibrate.py` and `ResidualDrivenController._update_residual()` now use `residual.state_indices` from the config.
   - Default residual metric: `[py, psi, vx, vy, r]`, excluding absolute `px`.

2. **Event trigger uses physical prediction error.**
   - The controller reconstructs the held Koopman prediction back to physical state and computes `E_z` on `residual.trigger_indices`.
   - This avoids comparing full RBF/polynomial lifted errors against physical thresholds.

3. **EDMD nonlinear lifting no longer uses absolute position.**
   - `EDMDLifting` now supports `feature_indices`.
   - Default nonlinear features are built only from `[psi, vx, vy, r]`.
   - Full `[px, py, psi, vx, vy, r]` is still preserved as the identity block for reconstruction and MPC tracking.

4. **CBF cannot learn the px artifact.**
   - `NeuralCBF` now supports `feature_indices` and can be called with the full state while internally using only selected safety features.
   - Default CBF input: `[py, vy]`.
   - `train_cbf.py` saves CBF metadata and prints a px-invariance diagnostic.

5. **Stale artifact protection.**
   - `run_closed_loop.py` will raise a clear error if you try to load an old full-state CBF artifact into the reduced-input CBF.
   - Old generated artifacts were removed from this zip except the collected dataset. Re-run the pipeline from zero.

6. **Training speed fix.**
   - `train_cbf.py` sets `torch.set_num_threads(1)` to avoid CPU thread-pool overhead for the small spectral-normalized MLP.
   - It also supports `--max_train_per_class` and `--max_transitions` for faster balanced CBF training.

## Required run order

From the project root:

```powershell
python scripts/train_koopman.py --config configs/default.yaml --data artifacts/data/bicycle_dataset.npz
python scripts/train_cbf.py --config configs/default.yaml --data artifacts/data/bicycle_dataset.npz
python scripts/calibrate.py --config configs/default.yaml --data artifacts/data/bicycle_dataset.npz
python scripts/run_closed_loop.py --config configs/default.yaml --baseline ours --scenario lane_keeping --name ours_lane_keeping
python scripts/analyze_outputs.py --runs runs/ours_lane_keeping.npz --out analysis_outputs
```

If imports fail on Windows, run from the project root or install editable mode:

```powershell
pip install -e .
```

## What I could test here

I could run all numpy-only tests and the Koopman/calibration scripts. I could not run closed-loop MPC here because this container does not have `cvxpy` installed. The code path is still guarded by compile tests and the existing pytest smoke test skips the full MPC when `cvxpy` is unavailable.
