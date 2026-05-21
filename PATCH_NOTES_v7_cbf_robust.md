# v7 CBF-robust patch

This patch keeps the CBF component, but fixes the remaining logical safety
problem found in `analysis_outputs1.zip`: the learned CBF could return `h > 0`
for states tens of metres outside the lane.

## Main changes

1. **Neural CBF with physical lane prior**
   - `NeuralCBF` now supports `prior_mode: lane_saturating`.
   - The default barrier is
     `h = h_prior(py, vy) - correction_scale * softplus(g_theta(py, vy))`.
   - The neural network remains trainable, but can only tighten the safe set;
     it cannot make physical lane-departure states falsely safe.

2. **Stronger CBF training**
   - `scripts/train_cbf.py` now augments training with symmetric safe/unsafe
     py/vy samples, boundary bands, far lane departures, and high lateral
     velocity samples.
   - The CBF still sees only `[py, vy]`, not `px`.
   - A strict py/vy grid validation runs after training and raises an error if
     any physical lane-departure grid point has `h > 0`.

3. **Fixed barrier-transition loader bug**
   - `train_neural_cbf` no longer recreates `iter(tx_loader)` on every batch.
     The old code repeatedly used the first transition batch.

4. **Emergency recovery improved**
   - Emergency fallback now uses lane-recovery steering plus controlled braking,
     instead of straight-line/max braking.

5. **Analysis reports physical lane safety**
   - `analyze_outputs.py` now reports actual lane violations based on
     `abs(py) > 1.75`, not only `h < 0`.
   - Added `physical_lane_safety.png` and metrics:
     `physical_lane_violations_abs_py_gt_1p75`,
     `lane_safe_rate_abs_py_le_1p75`, `max_lane_excess_m`, and
     `min_lane_margin_m`.

## Included regenerated artifacts

The zip includes regenerated artifacts from the included dataset:

- `artifacts/data/koopman.npz`
- `artifacts/data/conformal.npz`
- `artifacts/data/cbf.pt`

CBF validation from the included artifact:

- `false_safe_lane_count = 0`
- `max_h_outside_lane = -0.575`
- `h_center = 2.828`
- `px_invariance_span = 0`

## What still must be checked on the user's machine

This environment does not have `cvxpy`, so the full closed-loop MPC/QP rollout
cannot be executed here. After extracting, run:

```powershell
python scripts/run_closed_loop.py --config configs/default.yaml --baseline ours --scenario lane_keeping --name ours_lane_keeping_v7
python scripts/analyze_outputs.py --data-dir artifacts/data --runs-dir runs --out-dir analysis_outputs_v7
```

The run is not acceptable unless the physical lane metrics are good, especially:

- `physical_lane_violations_abs_py_gt_1p75 = 0`
- `max_abs_y < 1.75`
- `emergency_count` near zero
- `mean_speed` near nominal driving speed
