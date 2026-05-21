# v8 CBF margin fix

Why v7 still looked wrong:
- CBF itself was no longer falsely safe, but emergency_count was 357/400.
- Root cause: rho used the global CBF Lipschitz constant (~8) times the adaptive bound from [py,psi,vx,vy,r].
- When w_bar rose to ~0.49, rho became ~4.37, larger than h at lane center (~2.83), making the safety filter effectively infeasible even when the vehicle was physically safe.

Fixes in v8:
1. CBF is NOT removed.
2. Conformal robust-margin residual is now in the same coordinates as CBF: [py, vy].
3. Trigger E_z still uses the broader physical metric [py, psi, vx, vy, r].
4. rho uses local ||grad h(x)|| rather than global spectral Lh during normal operation.
5. emergency braking is no longer triggered by slack alone while the car is inside the lane.
6. emergency_brake reduced from -3.0 to -1.5 and emergency_slack raised from 3.0 to 6.0.
7. calibrate.py and run_closed_loop.py now prefer current config residual indices so stale old artifacts do not silently reuse [py,psi,vx,vy,r].

Regenerated artifacts:
- artifacts/data/koopman.npz
- artifacts/data/conformal.npz

Validation available in this environment:
- python -m pytest -q tests/test_sanity.py tests/test_cbf_prior.py  -> 7 passed
- python -m compileall -q cr_koop_mpc scripts -> passed
- train_koopman.py -> A spectral radius 1.0237, Frobenius 4.03
- calibrate.py -> w_bar0 0.0443 for [py, vy]

Need user-side validation:
- Run closed_loop because cvxpy is not installed here.
