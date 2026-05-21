# Conformal Residual-Driven Koopman MPC for Safe Autonomous Driving

Reference implementation for the paper *Conformal Residual-Driven
Event-Triggered Koopman MPC for Safe Autonomous Driving* (Dezhdar, 2026).

## What this does

A predictive controller for autonomous driving that:

1. Lifts nonlinear vehicle dynamics into a Koopman space and predicts with a
   learned linear model `z_{k+1} = A z_k + B u_k + w_k`.
2. Bounds the residual `w_k` online with **adaptive conformal inference**
   (Gibbs & Candès, 2021) — long-run marginal coverage `1 − α` under
   non-stationary driving.
3. Uses that bound to drive **three** decisions:
   - when to re-solve the MPC (event-triggering),
   - how much to tighten the safety margin around a **Lipschitz-certified**
     neural CBF (spectral normalization),
   - whether the previous control action is still trustworthy.
4. Filters every actuator command through a conservative discrete-time CBF QP.

## Repo layout

```
cr_koop_mpc/
├── envs/         # Simulator backends (bicycle, MetaDrive, CARLA)
├── models/       # Koopman lifting, predictor, adaptive conformal
├── safety/       # Spectral-normalized neural CBF, QP safety filter
├── control/      # Koopman MPC, baseline controllers
├── triggering/   # Four-condition residual-driven trigger
└── utils/        # config, logger

configs/default.yaml     # one place to change all hyperparameters
scripts/                 # collect_data → train → calibrate → run_closed_loop → analyze_outputs
tests/                   # smoke test on the bicycle backend
```

## Install

```bash
pip install -e .
# Optional simulators:
pip install metadrive-simulator      # easy local install
# CARLA: install the 0.9.x .egg matching your CARLA server
```

The bicycle backend has zero external sim dependencies and is sufficient to
reproduce the algorithmic results and run the test suite in CI.

## Swap simulators

The simulator is a single interface, `EnvBackend`, in
`cr_koop_mpc/envs/base.py`. Three backends ship out of the box:

- `BicycleEnv`   — pure-Python kinematic + dynamic bicycle for theory/CI tests
- `MetaDriveEnv` — wraps `metadrive` for fast scenario rollouts
- `CarlaEnv`     — wraps the CARLA 0.9.x Python client

Switching is one line in the config:

```yaml
env:
  backend: metadrive   # → bicycle | metadrive | carla
```

The controller code never knows which sim is underneath. The contract is six
methods: `reset`, `step`, `get_reference`, `scenario_done`, `close`,
`safety_features`.



## Important v10 note: paper-aligned diagnostics

This v10 release separates true residual/safety/input events from pure watchdog
replans.  A watchdog event (`STALE`) is not counted as evidence of residual-
driven triggering.  The default watchdog horizon is now `N_max=120` steps; use
`configs/event_pure.yaml` to disable it for diagnostic tests.

The implementation uses projected physical residuals, a prior-anchored neural
CBF, and a local CBF-gradient margin.  See:

- `docs/PAPER_PATCH_V10.md`
- `docs/PAPER_IMPLEMENTATION_ALIGNED_SECTIONS_V10.md`
- `docs/paper_code_alignment_report.md`

Run the recommended v10 pipeline:

```bash
python scripts/run_full_pipeline.py --baseline ours --scenario lane_keeping --name ours_lane_keeping_v10
python scripts/run_benchmark_suite.py --config configs/default.yaml --baseline ours --prefix v10 --analyze --out-dir analysis_outputs_v10_suite
```

## Important v6_fixed note

This version intentionally computes conformal residuals in **physical state
coordinates**, not in the full lifted EDMD/RBF space.  The default residual
metric is `[py, psi, vx, vy, r]`, and the neural CBF sees only `[py, vy]`.
Absolute `px` is excluded from nonlinear lifting and from the CBF safety model
because it is a translation coordinate and caused the v5 emergency-brake
lock-in.

Old generated artifacts are not compatible with this code.  Regenerate
`koopman.npz`, `cbf.pt`, and `conformal.npz` before running closed loop.

## Run

```bash
# You may skip collect_data.py initially because artifacts/data/bicycle_dataset.npz is included.
python scripts/train_koopman.py    --config configs/default.yaml --data artifacts/data/bicycle_dataset.npz
python scripts/train_cbf.py        --config configs/default.yaml --data artifacts/data/bicycle_dataset.npz
python scripts/calibrate.py        --config configs/default.yaml --data artifacts/data/bicycle_dataset.npz
python scripts/run_closed_loop.py  --config configs/default.yaml --baseline ours --scenario lane_keeping --name ours_lane_keeping
python scripts/analyze_outputs.py --runs runs/ours_lane_keeping.npz --out analysis_outputs
```

Closed-loop runs are saved in `<project>/runs/` and analysis reports/plots are
saved in `<project>/analysis_outputs/`, even if the command is launched from a
different working directory.

## Reproducing paper baselines

`scripts/run_closed_loop.py --baseline {nmpc, periodic, static_et, no_filter,
cbf_qp, tube, ours}` switches the controller stack while keeping the
simulator, the lifting, and the CBF identical, so the ablation is clean.

## Paper-to-code map

| Paper section / equation | Code |
|---|---|
| §III, eqs. (7)–(13) — Koopman + adaptive conformal | `models/koopman.py`, `models/conformal.py` |
| §IV, eqs. (14)–(15) — Lipschitz-certified neural CBF | `safety/neural_cbf.py` |
| §V, eqs. (16)–(18) — Koopman MPC | `control/koopman_mpc.py` |
| §VI, eqs. (19)–(22) — Four-condition trigger | `triggering/event_trigger.py` |
| §VII, eq. (23) — Safety filter QP | `safety/qp_filter.py` |
| §IX — Offline + online pipeline | `scripts/` + `control/controller.py` |
| §X — Baselines | `control/baselines.py` |

## v7 note: CBF is kept, but made physically robust

The CBF is not removed.  It is now a neural CBF with a smooth physical lane
prior and a learned neural risk tightening.  Before running closed-loop tests,
regenerate or verify artifacts with:

```bash
python scripts/train_koopman.py --config configs/default.yaml --data artifacts/data/bicycle_dataset.npz
python scripts/train_cbf.py --config configs/default.yaml --data artifacts/data/bicycle_dataset.npz
python scripts/calibrate.py --config configs/default.yaml --data artifacts/data/bicycle_dataset.npz
```

`train_cbf.py` will fail intentionally if the CBF is falsely safe outside the
physical lane on its py/vy validation grid.
