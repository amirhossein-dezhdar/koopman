# CR-Koopman-MPC v10 paper-aligned diagnostic release

This version fixes the v9 issue where events looked equally spaced because the
`STALE` watchdog (`N_max=15`) was firing every 0.75 s.  The watchdog is now 120
steps by default and the logger/analyzer separates true residual/safety/input
events from pure watchdog replans.

## Recommended run

```powershell
python scripts/run_full_pipeline.py --baseline ours --scenario lane_keeping --name ours_lane_keeping_v10
python scripts/analyze_outputs.py --data-dir artifacts/data --runs-dir runs --out-dir analysis_outputs_v10
```

## Pure event diagnostic

```powershell
python scripts/run_closed_loop.py --config configs/event_pure.yaml --baseline ours --scenario lane_keeping --name pure_event_lane
python scripts/analyze_outputs.py --data-dir artifacts/data --runs-dir runs --out-dir analysis_outputs_v10_pure
```

## Multi-scenario benchmark

```powershell
python scripts/run_benchmark_suite.py --config configs/default.yaml --baseline ours --prefix v10 --analyze --out-dir analysis_outputs_v10_suite
```

## Paper alignment

The uploaded paper PDF is not editable source.  I added these files instead:

- `docs/PAPER_PATCH_V10.md`
- `docs/PAPER_IMPLEMENTATION_ALIGNED_SECTIONS_V10.md`
- `docs/paper_code_alignment_report.md`

These specify the exact paper edits needed for the v10 implementation.

## New metrics to check

In `analysis_outputs_v10/comparisons/run_metrics.csv`, check:

- `physical_lane_violations_abs_py_gt_1p75`
- `actual_conformal_coverage`
- `trigger_nonstale_count`
- `trigger_pure_stale_count`
- `watchdog_fraction_of_replans`
- `solve_time_on_solved_mean_ms`
- `mean_inter_event_steps`
