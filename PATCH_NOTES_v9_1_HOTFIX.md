# v9.1 hotfix

Fixes a logger-only crash in `scripts/run_closed_loop.py`:

- `TriggerDecision` stores the adaptive trigger threshold as `sigma_k`.
- v9.0 logger tried to read `log.trigger.sigma`, causing:
  `AttributeError: 'TriggerDecision' object has no attribute 'sigma'`.
- v9.1 now logs `sigma=log.trigger.sigma_k` and also adds a backward-compatible
  `TriggerDecision.sigma` property.

No MPC/CBF/conformal formulas were changed in this hotfix.

Validation run in this environment:

```text
python -m compileall -q cr_koop_mpc scripts tests
pytest -q
# 9 passed, 1 skipped
```
