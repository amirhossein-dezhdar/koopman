# Paper-Code Alignment Report

Config: `configs/default.yaml`

## Implementation claims
- **state**: `[px, py, psi, vx, vy, r]`
- **nonlinear_lifting_state_indices**: `[2, 3, 4, 5]`
- **safety_conformal_residual_indices**: `[1, 4]`
- **trigger_model_error_indices**: `[1, 2, 3, 4, 5]`
- **cbf_input_indices**: `[1, 4]`
- **prior_anchored_neural_cbf**: `True`
- **local_lipschitz_margin**: `True`
- **watchdog_N_max_steps**: `120`
- **watchdog_N_max_seconds_at_dt**: `6.0`

## Required paper edits
- Eq. (7): replace full lifted score ||z_{k+1}-Az_k-Bu_k|| with projected physical score ||P_safety Psi(z_{k+1}-Az_k-Bu_k)|| for safety conformal calibration; mention a separate P_trigger metric for E_x.
- Eq. (12): define rho_k with L_{h,k}=||grad h(x_k)|| when local_lipschitz is enabled, and state that global L_h remains an optional conservative bound.
- Eq. (17): replace lifted-state E_z with physical projected E_x over [py, psi, vx, vy, r].
- Table I / Eq. (21): identify k-t_i>=N_max as watchdog/staleness, and report pure watchdog replans separately from residual/safety/input events.
- NCBF section: state that the implementation uses a prior-anchored neural CBF h=h_prior-softplus(g_theta)*scale, so the network can tighten but not falsely enlarge the lane safe set.
- Evaluation section: report physical lane violations |py|>1.75 and actual conformal coverage mean(score_k<=wbar_{k-1}), not only h<0 or rolling-window coverage.

## Warnings
- Paper text still contains old implementation-inaccurate phrase: full_lifted_score_eq7
- Paper text still contains old implementation-inaccurate phrase: lifted_Ez_eq17
- Paper text still contains old implementation-inaccurate phrase: global_margin_eq12