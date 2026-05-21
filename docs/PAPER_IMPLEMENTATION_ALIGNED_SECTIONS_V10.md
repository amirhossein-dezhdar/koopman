# Implementation-aligned paper sections for v10

Use these replacements in the paper source before claiming the code and paper are aligned.

## Replacement for Section IV.A score definition

Let `Psi(z)` denote the canonical projection onto physical state coordinates.
The full lifted residual is

```text
w^z_k = z_{k+1} - A z_k - B u_k.
```

For numerical diagnostics we may inspect `||w^z_k||`, but the conformal score
used by the safety margin is computed in physical coordinates:

```text
r^x_k = x_{k+1} - Psi(A Phi(x_k) + B u_k),
s^s_k = || P_s r^x_k ||_2,     P_s x = [p_y, v_y]^T.
```

A separate trigger-space score can be used for model-trust diagnostics:

```text
s^t_k = || P_t r^x_k ||_2,     P_t x = [p_y, psi, v_x, v_y, r]^T.
```

This projected-score choice is essential because RBF/polynomial lifted
coordinates do not have the same units as the physical barrier.

## Replacement for Section V margin

The residual-tightened margin used in the implementation is

```text
rho_k = L_{h,k} (L_x wbar_k + eps_rec + eps_sens) + rho_lin,
L_{h,k} = || grad h_theta(x_k) ||_2.
```

The global certified Lipschitz constant of the spectral-normalized NCBF remains
a valid conservative fallback.  The reported implementation uses the local
on-trajectory gradient norm because the global bound is too conservative for
normal lane-center driving.

## Replacement for NCBF architecture paragraph

The implementation uses a prior-anchored neural control barrier function:

```text
h_theta(x) = h_prior(P_s x) - c softplus(g_theta(P_s x)),
P_s x = [p_y, v_y]^T.
```

The learned neural term can tighten the safe set but cannot make physically
outside-lane states falsely safe.  This preserves the NCBF component while
removing the OOD extrapolation failure observed when an unconstrained MLP saw
only limited lane-departure data.

## Replacement for Section VII trigger diagnostic

The model-trust trigger is evaluated as

```text
E_x(k) = || P_t (x_k - xhat_{k|t_i}) ||_2,
```

not as the norm of the full lifted RBF/polynomial error.  The threshold remains

```text
sigma_k = clip(sigma_max - lambda_w wbar_k, sigma_min, sigma_max).
```

The event condition is

```text
T_k = 1 iff (E_x >= sigma_k) or (M_h <= eps_s) or (I_u >= eps_u) or (k-t_i >= N_max).
```

The last term is a watchdog/staleness replan.  It must be reported separately
from residual/safety/input-triggered events.

## Replacement for evaluation metrics

Report at least:

```text
max_abs_y
physical_lane_violations = sum(1{|p_y| > 1.75})
lane_safe_rate = mean(1{|p_y| <= 1.75})
actual_conformal_coverage = mean(1{s_k <= wbar_{k-1}})
model_trigger_count
safety_trigger_count
input_trigger_count
pure_watchdog_replan_count
nonstale_event_count
solve_time_on_solved_mean_ms
```

Do not use `h<0` alone as the safety metric, because a learned CBF can be wrong
out of distribution.  Physical lane violation must always be reported.
