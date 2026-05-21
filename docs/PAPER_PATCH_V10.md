# Paper patch required for v10 implementation alignment

The uploaded PDF is a compiled PDF, not an editable LaTeX/DOCX source.  I did
not destructively edit the PDF.  These are the exact conceptual edits needed so
that the paper matches the v10 code.

## 1. Section IV.A / Eq. (7): conformal score

Old wording says the realised score is the full lifted residual:

```text
s_k = || z_{k+1} - A z_k - B u_k ||_2
```

Replace with projected physical scores:

```text
r^x_k = Psi(z_{k+1}) - Psi(A z_k + B u_k)
s^safety_k = || P_s r^x_k ||_2,   P_s selects [p_y, v_y].
s^trigger_k = || P_t r^x_k ||_2,  P_t selects [p_y, psi, v_x, v_y, r].
```

The safety score is used for conformal calibration and the CBF margin.  The
trigger score/metric is used for model-trust replanning.  This avoids mixing
RBF/polynomial lifted coordinates with meter/radian safety margins.

## 2. Section V / Eq. (12): robust CBF margin

Old paper uses a global Lipschitz margin:

```text
rho_k = L_h (L_x wbar_k + eps_rec + eps_sens) + rho_lin.
```

Implementation-aligned text:

```text
rho_k = L_{h,k} (L_x wbar_k + eps_rec + eps_sens) + rho_lin,
L_{h,k} = || grad h_theta(x_k) ||_2.
```

The global spectral Lipschitz bound remains available as a conservative fallback
or for theorem statements.  The reported implementation uses the local gradient
bound to avoid vacuous over-tightening at lane center.

## 3. Section V: prior-anchored neural CBF

Add this implementation detail:

```text
The reported lane-keeping implementation uses a prior-anchored NCBF
h_theta(x) = h_prior(P_s x) - c softplus(g_theta(P_s x)),
where P_s x = [p_y, v_y].  The neural correction can tighten the safe set but
cannot make physical lane-departure states falsely safe.  This retains a neural
CBF while preventing OOD extrapolation failures.
```

## 4. Section VII / Eq. (17): model-trigger diagnostic

Old paper says the implementation uses Euclidean lifted-state error.  Replace
with:

```text
E_x(k) = || P_t ( x_k - \hat{x}_{k|t_i} ) ||_2,
```

where `P_t` selects `[p_y, psi, v_x, v_y, r]`.  Full lifted errors are still
useful as offline EDMD diagnostics but are not used as the event threshold in
closed loop.

## 5. Table I and Eq. (21): watchdog vs true event

Keep the staleness term, but define it explicitly as a watchdog:

```text
k - t_i >= N_max  (watchdog/staleness replan)
```

Evaluation tables must report:

```text
model_trigger_count
safety_trigger_count
input_trigger_count
stale_trigger_count
pure_watchdog_replan_count
nonstale_event_count
```

Do not describe pure watchdog replans as residual-driven events.

## 6. Section XI: metrics

Add the physical safety metric:

```text
physical lane violation = 1{|p_y| > 1.75 m}
lane safe rate = mean(1{|p_y| <= 1.75 m})
```

And distinguish conformal diagnostics:

```text
actual coverage = mean(1{s_k <= wbar_{k-1}})
rolling/window coverage = mean over the current ACI score buffer
```

The paper equation for empirical coverage should refer to actual coverage, not
alpha or the rolling-window diagnostic.

## 7. Evaluation wording

The current code is a bicycle-backend reproducibility implementation with
MetaDrive/CARLA-compatible interfaces.  Do not claim CARLA quantitative results
unless you actually run CARLA and include the logs.

Use wording like:

```text
We evaluate the core controller in the lightweight bicycle backend and expose
MetaDrive/CARLA-compatible environment adapters for future high-fidelity runs.
```

until those high-fidelity experiments are produced.
