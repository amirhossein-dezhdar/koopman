"""Conformal residual bound (split + adaptive variants).

Implements eqs. (8)-(13) of the paper:

    s_i  = ||Π_x(z_{i+1} − A z_i − B u_i)||      (physical residual score)
    w̄_k = Quantile_{1−α_k}(s_1, …, s_N)          (conformal bound)

The class stores scalar scores only.  The caller is responsible for computing
those scores in physical units, not full lifted-feature units.

For the adaptive variant we use the Gibbs-Candès rule (NeurIPS 2021):

    e_k^conf = 𝟙{s_k > w̄_k}
    α_{k+1}  = clip( α_k + η(α − e_k^conf), [α_min, α_max] )

so that the long-run empirical miscoverage converges to the target ``α``
even under distribution shift — which is exactly the regime we care about
(low-friction roads, OOD maneuvers, sensor noise).
"""

from __future__ import annotations

from collections import deque

import numpy as np


class ConformalResidual:
    """Conformal bound on ``||w_k||`` for the Koopman predictor.

    Parameters
    ----------
    alpha : float
        Target miscoverage in (0, 1).
    adaptive : bool
        If True, update α online (Gibbs-Candès); else fixed split-conformal.
    gamma : float
        Adaptation gain ``η`` for the α update.
    window : int
        Rolling-window size for online residual scores.
    alpha_min, alpha_max : float
        Projection interval for α to prevent runaway in adaptive mode.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        adaptive: bool = True,
        gamma: float = 0.02,
        window: int = 500,
        alpha_min: float = 1e-3,
        alpha_max: float = 0.5,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha_target = alpha
        self.alpha = alpha
        self.adaptive = adaptive
        self.gamma = gamma
        self.window = window
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self._scores: deque[float] = deque(maxlen=window)
        self._calibrated = False
        self._w_bar = float("inf")
        # Last online-update diagnostics.  These are used by run_closed_loop.py
        # and analyze_outputs.py to report actual conformal coverage rather
        # than confusing it with alpha or target coverage.
        self.last_score = float("nan")
        self.last_bound_before_update = float("nan")
        self.last_covered = float("nan")
        self.last_miscovered = float("nan")

    # ---------------- calibration ----------------

    def calibrate(self, calibration_residual_norms: np.ndarray) -> float:
        """Initialise the bound from a held-out calibration set of scores."""
        if calibration_residual_norms.ndim != 1:
            raise ValueError("Provide scalar residual norms, shape (N,)")
        for s in calibration_residual_norms:
            self._scores.append(float(s))
        self._w_bar = self._quantile(self.alpha)
        self._calibrated = True
        return self._w_bar

    # ---------------- online use ----------------

    def bound(self) -> float:
        """Current bound ``w̄_k``."""
        if not self._calibrated:
            raise RuntimeError("ConformalResidual must be calibrated before use")
        return self._w_bar

    def update(self, observed_residual_norm: float) -> float:
        """Add a new score, update α (if adaptive), recompute the bound."""
        score = float(observed_residual_norm)
        # ACI rule uses the *previous* bound to label miscoverage.  Store that
        # exact comparison so the analyzer can report actual coverage
        #     mean(score_k <= wbar_{k-1})
        # rather than the in-window coverage after the update.
        bound_before = float(self._w_bar)
        if self._calibrated and np.isfinite(bound_before):
            err = 1.0 if score > bound_before else 0.0
            covered = 1.0 - err
        else:
            err = float("nan")
            covered = float("nan")
        self.last_score = score
        self.last_bound_before_update = bound_before
        self.last_covered = covered
        self.last_miscovered = err

        if self.adaptive and self._calibrated:
            err_for_update = 0.0 if not np.isfinite(err) else err
            self.alpha = float(
                np.clip(
                    self.alpha + self.gamma * (self.alpha_target - err_for_update),
                    self.alpha_min,
                    self.alpha_max,
                )
            )
        # Then we add the score and recompute the bound at the new α.
        self._scores.append(score)
        self._w_bar = self._quantile(self.alpha)
        if not self._calibrated:
            self._calibrated = True
        return self._w_bar

    # ---------------- diagnostics ----------------

    def empirical_coverage(self) -> float:
        """Fraction of in-window scores below current bound."""
        if not self._scores:
            return 0.0
        arr = np.fromiter(self._scores, dtype=float)
        return float(np.mean(arr <= self._w_bar))

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    # ---------------- internals ----------------

    def _quantile(self, alpha: float) -> float:
        """Empirical (1−α)-quantile with finite-sample conformal correction."""
        arr = np.fromiter(self._scores, dtype=float)
        n = len(arr)
        if n == 0:
            return float("inf")
        k = int(np.ceil((n + 1) * (1.0 - alpha)))
        k = min(max(k, 1), n)
        return float(np.partition(arr, k - 1)[k - 1])
