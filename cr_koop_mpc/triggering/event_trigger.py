"""Residual-driven four-condition event trigger.

Implementation-aligned version of the paper trigger:

    T_k = 1  iff   E_x(k) >= sigma_k      (model trustworthiness)
              or   M_h(k) <= eps_s        (safety proximity)
              or   I_u(k) >= eps_u        (input invalidity)
              or   k - t_i >= N_max       (watchdog/staleness)

Important implementation detail for the paper:
``E_x`` is computed in selected *physical* coordinates, not in the full lifted
RBF/polynomial coordinates.  Full lifted errors have artificial units and are
not meaningful for thresholding.  The controller passes the projected physical
vectors to :meth:`step`; this class only computes their norm.

The staleness term is a watchdog, not evidence of residual-driven triggering.
Diagnostics therefore separate non-stale residual/safety/input events from pure
watchdog replans.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag, auto

import numpy as np


class TriggerReason(IntFlag):
    NONE = 0
    MODEL = auto()    # E_x >= sigma_k
    SAFETY = auto()   # M_h <= eps_s
    INPUT = auto()    # I_u >= eps_u
    STALE = auto()    # k - t_i >= N_max (watchdog)


@dataclass
class TriggerDecision:
    fire: bool
    reason: TriggerReason
    E_z: float
    M_h: float
    I_u: float
    age: int
    sigma_k: float

    @property
    def sigma(self) -> float:
        """Backward-compatible alias for older diagnostics code."""
        return self.sigma_k

    @property
    def model(self) -> bool:
        return bool(self.reason & TriggerReason.MODEL)

    @property
    def safety(self) -> bool:
        return bool(self.reason & TriggerReason.SAFETY)

    @property
    def input(self) -> bool:
        return bool(self.reason & TriggerReason.INPUT)

    @property
    def stale(self) -> bool:
        return bool(self.reason & TriggerReason.STALE)

    @property
    def nonstale(self) -> bool:
        return self.model or self.safety or self.input

    @property
    def pure_stale(self) -> bool:
        return self.stale and not self.nonstale


class ResidualDrivenTrigger:
    """Stateful four-condition trigger with reason-level diagnostics."""

    def __init__(
        self,
        sigma_min: float,
        sigma_max: float,
        lambda_w: float,
        eps_safety: float,
        eps_filter: float,
        N_max: int,
    ) -> None:
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.lambda_w = float(lambda_w)
        self.eps_safety = float(eps_safety)
        self.eps_filter = float(eps_filter)
        self.N_max = int(N_max)
        self._age = 0

    def reset(self) -> None:
        self._age = 0

    def step(
        self,
        z_now: np.ndarray,
        z_pred: np.ndarray,
        min_safety_margin: float,
        last_input_change: float,
        w_bar: float,
    ) -> TriggerDecision:
        """Evaluate the four conditions and return a decision.

        ``age`` in the returned decision is the age *before* the post-fire
        reset.  v9 logged zero for firing steps, which made inter-event
        intervals impossible to diagnose.
        """
        E_z = float(np.linalg.norm(np.asarray(z_now) - np.asarray(z_pred)))
        I_u = float(last_input_change)
        sigma_k = float(
            np.clip(self.sigma_max - self.lambda_w * float(w_bar), self.sigma_min, self.sigma_max)
        )
        self._age += 1
        age_eval = int(self._age)

        reason = TriggerReason.NONE
        if E_z >= sigma_k:
            reason |= TriggerReason.MODEL
        if float(min_safety_margin) <= self.eps_safety:
            reason |= TriggerReason.SAFETY
        if I_u >= self.eps_filter:
            reason |= TriggerReason.INPUT
        if age_eval >= self.N_max:
            reason |= TriggerReason.STALE

        fire = reason != TriggerReason.NONE
        if fire:
            self._age = 0

        return TriggerDecision(
            fire=fire,
            reason=reason,
            E_z=E_z,
            M_h=float(min_safety_margin),
            I_u=I_u,
            age=age_eval,
            sigma_k=sigma_k,
        )
