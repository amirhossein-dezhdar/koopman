"""Koopman predictor in the lifted space.

Given a lifting ``Φ``, identifies matrices ``A, B`` from data such that
``z_{k+1} ≈ A z_k + B u_k`` by ridge least squares. Supports optional
recursive weighted least squares (RLS) for online ``[A B]`` adaptation —
the future-work hook the paper mentions in the Conclusion.
"""

from __future__ import annotations

import numpy as np

from .lifting import BaseLifting


class KoopmanPredictor:
    """Linear predictor in the lifted space."""

    def __init__(self, lifting: BaseLifting, n_input: int, ridge: float = 1e-2) -> None:
        self.lifting = lifting
        self.n_lift = lifting.n_lift
        self.n_input = n_input
        self.ridge = ridge
        self.A = np.eye(self.n_lift)
        self.B = np.zeros((self.n_lift, n_input))
        self._is_fit = False
        self._P: np.ndarray | None = None
        self._lambda_forget: float = 0.995

    # ---------------- offline identification ----------------

    def fit(self, X: np.ndarray, U: np.ndarray, X_next: np.ndarray) -> None:
        """Ridge LS fit of ``[A B]`` from triplets ``(x, u, x_next)``.

        The lifted-feature matrix ``Φ = [Z | U]`` is rescaled per-column
        before the normal-equation solve to prevent huge condition numbers
        when poly + RBF features have widely varying magnitudes. The
        rescaling is undone at the end so ``A`` and ``B`` apply directly
        to the raw lifted state.
        """
        Z = self.lifting.lift(X)
        Z_next = self.lifting.lift(X_next)
        Phi = np.concatenate([Z, U], axis=1)  # (N, n_lift + m)

        # Per-column standardisation: prevents poly-vs-RBF feature scale
        # imbalance from blowing up the condition number of Φ.T Φ.
        col_scale = np.maximum(np.std(Phi, axis=0), 1e-8)
        Phi_s = Phi / col_scale

        n = Phi_s.shape[1]
        G = Phi_s.T @ Phi_s + self.ridge * np.eye(n)
        Theta_s = np.linalg.solve(G, Phi_s.T @ Z_next).T  # (n_lift, n) in scaled coords
        # Undo the column scaling so [A B] applies to raw [Z | U].
        Theta = Theta_s / col_scale[None, :]
        self.A = Theta[:, : self.n_lift]
        self.B = Theta[:, self.n_lift :]
        self._is_fit = True

    # ---------------- prediction ----------------

    def predict_next(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.A @ z + self.B @ u

    def rollout(self, z0: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Multi-step lifted rollout from ``z0`` over input sequence ``U``."""
        N = U.shape[0]
        Z = np.zeros((N + 1, self.n_lift))
        Z[0] = z0
        for k in range(N):
            Z[k + 1] = self.A @ Z[k] + self.B @ U[k]
        return Z

    def residual(self, x: np.ndarray, u: np.ndarray, x_next: np.ndarray) -> np.ndarray:
        """Full lifted residual ``z_{k+1} − A z_k − B u_k``.

        This is useful as a numerical identification diagnostic, but it must
        not be used directly as the robust physical-state error in the CBF
        margin.  Lifted residuals contain artificial coordinates such as
        polynomial/RBF features whose units are not meters/radians/m/s.
        Use :meth:`physical_residual` or :meth:`residual_score` for conformal
        calibration and online safety margins.
        """
        z = self.lifting.lift(x)
        z_next = self.lifting.lift(x_next)
        return z_next - (self.A @ z + self.B @ u)

    def predict_state_next(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """One-step Koopman prediction projected back to physical state."""
        z = self.lifting.lift(x)
        z_next = self.A @ z + self.B @ u
        return self.lifting.reconstruct(z_next)

    def physical_residual(self, x: np.ndarray, u: np.ndarray, x_next: np.ndarray) -> np.ndarray:
        """Physical-state residual induced by the lifted prediction.

        Because all provided liftings prepend the identity state block, the
        first ``n_state`` entries of the lifted residual are exactly
        ``x_next - reconstruct(A lift(x) + B u)``.
        """
        pred = self.predict_state_next(x, u)
        return np.asarray(x_next, dtype=float) - pred

    def residual_score(
        self,
        x: np.ndarray,
        u: np.ndarray,
        x_next: np.ndarray,
        indices: list[int] | np.ndarray | None = None,
    ) -> float:
        """Scalar conformal residual score in physical units.

        ``indices`` should select only physically meaningful coordinates for
        the safety margin / trigger.  For the bicycle model the default config
        uses ``[py, psi, vx, vy, r]`` and excludes absolute ``px``.
        """
        r = self.physical_residual(x, u, x_next)
        if indices is not None:
            r = r[np.asarray(indices, dtype=int)]
        return float(np.linalg.norm(r))

    # ---------------- online RLS update (paper extension) ----------------

    def enable_online_update(self, lambda_forget: float = 0.995) -> None:
        d = self.n_lift + self.n_input
        self._P = 1e3 * np.eye(d)
        self._lambda_forget = lambda_forget

    def online_update(self, x: np.ndarray, u: np.ndarray, x_next: np.ndarray) -> None:
        """One RLS step refining ``[A B]`` with the latest transition.

        Standard exponentially-weighted RLS with forgetting factor
        ``self._lambda_forget``. Used when residual exceeds the
        ``residual_trigger_ratio * w̄_k`` threshold (config-controlled).
        """
        if self._P is None:
            self.enable_online_update(self._lambda_forget)
        z = self.lifting.lift(x)
        z_next = self.lifting.lift(x_next)
        phi = np.concatenate([z, u])
        Pp = self._P @ phi
        denom = self._lambda_forget + phi @ Pp
        K = Pp / denom
        Theta = np.concatenate([self.A, self.B], axis=1)
        innovation = z_next - Theta @ phi
        Theta = Theta + np.outer(innovation, K)
        self.A = Theta[:, : self.n_lift]
        self.B = Theta[:, self.n_lift :]
        self._P = (self._P - np.outer(K, Pp)) / self._lambda_forget

    @property
    def is_fit(self) -> bool:
        return self._is_fit
