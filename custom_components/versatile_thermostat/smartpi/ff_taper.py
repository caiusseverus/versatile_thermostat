"""
Feed-Forward Regime Taper for Smart-PI.

Provides continuous FF modulation (alpha) based on the current governance regime.
The taper is a secondary mechanism — it reduces, not replaces, the FF principal.

Authority:
  u_ff_eff = alpha * u_ff_base
  alpha in [1 - FF_TAPER_RHO_MAX, 1.0]  =>  [0.75, 1.0]

"""
from __future__ import annotations

import logging

from .const import (
    GovernanceRegime,
    FF_TAPER_RHO_MAX,
    clamp,
    DEADBAND_BELOW_C,
    DEADBAND_ABOVE_C,
)

_LOGGER = logging.getLogger(__name__)

# Floor for the FF taper (1 - rho_max)
_ALPHA_FLOOR = 1.0 - FF_TAPER_RHO_MAX


class FFTaper:
    """Computes the regime-based FF modulation factor alpha."""

    def __init__(self) -> None:
        self.alpha: float = 1.0         # Current modulation factor
        self._prev_alpha: float = 1.0   # Previous cycle's alpha (for bumpless reference)

    def compute_alpha(
        self,
        regime: GovernanceRegime,
        error: float,
        near_band_below_deg: float,
        near_band_above_deg: float,
    ) -> float:
        """Compute and return the FF modulation factor for this cycle.

        Args:
            regime: Current governance regime.
            error: SP - T_in (positive = below setpoint, negative = above setpoint).
            near_band_below_deg: Near-band width below setpoint (°C).
            near_band_above_deg: Near-band width above setpoint (°C).

        Returns:
            alpha in [_ALPHA_FLOOR, 1.0].
        """
        self._prev_alpha = self.alpha

        if regime == GovernanceRegime.SATURATED:
            # Spec 11.8: preserve last alpha — do not add taper on saturation
            return self.alpha

        if regime in (
            GovernanceRegime.EXCITED_STABLE,
            GovernanceRegime.WARMUP,
            GovernanceRegime.PERTURBED,
            GovernanceRegime.DEGRADED,
        ):
            self.alpha = 1.0
            return self.alpha

        if regime in (GovernanceRegime.DEAD_BAND, GovernanceRegime.HOLD):
            self.alpha = _ALPHA_FLOOR
            return self.alpha

        if regime == GovernanceRegime.NEAR_BAND:
            # Continuous interpolation: alpha decreases linearly from 1.0 (at near-band
            # boundary) to _ALPHA_FLOOR (at deadband boundary).
            # Use asymmetric near-band/deadband widths based on sign of error.
            abs_e = abs(error)
            if error >= 0.0:
                # Below setpoint: near_band_below_deg and DEADBAND_BELOW_C
                nb = near_band_below_deg
                db = DEADBAND_BELOW_C
            else:
                # Above setpoint: near_band_above_deg and DEADBAND_ABOVE_C
                nb = near_band_above_deg
                db = DEADBAND_ABOVE_C

            span = nb - db
            if span <= 0.0:
                # Degenerate config: use floor
                self.alpha = _ALPHA_FLOOR
                return self.alpha

            # progress: 0.0 at near-band boundary, 1.0 at deadband boundary
            progress = clamp((nb - abs_e) / span, 0.0, 1.0)
            self.alpha = 1.0 - FF_TAPER_RHO_MAX * progress
            return self.alpha

        # Unknown regime: no taper
        self.alpha = 1.0
        return self.alpha

    def apply(
        self,
        u_ff_base: float,
        regime: GovernanceRegime,
        error: float,
        near_band_below_deg: float,
        near_band_above_deg: float,
    ) -> tuple[float, float]:
        """Compute alpha and return (u_ff_eff, alpha).

        Args:
            u_ff_base: FF value after trim (u_ff_ab + u_ff_trim, clamped).
            regime: Current governance regime.
            error: SP - T_in.
            near_band_below_deg: Near-band width below setpoint (°C).
            near_band_above_deg: Near-band width above setpoint (°C).

        Returns:
            (u_ff_eff, alpha)
        """
        alpha = self.compute_alpha(
            regime=regime,
            error=error,
            near_band_below_deg=near_band_below_deg,
            near_band_above_deg=near_band_above_deg,
        )
        u_ff_eff = alpha * u_ff_base
        return u_ff_eff, alpha

    def reset(self) -> None:
        self.alpha = 1.0
        self._prev_alpha = 1.0
