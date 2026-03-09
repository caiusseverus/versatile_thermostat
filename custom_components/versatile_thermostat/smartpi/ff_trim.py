"""
Feed-Forward Slow Trim for Smart-PI.

The trim corrects a slow, persistent bias in u_ff_ab without replacing the FF principal.

Hierarchy:
  u_ff_base = clamp(u_ff_ab + u_ff_trim, 0, 1)

Authority:
  |u_ff_trim| <= rho_trim * max(u_ff_ab, FF_TRIM_EPSILON)

The trim is frozen under several conditions (see freeze()).
"""
from __future__ import annotations

import logging

from .const import (
    clamp,
    FF_TRIM_RHO,
    FF_TRIM_LAMBDA,
    FF_TRIM_EPSILON,
)

_LOGGER = logging.getLogger(__name__)


class FFTrim:
    """Slow trim correction on the FF principal u_ff_ab."""

    def __init__(self) -> None:
        self.u_ff_trim: float = 0.0
        self._frozen: bool = False
        self._freeze_reason: str = "none"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, delta_hold: float, u_ff_ab: float) -> None:
        """Update the trim using the hold error.

        Args:
            delta_hold: u_hold_meas - u_ff_ab (observed bias).
            u_ff_ab: Current FF principal value (used to compute authority budget).
        """
        if self._frozen:
            return

        authority = FF_TRIM_RHO * max(u_ff_ab, FF_TRIM_EPSILON)
        new_trim = (1.0 - FF_TRIM_LAMBDA) * self.u_ff_trim + FF_TRIM_LAMBDA * delta_hold
        self.u_ff_trim = clamp(new_trim, -authority, authority)

        _LOGGER.debug(
            "FFTrim: updated u_ff_trim=%.4f (delta_hold=%.4f, authority=%.4f)",
            self.u_ff_trim,
            delta_hold,
            authority,
        )

    def compute_ff_base(self, u_ff_ab: float) -> float:
        """Return u_ff_base = clamp(u_ff_ab + u_ff_trim, 0, 1)."""
        return clamp(u_ff_ab + self.u_ff_trim, 0.0, 1.0)

    def freeze(self, reason: str) -> None:
        """Freeze trim updates."""
        if not self._frozen:
            _LOGGER.debug("FFTrim: frozen (%s)", reason)
        self._frozen = True
        self._freeze_reason = reason

    def unfreeze(self) -> None:
        """Unfreeze trim updates."""
        if self._frozen:
            _LOGGER.debug("FFTrim: unfrozen")
        self._frozen = False
        self._freeze_reason = "none"

    def reset(self) -> None:
        """Full reset of trim state."""
        self.u_ff_trim = 0.0
        self._frozen = False
        self._freeze_reason = "none"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> dict:
        return {"u_ff_trim": self.u_ff_trim}

    def load_state(self, state: dict) -> None:
        self.u_ff_trim = float(state.get("u_ff_trim", 0.0))
