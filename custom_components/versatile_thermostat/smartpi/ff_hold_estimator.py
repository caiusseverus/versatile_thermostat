"""
Feed-Forward Hold Power Estimator for Smart-PI.

Estimates the empirical hold power u_hold_emp by observing quasi-stationary
episodes where the system is near setpoint with low thermal drift.

u_hold_emp is NOT a second FF principal. Its roles are:
  1. Coherence check vs u_ff_ab.
  2. Slow trim correction input.
  3. Conditional fallback when a,b confidence is AB_BAD.
"""
from __future__ import annotations

import logging
import statistics
from typing import TYPE_CHECKING

from .const import (
    GovernanceRegime,
    FF_HOLD_LAMBDA,
    FF_HOLD_E_MAX_C,
    FF_HOLD_SLOPE_MAX_H,
    FF_HOLD_DU_MAX,
    FF_HOLD_MIN_CYCLES,
    FF_HOLD_CONF_DECAY_PER_REJECTION,
)

if TYPE_CHECKING:
    pass

_LOGGER = logging.getLogger(__name__)


def _quantile(data: list[float], q: float) -> float:
    """Compute quantile q (0..1) of data using linear interpolation."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_data[lo] * (1.0 - frac) + sorted_data[hi] * frac


def _is_cycle_admissible(
    *,
    error: float,
    slope_h: float | None,
    regime: GovernanceRegime,
    ff_reason: str,
    sat_state: str,
) -> tuple[bool, str]:
    """Check whether a single cycle is admissible for hold learning.

    Returns:
        (admissible, reject_reason)
    """
    if abs(error) > FF_HOLD_E_MAX_C:
        return False, f"error_too_large({error:.3f})"

    if slope_h is not None and abs(slope_h) > FF_HOLD_SLOPE_MAX_H:
        return False, f"slope_too_large({slope_h:.3f})"

    if regime in (
        GovernanceRegime.SATURATED,
        GovernanceRegime.PERTURBED,
        GovernanceRegime.DEGRADED,
        GovernanceRegime.WARMUP,
    ):
        return False, f"regime_{regime.value}"

    if sat_state != "NO_SAT":
        return False, f"sat_{sat_state}"

    return True, "ok"


class HoldEstimator:
    """Estimates empirical hold power u_hold_emp from quasi-stationary episodes."""

    def __init__(self) -> None:
        self.u_hold_emp: float = 0.0
        self.hold_confidence: float = 0.0
        self.u_hold_meas: float = 0.0

        self._cycle_buffer: list[float] = []   # u_applied per admissible cycle
        self._consecutive_count: int = 0       # consecutive admissible cycles
        self._frozen: bool = False
        self._freeze_reason: str = "none"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_cycle(
        self,
        *,
        u_applied: float,
        error: float,
        slope_h: float | None,
        regime: GovernanceRegime,
        ff_reason: str,
        sat_state: str,
    ) -> None:
        """Record one completed cycle. Resets streak on non-admissible cycle."""
        admissible, reason = _is_cycle_admissible(
            error=error,
            slope_h=slope_h,
            regime=regime,
            ff_reason=ff_reason,
            sat_state=sat_state,
        )
        if admissible:
            self._cycle_buffer.append(u_applied)
            self._consecutive_count += 1
        else:
            _LOGGER.debug("HoldEstimator: cycle rejected (%s), resetting streak", reason)
            self._cycle_buffer = []
            self._consecutive_count = 0
            # Sustained rejections signal the system is not in the quasi-stationary
            # state that u_hold_emp assumes. Erode confidence slowly so that a stale
            # estimate (after season change or hardware replacement) loses trust over time.
            self.hold_confidence = max(0.0, self.hold_confidence - FF_HOLD_CONF_DECAY_PER_REJECTION)

    def try_learn(self) -> bool:
        """Attempt to update u_hold_emp if the window is valid.

        Must be called at the end of each cycle (e.g. in on_cycle_completed).
        Returns True if u_hold_emp was updated.
        """
        if self._frozen:
            return False

        if self._consecutive_count < FF_HOLD_MIN_CYCLES:
            return False

        buf = self._cycle_buffer
        if len(buf) < FF_HOLD_MIN_CYCLES:
            return False

        # Command stability check: Q95 - Q05
        q05 = _quantile(buf, 0.05)
        q95 = _quantile(buf, 0.95)
        delta_u = q95 - q05
        if delta_u > FF_HOLD_DU_MAX:
            _LOGGER.debug(
                "HoldEstimator: window rejected — command spread %.3f > %.3f",
                delta_u,
                FF_HOLD_DU_MAX,
            )
            return False

        # Robust estimate: median of the window
        u_meas = statistics.median(buf)
        self.u_hold_meas = u_meas

        # EMA update
        self.u_hold_emp = (1.0 - FF_HOLD_LAMBDA) * self.u_hold_emp + FF_HOLD_LAMBDA * u_meas

        # Confidence: grows toward 1.0 with each successful episode
        self.hold_confidence = min(self.hold_confidence + 0.1, 1.0)

        _LOGGER.debug(
            "HoldEstimator: learned u_hold_emp=%.4f (meas=%.4f, conf=%.2f)",
            self.u_hold_emp,
            self.u_hold_meas,
            self.hold_confidence,
        )
        return True

    def freeze(self, reason: str) -> None:
        """Freeze learning (trim and hold updates blocked)."""
        if not self._frozen:
            _LOGGER.debug("HoldEstimator: frozen (%s)", reason)
        self._frozen = True
        self._freeze_reason = reason

    def unfreeze(self) -> None:
        """Unfreeze learning."""
        if self._frozen:
            _LOGGER.debug("HoldEstimator: unfrozen")
        self._frozen = False
        self._freeze_reason = "none"

    def reset(self) -> None:
        """Full reset (call only on complete SmartPI reset, not on PI reset)."""
        self._cycle_buffer = []
        self._consecutive_count = 0
        # u_hold_emp and hold_confidence are intentionally NOT reset here:
        # they survive PI resets and are only cleared on full factory reset.

    def full_reset(self) -> None:
        """Factory reset: clear all state including u_hold_emp."""
        self.u_hold_emp = 0.0
        self.hold_confidence = 0.0
        self.u_hold_meas = 0.0
        self._cycle_buffer = []
        self._consecutive_count = 0
        self._frozen = False
        self._freeze_reason = "none"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> dict:
        return {
            "u_hold_emp": self.u_hold_emp,
            "hold_confidence": self.hold_confidence,
            "u_hold_meas": self.u_hold_meas,
        }

    def load_state(self, state: dict) -> None:
        self.u_hold_emp = float(state.get("u_hold_emp", 0.0))
        self.hold_confidence = float(state.get("hold_confidence", 0.0))
        self.u_hold_meas = float(state.get("u_hold_meas", 0.0))
        # Buffer not persisted: start fresh after reboot
        self._cycle_buffer = []
        self._consecutive_count = 0
