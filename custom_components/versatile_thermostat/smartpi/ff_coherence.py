"""
Feed-Forward Coherence Check for Smart-PI.

Compares u_hold_emp with u_ff_ab to detect persistent model drift.

States:
  OK   — model and empirical estimate agree within tolerance.
  WARN — notable divergence; trim proceeds with caution.
  BAD  — large divergence; trim is frozen and an alert diagnostic is raised.

A BAD state is a symptom, not a failure. Possible causes:
  - sensor fault, external disturbance, model (a,b) drift.
"""
from __future__ import annotations

import logging

from .const import (
    FFCoherenceState,
    FF_COH_WARN_THRESHOLD,
    FF_COH_BAD_THRESHOLD,
    FF_COH_MIN_CONFIDENCE,
)

_LOGGER = logging.getLogger(__name__)


class FFCoherence:
    """Evaluates coherence between u_hold_emp and u_ff_ab."""

    def __init__(self) -> None:
        self.state: FFCoherenceState = FFCoherenceState.OK
        self.error: float = 0.0          # e_ff_coh = u_hold_emp - u_ff_ab
        self._bad_persist_count: int = 0

    def evaluate(
        self,
        u_hold_emp: float,
        u_ff_ab: float,
        hold_confidence: float,
    ) -> FFCoherenceState:
        """Evaluate coherence and update internal state.

        Returns the new FFCoherenceState.
        """
        # Not enough data to evaluate — stay OK
        if hold_confidence < FF_COH_MIN_CONFIDENCE:
            self.state = FFCoherenceState.OK
            self.error = 0.0
            self._bad_persist_count = 0
            return self.state

        e = u_hold_emp - u_ff_ab
        self.error = e
        abs_e = abs(e)

        if abs_e <= FF_COH_WARN_THRESHOLD:
            new_state = FFCoherenceState.OK
        elif abs_e <= FF_COH_BAD_THRESHOLD:
            new_state = FFCoherenceState.WARN
        else:
            new_state = FFCoherenceState.BAD

        if new_state == FFCoherenceState.BAD:
            self._bad_persist_count += 1
            if self._bad_persist_count == 1:
                _LOGGER.warning(
                    "FFCoherence: BAD state detected (e_ff_coh=%.3f). "
                    "Possible sensor fault or model drift.",
                    e,
                )
        else:
            self._bad_persist_count = 0

        self.state = new_state
        return self.state

    def reset(self) -> None:
        self.state = FFCoherenceState.OK
        self.error = 0.0
        self._bad_persist_count = 0
