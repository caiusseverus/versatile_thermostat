"""
Feed-Forward Orchestrator for Smart-PI Algorithm.

Computes the full FF signal chain:
  ff_raw -> u_ff_ab -> u_ff_base (with trim) -> u_ff_eff (with taper) -> hard gate override

Source of truth for runtime FF is u_ff_eff, not ff_raw.

Chain:
  u_ff_eff + u_pi -> u_cmd -> u_limited -> u_applied -> e_eff
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .const import (
    GovernanceRegime,
    clamp,
)
from .ff_trim import FFTrim
from .ff_taper import FFTaper

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FFResult:
    """Complete FF computation result for one cycle."""

    ff_raw: float          # Raw FF signal before trim/taper (diagnostic)
    u_ff_ab: float         # FF principal derived from a,b (or fallback value)
    u_ff_trim: float       # Slow trim correction applied
    u_ff_base: float       # u_ff_ab + u_ff_trim, clamped to [0, 1]
    u_ff_eff: float        # FF effectively injected into the command (after taper + hard gate)
    ff_reason: str         # Diagnostic reason string for the current FF state
    ff_taper_alpha: float  # Taper modulation factor applied


def compute_ff(
    *,
    k_ff: float,
    target_temp_filt: float,
    ext_temp: float | None,
    warmup_scale: float,
    trim: FFTrim,
    taper: FFTaper,
    regime: GovernanceRegime,
    error: float,
    near_band_below_deg: float,
    near_band_above_deg: float,
    ab_fallback: float | None,
) -> FFResult:
    """Compute the full FF signal for one cycle.

    Args:
        k_ff: FF gain = b / a (loss-to-heating ratio).
        target_temp_filt: Filtered setpoint (°C).
        ext_temp: Outdoor temperature (°C), or None if unavailable.
        warmup_scale: Combined warmup scaling factor (learn_scale * time_scale * reliable_cap).
        trim: FFTrim instance (provides u_ff_trim and compute_ff_base).
        taper: FFTaper instance (provides regime-based alpha modulation).
        regime: Current governance regime.
        error: SP - T_in (positive = below setpoint, negative = above setpoint).
        near_band_below_deg: Near-band width below setpoint (°C).
        near_band_above_deg: Near-band width above setpoint (°C).
        ab_fallback: Fallback FF value from ABConfidence (or None = use u_ff_ab normally).

    Returns:
        FFResult with all FF signal components.
    """
    # --- Step 1: Compute u_ff_ab ---
    if ab_fallback is not None:
        # AB_BAD fallback: use empirical hold or 0
        u_ff_ab = ab_fallback
        ff_reason_prefix = "ff_ab_fallback"
    elif ext_temp is not None:
        u_ff_ab = clamp(k_ff * (target_temp_filt - ext_temp), 0.0, 1.0) * warmup_scale
        ff_reason_prefix = "ff_none"
    else:
        u_ff_ab = 0.0
        ff_reason_prefix = "ff_no_ext_temp"

    ff_raw = u_ff_ab  # Raw FF before trim/taper

    # --- Step 2: Apply trim -> u_ff_base ---
    u_ff_base = trim.compute_ff_base(u_ff_ab)

    # --- Step 3: Apply taper -> u_ff_eff ---
    u_ff_eff, alpha = taper.apply(
        u_ff_base=u_ff_base,
        regime=regime,
        error=error,
        near_band_below_deg=near_band_below_deg,
        near_band_above_deg=near_band_above_deg,
    )
    ff_reason = ff_reason_prefix

    return FFResult(
        ff_raw=ff_raw,
        u_ff_ab=u_ff_ab,
        u_ff_trim=trim.u_ff_trim,
        u_ff_base=u_ff_base,
        u_ff_eff=u_ff_eff,
        ff_reason=ff_reason,
        ff_taper_alpha=alpha,
    )
