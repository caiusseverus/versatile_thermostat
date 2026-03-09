"""Tests for the Feed-Forward orchestrator (smartpi/feedforward.py)."""

import pytest
from custom_components.versatile_thermostat.smartpi.feedforward import (
    compute_ff,
    FFResult,
)
from custom_components.versatile_thermostat.smartpi.ff_trim import FFTrim
from custom_components.versatile_thermostat.smartpi.ff_taper import FFTaper
from custom_components.versatile_thermostat.smartpi.const import GovernanceRegime


def _call(
    *,
    error: float,
    near_band_above_deg: float = 0.3,
    near_band_below_deg: float = 0.4,
    k_ff: float = 0.20,
    ext_temp: float | None = 5.0,
    target_temp_filt: float = 21.0,
    warmup_scale: float = 1.0,
    regime: GovernanceRegime = GovernanceRegime.EXCITED_STABLE,
    ab_fallback: float | None = None,
) -> FFResult:
    """Helper: call compute_ff with sensible defaults for behavioural tests."""
    return compute_ff(
        k_ff=k_ff,
        target_temp_filt=target_temp_filt,
        ext_temp=ext_temp,
        warmup_scale=warmup_scale,
        trim=FFTrim(),
        taper=FFTaper(),
        regime=regime,
        error=error,
        near_band_below_deg=near_band_below_deg,
        near_band_above_deg=near_band_above_deg,
        ab_fallback=ab_fallback,
    )


# ================================================================
# Hard Gate Tests (legacy behaviour preserved)
# ================================================================


class TestHardGate:
    """error < -near_band_above_deg => u_ff_eff = 0, reason = ff_cut_above_setpoint."""

    def test_above_setpoint_cuts_ff(self):
        result = _call(error=-0.5, near_band_above_deg=0.3)
        assert result.u_ff_eff == 0.0
        assert result.ff_reason == "ff_cut_above_setpoint"

    def test_at_setpoint_exact_no_cut(self):
        """error == 0 should NOT trigger hard gate."""
        result = _call(error=0.0, near_band_above_deg=0.3)
        assert result.u_ff_eff > 0.0
        assert result.ff_reason != "ff_cut_above_setpoint"

    def test_in_near_band_above_no_cut(self):
        """error in (-near_band_above_deg, 0) should NOT trigger hard gate."""
        result = _call(error=-0.1, near_band_above_deg=0.3)
        assert result.u_ff_eff > 0.0
        assert result.ff_reason != "ff_cut_above_setpoint"

    def test_hard_gate_with_zero_raw(self):
        """Hard gate still fires even when k_ff=0 (zero FF)."""
        result = _call(error=-1.0, near_band_above_deg=0.3, k_ff=0.0)
        assert result.u_ff_eff == 0.0
        assert result.ff_reason == "ff_cut_above_setpoint"

    def test_hard_gate_exact_boundary(self):
        """error == -near_band_above_deg: boundary is exclusive (< not <=)."""
        # error == -0.3, near_band_above_deg == 0.3 -> NOT cut
        result = _call(error=-0.3, near_band_above_deg=0.3)
        assert result.ff_reason != "ff_cut_above_setpoint"


# ================================================================
# Passthrough Tests
# ================================================================


class TestPassthrough:
    """When no gate fires, u_ff_eff reflects u_ff_base * alpha (alpha=1 in EXCITED_STABLE)."""

    def test_positive_error_passes_through(self):
        result = _call(error=1.0, k_ff=0.2, target_temp_filt=21.0, ext_temp=5.0)
        # u_ff_ab = clamp(0.2 * (21 - 5), 0, 1) * 1.0 = clamp(3.2, 0, 1) = 1.0
        assert result.u_ff_eff == pytest.approx(1.0)
        assert result.ff_reason == "ff_none"

    def test_small_positive_error_passes_through(self):
        result = _call(error=0.05, k_ff=0.05, target_temp_filt=21.0, ext_temp=20.0)
        # u_ff_ab = clamp(0.05 * 1.0, 0, 1) = 0.05
        assert result.u_ff_eff == pytest.approx(0.05)
        assert result.ff_reason == "ff_none"

    def test_no_ext_temp_gives_zero(self):
        result = _call(error=1.0, ext_temp=None)
        assert result.u_ff_eff == 0.0
        assert result.ff_reason == "ff_no_ext_temp"

    def test_zero_warmup_scale_gives_zero(self):
        result = _call(error=1.0, warmup_scale=0.0)
        assert result.u_ff_eff == 0.0


# ================================================================
# FFResult structure
# ================================================================


class TestFFResult:
    """Verify all FFResult fields are populated correctly."""

    def test_result_fields_present(self):
        result = _call(error=0.5, k_ff=0.1, target_temp_filt=21.0, ext_temp=15.0)
        assert isinstance(result, FFResult)
        assert hasattr(result, "ff_raw")
        assert hasattr(result, "u_ff_ab")
        assert hasattr(result, "u_ff_trim")
        assert hasattr(result, "u_ff_base")
        assert hasattr(result, "u_ff_eff")
        assert hasattr(result, "ff_reason")
        assert hasattr(result, "ff_taper_alpha")

    def test_no_trim_base_equals_ab(self):
        """With default (zero) trim, u_ff_base == u_ff_ab."""
        result = _call(error=1.0, k_ff=0.1, target_temp_filt=21.0, ext_temp=15.0)
        assert result.u_ff_base == pytest.approx(result.u_ff_ab)

    def test_excited_stable_alpha_one(self):
        """EXCITED_STABLE regime -> taper alpha == 1.0."""
        result = _call(error=1.0, regime=GovernanceRegime.EXCITED_STABLE)
        assert result.ff_taper_alpha == pytest.approx(1.0)

    def test_dead_band_alpha_floor(self):
        """DEAD_BAND regime -> taper alpha == 0.75."""
        result = _call(error=0.02, regime=GovernanceRegime.DEAD_BAND)
        assert result.ff_taper_alpha == pytest.approx(0.75)

    def test_ab_fallback_overrides_ab(self):
        """When ab_fallback is provided, u_ff_ab == fallback value."""
        result = _call(error=1.0, ab_fallback=0.30)
        assert result.u_ff_ab == pytest.approx(0.30)
        assert result.ff_reason == "ff_ab_fallback"


# ================================================================
# Invariant: 0 <= u_ff_eff <= 1
# ================================================================


class TestInvariants:
    """u_ff_eff must always be in [0, 1]."""

    @pytest.mark.parametrize("error", [-2.0, -0.5, -0.3, 0.0, 0.05, 0.5, 2.0])
    @pytest.mark.parametrize("k_ff", [0.0, 0.05, 0.20, 0.50, 2.0])
    def test_u_ff_eff_in_range(self, error, k_ff):
        result = _call(error=error, k_ff=k_ff)
        assert 0.0 <= result.u_ff_eff <= 1.0

    @pytest.mark.parametrize("error", [-2.0, -0.5, -0.3, 0.0, 0.05, 0.5, 2.0])
    def test_hard_gate_always_zero_when_beyond_band(self, error):
        if error < -0.3:
            result = _call(error=error, near_band_above_deg=0.3)
            assert result.u_ff_eff == 0.0
