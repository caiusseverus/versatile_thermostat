import pytest
from unittest.mock import MagicMock
from custom_components.versatile_thermostat.vtherm_hvac_mode import (
    VThermHvacMode_HEAT,
    VThermHvacMode_COOL,
)
from custom_components.versatile_thermostat.prop_algo_smartpi import SmartPI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_algo(integral: float = 0.5, i_mode: str = "I:RUN") -> SmartPI:
    """Return a pre-configured SmartPI instance for AW tracking tests."""
    hass = MagicMock()
    algo = SmartPI(
        hass=hass,
        cycle_min=10,
        minimal_activation_delay=0,
        minimal_deactivation_delay=0,
        name="test_aw",
        max_on_percent=1.0,
        deadband_c=0.0,
    )
    # PI gains
    algo.Ki = 0.05
    algo.Kp = 0.5
    # Controller state
    algo.ctl.integral = integral
    algo.ctl.last_i_mode = i_mode
    algo.ctl.u_ff = 0.0
    algo.ctl.last_error_p = 0.3   # u_model = 0 + 0.5*0.3 + 0.05*integral
    # Thermal context (Tin < SP → no thermal invariant active by default)
    algo._last_hvac_mode = VThermHvacMode_HEAT
    algo._last_current_temp = 20.0
    algo._last_target_temp = 21.0
    algo._in_deadband = False
    return algo


# u_model for default setup with integral=0.5:
# u_model = 0 + 0.5*0.3 + 0.05*0.5 = 0.15 + 0.025 = 0.175


# ---------------------------------------------------------------------------
# T1 — AW blocked in I:SKIP
# ---------------------------------------------------------------------------

def test_t1_aw_blocked_in_skip():
    """AW must not modify integral when last_i_mode is I:SKIP."""
    algo = make_algo(integral=0.5, i_mode="I:SKIP(SAT_LO)")
    algo._last_current_temp = 22.0  # Tin > SP
    before = algo.integral
    algo.update_realized_power(u_applied=0.0, dt_min=5.0, elapsed_ratio=1.0)
    assert algo.integral == before, "Integral must not change when I:SKIP is active"


# ---------------------------------------------------------------------------
# T2 — AW blocked in I:HOLD
# ---------------------------------------------------------------------------

def test_t2_aw_blocked_in_hold():
    """AW must not modify integral when last_i_mode is I:HOLD."""
    algo = make_algo(integral=0.5, i_mode="I:HOLD")
    before = algo.integral
    algo.update_realized_power(u_applied=0.05, dt_min=5.0, elapsed_ratio=1.0)
    assert algo.integral == before, "Integral must not change when I:HOLD is active"


# ---------------------------------------------------------------------------
# T3 — AW blocked in I:FREEZE
# ---------------------------------------------------------------------------

def test_t3_aw_blocked_in_freeze():
    """AW must not modify integral when last_i_mode is I:FREEZE."""
    algo = make_algo(integral=0.5, i_mode="I:FREEZE(deadband)")
    before = algo.integral
    algo.update_realized_power(u_applied=0.05, dt_min=5.0, elapsed_ratio=1.0)
    assert algo.integral == before, "Integral must not change when I:FREEZE is active"


# ---------------------------------------------------------------------------
# T4 — AW blocked in I:GUARD
# ---------------------------------------------------------------------------

def test_t4_aw_blocked_in_guard():
    """AW must not modify integral when last_i_mode is I:GUARD."""
    algo = make_algo(integral=0.5, i_mode="I:GUARD(freeze)")
    before = algo.integral
    algo.update_realized_power(u_applied=0.05, dt_min=5.0, elapsed_ratio=1.0)
    assert algo.integral == before, "Integral must not change when I:GUARD is active"


# ---------------------------------------------------------------------------
# T5 — AW blocked in I:CLAMP
# ---------------------------------------------------------------------------

def test_t5_aw_blocked_in_clamp():
    """AW must not modify integral when last_i_mode is I:CLAMP."""
    algo = make_algo(integral=0.5, i_mode="I:CLAMP(near_ovr)")
    before = algo.integral
    algo.update_realized_power(u_applied=0.05, dt_min=5.0, elapsed_ratio=1.0)
    assert algo.integral == before, "Integral must not change when I:CLAMP is active"


# ---------------------------------------------------------------------------
# T6 — AW discharge-only in HEAT when Tin > SP
# ---------------------------------------------------------------------------

def test_t6_aw_discharge_only_heat_overshoot():
    """When Tin > SP in HEAT mode, AW may only reduce (or keep) the integral."""
    algo = make_algo(integral=0.5, i_mode="I:RUN")
    algo._last_current_temp = 22.0   # Tin > SP=21.0 → thermal invariant: du <= 0
    # u_model = 0 + 0.5*0.3 + 0.05*0.5 = 0.175
    # val_normalized = 0.3 * 1.0 = 0.3 → du = 0.3 - 0.175 = +0.125 > 0
    # thermal invariant clamps du to min(0, du) = 0 → no integral change
    before = algo.integral
    algo.update_realized_power(u_applied=0.3, dt_min=5.0, elapsed_ratio=1.0)
    assert algo.integral <= before, "Integral must not increase when Tin > SP in HEAT"


# ---------------------------------------------------------------------------
# T7 — u_applied = 0 drives integral down
# ---------------------------------------------------------------------------

def test_t7_u_applied_zero_reduces_integral():
    """With u_applied=0, du = 0 - u_model < 0 → integral decreases."""
    algo = make_algo(integral=0.5, i_mode="I:RUN")
    # u_model = 0.175, val_normalized = 0 → du = -0.175 < 0 → dI < 0
    before = algo.integral
    algo.update_realized_power(u_applied=0.0, dt_min=5.0, elapsed_ratio=1.0)
    assert algo.integral < before, "Integral must decrease when u_applied=0 and u_model>0"


# ---------------------------------------------------------------------------
# T8 — Nominal case: u_applied > u_model, I:RUN, Tin < SP → AW free
# ---------------------------------------------------------------------------

def test_t8_nominal_above_model():
    """In nominal conditions (Tin < SP, I:RUN), AW applies freely when e_eff > u_model."""
    algo = make_algo(integral=0.5, i_mode="I:RUN")
    # u_model = 0.175, val_normalized = 0.4 → du = +0.225 → dI > 0
    before = algo.integral
    algo.update_realized_power(u_applied=0.4, dt_min=5.0, elapsed_ratio=1.0)
    assert algo.integral > before, "Integral must increase when e_eff > u_model in nominal conditions"


# ---------------------------------------------------------------------------
# T9 — COOL symmetry: Tin < SP → discharge only upward (integral must not decrease)
# ---------------------------------------------------------------------------

def test_t9_cool_symmetry_discharge_only():
    """In COOL mode, when Tin < SP, AW may only increase (or keep) the integral."""
    algo = make_algo(integral=-0.5, i_mode="I:RUN")
    algo._last_hvac_mode = VThermHvacMode_COOL
    algo._last_current_temp = 18.0   # Tin < SP=21.0
    # u_model = 0 + 0.5*0.3 + 0.05*(-0.5) = 0.15 - 0.025 = 0.125
    # val_normalized = 0.3 → du = 0.3 - 0.125 = +0.175 > 0
    # COOL + Tin < SP → du = max(0, du) = +0.175 → dI > 0 → integral increases
    before = algo.integral
    algo.update_realized_power(u_applied=0.3, dt_min=5.0, elapsed_ratio=1.0)
    assert algo.integral >= before, "Integral must not decrease when Tin < SP in COOL (discharge only upwards)"
