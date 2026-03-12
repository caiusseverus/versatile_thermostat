"""
Thermal Twin 1R1C — Digital twin based on (a, b) model + deadtime.
Spec v4.5 compliant.

Diagnostics-only: does NOT modify the control command.

Model: dT/dt = a·u(t-L) - b·(T - Text) + d(t)
  - a: heating efficacy (°C·min⁻¹ per unit of u)
  - b: loss coefficient (min⁻¹), tau = 1/b
  - L: dead time (seconds)
  - d(t): external perturbation (solar, occupants, ...), estimated online

Architecture v4 — two distinct predictions:
  - T_pred_pure (pure model, no d_hat): drives T_hat state and d_raw estimation
  - T_pred (corrected model, includes d_hat_ema): exposed diagnostics, RMSE, CUSUM

v4.1 fixes:
  - CUSUM reset after detection (no simultaneous gain+loss)
  - d_hat_ema physical clamp (|d_hat| ≤ d_hat_max)
  - d_hat slow relaxation toward 0 when u < u_min (avoids stale solar bias)
  - rmse_pure exposed (uncompensated model quality indicator)
  - u_buffer without maxlen (fixes infinite loop on deadtime increase)
  - Input validation on dt_s, gamma
  - Buffer normalization after load_state()

v4.2 robustness improvements:
  - NaN/Inf guard on tin_meas input and computed state (prevents permanent corruption)
  - NaN/Inf validation in load_state() (protects against corrupted persistence)
  - Default gamma=0.1 (observer enabled by default)
  - Luenberger nudge clamped to MAX_NUDGE_C per step (prevents wild T_hat jumps)
  - d_raw clamped to D_RAW_MAX before EMA (prevents spike contamination of d_hat)
  - Auto-reset on sustained divergence (RMSE > threshold for DIVERGE_RESET_THRESHOLD steps)
  - Incremental RMSE computation O(1) per step instead of O(n)
  - Innovation bias EMA tracking for slow drift detection
  - T_steady_max (at u=1) and setpoint_reachable_max diagnostics
  - Fix: update_with_eta uses correct deadtime based on power direction (heat/cool)

v4.3 improvements:
  - NaN/Inf guard on text_meas in _resolve_text (prevents NaN propagation to model)
  - NaN/Inf validation in reset() (prevents corrupted initialization)
  - last_tin_meas reset to None in load_state() (prevents stale dTin_dt after HA restart)
  - NaN guard on tin0/target in eta_best_case() (prevents indeterminate results)
  - Periodic RMSE recalibration every 1000 steps (prevents float drift accumulation)
  - Exponential backoff on auto-reset cooldown (prevents reset loops on structural divergence)
  - T_hat_error diagnostic (absolute gap between twin and measurement)
  - warming_up flag (indicates unreliable diagnostics during buffer fill)
  - reset_count diagnostic (tracks cumulative auto-resets for consumer awareness)

v4.4 T_steady audit & reliability:
  - Fix: T_steady_max_valid now returned in diagnostics (was computed but missing from dict)
  - Add u_eff to diagnostics (deadtime-delayed command that actually drives the plant now)
  - Add T_steady_immediate = Text + (a·u_eff + d)/b (physical equilibrium at current heating)
  - Add d_hat_fresh flag (_d_hat_active_steps counter, True when u_eff >= u_min for >= 3 steps)
  - Add T_steady_passive = Text + d/b (equilibrium at u=0, natural floor temperature)
  - Add T_steady_reliable composite flag (T_steady_valid AND model_reliable AND d_hat_fresh)

v4.5 post-reboot reliability:
  - save_state() now saves a Unix timestamp ("saved_at": time.time())
  - load_state() computes downtime and restores state adaptively in 3 levels:
      < 5 min  (REBOOT_SHORT_S):  full restore, unchanged behaviour
      5–60 min (REBOOT_MEDIUM_S): clear u_buffer (fill 0s), clear innovation buffers,
                                   reset CUSUM and _d_hat_active_steps, decay d_hat_ema
                                   proportionally to downtime
      > 1 h:                       all of the above + stronger d_hat decay +
                                   _needs_resync=True, _diverge_count reset
  - First step() after a long restart: if T_hat is > REBOOT_RESYNC_THRESHOLD from
    tin_meas, immediately re-synchronise T_hat and clear d_hat_ema
  - step() exposes "cold_start" (bool) and "downtime_s" (float|None) diagnostics

Discretisation: exact exponential (ZOH, unconditionally stable).
Observer: simplified Luenberger (post-prediction nudging on pure model).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from math import exp, isfinite, log, sqrt

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp(v: float, lo: float, hi: float) -> float:
    """Clamp v between lo and hi."""
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _safe_float(val, default: float = 0.0) -> float:
    """Convert to float, returning default if NaN/Inf/None."""
    if val is None:
        return default
    try:
        f = float(val)
        return f if isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _resolve_text(
    text_meas: float | None,
    policy: str,
    last_text: float | None,
) -> float | None:
    """Resolve Text according to policy. Returns None if unavailable.

    v4.3: NaN/Inf text_meas is treated as missing (falls back to hold_last).
    """
    if text_meas is not None and isfinite(text_meas):
        return float(text_meas)
    if policy == "hold_last" and last_text is not None:
        return last_text
    return None


# ---------------------------------------------------------------------------
# Physical bounds for T_steady guard (§15.4.2)
# ---------------------------------------------------------------------------
T_STEADY_MIN = -50.0
T_STEADY_MAX = 80.0

# ---------------------------------------------------------------------------
# Robustness constants (v4.2)
# ---------------------------------------------------------------------------
MAX_NUDGE_C = 2.0              # °C max Luenberger nudge per step
D_RAW_MAX = 0.5                # °C/min max plausible perturbation rate for d_raw
DIVERGE_RESET_THRESHOLD = 60   # steps before auto-reset (~1h @ 60s/step)
DIVERGE_RMSE_THRESHOLD = 1.5   # °C RMSE above which divergence is counted
BIAS_EMA_ALPHA = 0.02          # Innovation bias EMA learning rate
BIAS_WARN_THRESHOLD = 0.3      # °C bias threshold for warning

# ---------------------------------------------------------------------------
# Post-reboot reliability constants (v4.5)
# ---------------------------------------------------------------------------
REBOOT_SHORT_S  = 5 * 60       # < 5 min  : full restore (current behaviour)
REBOOT_MEDIUM_S = 60 * 60      # 5–60 min : partial clean (buffers, CUSUM, d_hat decay)
# > REBOOT_MEDIUM_S            : full resync (also _diverge_count reset, _needs_resync)
REBOOT_RESYNC_THRESHOLD = 2.0  # °C : T_hat vs tin_meas gap triggering immediate resync


# ---------------------------------------------------------------------------
# ThermalTwin1R1C
# ---------------------------------------------------------------------------


class ThermalTwin1R1C:
    """Digital twin 1R1C with dead-time buffer and Luenberger nudging.

    Spec v4.5 compliant — dual prediction architecture:
      - Pure model for state estimation (T_hat) and d_raw extraction
      - Corrected model (with d_hat_ema) for exposed diagnostics

    v4.4 T_steady: NaN-proof inputs/state/text, clamped nudge & d_raw,
    auto-reset with backoff, incremental RMSE with recalibration, bias tracking,
    T_steady_max_valid returned, T_steady_immediate (at u_eff), T_steady_passive
    (at u=0), d_hat_fresh flag, T_steady_reliable composite.

    v4.5 post-reboot: save_state() stores a Unix timestamp; load_state()
    computes downtime and applies 3-level adaptive restoration so that stale
    signals (u_buffer, CUSUM, innovation buffers, d_hat, T_hat) do not
    corrupt the twin after a long HA shutdown.

    The u_buffer has NO maxlen — size is controlled exclusively by code
    in step() and _resize_buffer(). This prevents the infinite loop that
    would occur if maxlen blocked growth when deadtime increases.
    """

    def __init__(
        self,
        dt_s: int = 60,
        gamma: float = 0.1,
        text_policy: str = "hold_last",
        u_clip: tuple[float, float] = (0.0, 1.0),
        max_deadtime_s: int = 4 * 3600,
        # --- Advanced diagnostics (§15) ---
        rmse_window: int = 30,
        rmse_reliable_threshold: float = 0.5,
        cusum_delta: float = 0.15,
        cusum_threshold: float = 2.0,
        N_sat: int = 20,
        dT_sat_threshold: float = 0.005,
        eps_sat: float = 0.3,
        d_hat_alpha: float = 0.05,
        d_hat_u_min: float = 0.05,  # v4 §15.6.4: freeze d_hat when u < this
        d_hat_max: float = 0.1,     # v4.1 §15.6.5: clamp |d_hat_ema| (°C/min)
        d_hat_relax: float = 0.005, # v4.1: slow relaxation rate toward 0 when frozen
    ) -> None:
        # ---- Input validation ----
        if dt_s <= 0:
            raise ValueError(f"dt_s must be > 0, got {dt_s}")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(
                f"gamma must be in [0, 1] for observer stability, got {gamma}"
            )

        self.dt_s: int = int(dt_s)
        self.dt_min: float = self.dt_s / 60.0
        self.gamma: float = float(gamma)
        self.text_policy: str = text_policy
        self.u_min: float = float(u_clip[0])
        self.u_max: float = float(u_clip[1])
        self.max_deadtime_s: int = int(max_deadtime_s)

        # Advanced diagnostics config
        self.rmse_window: int = rmse_window
        self.rmse_reliable_threshold: float = rmse_reliable_threshold
        self.cusum_delta: float = cusum_delta
        self.cusum_threshold: float = cusum_threshold
        self.N_sat: int = N_sat
        self.dT_sat_threshold: float = dT_sat_threshold
        self.eps_sat: float = eps_sat
        self.d_hat_alpha: float = d_hat_alpha
        self.d_hat_u_min: float = d_hat_u_min
        self.d_hat_max: float = d_hat_max
        self.d_hat_relax: float = d_hat_relax

        # Core state — u_buffer has NO maxlen (critical: prevents infinite loop)
        self.T_hat: float | None = None
        self.T_pred: float | None = None
        self.last_text: float | None = None
        self.dead_steps: int = 0
        self.u_buffer: deque[float] = deque([0.0])

        # Advanced diagnostics state
        self.innovation_buffer: deque[float] = deque(maxlen=rmse_window)
        self.innovation_pure_buffer: deque[float] = deque(maxlen=rmse_window)
        self.cusum_pos: float = 0.0
        self.cusum_neg: float = 0.0
        self.sat_count: int = 0
        self.last_tin_meas: float | None = None
        self.d_hat_ema: float = 0.0

        # v4.2 robustness state
        self._diverge_count: int = 0
        self._sum_sq: float = 0.0
        self._sum_sq_pure: float = 0.0
        self._innovation_bias_ema: float = 0.0

        # v4.3 state
        self._step_count: int = 0   # for periodic RMSE recalibration
        self._reset_count: int = 0  # for auto-reset cooldown backoff

        # v4.4 T_steady reliability state
        self._d_hat_active_steps: int = 0  # consecutive steps with u_eff >= d_hat_u_min

        # v4.5 post-reboot state
        self._downtime_s: float = 0.0    # downtime detected at load_state(); reset after 1st step
        self._needs_resync: bool = False  # True → re-sync T_hat at next step() if far from meas

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        tin_init: float,
        text_init: float | None = None,
        u_init: float = 0.0,
    ) -> None:
        """Initialise or warm-restart the twin.

        v4.3: validates tin_init and text_init against NaN/Inf.
        """
        if not isfinite(tin_init):
            raise ValueError(f"reset: tin_init must be finite, got {tin_init}")
        self.T_hat = float(tin_init)
        self.T_pred = None
        if text_init is not None:
            if not isfinite(text_init):
                _LOGGER.warning("ThermalTwin reset: text_init NaN/Inf, ignored")
            else:
                self.last_text = float(text_init)
        u_val = _clamp(float(u_init), self.u_min, self.u_max)
        n = self.dead_steps + 1
        # NO maxlen — size controlled by step() and _resize_buffer()
        self.u_buffer = deque([u_val] * n)
        # Reset advanced diagnostics state
        self.innovation_buffer.clear()
        self.innovation_pure_buffer.clear()
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.sat_count = 0
        self.last_tin_meas = None
        self.d_hat_ema = 0.0
        # v4.2 robustness state
        self._diverge_count = 0
        self._sum_sq = 0.0
        self._sum_sq_pure = 0.0
        self._innovation_bias_ema = 0.0
        # v4.3: reset step count but NOT _reset_count (persists across resets)
        self._step_count = 0
        # v4.4: reset d_hat freshness counter
        self._d_hat_active_steps = 0
        # v4.5: reset reboot flags
        self._downtime_s = 0.0
        self._needs_resync = False

    # ------------------------------------------------------------------
    # Dead-time buffer management
    # ------------------------------------------------------------------

    def _resize_buffer(self, new_dead_steps: int) -> None:
        """Resize u_buffer to match new_dead_steps, preserving recent history."""
        new_n = new_dead_steps + 1
        old_n = len(self.u_buffer)

        if new_n > old_n:
            pad_val = self.u_buffer[0] if self.u_buffer else 0.0
            for _ in range(new_n - old_n):
                self.u_buffer.appendleft(pad_val)
        elif new_n < old_n:
            while len(self.u_buffer) > new_n:
                self.u_buffer.popleft()

        self.dead_steps = new_dead_steps

    def _normalize_buffer(self) -> None:
        """Ensure u_buffer invariant: len == dead_steps + 1.

        Called after load_state() to guarantee consistency.
        """
        n = self.dead_steps + 1
        while len(self.u_buffer) < n:
            pad = self.u_buffer[0] if self.u_buffer else 0.0
            self.u_buffer.appendleft(pad)
        while len(self.u_buffer) > n:
            self.u_buffer.popleft()

    # ------------------------------------------------------------------
    # Step (one tick of simulation) — v4.1 dual prediction architecture
    # ------------------------------------------------------------------

    def step(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        self,
        tin_meas: float,
        text_meas: float | None,
        a: float,
        b: float,
        u_now: float,
        deadtime_s: float,
        sp: float | None = None,
    ) -> dict:
        """Simulate one time-step and return diagnostics dict."""
        if self.T_hat is None:
            return {"status": "not_initialized"}

        # ---- v4.2: NaN/Inf guard on tin_meas ----
        if not isfinite(tin_meas):
            _LOGGER.warning("ThermalTwin: tin_meas is NaN/Inf, skipping step")
            return {"status": "invalid_tin_meas", "T_hat_prev": self.T_hat}

        # ---- v4.5: first-step resync after long reboot ----
        if self._needs_resync:
            if abs(float(tin_meas) - self.T_hat) > REBOOT_RESYNC_THRESHOLD:
                _LOGGER.info(
                    "ThermalTwin v4.5: post-reboot resync T_hat %.2f → %.2f "
                    "(downtime %.0f s)",
                    self.T_hat, float(tin_meas), self._downtime_s,
                )
                self.T_hat = float(tin_meas)
                self.d_hat_ema = 0.0
            self._needs_resync = False

        # ---- Text policy ----
        text_used = _resolve_text(text_meas, self.text_policy, self.last_text)
        if text_meas is not None and isfinite(text_meas):
            self.last_text = float(text_meas)

        if text_used is None:
            return {"status": "missing_text", "T_hat_prev": self.T_hat}

        # ---- Params validation ----
        if not isfinite(a) or not isfinite(b) or b <= 0 or a < 0:
            return {
                "status": "invalid_params",
                "T_hat_prev": self.T_hat,
                "Text_used": text_used,
            }

        # ---- Deadtime buffer sizing ----
        dead_steps = int(round(
            _clamp(deadtime_s, 0, self.max_deadtime_s) / self.dt_s
        ))
        if dead_steps != self.dead_steps:
            self._resize_buffer(dead_steps)

        # ---- Push u and get u_eff ----
        n = self.dead_steps + 1
        u = _clamp(float(u_now), self.u_min, self.u_max)
        self.u_buffer.append(u)
        while len(self.u_buffer) > n:
            self.u_buffer.popleft()
        while len(self.u_buffer) < n:
            self.u_buffer.appendleft(
                self.u_buffer[0] if self.u_buffer else 0.0
            )
        u_eff = self.u_buffer[0]

        # ---- Exact discretisation (ZOH, unconditionally stable) ----
        t_prev = self.T_hat
        alpha = exp(-b * self.dt_min)
        one_minus_alpha = 1.0 - alpha

        # ── PURE prediction (drives T_hat and d_raw) ──
        t_eq_raw = text_used + (a / b) * u_eff
        t_pred_pure = t_eq_raw + (t_prev - t_eq_raw) * alpha

        # ── Innovation on pure model ──
        innovation_pure = float(tin_meas) - t_pred_pure

        # ── d_hat estimation (§15.6) with v4.2 d_raw clamping ──
        if u_eff >= self.d_hat_u_min and one_minus_alpha > 1e-12:
            # Active heating: estimate perturbation from innovation
            # v4.2: clamp d_raw to prevent spike contamination
            d_raw = _clamp(
                b * innovation_pure / one_minus_alpha,
                -D_RAW_MAX, D_RAW_MAX,
            )
            self.d_hat_ema = (
                self.d_hat_alpha * d_raw
                + (1.0 - self.d_hat_alpha) * self.d_hat_ema
            )
            # Physical clamp (v4.1 §15.6.5)
            self.d_hat_ema = _clamp(
                self.d_hat_ema, -self.d_hat_max, self.d_hat_max
            )
        else:
            # u too low: slow relaxation toward 0 instead of hard freeze (v4.1).
            # Avoids stale solar bias when heating OFF at night.
            # d_hat_relax=0.005 → 95% decay in ~600 steps (10h).
            self.d_hat_ema *= (1.0 - self.d_hat_relax)

        # ── CORRECTED prediction (diagnostics only) ──
        t_eq_corr = text_used + (a * u_eff + self.d_hat_ema) / b
        t_pred = t_eq_corr + (t_prev - t_eq_corr) * alpha

        # ── Innovation on corrected model (for RMSE, CUSUM) ──
        innovation = float(tin_meas) - t_pred

        # ---- Nudge (Luenberger, on PURE model — §4.2) ----
        # v4.2: clamp innovation to prevent wild T_hat jumps on sensor glitches
        if self.gamma > 0:
            max_inno = MAX_NUDGE_C / self.gamma
            clamped_inno_pure = _clamp(innovation_pure, -max_inno, max_inno)
            t_next = t_pred_pure + self.gamma * clamped_inno_pure
        else:
            t_next = t_pred_pure

        # ---- v4.2: NaN/Inf guard on computed state ----
        if not isfinite(t_next):
            _LOGGER.warning(
                "ThermalTwin: t_next is NaN/Inf, resyncing to tin_meas=%.2f",
                tin_meas,
            )
            t_next = float(tin_meas)
            self.d_hat_ema = 0.0

        # ---- Update state ----
        self.T_hat = t_next
        self.T_pred = t_pred if isfinite(t_pred) else t_next

        # ---- Advanced diagnostics (§15) ----
        adv = self._compute_advanced_diagnostics(
            innovation=innovation,
            innovation_pure=innovation_pure,
            a=a, b=b,
            u_now=u,
            u_eff=u_eff,
            tin_meas=float(tin_meas),
            text_used=text_used,
            sp=sp,
        )

        # ---- v4.2/v4.3: auto-reset on sustained divergence with backoff ----
        auto_reset_triggered = False
        if adv.get("_auto_reset_needed"):
            self._reset_count += 1
            _LOGGER.warning(
                "ThermalTwin: auto-reset #%d after %d divergent steps (RMSE=%.3f)",
                self._reset_count,
                self._diverge_count,
                adv.get("rmse_30", 0),
            )
            self.T_hat = float(tin_meas)
            self.T_pred = float(tin_meas)
            self.d_hat_ema = 0.0
            self._sum_sq = 0.0
            self._sum_sq_pure = 0.0
            self.innovation_buffer.clear()
            self.innovation_pure_buffer.clear()
            self.cusum_pos = 0.0
            self.cusum_neg = 0.0
            self._innovation_bias_ema = 0.0
            self._step_count = 0
            self._d_hat_active_steps = 0   # v4.4: d_hat is no longer fresh after reset
            # v4.3: exponential backoff — next reset requires more divergent steps
            # 60 → 120 → 240 → 360 (capped at 6h @ 60s/step)
            cooldown = min(
                DIVERGE_RESET_THRESHOLD * (2 ** min(self._reset_count - 1, 3)),
                360,
            )
            self._diverge_count = -cooldown  # negative = must count up to 0 then to threshold
            auto_reset_triggered = True
            t_next = float(tin_meas)

        # ---- v4.5: cold_start diagnostics (ephemeral — reset after first step) ----
        cold_start = self._downtime_s > REBOOT_SHORT_S
        downtime_diag = (
            round(self._downtime_s, 0)
            if self._downtime_s < float("inf")
            else None
        )
        self._downtime_s = 0.0  # consumed — subsequent steps report 0

        return {
            "status": "ok",
            "T_hat_prev": t_prev,
            "T_hat_next": t_next,
            "T_pred": self.T_pred,
            "Tin_meas": float(tin_meas),
            "Text_used": text_used,
            "a": float(a),
            "b": float(b),
            "tau_min": 1.0 / float(b),
            "deadtime_s": float(deadtime_s),
            "dead_steps": int(self.dead_steps),
            "u_now": u,
            "u_eff": u_eff,   # v4.4: deadtime-delayed command actually driving the plant
            "gamma": float(self.gamma),
            "innovation": innovation,
            "d_hat_ema": round(self.d_hat_ema, 6),
            "T_hat_error": round(abs(t_next - float(tin_meas)), 4),
            "auto_reset_triggered": auto_reset_triggered,
            # v4.5 reboot diagnostics
            "cold_start": cold_start,
            "downtime_s": downtime_diag,
            **adv,
        }

    # ------------------------------------------------------------------
    # Advanced diagnostics (§15) — v4.1
    # ------------------------------------------------------------------

    def _compute_advanced_diagnostics(  # pylint: disable=too-many-locals
        self,
        innovation: float,
        innovation_pure: float,
        a: float,
        b: float,
        u_now: float,
        u_eff: float,
        tin_meas: float,
        text_used: float,
        sp: float | None = None,
    ) -> dict:
        """Compute advanced diagnostics from innovation signals.

        v4.4 T_steady note:
          T_steady uses u_now (current command) — answers "where will we converge
          at the CURRENT COMMAND?" This is the correct quantity for setpoint planning.
          Deadtime shifts the transient, not the equilibrium.

          T_steady_immediate uses u_eff (deadtime-delayed) — answers "where is the
          CURRENT HEATING EFFECT driving us?" Useful to detect optimism/pessimism
          during command transitions.

          T_steady_passive uses u=0 — the natural floor temperature (no heating).
        """

        # ---- §15.1 Sliding RMSE (v4.2: incremental O(1)) ----
        # Subtract oldest element before it gets evicted by the append
        if len(self.innovation_buffer) == self.rmse_window:
            self._sum_sq -= self.innovation_buffer[0] ** 2
        self._sum_sq += innovation ** 2
        self._sum_sq = max(self._sum_sq, 0.0)  # guard against float drift
        self.innovation_buffer.append(innovation)

        if len(self.innovation_pure_buffer) == self.rmse_window:
            self._sum_sq_pure -= self.innovation_pure_buffer[0] ** 2
        self._sum_sq_pure += innovation_pure ** 2
        self._sum_sq_pure = max(self._sum_sq_pure, 0.0)
        self.innovation_pure_buffer.append(innovation_pure)

        n_inno = len(self.innovation_buffer)
        rmse = sqrt(self._sum_sq / n_inno) if n_inno >= 2 else None

        n_inno_pure = len(self.innovation_pure_buffer)
        rmse_pure = sqrt(self._sum_sq_pure / n_inno_pure) if n_inno_pure >= 2 else None

        model_reliable = rmse is not None and rmse < self.rmse_reliable_threshold

        # ---- §15.2 CUSUM with reset (v4.1) ----
        self.cusum_pos = max(0.0, self.cusum_pos + innovation - self.cusum_delta)
        self.cusum_neg = max(0.0, self.cusum_neg - innovation - self.cusum_delta)
        external_gain_detected = self.cusum_pos > self.cusum_threshold
        external_loss_detected = self.cusum_neg > self.cusum_threshold
        if external_gain_detected:
            self.cusum_pos = 0.0
        if external_loss_detected:
            self.cusum_neg = 0.0

        # ---- §15.3 Perturbation ----
        perturbation_dTdt = b * innovation

        # ---- §15.4 T_steady family (v4.4 audit) ----
        # T_steady: equilibrium at current COMMAND u_now (planning / setpoint reachability)
        # Deadtime only shifts the transient, not the steady-state → u_now is correct here.
        T_steady = text_used + (a * u_now + self.d_hat_ema) / b
        T_steady_valid = (T_STEADY_MIN <= T_steady <= T_STEADY_MAX)
        if not T_steady_valid:
            model_reliable = False

        setpoint_reachable = (
            (T_steady >= sp)
            if (sp is not None and T_steady_valid)
            else None
        )

        # T_steady_max: equilibrium at max power u=1 — "can the system ever reach sp?"
        T_steady_max = text_used + (a * 1.0 + self.d_hat_ema) / b
        T_steady_max_valid = (T_STEADY_MIN <= T_steady_max <= T_STEADY_MAX)
        setpoint_reachable_max = (
            (T_steady_max >= sp)
            if (sp is not None and T_steady_max_valid)
            else None
        )

        # v4.4: T_steady_immediate — equilibrium at u_eff (deadtime-delayed command).
        # Shows where the current HEATING EFFECT is driving the temperature.
        # Differs from T_steady during command transitions (e.g. u went 0→1 just now).
        T_steady_immediate = text_used + (a * u_eff + self.d_hat_ema) / b
        T_steady_immediate_valid = (T_STEADY_MIN <= T_steady_immediate <= T_STEADY_MAX)

        # v4.4: T_steady_passive — equilibrium with no heating (u=0).
        # Natural floor temperature: where the room converges without any control action.
        T_steady_passive = text_used + self.d_hat_ema / b
        T_steady_passive_valid = (T_STEADY_MIN <= T_steady_passive <= T_STEADY_MAX)

        # v4.4: d_hat freshness tracking
        if u_eff >= self.d_hat_u_min:
            self._d_hat_active_steps += 1
        else:
            self._d_hat_active_steps = 0
        d_hat_fresh = self._d_hat_active_steps >= 3

        # ---- §15.5 Emitter saturation ----
        if u_now >= 0.95:
            self.sat_count += 1
        else:
            self.sat_count = 0

        dTin_dt = None
        if self.last_tin_meas is not None:
            dTin_dt = (tin_meas - self.last_tin_meas) / self.dt_min
        self.last_tin_meas = tin_meas

        emitter_saturated = (
            sp is not None
            and self.sat_count >= self.N_sat
            and dTin_dt is not None
            and abs(dTin_dt) < self.dT_sat_threshold
            and tin_meas < sp - self.eps_sat
        )

        # ---- v4.3: periodic RMSE recalibration (every 1000 steps) ----
        self._step_count += 1
        if self._step_count % 1000 == 0:
            self._sum_sq = sum(e ** 2 for e in self.innovation_buffer)
            self._sum_sq_pure = sum(e ** 2 for e in self.innovation_pure_buffer)

        # ---- v4.3: warming_up flag ----
        warming_up = n_inno < self.rmse_window

        # ---- v4.4: T_steady_reliable composite flag ----
        # True only when all conditions are met:
        #   - T_steady is within physical bounds
        #   - Overall model RMSE is low (reliable parameters)
        #   - d_hat_ema is fresh (recently estimated under active heating)
        #   - Innovation buffer is full (no warm-up artefacts)
        T_steady_reliable = (
            T_steady_valid
            and model_reliable
            and d_hat_fresh
            and not warming_up
        )

        # ---- v4.2: Innovation bias tracking (slow drift detection) ----
        self._innovation_bias_ema = (
            BIAS_EMA_ALPHA * innovation
            + (1.0 - BIAS_EMA_ALPHA) * self._innovation_bias_ema
        )
        bias_warning = abs(self._innovation_bias_ema) > BIAS_WARN_THRESHOLD

        # ---- v4.2/v4.3: Divergence counter for auto-reset with backoff ----
        # During backoff cooldown (_diverge_count < 0): always increment toward 0.
        # After cooldown (_diverge_count >= 0): count consecutive high-RMSE steps.
        # This prevents the backoff from being bypassed by low-RMSE periods
        # immediately after a reset (when T_hat = tin_meas, first innovation is small).
        if self._diverge_count < 0:
            self._diverge_count += 1
        elif rmse is not None and rmse > DIVERGE_RMSE_THRESHOLD:
            self._diverge_count += 1
        else:
            self._diverge_count = 0
        auto_reset_needed = self._diverge_count >= DIVERGE_RESET_THRESHOLD

        return {
            "rmse_30": round(rmse, 4) if rmse is not None else None,
            "rmse_pure": round(rmse_pure, 4) if rmse_pure is not None else None,
            "model_reliable": model_reliable,
            "perturbation_dTdt": round(perturbation_dTdt, 6),
            "cusum_pos": round(self.cusum_pos, 4),
            "cusum_neg": round(self.cusum_neg, 4),
            "external_gain_detected": external_gain_detected,
            "external_loss_detected": external_loss_detected,
            # ---- T_steady family (v4.4) ----
            "T_steady": round(T_steady, 2),
            "T_steady_valid": T_steady_valid,
            "T_steady_reliable": T_steady_reliable,
            "setpoint_reachable": setpoint_reachable,
            "T_steady_max": round(T_steady_max, 2),
            "T_steady_max_valid": T_steady_max_valid,     # v4.4: was missing
            "setpoint_reachable_max": setpoint_reachable_max,
            "T_steady_immediate": round(T_steady_immediate, 2),   # v4.4: at u_eff
            "T_steady_immediate_valid": T_steady_immediate_valid,
            "T_steady_passive": round(T_steady_passive, 2),       # v4.4: at u=0
            "T_steady_passive_valid": T_steady_passive_valid,
            "d_hat_fresh": d_hat_fresh,                            # v4.4
            # ---- rest ----
            "emitter_saturated": emitter_saturated,
            "innovation_bias": round(self._innovation_bias_ema, 4),
            "bias_warning": bias_warning,
            "warming_up": warming_up,
            "reset_count": self._reset_count,
            "_auto_reset_needed": auto_reset_needed,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> dict:
        """Save twin state for persistence across restarts."""
        return {
            "T_hat": self.T_hat,
            "T_pred": self.T_pred,
            "last_text": self.last_text,
            "dead_steps": self.dead_steps,
            "u_buffer": list(self.u_buffer),
            "cusum_pos": self.cusum_pos,
            "cusum_neg": self.cusum_neg,
            "innovation_buffer": list(self.innovation_buffer),
            "innovation_pure_buffer": list(self.innovation_pure_buffer),
            "sat_count": self.sat_count,
            "last_tin_meas": self.last_tin_meas,
            "d_hat_ema": self.d_hat_ema,
            # v4.2 robustness state
            "diverge_count": self._diverge_count,
            "sum_sq": self._sum_sq,
            "sum_sq_pure": self._sum_sq_pure,
            "innovation_bias_ema": self._innovation_bias_ema,
            # v4.3 state
            "step_count": self._step_count,
            "reset_count": self._reset_count,
            # v4.4 T_steady reliability state
            "d_hat_active_steps": self._d_hat_active_steps,
            # v4.5 post-reboot: timestamp for downtime computation on next load
            "saved_at": time.time(),
        }

    def load_state(self, state: dict) -> None:
        """Restore twin state from persisted data.

        Normalizes u_buffer after loading to guarantee len == dead_steps + 1.
        v4.2: validates all loaded floats against NaN/Inf to prevent corruption.
        """
        if not state:
            return

        # v4.2: validate core floats — reset to safe defaults if corrupted
        t_hat_raw = state.get("T_hat")
        self.T_hat = float(t_hat_raw) if (t_hat_raw is not None and isfinite(float(t_hat_raw))) else None

        t_pred_raw = state.get("T_pred")
        self.T_pred = float(t_pred_raw) if (t_pred_raw is not None and isfinite(float(t_pred_raw))) else None

        lt_raw = state.get("last_text")
        self.last_text = float(lt_raw) if (lt_raw is not None and isfinite(float(lt_raw))) else None

        ds = state.get("dead_steps", 0)
        self.dead_steps = int(ds)
        buf = state.get("u_buffer", [])
        # v4.2: filter NaN/Inf from u_buffer
        clean_buf = [v for v in buf if isinstance(v, (int, float)) and isfinite(v)]
        self.u_buffer = deque(clean_buf) if clean_buf else deque([0.0] * (self.dead_steps + 1))
        self._normalize_buffer()

        self.cusum_pos = _safe_float(state.get("cusum_pos"), 0.0)
        self.cusum_neg = _safe_float(state.get("cusum_neg"), 0.0)
        self.sat_count = int(state.get("sat_count", 0))

        # v4.3: always reset last_tin_meas after load to avoid stale dTin_dt
        # after a long HA restart. The first step will compute dTin_dt = None.
        self.last_tin_meas = None

        self.d_hat_ema = _safe_float(state.get("d_hat_ema"), 0.0)

        # v4.2: filter NaN/Inf from innovation buffers
        inno_buf = state.get("innovation_buffer", [])
        clean_inno = [v for v in inno_buf if isinstance(v, (int, float)) and isfinite(v)]
        self.innovation_buffer = deque(clean_inno, maxlen=self.rmse_window)

        inno_pure_buf = state.get("innovation_pure_buffer", [])
        clean_inno_pure = [v for v in inno_pure_buf if isinstance(v, (int, float)) and isfinite(v)]
        self.innovation_pure_buffer = deque(clean_inno_pure, maxlen=self.rmse_window)

        # v4.2: restore robustness state
        self._diverge_count = int(state.get("diverge_count", 0))
        self._sum_sq = _safe_float(state.get("sum_sq"), 0.0)
        self._sum_sq_pure = _safe_float(state.get("sum_sq_pure"), 0.0)
        self._innovation_bias_ema = _safe_float(state.get("innovation_bias_ema"), 0.0)

        # v4.3 state
        self._step_count = int(state.get("step_count", 0))
        self._reset_count = int(state.get("reset_count", 0))

        # v4.4 T_steady reliability state
        self._d_hat_active_steps = int(state.get("d_hat_active_steps", 0))

        # v4.2: recompute sum_sq from buffers to ensure consistency after filtering
        self._sum_sq = sum(e ** 2 for e in self.innovation_buffer)
        self._sum_sq_pure = sum(e ** 2 for e in self.innovation_pure_buffer)

        # ---- v4.5: adaptive restoration based on downtime ----
        saved_at = _safe_float(state.get("saved_at"), 0.0)
        downtime_s = time.time() - saved_at if saved_at > 0.0 else float("inf")
        self._downtime_s = downtime_s
        self._needs_resync = False  # may be set below

        if downtime_s >= REBOOT_SHORT_S:
            # Level 2 (5 min – 1h) and Level 3 (> 1h):
            # The heating state during the downtime is unknown — fill u_buffer
            # with zeros (most conservative: assume heating was off).
            self.u_buffer = deque([0.0] * (self.dead_steps + 1))
            # Innovation buffers contain stale residuals that would corrupt RMSE.
            self.innovation_buffer.clear()
            self.innovation_pure_buffer.clear()
            self._sum_sq = 0.0
            self._sum_sq_pure = 0.0
            # CUSUM accumulators based on old residuals are meaningless.
            self.cusum_pos = 0.0
            self.cusum_neg = 0.0
            # d_hat freshness: must be re-earned from scratch.
            self._d_hat_active_steps = 0
            # Proportional d_hat decay: the longer the stop, the more d_hat
            # represents stale external conditions (solar, occupants).
            # Using the same per-step relax rate as in the frozen-u path.
            if self.dt_s > 0:
                steps_off = downtime_s / self.dt_s
                decay = (1.0 - self.d_hat_relax) ** steps_off
                self.d_hat_ema *= decay

        if downtime_s > REBOOT_MEDIUM_S:
            # Level 3 only (> 1h):
            # Backoff counter from a previous session is no longer relevant.
            self._diverge_count = 0
            # T_hat may be significantly off — flag for resync at first step().
            self._needs_resync = True
            _LOGGER.info(
                "ThermalTwin v4.5: long downtime %.0f s detected, "
                "_needs_resync=True, d_hat_ema decayed to %.4f",
                downtime_s, self.d_hat_ema,
            )

    # ------------------------------------------------------------------
    # Convenience: update twin + compute ETA in one call
    # ------------------------------------------------------------------

    def update_with_eta(
        self,
        tin: float,
        text: float | None,
        target: float,
        on_percent: float,
        tau_reliable: bool,
        a: float,
        b: float,
        mode: str,
        deadtime_heat_s: float | None,
        deadtime_cool_s: float | None,
        deadtime_heat_reliable: bool,
        deadtime_cool_reliable: bool,
    ) -> dict:
        """Update thermal twin and compute ETA best-case (diagnostics-only)."""
        if self.T_hat is None:
            if not tau_reliable:
                return {"status": "not_reliable"}
            self.reset(tin, text, u_init=on_percent)

        # v4.2: use correct deadtime based on power direction
        if on_percent > 0.01:
            deadtime_s = deadtime_heat_s or 0.0
        else:
            deadtime_s = deadtime_cool_s or 0.0
        twin_result = self.step(
            tin_meas=tin,
            text_meas=text,
            a=a, b=b,
            u_now=on_percent,
            deadtime_s=deadtime_s,
            sp=target,
        )

        if tau_reliable:
            eta_result = eta_best_case(
                tin0=tin, text=text, target=target,
                a=a, b=b, mode=mode,
                deadtime_heat_s=deadtime_heat_s,
                deadtime_cool_s=deadtime_cool_s,
                deadtime_heat_ok=deadtime_heat_reliable,
                deadtime_cool_ok=deadtime_cool_reliable,
                last_text=self.last_text,
                d_hat_ema=self.d_hat_ema,
            )
        else:
            eta_result = {"eta_s": None, "reason": "not_reliable"}

        return {
            **twin_result,
            **{f"eta_{k}": v for k, v in eta_result.items()},
        }


# ---------------------------------------------------------------------------
# ETA best-case (standalone function)
# ---------------------------------------------------------------------------

EPS_DENOM = 1e-6
MAX_ETA_S = 48 * 3600


def eta_best_case(  # pylint: disable=too-many-arguments,too-many-return-statements
    tin0: float,
    text: float | None,
    target: float,
    a: float,
    b: float,
    mode: str,
    deadtime_heat_s: float | None,
    deadtime_cool_s: float | None,
    deadtime_heat_ok: bool,
    deadtime_cool_ok: bool,
    eps: float = 0.1,
    eps_denom: float = EPS_DENOM,
    max_eta_s: float = MAX_ETA_S,
    text_policy: str = "hold_last",
    last_text: float | None = None,
    d_hat_ema: float = 0.0,
) -> dict:
    """Compute best-case ETA to reach *target* from *tin0*."""
    # v4.3: NaN guard on inputs
    if not isfinite(tin0) or not isfinite(target):
        return {"eta_s": None, "reason": "invalid_input"}

    if abs(tin0 - target) <= eps:
        return {"eta_s": 0.0, "reason": "already_reached"}

    text_used = _resolve_text(text, text_policy, last_text)
    if text_used is None:
        return {"eta_s": None, "reason": "missing_text"}

    if not isfinite(a) or not isfinite(b) or b <= 0 or a < 0:
        return {"eta_s": None, "reason": "invalid_params"}

    tau_min = 1.0 / b

    if mode == "heat":
        u = 1.0
        l_s = (deadtime_heat_s
               if (deadtime_heat_ok and deadtime_heat_s is not None)
               else 0.0)
    else:
        u = 0.0
        l_s = (deadtime_cool_s
               if (deadtime_cool_ok and deadtime_cool_s is not None)
               else 0.0)

    t_inf = text_used + (a * u + d_hat_ema) / b

    denom = tin0 - t_inf
    if abs(denom) < eps_denom:
        return {"eta_s": None, "reason": "unreachable", "T_inf": t_inf}

    rho = (target - t_inf) / denom
    if rho <= 0 or rho >= 1:
        return {"eta_s": None, "reason": "unreachable",
                "T_inf": t_inf, "rho": rho}

    t_reach_min = (l_s / 60.0) - tau_min * log(rho)
    eta_s = max(0.0, t_reach_min * 60.0)

    if eta_s > max_eta_s:
        return {"eta_s": eta_s, "reason": "too_far", "T_inf": t_inf,
                "tau_min": tau_min, "L_s": l_s, "u": u, "rho": rho}

    return {"eta_s": eta_s, "reason": "ok", "T_inf": t_inf,
            "tau_min": tau_min, "L_s": l_s, "u": u, "rho": rho}
