"""
ABLearningLogger — Diagnostic CSV logger for SmartPI a/b learning.

Wires on_window_submit and on_learning_event callbacks onto a SmartPI
instance and writes per-thermostat CSV files for offline analysis.

Usage
-----
Instantiate one logger per SmartPI instance (e.g. in SmartPI.__init__
when debug_mode is True), then call attach():

    if debug_mode:
        self._ab_logger = ABLearningLogger(name, output_dir="/config/smartpi_logs")
        self._ab_logger.attach(self)

Three files are written under output_dir, prefixed with the thermostat name:

  <name>_window_submits.csv
      One row per learning window submitted to ABEstimator.learn().
      Contains window-level metadata: duration, u_eff, start temperatures,
      computed OLS slope, and trim parameters.

  <name>_window_samples.csv
      One row per raw (timestamp, temperature) sample within each submitted
      window.  Timestamps are expressed as elapsed seconds from window start
      so windows from different wall-clock times are directly comparable.
      Plot temp_c vs elapsed_s for each window to inspect the OLS fit quality
      and confirm whether early fast transients are contaminating b estimates.

  <name>_learning_events.csv
      One row per call to ABEstimator.learn(), including both accepted and
      rejected samples.  Contains the computed measurement (a_meas or b_meas),
      whether it passed the MAD outlier gate, and the resulting a/b values.

All three files use wall-clock time (Unix timestamp) so rows can be
correlated across files and with external data sources.
"""
from __future__ import annotations

import csv
import logging
import os
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .learning import LearningEvent
    from .learning_window import WindowSubmitEvent

_LOGGER = logging.getLogger(__name__)


class ABLearningLogger:
    """
    Diagnostic CSV logger for SmartPI a/b learning.

    One instance per SmartPI thermostat.  Safe to attach to multiple
    thermostats simultaneously provided each has its own logger instance.
    """

    def __init__(self, name: str, output_dir: str) -> None:
        """
        Args:
            name:       Thermostat entity name.  Used as filename prefix and
                        in the 'name' column of every row.
            output_dir: Directory to write CSV files into.  Created if absent.
                        Must be writable by the HA process.
        """
        self._name = name
        # Sanitise name for use in filenames
        safe = name.replace(" ", "_").replace("/", "_").replace(".", "_")

        os.makedirs(output_dir, exist_ok=True)

        self._raw_f = open(
            os.path.join(output_dir, f"{safe}_window_submits.csv"),
            "w", newline="", buffering=1,  # line-buffered: flush on every newline
        )
        self._samples_f = open(
            os.path.join(output_dir, f"{safe}_window_samples.csv"),
            "w", newline="", buffering=1,
        )
        self._events_f = open(
            os.path.join(output_dir, f"{safe}_learning_events.csv"),
            "w", newline="", buffering=1,
        )

        self._raw_w = csv.writer(self._raw_f)
        self._samples_w = csv.writer(self._samples_f)
        self._events_w = csv.writer(self._events_f)

        self._raw_w.writerow([
            "name", "wall_time",
            "param", "u_eff", "window_s",
            "t_int_start", "t_ext_start", "delta_start",
            "slope", "slope_method", "trim_start_frac", "n_samples",
        ])
        self._samples_w.writerow([
            "name", "wall_time",
            "param", "elapsed_s", "temp_c",
        ])
        self._events_w.writerow([
            "name", "wall_time",
            "param", "accepted", "reject_reason",
            "dTdt", "u", "t_int", "t_ext", "delta",
            "meas", "a", "b",
            "ok_a", "ok_b",
        ])

        _LOGGER.debug("ABLearningLogger: logging '%s' to %s", name, output_dir)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def on_window_submit(self, ev: "WindowSubmitEvent") -> None:
        """Called by LearningWindowManager just before each learn() submission."""
        wall = round(time.time(), 3)
        self._raw_w.writerow([
            self._name, wall,
            ev.param, round(ev.u_eff, 4), round(ev.window_s, 1),
            round(ev.t_int_start, 3), round(ev.t_ext_start, 3),
            round(ev.t_int_start - ev.t_ext_start, 3),
            round(ev.slope, 6), ev.slope_method,
            ev.trim_start_frac, len(ev.samples),
        ])
        # Write individual temperature samples, elapsed from window start
        if ev.samples:
            t0 = ev.samples[0][0]
            for ts, temp in ev.samples:
                self._samples_w.writerow([
                    self._name, wall,
                    ev.param, round(ts - t0, 1), round(temp, 4),
                ])

    def on_learning_event(self, ev: "LearningEvent") -> None:
        """Called by ABEstimator.learn() for every accepted or rejected sample."""
        self._events_w.writerow([
            self._name, round(time.time(), 3),
            ev.param, ev.accepted, ev.reject_reason,
            round(ev.dTdt, 6), round(ev.u, 4),
            round(ev.t_int, 3), round(ev.t_ext, 3), round(ev.delta, 3),
            round(ev.meas, 6), round(ev.a, 6), round(ev.b, 6),
            ev.learn_ok_count_a, ev.learn_ok_count_b,
        ])

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def attach(self, algo: object) -> None:
        """
        Wire both callbacks onto a SmartPI instance.

        Args:
            algo: SmartPI instance exposing .learn_win and .est attributes.
        """
        algo.learn_win.on_window_submit = self.on_window_submit
        algo.est.on_learning_event = self.on_learning_event

    def detach(self, algo: object) -> None:
        """Remove callbacks from a SmartPI instance."""
        if getattr(algo.learn_win, "on_window_submit", None) is self.on_window_submit:
            algo.learn_win.on_window_submit = None
        if getattr(algo.est, "on_learning_event", None) is self.on_learning_event:
            algo.est.on_learning_event = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close all file handles."""
        for f in (self._raw_f, self._samples_f, self._events_f):
            try:
                f.close()
            except OSError as exc:
                _LOGGER.debug("ABLearningLogger: error closing file: %s", exc)
