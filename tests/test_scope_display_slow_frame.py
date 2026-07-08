# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for the per-spike frame-drop attribution logger.

`_check_slow_frame` is the "uneven video" instrument: it emits one WARN when
the gap to the PREVIOUS displayed frame spikes above its recent-window
baseline, with a grab/proc/eng breakdown so a bench bundle can attribute each
stutter. The contract these tests lock:

  * It is called only on STATUS_OK and keys off _last_ok_frame_time, so the
    interval is between DISPLAYED frames -- duplicate/empty loop iterations
    never collapse it (the original bug: measuring inter-iteration time made
    the median collapse to ~1ms on a slow camera, so it never fired).
  * The reported grab/proc/eng are the PREVIOUS frame's -- the display-path
    work that actually ran during the interval -- not this frame's (whose
    compute happens after the interval ends). gap = interval - that work.
  * It fires only when the interval beats BOTH the floor AND ratio-x-median,
    so a uniformly slow stream (median rises to match) does not fire.
  * A fast->slow transition is rate-limited so it does not log every frame
    until the median catches up.

The real ScopeDisplay is a Kivy widget needing a GL context; the logic touches
only a handful of instance attributes + module constants, so a minimal stand-in
borrowing the real methods exercises the exact code without constructing it.
"""

import logging
import sys
from collections import deque
from types import ModuleType
from unittest.mock import MagicMock


class _StubWidget:
    def __init__(self, **kwargs):
        pass


def _real_base_module(name, **attrs):
    mod = ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod


for _name in (
    'kivy.uix',
    'kivy.graphics',
    'kivy.graphics.texture',
    'kivy.metrics',
    'kivy.properties',
    'kivy.input',
    'kivy.clock',
):
    sys.modules.setdefault(_name, MagicMock())

_real_base_module('kivy.uix.image', Image=_StubWidget)
_real_base_module('kivy.uix.widget', Widget=_StubWidget)

from ui.scope_display import (
    FRAME_SPIKE_LOG_MIN_GAP_S,
    FRAME_SPIKE_RATIO,
    FRAME_SPIKE_WINDOW,
    ScopeDisplay,
)

# A representative previous-frame compute (grab, proc, eng) in ms; eng dominant
# (the engineering-stats hitch) so the breakdown is easy to assert.
_PREV_COMPUTE = (2.0, 8.0, 51.0)


class _Stand:
    """Carries only the slow-frame state + the real methods under test."""

    _check_slow_frame = ScopeDisplay._check_slow_frame
    _spike_median = ScopeDisplay._spike_median

    def __init__(self):
        self._spike_interval_window = deque(maxlen=FRAME_SPIKE_WINDOW)
        self._last_ok_frame_time = None
        self._last_ok_compute = None
        self._spike_median_cache = None
        self._spike_median_refresh = 0.0
        self._slow_frame_last_log = 0.0


def _primed(baseline_ms, *, prev_time=1000.0, prev_compute=_PREV_COMPUTE):
    """A stand with the median window pre-filled to a known baseline and a
    previous OK frame already recorded (so the next call computes an interval)."""
    stand = _Stand()
    stand._spike_interval_window = deque(
        [baseline_ms] * FRAME_SPIKE_WINDOW, maxlen=FRAME_SPIKE_WINDOW
    )
    stand._last_ok_frame_time = prev_time
    stand._last_ok_compute = prev_compute
    return stand


def _feed(stand, cycle_start, caplog, *, grab=1.0, proc=3.0, eng=0.0):
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger='LVP.ui.scope_display'):
        stand._check_slow_frame(cycle_start, grab_ms=grab, proc_ms=proc, eng_ms=eng)
    return [r.getMessage() for r in caplog.records if '[SLOW FRAME]' in r.getMessage()]


def test_first_frame_is_silent_and_seeds_state(caplog):
    # No previous OK frame -> no interval to judge; just record state.
    stand = _Stand()
    assert _feed(stand, 1000.0, caplog) == []
    assert stand._last_ok_frame_time == 1000.0
    assert stand._last_ok_compute == (1.0, 3.0, 0.0)


def test_silent_until_window_has_min_samples(caplog):
    # A clear 250ms spike, but only a handful of baseline samples -> the median
    # is not yet meaningful, so no warn.
    stand = _Stand()
    stand._spike_interval_window = deque([33.0] * 5, maxlen=FRAME_SPIKE_WINDOW)
    stand._last_ok_frame_time = 1000.0
    stand._last_ok_compute = _PREV_COMPUTE
    assert _feed(stand, 1000.250, caplog) == []


def test_fires_and_attributes_to_previous_frame(caplog):
    stand = _primed(33.0)  # 2x = 66ms > 50ms floor -> threshold 66ms
    # 250ms gap. This frame's own compute is small; the WARN must report the
    # PREVIOUS frame's compute (the work that ran during the interval).
    records = _feed(stand, 1000.250, caplog, grab=1.0, proc=3.0, eng=0.0)
    assert len(records) == 1
    msg = records[0]
    assert 'interval=250ms' in msg
    assert 'median=33ms' in msg
    assert 'grab=2.0ms proc=8.0ms eng=51.0ms' in msg  # the PREVIOUS frame's
    # gap = 250 - (2 + 8 + 51) = 189ms -> the non-display remainder (upstream).
    assert 'gap=189ms' in msg


def test_previous_frame_attribution_across_two_real_calls(caplog):
    # Locks the off-by-one fix end to end: feed a normal frame carrying compute
    # A, then a slow frame carrying compute B. The WARN must blame A (ran during
    # the interval), never B (this frame's own compute, after the interval).
    stand = _primed(33.0, prev_compute=(99.0, 99.0, 99.0))
    # Normal frame (30ms < 66ms threshold): silent, but records compute A.
    assert _feed(stand, 1000.030, caplog, grab=2.0, proc=8.0, eng=51.0) == []
    # Slow frame 250ms later, carrying a deliberately different compute B.
    records = _feed(stand, 1000.280, caplog, grab=7.0, proc=7.0, eng=7.0)
    assert len(records) == 1
    assert 'grab=2.0ms proc=8.0ms eng=51.0ms' in records[0]  # compute A, not B
    assert '7.0ms' not in records[0]


def test_silent_just_below_threshold(caplog):
    stand = _primed(33.0)  # threshold = 66ms
    assert _feed(stand, 1000.065, caplog) == []  # 65ms interval


def test_floor_suppresses_fast_stream_jitter(caplog):
    # Fast stream: median 10ms, 2x = 20ms, but the 50ms floor dominates, so a
    # 30ms blip (3x median) does NOT fire -- it is not a real stutter.
    assert FRAME_SPIKE_RATIO * 10.0 < 50.0
    assert _feed(_primed(10.0), 1000.030, caplog) == []
    # ...but a genuine 200ms stall on the same fast stream does fire.
    assert len(_feed(_primed(10.0), 1000.200, caplog)) == 1


def test_uniformly_slow_stream_does_not_fire(caplog):
    # A steadily slow stream raises the median to match, so no single frame
    # beats 2x median. This is the case capture_fps owns, not this instrument.
    stand = _primed(2000.0)  # ~0.5 fps, the old fps-bug symptom
    assert _feed(stand, 1002.0, caplog) == []  # 2000ms interval
    assert _feed(stand, 1004.1, caplog) == []  # 2100ms interval


def test_rate_limit_caps_a_fast_to_slow_burst(caplog):
    stand = _primed(33.0)
    # First spike fires.
    assert len(_feed(stand, 1000.250, caplog)) == 1
    # A second spike within FRAME_SPIKE_LOG_MIN_GAP_S is suppressed (the median
    # has not yet caught up to the new slow rate, but we don't flood the log).
    assert _feed(stand, 1000.500, caplog) == []
    # Once the min gap has elapsed, a distinct later stutter logs again.
    later = 1000.500 + FRAME_SPIKE_LOG_MIN_GAP_S + 0.300
    assert len(_feed(stand, later, caplog)) == 1
