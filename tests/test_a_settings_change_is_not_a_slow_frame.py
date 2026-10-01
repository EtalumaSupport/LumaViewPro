# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A camera setting change is not a slow frame; a stall still is one.

The live-view slow-frame warning compared each displayed-frame interval with
a rolling median of the last 120 intervals. After a change from a short
exposure to a long one, that median described a cadence the camera no longer
ran, and every frame at the new exposure was reported as a stall until it
caught up: about two minutes at 1000 ms, one warning every two seconds. A
floor at the exposure did not stop it, because a frame takes the exposure
plus the readout and the interval lands just over the floor. Bench and
simulator logs were full of these lines, so the real ones were ignored.

The detector now takes ``settled`` from the API's frame validity: frames
delivered while a change is still switching over are not judged, and the
baseline is rebuilt from a few frames at the new setting. These tests drive
the real method through the frame sequences the logs showed.
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

from modules.frame_validity import FrameValidity
from ui.scope_display import (
    FRAME_SPIKE_MIN_SAMPLES,
    FRAME_SPIKE_RESTART_SAMPLES,
    FRAME_SPIKE_WINDOW,
    ScopeDisplay,
)

# The simulator's frame period is the exposure plus this; the log line
# `interval=151ms (median=34ms)` at a 134 ms exposure is exactly that.
_READOUT_MS = 17.0


class _Stand:
    """The detector's state and the real methods under test."""

    _check_slow_frame = ScopeDisplay._check_slow_frame
    _spike_median = ScopeDisplay._spike_median

    def __init__(self):
        self._spike_interval_window = deque(maxlen=FRAME_SPIKE_WINDOW)
        self._last_ok_frame_time = None
        self._last_ok_compute = None
        self._spike_median_cache = None
        self._spike_median_refresh = 0.0
        self._spike_min_samples = FRAME_SPIKE_MIN_SAMPLES
        self._slow_frame_last_log = 0.0
        self.now = 1000.0


class _Warnings(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        message = record.getMessage()
        if '[SLOW FRAME]' in message:
            self.lines.append(message)


def _frames(stand, warnings, count, period_ms, *, settled=True):
    for _ in range(count):
        stand.now += period_ms / 1000.0
        stand._check_slow_frame(stand.now, grab_ms=0.3, proc_ms=0.3, eng_ms=0.0, settled=settled)
    return warnings.lines


def _rig():
    warnings = _Warnings()
    logger = logging.getLogger('LVP.ui.scope_display')
    logger.addHandler(warnings)
    logger.setLevel(logging.DEBUG)
    return _Stand(), warnings, lambda: logger.removeHandler(warnings)


def _change(stand, warnings, old_period_ms, new_period_ms, switch_over=3):
    """One setting change as the camera delivers it: the frame in flight at
    the old cadence, then the rest of the switch-over at the new one, all
    while validity is pending."""
    _frames(stand, warnings, 1, old_period_ms, settled=False)
    _frames(stand, warnings, switch_over - 1, new_period_ms, settled=False)


def test_short_to_long_exposure_is_not_a_slow_frame():
    stand, warnings, done = _rig()
    try:
        _frames(stand, warnings, 200, 17.0 + _READOUT_MS)
        _change(stand, warnings, 17.0 + _READOUT_MS, 1000.0 + _READOUT_MS)
        lines = _frames(stand, warnings, 180, 1000.0 + _READOUT_MS)
        assert lines == [], f'three minutes at the new exposure warned: {lines[:2]}'
    finally:
        done()


def test_long_to_short_exposure_is_not_a_slow_frame():
    # The 2026-09-29 bench: 1 ms, 1000 ms for a few seconds, back to 1 ms. The
    # long exposure's last frame arrived after the change back, as
    # `interval=996ms (median=34ms)`.
    stand, warnings, done = _rig()
    try:
        _frames(stand, warnings, 200, 1.0 + _READOUT_MS)
        _change(stand, warnings, 1.0 + _READOUT_MS, 1000.0 + _READOUT_MS)
        _frames(stand, warnings, 8, 1000.0 + _READOUT_MS)
        _change(stand, warnings, 1000.0 + _READOUT_MS, 1.0 + _READOUT_MS)
        lines = _frames(stand, warnings, 300, 1.0 + _READOUT_MS)
        assert lines == [], f'the last long frame was called a stall: {lines[:2]}'
    finally:
        done()


def test_a_run_switching_exposure_at_each_step_is_not_a_slow_frame():
    # The 2026-09-28 sim walk: a run alternating 18 and 134 ms, a few
    # seconds per step, gave 16 warnings of `interval=151ms`.
    stand, warnings, done = _rig()
    try:
        period = 18.0 + _READOUT_MS
        _frames(stand, warnings, 100, period)
        for _ in range(10):
            new_period = (134.0 if period < 100 else 18.0) + _READOUT_MS
            _change(stand, warnings, period, new_period)
            _frames(stand, warnings, 20, new_period)
            period = new_period
        assert warnings.lines == []
    finally:
        done()


def test_a_stall_at_a_steady_setting_is_one_warning():
    stand, warnings, done = _rig()
    try:
        _frames(stand, warnings, 100, 34.0)
        lines = _frames(stand, warnings, 1, 400.0)
        assert len(lines) == 1
        assert 'interval=400ms (median=34ms)' in lines[0]
    finally:
        done()


def test_a_stall_soon_after_a_change_is_reported():
    # The rebuilt baseline judges after FRAME_SPIKE_RESTART_SAMPLES frames,
    # not after the 30 a cold start needs: a stall a few frames past an
    # exposure change still reports.
    stand, warnings, done = _rig()
    try:
        _frames(stand, warnings, 100, 34.0)
        _change(stand, warnings, 34.0, 100.0 + _READOUT_MS)
        _frames(stand, warnings, FRAME_SPIKE_RESTART_SAMPLES, 100.0 + _READOUT_MS)
        lines = _frames(stand, warnings, 1, 600.0)
        assert len(lines) == 1
        assert 'interval=600ms (median=117ms)' in lines[0]
    finally:
        done()


def test_a_stall_while_the_stage_moves_is_reported():
    """A move does not change the frame cadence, so it leaves the detector
    judging: the caller asks for validity with the motion sources excluded,
    and a pending move reads as settled while a pending exposure does not."""
    validity = FrameValidity(lambda: 0)
    validity.set_settle_check(lambda source: False)  # the stage is still moving
    validity.invalidate('xy_move')
    validity.invalidate('z_move')
    motion = tuple(FrameValidity.MOTION_SOURCES)
    assert validity.frames_until_valid(exclude_sources=motion) == 0
    settled_while_moving = validity.frames_until_valid(exclude_sources=motion) == 0
    assert settled_while_moving

    stand, warnings, done = _rig()
    try:
        _frames(stand, warnings, 100, 34.0, settled=settled_while_moving)
        lines = _frames(stand, warnings, 1, 400.0, settled=settled_while_moving)
        assert len(lines) == 1
    finally:
        done()

    validity.invalidate('exposure')
    assert validity.frames_until_valid(exclude_sources=motion) > 0
