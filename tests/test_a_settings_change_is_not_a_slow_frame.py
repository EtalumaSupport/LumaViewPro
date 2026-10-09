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

import ast
import logging
from collections import deque
from types import SimpleNamespace


from modules.frame_validity import FrameValidity
from tests.ast_seams import find_def
from ui.scope_display import (
    FRAME_SPIKE_MIN_SAMPLES,
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
    # The rebuilt baseline judges after three frames at the new setting, not
    # after the 30 a cold start needs: a stall three frames past an exposure
    # change still reports.
    stand, warnings, done = _rig()
    try:
        _frames(stand, warnings, 100, 34.0)
        _change(stand, warnings, 34.0, 100.0 + _READOUT_MS)
        _frames(stand, warnings, 3, 100.0 + _READOUT_MS)
        lines = _frames(stand, warnings, 1, 600.0)
        assert len(lines) == 1
        assert 'interval=600ms (median=117ms)' in lines[0]
    finally:
        done()


def test_a_baseline_cached_before_the_change_is_not_used_after_it():
    # The median is recomputed at most every half second. A baseline cached
    # a moment before a change from a 100 ms cadence to a 30 ms one would
    # judge the new stream against 200 ms for that half second, and a 180 ms
    # stall at 30 ms would go unreported.
    stand, warnings, done = _rig()
    try:
        _frames(stand, warnings, 100, 100.0)
        stand._spike_median_cache = 100.0
        stand._spike_median_refresh = stand.now
        _change(stand, warnings, 100.0, 30.0)
        _frames(stand, warnings, 3, 30.0)
        lines = _frames(stand, warnings, 1, 180.0)
        assert len(lines) == 1
        assert 'interval=180ms (median=30ms)' in lines[0]
    finally:
        done()


def _imaging(validity):
    return SimpleNamespace(frames_until_valid=validity.frames_until_valid)


def test_the_live_view_asks_validity_with_stage_motion_left_out():
    """A move does not change the frame cadence, so the detector keeps
    judging while the stage moves; a pending exposure stops it."""
    validity = FrameValidity(lambda: 0)
    validity.set_settle_check(lambda source: False)  # the stage is still moving
    validity.invalidate('xy_move')
    validity.invalidate('z_move')
    validity.invalidate('turret')
    assert ScopeDisplay._camera_settled(_imaging(validity)) is True

    validity.invalidate('exposure')
    assert ScopeDisplay._camera_settled(_imaging(validity)) is False


def test_the_render_loop_hands_the_detector_the_camera_s_settled_state():
    # The render loop is a Kivy path no test can run; its one wiring line is
    # pinned on the AST: the slow-frame check gets settled from
    # _camera_settled, never a constant.
    render = find_def('ui/scope_display.py', '_render_one_frame', class_name='ScopeDisplay')
    calls = [
        node
        for node in ast.walk(render)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == '_check_slow_frame'
    ]
    assert len(calls) == 1
    settled = next(kw.value for kw in calls[0].keywords if kw.arg == 'settled')
    assert isinstance(settled, ast.Call)
    assert isinstance(settled.func, ast.Attribute)
    assert settled.func.attr == '_camera_settled'


def test_a_stall_while_the_stage_moves_is_reported():
    validity = FrameValidity(lambda: 0)
    validity.set_settle_check(lambda source: False)
    validity.invalidate('xy_move')
    moving = ScopeDisplay._camera_settled(_imaging(validity))

    stand, warnings, done = _rig()
    try:
        _frames(stand, warnings, 100, 34.0, settled=moving)
        lines = _frames(stand, warnings, 1, 400.0, settled=moving)
        assert len(lines) == 1
    finally:
        done()
