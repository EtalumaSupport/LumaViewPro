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

import functools
import logging
from collections import deque
from types import SimpleNamespace


from _lumascope.modules.frame_validity import Counter, FrameValidity
from _lumascope.modules.lumascope_api.imaging import ImagingAPI
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
    _camera_settled = ScopeDisplay._camera_settled

    def __init__(self):
        self._spike_interval_window = deque(maxlen=FRAME_SPIKE_WINDOW)
        self._last_ok_frame_time = None
        self._last_ok_compute = None
        self._spike_median_cache = None
        self._spike_median_refresh = 0.0
        self._spike_min_samples = FRAME_SPIKE_MIN_SAMPLES
        self._slow_frame_last_log = 0.0
        self._changes_at_last_frame = None
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
    level = logger.level
    logger.addHandler(warnings)
    logger.setLevel(logging.DEBUG)

    def done():
        logger.removeHandler(warnings)
        logger.setLevel(level)

    return _Stand(), warnings, done


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
    """An imaging whose ``changes_in_force`` and ``recorded_changes`` are the real members over ``validity``.

    A stand-in by design: the subject is the display's read, and its
    collaborators are the real ``ImagingAPI`` members on a self that holds
    only the one store they read.
    """
    owner = SimpleNamespace(_frame_validity=validity)
    return SimpleNamespace(
        changes_in_force=functools.partial(ImagingAPI.changes_in_force, owner),
        recorded_changes=functools.partial(ImagingAPI.recorded_changes, owner),
    )


def _exposure_changed(frames_counted: int) -> FrameValidity:
    """A frame validity with an exposure change and ``frames_counted`` frames counted after it."""
    delivered = {'n': 0}
    validity = FrameValidity(lambda: delivered['n'])
    validity.invalidate('exposure', began_ns=None)
    for _ in range(frames_counted):
        delivered['n'] += 1
        validity.count_frame(delivered['n'], by=Counter.PREVIEW)
    return validity


def test_a_change_still_owed_frames_has_not_settled():
    """A change is in force once its count of frames has come: the
    slow-frame check judges no frame before that."""
    validity = _exposure_changed(FrameValidity.SKIP_FRAMES['exposure'] - 1)
    assert _Stand()._camera_settled(_imaging(validity)) is False


def test_a_change_whose_count_has_come_has_settled():
    validity = _exposure_changed(FrameValidity.SKIP_FRAMES['exposure'])
    assert _Stand()._camera_settled(_imaging(validity)) is True


def test_the_live_view_asks_validity_with_stage_motion_left_out():
    """A move does not change the frame cadence, so the detector keeps
    judging while the stage moves; a pending exposure stops it."""
    validity = FrameValidity(lambda: 0)
    validity.set_settle_check(lambda source: False)  # the stage is still moving
    validity.invalidate('xy_move', began_ns=None)
    validity.invalidate('z_move', began_ns=None)
    validity.invalidate('turret', began_ns=None)
    stand = _Stand()
    assert stand._camera_settled(_imaging(validity)) is True

    validity.invalidate('exposure', began_ns=None)
    assert stand._camera_settled(_imaging(validity)) is False


def test_a_stall_while_the_stage_moves_is_reported():
    validity = FrameValidity(lambda: 0)
    validity.set_settle_check(lambda source: False)
    validity.invalidate('xy_move', began_ns=None)
    moving = _Stand()._camera_settled(_imaging(validity))

    stand, warnings, done = _rig()
    try:
        _frames(stand, warnings, 100, 34.0, settled=moving)
        lines = _frames(stand, warnings, 1, 400.0, settled=moving)
        assert len(lines) == 1
    finally:
        done()


class _Scope:
    """A frame validity the camera counts frames into, read as the live view reads it."""

    def __init__(self):
        self._delivered = 0
        self.validity = FrameValidity(lambda: self._delivered)
        self.imaging = _imaging(self.validity)

    def change(self, source):
        """Record a change and count frames until it is in force, as the camera delivers them."""
        self.validity.invalidate(source, began_ns=None)
        for _ in range(FrameValidity.SKIP_FRAMES[source]):
            self._delivered += 1
            self.validity.count_frame(self._delivered, by=Counter.PREVIEW)


def _drawn(stand, scope, count, period_ms, *, held_ms=0.0):
    """Frames the live view draws, each judged with the display's own settled read."""
    for _ in range(count):
        stand.now += (period_ms + held_ms) / 1000.0
        stand._check_slow_frame(
            stand.now,
            grab_ms=0.3,
            proc_ms=0.3,
            eng_ms=0.0,
            settled=stand._camera_settled(scope.imaging),
            held_ms=held_ms,
        )


_CAPTURE_HOLD_MS = 500.0


def test_a_change_that_comes_into_force_between_two_drawn_frames_is_not_a_slow_frame():
    """A protocol's channel change lands inside a capture's hold, when the
    live view draws nothing, and is in force before the next drawn frame.
    The frames at the new channel's cadence are not judged against the old
    channel's baseline (the field bundle's one warning a well at Ex589)."""
    stand, warnings, done = _rig()
    scope = _Scope()
    try:
        _drawn(stand, scope, 100, 34.0)
        scope.change('exposure')
        _drawn(stand, scope, 1, 178.0, held_ms=_CAPTURE_HOLD_MS)
        _drawn(stand, scope, 20, 178.0)
        assert warnings.lines == []
    finally:
        done()


def test_a_stall_after_a_change_made_during_a_hold_is_reported():
    stand, warnings, done = _rig()
    scope = _Scope()
    try:
        _drawn(stand, scope, 100, 34.0)
        scope.change('exposure')
        _drawn(stand, scope, 1, 178.0, held_ms=_CAPTURE_HOLD_MS)
        _drawn(stand, scope, 3, 178.0)
        _drawn(stand, scope, 1, 600.0)
        assert len(warnings.lines) == 1
        assert 'interval=600ms (median=178ms)' in warnings.lines[0]
    finally:
        done()


def test_a_move_recorded_between_drawn_frames_does_not_reset_the_baseline():
    """A move leaves the cadence alone, so one recorded between two drawn
    frames does not make the next frame unsettled."""
    scope = _Scope()
    scope.validity.set_settle_check(lambda source: False)  # the stage is still moving
    stand = _Stand()
    assert stand._camera_settled(scope.imaging) is True
    scope.validity.invalidate('xy_move', began_ns=None)
    assert stand._camera_settled(scope.imaging) is True


def test_an_interval_spanning_a_hold_across_two_changes_is_not_judged():
    """Two channel changes in one hold, a long exposure and then a short
    one: the first drawn frame after it spans the long one's cadence and is
    not judged against the short baseline from before (the sim's Ex405)."""
    stand, warnings, done = _rig()
    scope = _Scope()
    try:
        _drawn(stand, scope, 100, 34.0)
        scope.change('exposure')
        scope.change('exposure')
        _drawn(stand, scope, 1, 178.0, held_ms=_CAPTURE_HOLD_MS)
        _drawn(stand, scope, 20, 25.0)
        assert warnings.lines == []
    finally:
        done()
