"""Regression: the live-view display loop warns when frames stop
advancing while the camera is active.

#676 was a report of the live image freezing with no log signal. The
diagnostic build could not reproduce a hard stall, so instead of shipping
the raw 1 Hz heartbeat the loop gained a stall watchdog: one WARNING per
stall episode when no new frame (STATUS_OK) arrives for longer than
STALL_WARN_SECONDS while the camera is active, logging camera + display
state. It re-arms when frames resume and stays quiet when the camera is
inactive.
"""

from __future__ import annotations

import logging
import types

from modules.scope_display_thread import (
    STALL_WARN_SECONDS,
    STATUS_EMPTY,
    STATUS_OK,
    ScopeDisplayThread,
)


def _ctx(*, camera_active=True, camera_connected=True):
    scope = types.SimpleNamespace(
        imaging=types.SimpleNamespace(camera_active=camera_active),
        camera_connected=camera_connected,
    )
    return types.SimpleNamespace(scope=scope)


def _count_stall_warnings(records):
    return sum(
        1 for r in records if r.levelno == logging.WARNING and 'frames stalled' in r.getMessage()
    )


def test_warns_once_per_stall_episode(caplog):
    thread = ScopeDisplayThread()
    ctx = _ctx()
    t0 = 1000.0

    with caplog.at_level(logging.WARNING, logger='LVP.modules.scope_display_thread'):
        thread._check_frame_stall(STATUS_OK, t0, ctx)  # frames flowing
        thread._check_frame_stall(STATUS_EMPTY, t0 + 5, ctx)  # within window
        assert _count_stall_warnings(caplog.records) == 0

        thread._check_frame_stall(STATUS_EMPTY, t0 + STALL_WARN_SECONDS + 1, ctx)
        assert _count_stall_warnings(caplog.records) == 1

        # Still stalled -- no second warning for the same episode.
        thread._check_frame_stall(STATUS_EMPTY, t0 + STALL_WARN_SECONDS + 5, ctx)
        assert _count_stall_warnings(caplog.records) == 1


def test_recovery_rearms_the_warning(caplog):
    thread = ScopeDisplayThread()
    ctx = _ctx()
    t0 = 0.0

    with caplog.at_level(logging.WARNING, logger='LVP.modules.scope_display_thread'):
        thread._check_frame_stall(STATUS_OK, t0, ctx)
        thread._check_frame_stall(STATUS_EMPTY, t0 + STALL_WARN_SECONDS + 1, ctx)
        assert _count_stall_warnings(caplog.records) == 1

        # Frames resume, then stall again -- a new episode warns again.
        thread._check_frame_stall(STATUS_OK, t0 + 20, ctx)
        thread._check_frame_stall(STATUS_EMPTY, t0 + 20 + STALL_WARN_SECONDS + 1, ctx)
        assert _count_stall_warnings(caplog.records) == 2


def test_no_warning_when_camera_inactive(caplog):
    thread = ScopeDisplayThread()
    ctx = _ctx(camera_active=False)
    t0 = 0.0

    with caplog.at_level(logging.WARNING, logger='LVP.modules.scope_display_thread'):
        # Even well past the threshold, an inactive camera never warns
        # (disconnect is surfaced by the recovery contract elsewhere).
        thread._check_frame_stall(STATUS_EMPTY, t0, ctx)
        thread._check_frame_stall(STATUS_EMPTY, t0 + STALL_WARN_SECONDS * 3, ctx)
        assert _count_stall_warnings(caplog.records) == 0


def test_warns_when_stream_never_delivers_a_frame(caplog):
    """No STATUS_OK ever -- the clock starts on the first active
    iteration so a never-starting stream is still caught."""
    thread = ScopeDisplayThread()
    ctx = _ctx()
    t0 = 500.0

    with caplog.at_level(logging.WARNING, logger='LVP.modules.scope_display_thread'):
        thread._check_frame_stall(STATUS_EMPTY, t0, ctx)  # starts the clock
        assert _count_stall_warnings(caplog.records) == 0
        thread._check_frame_stall(STATUS_EMPTY, t0 + STALL_WARN_SECONDS + 1, ctx)
        assert _count_stall_warnings(caplog.records) == 1
