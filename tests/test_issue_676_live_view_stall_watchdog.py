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


def _ctx(*, active_cached=True, camera_connected=True):
    scope = types.SimpleNamespace(
        imaging=types.SimpleNamespace(active_cached=active_cached),
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
    ctx = _ctx(active_cached=False)
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


def test_stall_fires_a_user_notification_once_per_episode(monkeypatch):
    """The stall must reach the user as a notification, not only the log.
    A frozen live image with nothing but a log line is the original report;
    the popup fires once per episode (same gate as the log) and re-arms."""
    from modules import scope_display_thread as sdt

    calls = []
    monkeypatch.setattr(sdt.notifications, 'warning', lambda *a, **k: calls.append((a, k)))

    thread = ScopeDisplayThread()
    ctx = _ctx()
    t0 = 0.0

    thread._check_frame_stall(STATUS_OK, t0, ctx)
    thread._check_frame_stall(STATUS_EMPTY, t0 + STALL_WARN_SECONDS + 1, ctx)
    assert len(calls) == 1, 'stall did not fire exactly one user notification'

    # Same episode -- no second popup.
    thread._check_frame_stall(STATUS_EMPTY, t0 + STALL_WARN_SECONDS + 5, ctx)
    assert len(calls) == 1


def test_suppressed_watchdog_does_not_warn(caplog):
    """suppress_stall_warnings(True) mutes the watchdog. An operation that
    deliberately monopolizes the camera (e.g. characterization driving forced
    grabs) makes frame delivery gap on purpose, so the warning would be a
    false alarm. The display keeps rendering; only the warning is withheld."""
    thread = ScopeDisplayThread()
    ctx = _ctx()
    t0 = 0.0

    with caplog.at_level(logging.WARNING, logger='LVP.modules.scope_display_thread'):
        thread._check_frame_stall(STATUS_OK, t0, ctx)
        thread.suppress_stall_warnings(True)
        # Well past the threshold, but muted -- no warning.
        thread._check_frame_stall(STATUS_EMPTY, t0 + STALL_WARN_SECONDS * 3, ctx)
        assert _count_stall_warnings(caplog.records) == 0


def test_suppression_lift_rearms_on_a_fresh_window(caplog):
    """When the mute lifts, the watchdog re-arms on a fresh window rather than
    firing immediately on the gap that accumulated while it was muted."""
    thread = ScopeDisplayThread()
    ctx = _ctx()
    t0 = 0.0

    with caplog.at_level(logging.WARNING, logger='LVP.modules.scope_display_thread'):
        thread.suppress_stall_warnings(True)
        thread._check_frame_stall(STATUS_EMPTY, t0, ctx)
        thread._check_frame_stall(STATUS_EMPTY, t0 + STALL_WARN_SECONDS * 5, ctx)
        thread.suppress_stall_warnings(False)
        # First post-lift check starts the clock fresh -- no instant warning.
        thread._check_frame_stall(STATUS_EMPTY, t0 + STALL_WARN_SECONDS * 5 + 1, ctx)
        assert _count_stall_warnings(caplog.records) == 0
        # A real stall AFTER the fresh window still warns.
        thread._check_frame_stall(STATUS_EMPTY, t0 + STALL_WARN_SECONDS * 7, ctx)
        assert _count_stall_warnings(caplog.records) == 1
