# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for MetricsLogger FRAME FLOW heartbeat sticky-failure.

The previous implementation fired the user-facing notification once
per stall episode and then went silent forever. Rule 14 requires
persistent faults to resurface; this suite locks in the new behavior:

  * threshold-tick: first WARNING + first popup
  * +RENOTIFY ticks: WARNING + popup again
  * +CRITICAL ticks (once): critical-severity popup escalation
  * fps recovery: all counters reset; next stall fires fresh
"""

from unittest.mock import MagicMock

import pytest

from modules import app_context as _app_ctx
from modules import metrics_logger as ml


class _FakeCamera:
    def __init__(self, active=True, grabbing=True):
        self.active = active
        self._grabbing = grabbing

    def is_grabbing(self):
        return self._grabbing


class _FakeScope:
    def __init__(self, camera):
        self.camera = camera


class _FakeScopeDisplay:
    def __init__(self, fps=0.0):
        self._capture_fps_value = fps


@pytest.fixture
def stalled_logger(monkeypatch):
    """Logger pre-wired with a stalled camera (active+grabbing, fps=0)
    and a captured-notifications shim so tests can assert on calls."""
    cam = _FakeCamera()
    scope = _FakeScope(cam)
    bundle = MagicMock()
    settings = {}
    log = ml.MetricsLogger(scope=scope, executor_bundle=bundle, settings=settings)

    # Wire scope_display into app_context so the heartbeat finds fps=0.
    fake_ctx = MagicMock()
    fake_ctx.scope_display = _FakeScopeDisplay(fps=0.0)
    saved = _app_ctx.ctx
    _app_ctx.ctx = fake_ctx

    # Capture notifications fired by the heartbeat.
    notifications_calls = []

    class _FakeNotifications:
        def warning(self, *args):
            notifications_calls.append(('warning', args))

        def critical(self, *args):
            notifications_calls.append(('critical', args))

    fake_notifications = _FakeNotifications()
    # The heartbeat does a function-local import; patch the module
    # the import resolves through.
    from modules import notification_center

    monkeypatch.setattr(notification_center, 'notifications', fake_notifications)

    yield log, cam, fake_ctx, notifications_calls

    _app_ctx.ctx = saved


def test_first_threshold_tick_fires_warning(stalled_logger):
    log, _, _, calls = stalled_logger
    # Tick 1: stalled_ticks=1, below threshold (2) -- no notify yet.
    log._check_frame_flow_heartbeat()
    assert calls == []
    # Tick 2: stalled_ticks=2 == threshold -- warning fires.
    log._check_frame_flow_heartbeat()
    assert len(calls) == 1
    assert calls[0][0] == 'warning'


def test_refire_every_renotify_ticks_while_stall_persists(stalled_logger):
    log, _, _, calls = stalled_logger
    # Pump enough ticks to hit threshold + at least one refire cycle.
    n_ticks = ml._FRAME_FLOW_STALL_TICK_THRESHOLD + ml._FRAME_FLOW_STALL_RENOTIFY_TICKS + 1
    for _ in range(n_ticks):
        log._check_frame_flow_heartbeat()
    # Should have AT LEAST two warning popups (initial + one refire).
    warnings = [c for c in calls if c[0] == 'warning']
    assert len(warnings) >= 2, (
        f'persistent stall must resurface popup; got '
        f'{len(warnings)} warning(s) over {n_ticks} ticks'
    )


def test_escalates_to_critical_once_after_extended_stall(stalled_logger):
    log, _, _, calls = stalled_logger
    # Drive past the critical threshold to trigger the one-shot escalation.
    n_ticks = ml._FRAME_FLOW_STALL_CRITICAL_TICKS + 2
    for _ in range(n_ticks):
        log._check_frame_flow_heartbeat()
    criticals = [c for c in calls if c[0] == 'critical']
    assert len(criticals) == 1, (
        f'critical escalation must fire exactly once per episode; got {len(criticals)} critical(s)'
    )
    # Pump more ticks; critical must NOT re-fire (one-shot escalation).
    for _ in range(ml._FRAME_FLOW_STALL_RENOTIFY_TICKS * 2):
        log._check_frame_flow_heartbeat()
    criticals_after = [c for c in calls if c[0] == 'critical']
    assert len(criticals_after) == 1, (
        f'critical escalation must NOT re-fire after the first; '
        f'got {len(criticals_after)} critical(s) total'
    )


def test_recovery_resets_state_for_next_stall(stalled_logger):
    log, _, fake_ctx, calls = stalled_logger
    # Stall, escalate to critical, then recover, then stall again.
    for _ in range(ml._FRAME_FLOW_STALL_CRITICAL_TICKS + 2):
        log._check_frame_flow_heartbeat()
    first_episode_criticals = sum(1 for c in calls if c[0] == 'critical')
    assert first_episode_criticals == 1

    # Recovery: fps > threshold for one tick.
    fake_ctx.scope_display = _FakeScopeDisplay(fps=10.0)
    log._check_frame_flow_heartbeat()
    assert log._frame_flow_stalled_ticks == 0
    assert log._frame_flow_stall_last_notified_tick == -1
    assert log._frame_flow_stall_critical_fired is False

    # Re-stall: fresh episode must be able to escalate again.
    fake_ctx.scope_display = _FakeScopeDisplay(fps=0.0)
    for _ in range(ml._FRAME_FLOW_STALL_CRITICAL_TICKS + 2):
        log._check_frame_flow_heartbeat()
    second_episode_criticals = sum(1 for c in calls if c[0] == 'critical')
    assert second_episode_criticals == 2, (
        f'second stall episode must also escalate; got '
        f'{second_episode_criticals - first_episode_criticals} criticals '
        f'in the second episode'
    )


def test_camera_inactive_resets_without_notifying(stalled_logger):
    log, cam, _, calls = stalled_logger
    # Stall first.
    for _ in range(ml._FRAME_FLOW_STALL_TICK_THRESHOLD + 1):
        log._check_frame_flow_heartbeat()
    assert len(calls) >= 1
    notify_count_after_stall = len(calls)

    # Camera goes inactive -- counters reset, no new notify.
    cam.active = False
    log._check_frame_flow_heartbeat()
    assert log._frame_flow_stalled_ticks == 0
    assert len(calls) == notify_count_after_stall, 'inactive camera must reset counters silently'
