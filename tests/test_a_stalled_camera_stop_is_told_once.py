# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A camera stop that outlasts every exposure the camera takes is told once, and kept on the status.

The driver times its SDK's stop and keeps the session's longest; the imaging
API's stream check tells ``CameraStopStalledNotice`` unsolicited the first
time that stop outlasts the camera's maximum exposure by its margin, once a
session, whether or not the camera is streaming at the tick, and the
session's ``Status.slow_camera_stop`` keeps it for a client that was not
listening. A stop under the margin tells nothing.

The margin and the check's cadence are shortened, as the stream stall's own
tests shorten theirs, so a stalled stop takes 2 s rather than the 6 s past
the simulated camera's 1 s maximum exposure that the shipped 5 s margin needs.
"""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from _lumascope.modules.exceptions import CameraStopStalledNotice
from _lumascope.modules.lumascope_api.imaging import ImagingAPI, SlowCameraStop
from _lumascope.modules.notification_center import OutcomeKind, Severity, notifications
from _lumascope.modules.scope_session import ScopeSession
from _lumascope.rest.app import build_app
from tests.settings_fixtures import complete_settings

# The simulated camera's maximum exposure is 1 s: with a 0.5 s margin a stop
# is slow past 1.5 s.
_MARGIN_S = 0.5
_STALL_S = 2.0
_UNDER_THE_MARGIN_S = 1.2
# Several ticks of the shortened check, for a notice that must not come.
_TICKS_S = 0.6


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setattr(ImagingAPI, '_SLOW_STOP_MARGIN_S', _MARGIN_S)
    monkeypatch.setattr(ImagingAPI, '_STREAM_CHECK_INTERVAL_S', 0.1)
    session = ScopeSession.create(
        complete_settings(microscope='LS850T'), simulate=True, warn_pre_release=False
    )
    yield session
    session.shutdown()


class _Told:
    def __init__(self):
        self.seen = []
        self.first = threading.Event()

    def __call__(self, notification):
        if notification.title == CameraStopStalledNotice.title:
            self.seen.append(notification)
            self.first.set()


@pytest.fixture
def told():
    listener = _Told()
    notifications.add_listener(listener, min_severity=Severity.DEBUG)
    yield listener
    notifications.remove_listener(listener)


def test_a_stalled_stop_is_told_once_unsolicited_and_kept_on_the_status(session, told):
    scope = session.scope
    assert session.status.slow_camera_stop is None
    scope.simulation.slow_next_stop(_STALL_S)

    scope.camera.set_frame_size(1024, 1024)
    # A fast stop straight after the stalled one: the longest is kept.
    scope.camera.set_frame_size(1200, 1200)

    assert told.first.wait(5.0), 'the stalled stop was not told'
    (notice,) = told.seen
    assert notice.kind == OutcomeKind.NOTICE
    assert notice.reason == 'camera_stop_stalled'
    assert notice.solicited is False
    # The simulated camera is not reset at connect: the power-cycle sentence.
    assert 'Turning the scope off and on again' in notice.message
    kept = session.status.slow_camera_stop
    assert isinstance(kept, SlowCameraStop)
    assert _STALL_S <= kept.stop_s < _STALL_S + 1.0
    assert kept.resets_at_connect is False

    scope.simulation.slow_next_stop(_STALL_S)
    scope.camera.set_frame_size(1024, 1024)
    time.sleep(_TICKS_S)

    assert len(told.seen) == 1, 'a second slow stop was told again'
    assert session.status.slow_camera_stop is kept


def test_a_slow_stop_that_leaves_the_camera_stopped_is_told(session, told):
    scope = session.scope
    scope.simulation.slow_next_stop(_STALL_S)

    scope.imaging.disable_streaming()

    assert told.first.wait(5.0), 'a stalled stop with nothing streaming was not told'
    assert not scope.imaging.read_streaming()


def test_a_stop_under_the_margin_tells_nothing(session, told):
    scope = session.scope
    scope.simulation.slow_next_stop(_UNDER_THE_MARGIN_S)

    scope.camera.set_frame_size(1024, 1024)
    time.sleep(_TICKS_S)

    assert told.seen == []
    assert session.status.slow_camera_stop is None


def test_a_rest_client_reads_the_slow_stop_on_the_status(session, told):
    session.scope.simulation.slow_next_stop(_STALL_S)
    session.scope.camera.set_frame_size(1024, 1024)
    assert told.first.wait(5.0)

    with TestClient(build_app(session)) as client:
        answer = client.get('/api/v1/status')

    assert answer.status_code == 200
    assert answer.json()['slow_camera_stop']['stop_s'] >= _STALL_S
