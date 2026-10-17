# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A camera stream start that fails raises its fault to whoever started it.

The FX2's start raises its typed fault (``camera_stream_start_failed`` for a
START the device refused). A start that raised did not start, so the grab is
no longer wanted: the caller reports the fault, and the stream check does not
report the same failure again as a stall. The camera stays connected, and the
next start tries again.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

import modules.video_cadence as video_cadence
from drivers.exceptions import CameraStreamStartFailedError
from modules.exceptions import CameraStreamStalledError
from modules.lumascope_api.imaging import ImagingAPI
from modules.notification_center import Severity, notifications
from modules.scope_session import ScopeSession
from rest.app import build_app
from tests.settings_fixtures import complete_settings


@pytest.fixture
def fx2_session(monkeypatch):
    # The stall bound and the check's cadence shortened, as the stall's own
    # tests do, so a stall would be reported well inside the wait below.
    monkeypatch.setattr(video_cadence, 'STALL_FLOOR_S', 0.5)
    monkeypatch.setattr(ImagingAPI, '_STREAM_CHECK_INTERVAL_S', 0.1)
    session = ScopeSession.create(
        complete_settings(microscope='LS620'), simulate=True, warn_pre_release=False
    )
    session.scope.imaging.disable_streaming()
    session.scope.simulation.refuse_stream_start()
    yield session
    session.shutdown()


@pytest.fixture
def stalls():
    seen = []

    def listener(notification):
        if notification.title == CameraStreamStalledError.title:
            seen.append(notification)

    notifications.add_listener(listener, min_severity=Severity.DEBUG)
    yield seen
    notifications.remove_listener(listener)


def test_a_refused_start_raises_its_fault_and_is_not_reported_again_as_a_stall(fx2_session, stalls):
    imaging = fx2_session.scope.imaging

    with pytest.raises(CameraStreamStartFailedError) as raised:
        imaging.enable_streaming()

    assert raised.value.reason == 'camera_stream_start_failed'
    assert 'refused' in str(raised.value)
    assert fx2_session.scope.camera_connected
    assert not imaging.is_streaming()
    # Three stall bounds: a grab still wanted after the refusal is reported
    # inside one (the stall's own tests).
    time.sleep(1.5)
    assert stalls == []


def test_the_start_after_a_refused_one_streams(fx2_session):
    imaging = fx2_session.scope.imaging
    with pytest.raises(CameraStreamStartFailedError):
        imaging.enable_streaming()

    imaging.enable_streaming()

    assert imaging.is_streaming()


def test_a_rest_caller_is_answered_with_the_refused_start(fx2_session):
    with TestClient(build_app(fx2_session)) as client:
        answered = client.post('/api/v1/scope/imaging/enable_streaming')

    assert answered.json()['reason'] == 'camera_stream_start_failed'
    assert answered.status_code >= 500
