# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A camera stream that stops delivering is reported by the imaging API, in every host.

A camera can stall without being removed: it stays connected and grabbing,
and no frame arrives. Nobody is waiting on the stream when that happens, so
the imaging API watches its own frame count on the session's scheduler and
reports ``CameraStreamStalledError`` unsolicited, once per stall. The check
used to live in the GUI's display thread, so a headless session -- REST, a
script -- had no stall report at all; every session here is headless, with
no display thread running.

The stall bound is the recording feed's: the larger of a floor and ten
frames at the current exposure, so a long exposure is never a stall. The
floor and the check's cadence are shortened here so a stall is seen in a
fraction of a second.
"""

from __future__ import annotations

import logging
import time

import pytest

import modules.video_cadence as video_cadence
from drivers.simulated_camera import SimulatedStall
from modules.exceptions import CameraStreamStalledError
from modules.lumascope_api.imaging import ImagingAPI
from modules.notification_center import Severity, notifications
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings

STALL_TITLE = CameraStreamStalledError.title


@pytest.fixture
def fast_check(monkeypatch):
    monkeypatch.setattr(video_cadence, 'STALL_FLOOR_S', 0.5)
    monkeypatch.setattr(ImagingAPI, '_STREAM_CHECK_INTERVAL_S', 0.1)


@pytest.fixture
def session(fast_check):
    session = ScopeSession.create(complete_settings(), simulate=True)
    try:
        yield session
    finally:
        session.shutdown()


@pytest.fixture
def shown():
    seen = []

    def listener(notification):
        if notification.shown and notification.title == STALL_TITLE:
            seen.append(notification)

    notifications.add_listener(listener, min_severity=Severity.WARNING)
    try:
        yield seen
    finally:
        notifications.remove_listener(listener)


def _stall_lines(caplog):
    """The reporter's one log line per stall reported (shown or muted)."""
    return [
        r
        for r in caplog.records
        if r.levelno == logging.ERROR and 'raised CameraStreamStalledError' in r.getMessage()
    ]


def _wait_for(predicate, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


class TestAStallIsReported:
    def test_a_stall_is_shown_once_as_an_unsolicited_camera_fault(self, session, shown, caplog):
        camera = session.scope._camera_driver
        with caplog.at_level(logging.INFO):
            camera.hold_frames(SimulatedStall(after_s=0.2, for_s=1.5))
            assert _wait_for(lambda: shown, 3.0), 'a stall was not reported'
            time.sleep(0.6)  # still inside the stall: no second report
        assert len(shown) == 1
        notice = shown[0]
        assert notice.severity == Severity.ERROR
        assert notice.category == 'Camera'
        assert 'no new frame' in notice.message
        assert notice.solicited is False
        assert len(_stall_lines(caplog)) == 1

    def test_the_check_rearms_when_frames_resume(self, session, caplog):
        camera = session.scope._camera_driver
        with caplog.at_level(logging.INFO):
            camera.hold_frames(SimulatedStall(after_s=0.1, for_s=1.0))
            assert _wait_for(lambda: len(_stall_lines(caplog)) == 1, 3.0)
            time.sleep(1.2)  # past the end of the first stall: frames flow again
            camera.hold_frames(SimulatedStall(after_s=0.1, for_s=1.0))
            assert _wait_for(lambda: len(_stall_lines(caplog)) == 2, 3.0), (
                'a second stall after frames resumed was not seen'
            )


class TestWhatIsNotAStall:
    def test_a_long_exposure_is_not_a_stall(self, session, shown, caplog):
        # 800 ms between frames is past the 0.5 s floor; ten of them is the bound.
        imaging = session.scope.imaging
        imaging.set_exposure_ms(800.0)
        assert imaging.exposure_ms_cached == 800.0, 'precondition: the long exposure took'
        before = imaging._frames_delivered()
        with caplog.at_level(logging.INFO):
            time.sleep(2.5)
        delivered = imaging._frames_delivered() - before
        assert 1 <= delivered <= 4, (
            f'precondition: frames arrived about 0.8 s apart, past the floor ({delivered} in 2.5 s)'
        )
        assert shown == []
        assert _stall_lines(caplog) == []

    def test_a_camera_that_is_not_streaming_is_not_stalled(self, session, shown, caplog):
        session.scope.imaging.stop_streaming()
        with caplog.at_level(logging.INFO):
            time.sleep(1.5)
        assert shown == []
        assert _stall_lines(caplog) == []


class TestWhereItIsNotShown:
    def test_an_unattended_run_logs_the_stall_and_shows_nothing(self, session, shown, caplog):
        camera = session.scope._camera_driver
        notifications.open_run_scope(attended=False)
        try:
            with caplog.at_level(logging.INFO):
                camera.hold_frames(SimulatedStall(after_s=0.1, for_s=1.5))
                assert _wait_for(lambda: _stall_lines(caplog), 3.0), 'the stall was not seen'
                time.sleep(0.3)
        finally:
            notifications.close_run_scope()
        assert shown == [], 'an unattended run showed a non-fatal fault'
        assert len(_stall_lines(caplog)) == 1, 'the muted stall was not logged once as a fault'


class TestTheCheckEndsWithTheScope:
    def test_disconnect_stops_the_check(self, fast_check):
        session = ScopeSession.create(complete_settings(), simulate=True)
        imaging = session.scope.imaging
        assert imaging._stream_check_handle is not None, 'bring-up did not start the check'
        session.shutdown()
        assert imaging._stream_check_handle is None
