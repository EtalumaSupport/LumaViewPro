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

import itertools
import logging
import threading
import time

import pytest

from drivers.simulated_camera import SimulatedCamera
import modules.video_cadence as video_cadence
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
        with caplog.at_level(logging.INFO):
            session.scope.simulation.stall_stream(0.2, 1.5)
            assert _wait_for(lambda: shown, 3.0), 'a stall was not reported'
            time.sleep(0.6)  # still inside the stall: no second report
        assert len(shown) == 1
        notice = shown[0]
        assert notice.severity == Severity.ERROR
        assert notice.category == 'Camera'
        assert 'no new frame' in notice.message
        # A pulled cable meets this check before its driver reports the
        # removal, so the words point at the cable and never claim a connection.
        assert 'USB cable' in notice.message
        assert 'connected' not in notice.message
        assert notice.solicited is False
        assert len(_stall_lines(caplog)) == 1

    def test_frames_handed_over_and_never_stored_are_a_stall(self, session, shown):
        """A store stage that has wedged while the camera still hands frames
        over -- Pylon's worker, the IDS unpack worker, the FX2 grab loop --
        delivers nothing a reader can have, so it is a stall. The simulator's
        backlog holds every frame between its hand-over and the store, as
        such a wedge does: the hand-over count keeps rising and the stored
        count does not."""
        session.scope.simulation.delay_delivery(1_000_000)
        assert _wait_for(lambda: shown, 3.0), 'frames handed over and never stored were not a stall'

    def test_the_check_rearms_when_frames_resume(self, session, caplog):
        with caplog.at_level(logging.INFO):
            session.scope.simulation.stall_stream(0.1, 1.0)
            assert _wait_for(lambda: len(_stall_lines(caplog)) == 1, 3.0)
            # The check re-arms once it sees frames again -- the state this
            # test is about; nothing public states it, and a clock only
            # guesses when the check has looked.
            imaging = session.scope.imaging
            assert _wait_for(lambda: not imaging._stream_stall_reported, 10.0), (
                'the check never saw frames resume'
            )
            session.scope.simulation.stall_stream(0.1, 1.0)
            assert _wait_for(lambda: len(_stall_lines(caplog)) == 2, 3.0), (
                'a second stall after frames resumed was not seen'
            )


class TestAGrabThatStoppedUnaskedIsReported:
    def test_a_grab_the_camera_stopped_itself_is_reported_once(self, session, shown, caplog):
        """A grab that stopped when nobody asked -- Pylon's SDK after
        OnGrabError, an FX2 grab thread that died -- left the camera
        connected and not grabbing, which the check took for a deliberate
        stop: no stall was ever reported, and every capture was refused
        camera_not_streaming with nothing saying why."""
        driver = session.scope._camera_driver
        with caplog.at_level(logging.INFO):
            # A stand-in by design: the simulator cannot yet stop its own
            # grab, so the SDK's stop is made here, past stop_grabbing.
            driver._grabbing = False
            driver._stop_acquisition()
            assert _wait_for(lambda: shown, 3.0), 'a grab stopped unasked was not reported'
            time.sleep(0.6)
        assert len(shown) == 1
        assert len(_stall_lines(caplog)) == 1


class TestWhatIsNotAStall:
    def test_a_long_exposure_is_not_a_stall(self, session, shown, caplog):
        # 800 ms between frames is past the 0.5 s floor; ten of them is the bound.
        scope = session.scope
        camera = scope.camera
        camera.set_exposure_ms(800.0)
        assert camera.exposure_ms == 800.0, 'precondition: the long exposure took'
        # The frames the camera had started come at the old, short exposure,
        # counted from where the gate recorded the change, after the write:
        # frames that arrive while a slow write is still in flight are short
        # too, and a count taken before the write would spend the pipeline's
        # frames on them.
        at_change = scope.imaging.recorded_changes()['exposure'].at_seq
        in_flight = SimulatedCamera.SENSOR_PIPELINE_FRAMES
        assert _wait_for(lambda: scope._frames_delivered() > at_change + in_flight, 3.0)
        arrivals = []
        three = threading.Event()

        def arrived(frame):
            arrivals.append(time.monotonic())
            if len(arrivals) >= 3:
                three.set()

        scope.imaging.add_frame_listener(arrived)
        try:
            with caplog.at_level(logging.INFO):
                assert three.wait(10.0), 'precondition: the long-exposure stream delivered'
        finally:
            scope.imaging.remove_frame_listener(arrived)
        # The check saw the stream silent past the floor and reported nothing.
        # Measured between arrivals, the clock the check runs on: frames come
        # on the camera's schedule, so a frame a paused host took late is
        # followed at once by the next one due, and a span over the frames
        # can be shorter than their exposures; the longest silence is never
        # shorter.
        longest = max(b - a for a, b in itertools.pairwise(arrivals))
        assert longest > video_cadence.STALL_FLOOR_S, (
            f'precondition: the stream was never silent past the floor (longest {longest:.2f} s)'
        )
        assert shown == []
        assert _stall_lines(caplog) == []

    def test_the_frames_exposed_before_a_drop_to_a_short_exposure_are_not_a_stall(
        self, session, shown, caplog
    ):
        # A sensor still delivers the frames it exposed before the change, at
        # the old exposure: the simulated camera's come 1 s apart after the
        # drop, past the short exposure's 0.5 s bound and inside the long
        # one's 10 s. Once the camera's pipeline of frames has come the short
        # bound applies: a hold of 1.5 s after it is a stall.
        scope = session.scope
        camera = scope.camera
        camera.set_exposure_ms(1000.0)
        before = scope._frames_delivered()
        assert _wait_for(lambda: scope._frames_delivered() > before, 3.0), (
            'precondition: the long-exposure stream delivered'
        )
        time.sleep(0.3)  # the check has run at the long exposure's bound
        with caplog.at_level(logging.INFO):
            camera.set_exposure_ms(5.0)
            # Counted from the gate's record of the change, as above.
            at_drop = scope.imaging.recorded_changes()['exposure'].at_seq
            in_flight = SimulatedCamera.SENSOR_PIPELINE_FRAMES
            # The frames in flight, and one at the short exposure behind them.
            assert _wait_for(lambda: scope._frames_delivered() > at_drop + in_flight + 1, 5.0)
            assert shown == [], 'the frames in the pipeline were reported as a stall'
            assert _stall_lines(caplog) == []

            session.scope.simulation.stall_stream(0.3, 1.5)
            assert _wait_for(lambda: shown, 3.0), 'the short bound never came back'

    def test_a_camera_that_is_not_streaming_is_not_stalled(self, session, shown, caplog):
        session.scope.imaging.disable_streaming()
        with caplog.at_level(logging.INFO):
            time.sleep(1.5)
        assert shown == []
        assert _stall_lines(caplog) == []


class TestWhereItIsNotShown:
    def test_an_unattended_run_logs_the_stall_and_shows_nothing(self, session, shown, caplog):
        notifications.open_run_scope(attended=False)
        try:
            with caplog.at_level(logging.INFO):
                session.scope.simulation.stall_stream(0.1, 1.5)
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
