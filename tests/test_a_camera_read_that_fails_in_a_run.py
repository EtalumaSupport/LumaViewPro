# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A camera that does not answer a read during a run is a camera fault.

Two reads a run makes can fail. Inside a step, the frame's record reads the
camera's black level (and, on cameras without chunk data, its gain and
exposure); a failed read is a failed capture, a strike on the writer's
counter with its cause in ``captures.failed``, and three in a row end the
run ``failed`` / ``camera_failure``. At run start, the snapshot the run
restores at its end reads the frame size, binning, pixel format and black
level; a failed read there means the run did not begin, ``failed_at_start``
/ ``camera_read_failed``, with the scope left as it was. Before this, the
step's read fell through to the scan's transient retry (a run that
completed, or ``consecutive_scan_failures`` blaming the cable) and the
snapshot's to the run loop's catch-all (``run_loop_crashed``).

A read that fails is met by a reader, never by the simulated camera's own
frame loop, so an armed pixel-format read leaves the frames coming and a run
past it goes on.

Driven end to end on the simulated scope, the fault armed through
``scope.simulation.refuse_camera_read``.
"""

import pathlib
import threading

import pytest

from drivers.exceptions import CameraReadFailedError
from modules.exceptions import CaptureFailedError
from modules.lumascope_api.illumination import LedState
from modules.notification_center import Severity, notifications
from modules.run_events import RunEvents

from tests.test_a_late_write_records_its_frame import _step
from tests.test_a_run_reports_what_it_captured import _run
from tests.test_composite_run_e2e import headless_settings, open_composite_session

# Enough failed reads that every read from the arming on meets one.
EVERY_TIME = 50
# A bound on the wait for the run thread's telling, which follows its cleanup.
TOLD_WAIT_S = 30.0
# Frames the feed delivers after the arming, as proof it still streams; and
# a bound on the wait for them, far past the fast tier's frame interval.
FRAMES_AFTER = 3
FRAMES_WAIT_S = 10.0


@pytest.fixture
def run_session(tmp_path):
    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        yield session, runner, tmp_path / 'runs'


@pytest.fixture
def read_failure_told():
    """Set when a notification titled 'Camera Read Failed' is posted.

    The run thread tells a run that did not begin after its cleanup, so
    after the run's outcome has resolved: the test waits for the telling.
    """
    told = threading.Event()

    def listen(notification):
        if notification.title == 'Camera Read Failed':
            told.set()

    notifications.add_listener(listen, min_severity=Severity.DEBUG)
    try:
        yield told
    finally:
        notifications.remove_listener(listen)


def _four_steps():
    return [_step(f'C{i}', i, x=20.0 + 10 * i, gain=1.0) for i in range(4)]


def _armed_at_scan_start(session, node, count):
    """Events that arm the read fault once the run has taken the camera.

    Armed at the first scan's start only: a scan that fails is run again, and
    its start must not arm a second fault.
    """
    armed = []

    def arm(*_scan):
        if not armed:
            armed.append(True)
            session.scope.simulation.refuse_camera_read(node, 'raise', count=count)

    return RunEvents(scan_started=arm)


class TestTheSimulatorsOwnFeed:
    """An injected failed read is met by a reader, never by the simulated
    camera's frame loop: a real driver tags a frame from the frame, not by
    reading a node."""

    def test_an_armed_pixel_format_read_is_the_next_reader_s_and_the_frames_keep_coming(
        self, sim_scope
    ):
        camera = sim_scope.camera
        arrived = threading.Event()
        after = []

        def count(_frame):
            after.append(True)
            if len(after) >= FRAMES_AFTER:
                arrived.set()

        sim_scope.simulation.refuse_camera_read('pixel_format', 'raise', count=1)
        sim_scope.imaging.add_frame_listener(count, name='frames after the arming')
        try:
            assert arrived.wait(FRAMES_WAIT_S), (
                f'the feed stopped: {len(after)} frames after the arming'
            )
        finally:
            sim_scope.imaging.remove_frame_listener(count)

        # Frames were made past the arming, and the fault is still the reader's.
        with pytest.raises(CameraReadFailedError, match='pixel format'):
            camera.read_pixel_format()
        assert camera.read_pixel_format() == 'Mono8', 'the one failed read was not one'

    def test_a_run_goes_on_past_one_failed_pixel_format_read(self, run_session):
        session, runner, run_parent = run_session

        outcome = _run(
            runner, run_parent, _four_steps(), _armed_at_scan_start(session, 'pixel_format', 1)
        )

        assert outcome.status == 'completed', outcome
        assert outcome.captures.captured == 4, outcome.captures
        # No capture reads the format; the run-end restore does, so the one
        # failed read is the restore's, listed by cleanup.
        assert outcome.cleanup_failures, outcome


class TestInsideAStep:
    def test_one_failed_record_read_is_that_step_s_failed_capture(self, run_session):
        session, runner, run_parent = run_session

        outcome = _run(
            runner, run_parent, _four_steps(), _armed_at_scan_start(session, 'black_level', 1)
        )

        assert outcome.status == 'incomplete', outcome
        assert outcome.captures.captured == 3, outcome.captures
        (failed,) = outcome.captures.failed
        assert failed.cause == str(CaptureFailedError('record_read_failed')), failed

    def test_record_reads_that_keep_failing_end_the_run_as_a_camera_failure(self, run_session):
        session, runner, run_parent = run_session

        outcome = _run(
            runner,
            run_parent,
            _four_steps(),
            _armed_at_scan_start(session, 'black_level', EVERY_TIME),
        )

        assert (outcome.status, outcome.reason) == ('failed', 'camera_failure'), outcome
        assert len(outcome.captures.failed) == 3, outcome.captures
        assert str(CaptureFailedError('record_read_failed')) in outcome.message, outcome.message
        # The run-end restore reads the camera too; its failure is cleanup's
        # to list and does not change the ending.


class TestAtRunStart:
    # parametrize-justified: each node is a different read of the snapshot,
    # and each needs its own session, since a read fault outlives the run.
    @pytest.mark.parametrize('node', ['frame_size', 'binning', 'pixel_format', 'black_level'])
    def test_a_failed_snapshot_read_ends_the_run_failed_at_start_leaving_the_scope_be(
        self, run_session, read_failure_told, node
    ):
        session, runner, run_parent = run_session
        scope = session.scope
        scope.illumination.led_on('BF', 20.0)
        scope.simulation.refuse_camera_read(node, 'raise', count=EVERY_TIME)

        outcome = _run(runner, run_parent, _four_steps())

        assert (outcome.status, outcome.reason) == ('failed_at_start', 'camera_read_failed'), (
            outcome
        )
        assert outcome.title == 'Camera Read Failed'
        assert 'simulated' not in outcome.message, 'the SDK text reached the person'
        assert outcome.captures.captured == 0
        assert read_failure_told.wait(TOLD_WAIT_S), 'the run that did not begin was not told'
        assert scope.illumination.get_led_state('BF').state is LedState.LIT, (
            'a run that did not begin darkened the LED the user lit'
        )
        empty = [
            d for d in pathlib.Path(run_parent).rglob('*') if d.is_dir() and not any(d.iterdir())
        ]
        assert empty == [], 'an empty run folder was left'

    def test_a_camera_gone_from_the_bus_at_the_snapshot_is_a_disconnect(self, run_session):
        session, runner, run_parent = run_session
        session.scope.simulation.refuse_camera_read('black_level', 'removed')

        outcome = _run(runner, run_parent, _four_steps())

        assert (outcome.status, outcome.reason) == (
            'failed_at_start',
            'hardware_disconnected',
        ), outcome
        assert outcome.captures.captured == 0
