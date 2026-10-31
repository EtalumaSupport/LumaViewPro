# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A capture returns a frame or raises why it got none.

``capture_and_wait`` and ``get_image`` once answered every failure with
None and kept the reason in a side record, whose ladder ended in "camera
inactive or not grabbing" -- the wrong cause for the commonest fault, a
stalled stream on a grabbing camera. Now a capture that got no frame raises
``CaptureFailedError`` with one reason per cause, a camera that is not there
is the ``not_connected`` refusal every camera command raises, and a camera
that is not streaming is refused ``camera_not_streaming``. Nothing inside
the API reports a removal as well: the raise is the report.
"""

from __future__ import annotations

import logging
import re

import pytest

from drivers.exceptions import CameraReadFailedError
from modules.exceptions import CaptureFailedError, HardwareCommandRefusedError
from modules.lumascope_api import imaging
from modules.notification_center import Severity, notifications
from tests.log_capture import records_of

# The facts a give-up line appends: the wait, the feed's two counts across
# it, and the last stored frame's age at the give-up.
_FEED_ACCOUNT = re.compile(
    r'waited (?P<waited>\d+\.\d+)s from steady \d+\.\d+s; '
    r'handed over (?P<handed_from>\d+)->(?P<handed_to>\d+), '
    r'stored (?P<stored_from>\d+)->(?P<stored_to>\d+); '
    r'last stored seq=\d+, (?P<age>\d+\.\d+)s before'
)


def _give_up_lines(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in records_of(caplog, imaging)
        if r.levelno >= logging.WARNING and 'waited ' in r.getMessage()
    ]


AUTO_GAIN = {
    'target_brightness': 0.5,
    'min_gain_db': 0.0,
    'max_gain_db': 20.0,
    'max_exposure_ms': 100.0,
    'min_exposure_ms': 0.1,
}


@pytest.fixture
def posted():
    """Every notification posted while the test runs."""
    seen = []

    def listen(notification):
        seen.append(notification.title)

    notifications.add_listener(listen, min_severity=Severity.DEBUG)
    try:
        yield seen
    finally:
        notifications.remove_listener(listen)


class TestAStalledStream:
    @pytest.mark.parametrize('pending', [True, False], ids=['draining', 'grabbing'])
    def test_it_raises_no_frame_whether_or_not_a_change_was_pending(self, sim_scope, pending):
        # One fault, one reason: the drain leg and the grab leg once named
        # the same stall under different causes.
        if pending:
            sim_scope.imaging.frame_validity.invalidate('exposure', began_ns=None)
        sim_scope.simulation.stall_stream(0.0, 30.0)

        with pytest.raises(CaptureFailedError) as failed:
            sim_scope.imaging.capture_and_wait(timeout_s=1.0)

        assert failed.value.reason == 'no_frame'
        assert str(failed.value) == str(CaptureFailedError('no_frame'))

    def test_get_image_raises_it_too(self, sim_scope):
        sim_scope.simulation.stall_stream(0.0, 30.0)

        with pytest.raises(CaptureFailedError) as failed:
            sim_scope.imaging.get_image(
                force_new_capture=True, timeout_s=0.5, new_capture_timeout_s=0.5
            )

        assert failed.value.reason == 'no_frame'


class TestAGiveUpStatesItsWait:
    """A capture that gives up waiting for a frame logs how long it waited and
    what the camera's feed did meanwhile, so a red or a field log names whether
    the feed stopped, the store stopped, or frames were stored and not taken.
    The line is the product here: it is what a red's report carries."""

    @pytest.mark.parametrize('pending', [True, False], ids=['draining', 'grabbing'])
    def test_a_held_feed_names_the_wait_no_frame_in_it_and_the_last_frames_age(
        self, sim_scope, caplog, pending
    ):
        sim_scope.imaging.capture_and_wait(timeout_s=1.0)  # a frame stored before the hold
        if pending:
            sim_scope.imaging.frame_validity.invalidate('exposure', began_ns=None)
        sim_scope.simulation.stall_stream(0.0, 30.0)
        caplog.clear()

        with pytest.raises(CaptureFailedError):
            sim_scope.imaging.capture_and_wait(timeout_s=1.0)

        lines = _give_up_lines(caplog)
        assert len(lines) == 1, f'one give-up line, stating its wait; got {lines}'
        account = _FEED_ACCOUNT.search(lines[0])
        assert account is not None, f'the line does not state the wait: {lines[0]}'
        assert float(account['waited']) >= 1.0
        assert account['handed_from'] == account['handed_to'], 'a frame handed over in a hold'
        assert account['stored_from'] == account['stored_to'], 'a frame stored in a hold'
        assert float(account['age']) >= float(account['waited']), (
            'the last frame arrived before the wait began'
        )

    def test_a_feed_released_inside_the_bound_captures_and_states_no_give_up(
        self, sim_scope, caplog
    ):
        sim_scope.simulation.stall_stream(0.0, 0.3)

        sim_scope.imaging.capture_and_wait(timeout_s=1.0)

        assert _give_up_lines(caplog) == []


class TestARemovedCamera:
    def test_a_removal_mid_capture_is_the_not_connected_refusal_reported_by_no_one_inside(
        self, sim_scope, posted
    ):
        # The camera leaves between the two frames of a sum: past the lane's
        # presence question, inside the grab.
        with pytest.raises(HardwareCommandRefusedError) as refused:
            sim_scope.imaging.capture_and_wait(
                sum_count=2,
                sum_iteration_callback=sim_scope.simulation.remove_camera,
                timeout_s=0.5,
            )

        assert refused.value.reason == 'not_connected'
        assert posted == [], f'the raise is the report; the API also posted {posted}'

    def test_under_a_live_view_auto_gain_arm_it_is_not_reported_again_by_the_re_arm(
        self, sim_scope, posted
    ):
        sim_scope.camera.set_auto_gain(True, AUTO_GAIN)
        posted.clear()

        with pytest.raises(HardwareCommandRefusedError) as refused:
            sim_scope.imaging.capture_and_wait(
                sum_count=2,
                sum_iteration_callback=sim_scope.simulation.remove_camera,
                timeout_s=0.5,
            )

        assert refused.value.reason == 'not_connected'
        assert posted == [], f'one removal, told by its raise; the API also posted {posted}'

    def test_get_image_after_the_removal_latch_is_refused_at_once(self, sim_scope, posted):
        # The latch is set the moment a removal is seen and the handle is
        # released later, off-thread: ``active`` is still truthy here.
        sim_scope.simulation.remove_camera()
        assert sim_scope._camera_driver.active

        with pytest.raises(HardwareCommandRefusedError) as refused:
            sim_scope.imaging.get_image(force_new_capture=True)

        assert refused.value.reason == 'not_connected'
        assert posted == []


class TestARecordReadTheCameraDoesNotAnswer:
    def test_it_fails_the_capture_chained_from_the_read(self, sim_scope):
        sim_scope.simulation.refuse_camera_read('black_level', 'failed')

        with pytest.raises(CaptureFailedError) as failed:
            sim_scope.imaging.capture_and_wait(timeout_s=1.0)

        assert failed.value.reason == 'record_read_failed'
        assert isinstance(failed.value.__cause__, CameraReadFailedError)

    def test_from_a_camera_gone_from_the_bus_it_is_the_not_connected_refusal(self, sim_scope):
        sim_scope.simulation.refuse_camera_read('black_level', 'removed')

        with pytest.raises(HardwareCommandRefusedError) as refused:
            sim_scope.imaging.capture_and_wait(timeout_s=1.0)

        assert refused.value.reason == 'not_connected'


def test_a_camera_that_is_not_streaming_is_refused_as_that(sim_scope):
    sim_scope.imaging.disable_streaming()

    with pytest.raises(HardwareCommandRefusedError) as refused:
        sim_scope.imaging.capture_and_wait(timeout_s=1.0)

    assert refused.value.reason == 'camera_not_streaming'
    assert refused.value.title == 'Camera Not Streaming'


def test_a_no_frame_capture_under_a_live_arm_leaves_the_arm_re_armed(sim_scope):
    # The camera is still there, so the live view keeps adjusting after a
    # capture that got nothing; only a removal skips the re-arm.
    sim_scope.camera.set_auto_gain(True, AUTO_GAIN)

    def stall():
        sim_scope.simulation.stall_stream(0.0, 30.0)

    with pytest.raises(CaptureFailedError):
        sim_scope.imaging.capture_and_wait(sum_count=2, sum_iteration_callback=stall, timeout_s=0.5)

    # A lock takes the arm that stands; with none it locks nothing.
    assert sim_scope.camera.lock_auto_gain().state is not None, 'the arm was not re-armed'
