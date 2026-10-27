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

import pytest

from drivers.exceptions import CameraReadFailedError
from modules.exceptions import CaptureFailedError, HardwareCommandRefusedError
from modules.notification_center import Severity, notifications

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
