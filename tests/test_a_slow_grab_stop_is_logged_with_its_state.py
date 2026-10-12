"""A Basler grab stop is logged with the state it stopped from and how long it took.

The contract is the pylon driver's ``stop_grabbing``: the camera log carries,
before the stop, what was in flight, and after it, the stop's elapsed time,
a WARNING at ``_SLOW_GRAB_STOP_S`` or more so a slow stop reaches the errors
log and stands out in a support bundle. The log is the product here: it is
what a bench run on a stalling camera is read from.

A stand-in by design: no simulated Pylon camera exists, so the SDK camera is
``camera_fakes.bare_pylon_camera``'s MagicMock, and the camera log is a real
logger with a list handler patched over ``pyloncamera._cam_log`` (the suite
mocks ``lvp_logger``, so ``caplog`` would see nothing).
"""

import logging
import time
import types
from unittest import mock

from pypylon import genicam

from tests.camera_fakes import bare_pylon_camera


class _Records(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _stop_with_log(cam, stop_takes_s):
    """Run ``cam.stop_grabbing()`` with a clock the stop advances by ``stop_takes_s``."""
    from drivers import pyloncamera

    clock = [100.0]

    def stop():
        clock[0] += stop_takes_s

    if cam.active is not None:
        cam.active.StopGrabbing.side_effect = stop
    fake_time = types.SimpleNamespace(
        perf_counter=lambda: clock[0], monotonic=time.monotonic, sleep=time.sleep
    )
    handler = _Records()
    log = logging.getLogger('test.camera_stop')
    log.handlers = [handler]
    log.setLevel(logging.DEBUG)
    log.propagate = False
    with (
        mock.patch.object(pyloncamera, '_cam_log', log),
        mock.patch.object(pyloncamera, 'time', fake_time),
    ):
        cam.stop_grabbing()
    return handler.records


def test_a_slow_stop_is_a_warning_with_its_elapsed_time_after_the_state_it_stopped_from():
    cam = bare_pylon_camera()
    cam.cam_image_handler = types.SimpleNamespace(frames_delivered=42)
    cam.active.GetStreamGrabberNodeMap.return_value.GetNode.return_value.GetValue.return_value = 7
    cam.active.GetInstantCameraNodeMap.return_value.GetNode.return_value.GetValue.return_value = 3
    cam.active.GetNodeMap.return_value.GetNode.return_value.GetValue.return_value = 'Complete'

    records = _stop_with_log(cam, stop_takes_s=11.02)

    cam.active.StopGrabbing.assert_called_once_with()
    pre = [r.getMessage() for r in records if r.getMessage().startswith('pylon StopGrabbing() [')]
    assert len(pre) == 1, [r.getMessage() for r in records]
    for field in (
        'frames_delivered=42 ',
        'grabber_status=7 ',
        'Statistic_Total_Buffer_Count=7 ',
        'NumQueuedBuffers=3 ',
        'BslAcquisitionStopMode=Complete ',
        'AcquisitionActive=Complete ExposureActive=Complete live_read_ms=0.0',
    ):
        assert field in pre[0], f'{field} missing from the pre-stop line: {pre[0]}'
    warnings = [r.getMessage() for r in records if r.levelno == logging.WARNING]
    assert warnings == ['pylon StopGrabbing [stop_grabbing] returned after 11020.0 ms'], warnings


def test_a_fast_stop_logs_its_elapsed_time_and_no_warning():
    cam = bare_pylon_camera()

    records = _stop_with_log(cam, stop_takes_s=0.02)

    assert not [r for r in records if r.levelno >= logging.WARNING]
    assert 'pylon StopGrabbing [stop_grabbing] returned after 20.0 ms' in [
        r.getMessage() for r in records
    ]


def test_a_node_the_camera_lacks_is_named_and_the_stop_still_runs():
    cam = bare_pylon_camera()
    cam.active.GetNodeMap.return_value.GetNode.side_effect = genicam.LogicalErrorException(
        'Node not existing'
    )

    records = _stop_with_log(cam, stop_takes_s=0.02)

    cam.active.StopGrabbing.assert_called_once_with()
    pre = next(
        r.getMessage() for r in records if r.getMessage().startswith('pylon StopGrabbing() [')
    )
    assert 'BslAcquisitionStopMode=<missing>' in pre, pre


def test_a_stop_with_no_camera_logs_its_failure_and_raises_nothing():
    cam = bare_pylon_camera()
    cam.active = None

    records = _stop_with_log(cam, stop_takes_s=0.0)

    messages = [r.getMessage() for r in records]
    assert 'pylon StopGrabbing() [stop_grabbing]: no camera' in messages
    assert any(
        r.levelno == logging.WARNING and 'FAILED after' in r.getMessage() for r in records
    ), messages
