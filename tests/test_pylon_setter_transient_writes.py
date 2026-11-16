# Copyright Etaluma, Inc.
"""A Pylon setting write that raises RuntimeException is that write's failure,
not the camera's removal.

The camera is marked removed only on a definite signal -- the SDK's removal
callback, a grab result of DEVICE_NOT_FOUND, the consecutive-failure cascade, a
released handle -- and a removal has no reset short of a reconnect: frames
stop and every later camera command is refused. One RuntimeException from one
node write is none of those, so the write raises as one the camera did not
take (``CameraWriteFailedError``) and the camera stays connected. The
getters' half of the same contract is ``test_pylon_getter_transient_reads``.

One test walks every Pylon setting writer and names each that marks the
camera removed or answers the failure as applied. Every writer writes through
``write_node`` (``tests/guards/test_a_camera_node_has_one_writer.py``), so
its one failure form is the one these share.
"""

from unittest.mock import MagicMock

import pytest
from pypylon import genicam

from _lumascope.drivers.exceptions import (
    CameraNoAnswerError,
    CameraReadFailedError,
    CameraWriteFailedError,
)
from tests.camera_fakes import bare_pylon_camera, nodemaps_answer_attributes

_WRITES = frozenset({'SetValue', 'SetIntValue', 'Execute'})


def _comms_lost(*_args, **_kwargs):
    raise genicam.RuntimeException('comms lost')


def _timed_out(*_args, **_kwargs):
    raise genicam.TimeoutException('no answer in time')


class _WritesFail(MagicMock):
    """A camera handle whose every node write raises ``failure`` (RuntimeException).

    a stand-in by design: the SDK below the driver is the subject's
    collaborator, and no simulated camera runs pyloncamera.py. Reads answer
    as a MagicMock does; every write -- a SetValue / SetIntValue / Execute
    call or a ``.Value`` assignment, on any node however reached -- raises.
    """

    failure = staticmethod(_comms_lost)

    def _get_child_mock(self, **kw):
        child = type(self)(**kw)
        if kw.get('name') in _WRITES:
            child.side_effect = self.failure
        return child

    def __setattr__(self, name, value):
        if name == 'Value':
            self.failure()
        super().__setattr__(name, value)


class _WritesTimeOut(_WritesFail):
    """A camera handle whose every node write is not answered in time."""

    failure = staticmethod(_timed_out)


def _camera(handle=_WritesFail):
    """A bare PylonCamera with the real removal latch and a failing handle."""
    cam = bare_pylon_camera()
    del cam._mark_disconnected
    cam._device_removed = False
    cam.active = handle()
    nodemaps_answer_attributes(cam)
    cam.active.IsGrabbing.return_value = False
    cam.active.PixelFormat.GetSymbolics.return_value = ('Mono8', 'Mono12')
    cam.active.PixelFormat.GetValue.return_value = 'Mono8'
    cam.active.Gain.GetValue.return_value = 0.0
    cam.active.ExposureTime.Min = 20.0
    cam.active.ExposureTime.GetValue.return_value = 10000.0
    for axis, size in (('Width', 4096), ('Height', 3000)):
        node = getattr(cam.active, axis)
        node.GetInc.return_value = 4
        node.GetMin.return_value = 16
        node.GetMax.return_value = size
    cam.profile = None  # the default exposure ceiling, well above the 20 ms asked for
    cam._use_camera_emulation = False
    return cam


# (the writer, the call a caller makes, the failure answer). 'raises' is the
# answer for a write not taken ('read_failed' for latch_clock, whose write
# starts a read). init_camera_config's steps are each named in
# config_not_applied (test_a_camera_config_step_not_taken_is_reported); with
# every write failing, its orientation write fails too, and a camera whose
# orientation is not taken raises.
_CASES = [
    ('init_camera_config', lambda c: c.init_camera_config(), 'raises'),
    ('_write_reverse', lambda c: c._write_reverse('ReverseY', True), 'raises'),
    # A bench knob; each knob's answer to a node it lacks is
    # test_a_pylon_node_the_camera_lacks.
    ('set_gev_packet_size', lambda c: c.set_gev_packet_size(1500), 'raises'),
    ('set_pixel_format', lambda c: c.set_pixel_format('Mono12'), 'raises'),
    ('_set_hardware_binning', lambda c: c.set_binning_size(2), 'raises'),
    ('set_conversion_gain_mode', lambda c: c.set_conversion_gain_mode('High'), 'raises'),
    ('set_line_noise_reduction', lambda c: c.set_line_noise_reduction(True), 'raises'),
    ('set_black_level', lambda c: c.set_black_level(2.0), 'raises'),
    ('init_auto_gain_focus', lambda c: c.init_auto_gain_focus(), 'raises'),
    (
        'update_auto_gain_target_brightness',
        lambda c: c.update_auto_gain_target_brightness(0.4),
        'raises',
    ),
    ('update_auto_gain_min_max', lambda c: c.update_auto_gain_min_max(0.0, 12.0), 'raises'),
    ('_set_hardware_window', lambda c: c.set_frame_size(1024, 1024), 'raises'),
    ('gain', lambda c: c.gain(12.0), 'raises'),
    ('auto_gain', lambda c: c.auto_gain(True), 'raises'),
    ('auto_gain_once', lambda c: c.auto_gain_once(True), 'raises'),
    ('exposure_t', lambda c: c.exposure_t(20.0), 'raises'),
    ('auto_exposure_t', lambda c: c.auto_exposure_t(True), 'raises'),
    ('set_test_pattern', lambda c: c.set_test_pattern(pattern='White'), 'raises'),
    ('_set_auto_exposure_time_bounds', lambda c: c._set_auto_exposure_time_bounds(50.0), 'raises'),
    ('latch_clock', lambda c: c.latch_clock(), 'read_failed'),
]


def _answered(answer, call, cam):
    """Whether ``call`` gave the failure answer ``answer``; the reason if not."""
    try:
        result = call(cam)
    except CameraWriteFailedError:
        return answer == 'raises', 'raised CameraWriteFailedError'
    except CameraReadFailedError:
        return answer == 'read_failed', 'raised CameraReadFailedError'
    except Exception as e:  # the answer under test: any other raise is named
        return False, f'raised {type(e).__name__}: {e}'
    return False, f'answered {result!r}'


@pytest.mark.filterwarnings('ignore::pytest.PytestUnraisableExceptionWarning')
def test_a_runtime_exception_on_a_write_is_its_failure_not_a_removal():
    offenders = []
    for name, call, answer in _CASES:
        cam = _camera()
        ok, said = _answered(answer, call, cam)
        if cam.is_device_removed():
            offenders.append(f'{name}: marked the camera removed ({said})')
        elif not ok:
            offenders.append(f'{name}: expected {answer}, {said}')
    assert not offenders, '\n'.join(offenders)


@pytest.mark.filterwarnings('ignore::pytest.PytestUnraisableExceptionWarning')
def test_a_write_the_camera_does_not_answer_is_no_answer_from_every_writer():
    """A timeout is answered one way by every writer: ``CameraNoAnswerError``,
    which the API sends once more, never a refusal and never a removal
    (``latch_clock``'s write starts a read, so it is that read's failure)."""
    offenders = []
    for name, call, answer in _CASES:
        cam = _camera(_WritesTimeOut)
        expected = CameraReadFailedError if answer == 'read_failed' else CameraNoAnswerError
        try:
            result = call(cam)
        except expected:
            pass
        except Exception as e:  # the answer under test: any other raise is named
            offenders.append(f'{name}: expected {expected.__name__}, raised {type(e).__name__}')
        else:
            offenders.append(f'{name}: expected {expected.__name__}, answered {result!r}')
        if cam.is_device_removed():
            offenders.append(f'{name}: marked the camera removed')
    assert not offenders, '\n'.join(offenders)
