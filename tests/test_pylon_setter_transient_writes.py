# Copyright Etaluma, Inc.
"""A Pylon setting write that raises RuntimeException is that write's failure,
not the camera's removal.

The camera is marked removed only on a definite signal -- the SDK's removal
callback, a grab result of DEVICE_NOT_FOUND, the consecutive-failure cascade, a
released handle -- and a removal has no reset short of a reconnect: frames
stop and every later camera command is refused. One RuntimeException from one
node write is none of those, so the write answers as one the camera did not
take (False, or ``CameraWriteFailedError``) and the camera stays connected.
The getters' half of the same contract is ``test_pylon_getter_transient_reads``.

One test walks every Pylon method that writes a node inside a ``try`` and
names each that marks the camera removed or answers the failure as applied.
The table is checked against the driver's source, so a new writer cannot be
left out.
"""

import ast
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pypylon import genicam

from drivers.exceptions import CameraReadFailedError, CameraWriteFailedError
from tests.camera_fakes import bare_pylon_camera

_WRITES = frozenset({'SetValue', 'SetIntValue', 'Execute'})


def _comms_lost(*_args, **_kwargs):
    raise genicam.RuntimeException('comms lost')


class _WritesFail(MagicMock):
    """A camera handle whose every node write raises RuntimeException.

    a stand-in by design: the SDK below the driver is the subject's
    collaborator, and no simulated camera runs pyloncamera.py. Reads answer
    as a MagicMock does; every write -- a SetValue / SetIntValue / Execute
    call or a ``.Value`` assignment, on any node however reached -- raises.
    """

    def _get_child_mock(self, **kw):
        child = _WritesFail(**kw)
        if kw.get('name') in _WRITES:
            child.side_effect = _comms_lost
        return child

    def __setattr__(self, name, value):
        if name == 'Value':
            _comms_lost()
        super().__setattr__(name, value)


def _camera():
    """A bare PylonCamera with the real removal latch and a failing handle."""
    cam = bare_pylon_camera()
    del cam._mark_disconnected
    cam._device_removed = False
    cam.active = _WritesFail()
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


# (the writer the handler is in, the call a caller makes, the failure answer).
# 'false' / 'raises' are the two answers for a write not taken ('read_failed'
# for latch_clock, whose write starts a read); 'not_applied'
# is init_camera_config's: the step is named in config_not_applied. 'none' is
# the two writers with no caller, which answer None for every failure.
_CASES = [
    ('init_camera_config', lambda c: c.init_camera_config(), 'not_applied'),
    (
        'set_max_acquisition_frame_rate',
        lambda c: c.set_max_acquisition_frame_rate(True, 5.0),
        'none',
    ),
    (
        'set_device_link_throughput_limit',
        lambda c: c.set_device_link_throughput_limit('Off'),
        'raises',
    ),
    ('_set_stream_grabber_int_node', lambda c: c.set_max_transfer_size(262144), 'raises'),
    ('set_acquisition_stop_mode', lambda c: c.set_acquisition_stop_mode('Complete'), 'raises'),
    ('set_bandwidth_reserve_mode', lambda c: c.set_bandwidth_reserve_mode('Performance'), 'raises'),
    ('set_gev_packet_size', lambda c: c.set_gev_packet_size(1500), 'raises'),
    ('set_gev_inter_packet_delay', lambda c: c.set_gev_inter_packet_delay(100), 'raises'),
    ('set_pixel_format', lambda c: c.set_pixel_format('Mono12'), 'raises'),
    ('_set_hardware_binning', lambda c: c.set_binning_size(2), 'raises'),
    ('set_conversion_gain_mode', lambda c: c.set_conversion_gain_mode('High'), 'raises'),
    ('set_line_noise_reduction', lambda c: c.set_line_noise_reduction(True), 'raises'),
    ('set_black_level', lambda c: c.set_black_level(2.0), 'false'),
    ('init_auto_gain_focus', lambda c: c.init_auto_gain_focus(), 'raises'),
    (
        'update_auto_gain_target_brightness',
        lambda c: c.update_auto_gain_target_brightness(0.4),
        'false',
    ),
    ('update_auto_gain_min_max', lambda c: c.update_auto_gain_min_max(0.0, 12.0), 'false'),
    ('_set_hardware_window', lambda c: c.set_frame_size(1024, 1024), 'false'),
    ('gain', lambda c: c.gain(12.0), 'false'),
    ('auto_gain', lambda c: c.auto_gain(True), 'false'),
    ('auto_gain_once', lambda c: c.auto_gain_once(True), 'false'),
    ('exposure_t', lambda c: c.exposure_t(20.0), 'false'),
    ('auto_exposure_t', lambda c: c.auto_exposure_t(True), 'false'),
    ('set_test_pattern', lambda c: c.set_test_pattern(pattern='White'), 'none'),
    ('_set_auto_exposure_time_bounds', lambda c: c._set_auto_exposure_time_bounds(50.0), 'false'),
    ('set_max_num_buffer', lambda c: c.set_max_num_buffer(16), 'false'),
    ('latch_clock', lambda c: c.latch_clock(), 'read_failed'),
]

# Methods the source rule finds whose answer is pinned where they are owned:
# connect answers a bring-up that failed; the reads write only a selector
# before reading (test_a_failed_temperature_read_raises, the chunk and
# diagnostic probes' tests); _enable_validity_chunks logs each chunk it could
# not enable by design and is a step of init_camera_config, in the table.
_ANSWERED_ELSEWHERE = frozenset(
    {
        'connect',
        'get_all_temperatures',
        'probe_chunk_capabilities',
        'read_diagnostic_snapshot',
        '_enable_validity_chunks',
    }
)


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
    if answer == 'not_applied':
        return bool(cam.config_not_applied), f'config_not_applied={cam.config_not_applied!r}'
    if answer == 'false':
        return result is False, f'answered {result!r}'
    if answer == 'none':
        return result is None, f'answered {result!r}'
    return False, f'answered {result!r}'


def _writers_inside_a_try() -> set[str]:
    """PylonCamera methods with a node write inside a ``try`` that has handlers."""
    source = Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PylonCamera')
    found = set()
    for fn in cls.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        tries = [t for t in ast.walk(fn) if isinstance(t, ast.Try) and t.handlers]
        if any(_writes(n) for t in tries for stmt in t.body for n in ast.walk(stmt)):
            found.add(fn.name)
    return found


def _writes(node) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _WRITES
    ) or (
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == 'Value' for t in node.targets)
    )


def test_the_table_is_every_writer_inside_a_try():
    assert {name for name, _, _ in _CASES} == _writers_inside_a_try() - _ANSWERED_ELSEWHERE


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
