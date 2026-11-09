# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A camera driver raises a failed read, and lets it through the writes that make it.

The base ``Camera`` getters' contract: a value the camera gave, the getter's
not-connected answer, or ``CameraReadFailedError`` where the read fails,
the SDK's words carried and the camera kept. Pylon's getters are walked in
``test_pylon_getter_transient_reads.py``; IDS's are here.

A setter that meets that raise -- the config guard's grab-state read at its
entry, a gain selector it resolves -- lets it out as itself, with nothing
written, so the API refuses the write as one not sent rather than the
camera's refusal or a write it did not answer. And the base keeps the frame
window when a read failed before anything was sent.

The SDK stand-ins are the suite's ``pypylon`` and ``ids_peak`` modules,
whose exception classes are the vendors'.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from ids_peak import ids_peak

from _lumascope.drivers.camera import FrameWindow
from _lumascope.drivers.exceptions import CameraReadFailedError
from _lumascope.drivers.simulated_camera import SimulatedCamera
from tests.camera_fakes import bare_ids_camera, bare_pylon_camera

# getter: its answer when the camera is not connected
_IDS_NOT_CONNECTED = {
    'get_gain': -1,
    'get_exposure_t': -1,
    'get_binning_size': 1,
    'get_pixel_format': None,
    'get_supported_pixel_formats': (),
    'get_max_frame_size': {},
    'get_acquired_aoi': None,
    '_frame_grid': None,
}


def test_every_ids_getter_raises_a_failed_read_with_the_sdks_words_and_keeps_the_camera():
    offenders = []
    for unanswered in (
        ids_peak.Exception('node read failed'),
        ids_peak.TimeoutException('GC_ERR_TIMEOUT: node read failed'),
    ):
        for getter in _IDS_NOT_CONNECTED:
            cam = bare_ids_camera()
            cam._sensor_max = None
            cam.remote_nodemap.FindNode.side_effect = unanswered
            cam.remote_nodemap.HasNode.side_effect = unanswered
            try:
                answer = getattr(cam, getter)()
            except CameraReadFailedError as e:
                if 'node read failed' not in str(e):
                    offenders.append(f"{getter}: dropped the SDK's words: {e}")
            else:
                offenders.append(f'{getter} ({type(unanswered).__name__}): answered {answer!r}')
            if cam._mark_disconnected.called:
                offenders.append(f'{getter}: marked the camera removed')
    assert not offenders, '\n'.join(offenders)


def test_every_ids_getter_answers_its_not_connected_value_and_asks_the_sdk_nothing():
    offenders = []
    for getter, expected in _IDS_NOT_CONNECTED.items():
        cam = bare_ids_camera()
        cam.active = None
        answer = getattr(cam, getter)()
        if answer != expected:
            offenders.append(f'{getter}: {answer!r}, not {expected!r}')
        if cam.remote_nodemap.FindNode.called:
            offenders.append(f'{getter}: asked the SDK')
    assert not offenders, '\n'.join(offenders)


def test_an_ids_gain_factor_no_camera_holds_raises_not_zero_db():
    cam = bare_ids_camera()
    cam.remote_nodemap.FindNode.return_value.Value.return_value = 0.0
    with pytest.raises(CameraReadFailedError, match=r'factor of 0\.0'):
        cam.get_gain()


def test_an_ids_frame_grid_lets_a_defect_of_ours_out_as_itself():
    cam = bare_ids_camera()
    cam.remote_nodemap.FindNode.return_value.Maximum.return_value = 1000
    cam.remote_nodemap.FindNode.return_value.Minimum.return_value = 0
    cam.remote_nodemap.FindNode.return_value.Increment.return_value = 4

    def defect():
        raise TypeError('scripted defect')

    cam._optical_center_bias = defect
    with pytest.raises(TypeError, match='scripted defect'):
        cam._frame_grid()


# --- the setters let the guard's read through ------------------------------------------


def _guard_read_fails(cam) -> CameraReadFailedError:
    """The config guard's grab-state read raises, its words a timeout's."""
    failed = CameraReadFailedError('The camera did not answer a read of IsGrabbing: timeout')

    def is_grabbing():
        raise failed

    cam.is_grabbing = is_grabbing
    return failed


def _pylon_format_write(cam):
    cam.active.PixelFormat.GetSymbolics.return_value = ('Mono8', 'Mono12')
    cam.active.PixelFormat.GetValue.return_value = 'Mono12'
    return cam.set_pixel_format('Mono8'), (cam.active.PixelFormat.SetValue,)


def _pylon_binning_write(cam):
    return cam._set_hardware_binning(2), (
        cam.active.BinningVertical.SetValue,
        cam.active.BinningHorizontal.SetValue,
    )


def _pylon_conversion_gain_write(cam):
    return cam.set_conversion_gain_mode('High'), (cam.active.BslConversionGainMode.SetValue,)


def _pylon_line_noise_write(cam):
    return cam.set_line_noise_reduction(True), (cam.active.BslLineNoiseReductionEnable.SetValue,)


def _pylon_window_write(cam):
    plan = SimpleNamespace(acq_width=640, acq_height=480)
    return cam._set_hardware_window(plan), (cam.active.Width.SetValue, cam.active.Height.SetValue)


def _ids_window_write(cam):
    plan = SimpleNamespace(acq_width=640, acq_height=480, offset_x=0, offset_y=0)
    return cam._set_hardware_window(plan), (cam.remote_nodemap.FindNode.return_value.SetValue,)


def _ids_format_write(cam):
    cam._pixel_format_cache = 'Mono12'
    cam._read_supported_pixel_formats = lambda: ('Mono8', 'Mono12')
    entry = cam.remote_nodemap.FindNode.return_value.SetCurrentEntry
    return cam.set_pixel_format('Mono8'), (entry,)


def _ids_binning_write(cam):
    return cam._set_hardware_binning(2), (cam.remote_nodemap.FindNode.return_value.SetValue,)


_PYLON_SETTERS = {
    'set_pixel_format': _pylon_format_write,
    '_set_hardware_binning': _pylon_binning_write,
    'set_conversion_gain_mode': _pylon_conversion_gain_write,
    'set_line_noise_reduction': _pylon_line_noise_write,
    '_set_hardware_window': _pylon_window_write,
}

_IDS_SETTERS = {
    '_set_hardware_window': _ids_window_write,
    'set_pixel_format': _ids_format_write,
    '_set_hardware_binning': _ids_binning_write,
}


def test_every_setter_lets_its_guards_failed_read_out_as_itself_with_nothing_written():
    offenders = []
    for driver, setters, make in (
        ('pylon', _PYLON_SETTERS, bare_pylon_camera),
        ('ids', _IDS_SETTERS, bare_ids_camera),
    ):
        for name, write in setters.items():
            cam = make()
            failed = _guard_read_fails(cam)
            written = ()
            try:
                answer, written = write(cam)
            except CameraReadFailedError as e:
                if e is not failed:
                    offenders.append(f'{driver} {name}: raised another read failure: {e}')
            except Exception as e:
                offenders.append(f'{driver} {name}: raised {type(e).__name__}: {e}')
            else:
                offenders.append(f'{driver} {name}: answered {answer!r}')
            for node in written:
                if node.called:
                    offenders.append(f'{driver} {name}: wrote after the read failed')
    assert not offenders, '\n'.join(offenders)


def test_an_ids_gain_write_whose_selector_read_fails_lets_it_out_with_nothing_written():
    cam = bare_ids_camera()
    cam._gain_selector = None
    cam.remote_nodemap.HasNode.side_effect = ids_peak.TimeoutException('GC_ERR_TIMEOUT')

    with pytest.raises(CameraReadFailedError, match='GainSelector'):
        cam.gain(6.0)
    cam.remote_nodemap.FindNode.return_value.SetValue.assert_not_called()
    cam.remote_nodemap.FindNode.return_value.SetCurrentEntry.assert_not_called()


# --- the base keeps the window a failed read leaves ------------------------------------


def _camera_with_a_window() -> tuple[SimulatedCamera, FrameWindow]:
    cam = SimulatedCamera()
    cam.set_frame_size(800, 600)
    return cam, cam._frame_window


def test_a_frame_size_write_whose_guard_read_fails_keeps_the_window_held_before():
    cam, window = _camera_with_a_window()
    _guard_read_fails(cam)
    with pytest.raises(CameraReadFailedError):
        cam.set_frame_size(400, 300)
    assert cam._frame_window == window


def test_a_binning_write_whose_guard_read_fails_leaves_the_window_untouched():
    cam, window = _camera_with_a_window()
    _guard_read_fails(cam)
    with pytest.raises(CameraReadFailedError):
        cam.set_binning_size(2)
    assert cam._frame_window == window


def test_a_frame_size_write_that_fails_otherwise_drops_the_window():
    """A pin of the base's answer to a write that may have half-landed,
    unchanged beside the failed read's."""
    cam, _window = _camera_with_a_window()
    cam._set_hardware_window = MagicMock(side_effect=RuntimeError('half-written'))
    with pytest.raises(RuntimeError):
        cam.set_frame_size(400, 300)
    assert cam._frame_window is None


def test_a_pylon_grab_start_reads_no_format_and_starts_its_poller(caplog):
    """The started grab's depth line reads the format the driver holds: a log
    line asks the camera nothing, so a format read never fails a grab that
    started or leaves it without its stats poller."""
    from pypylon import genicam

    cam = bare_pylon_camera()
    cam._grab_strategy_name = 'LatestImageOnly'
    cam._log_stream_grabber_status = MagicMock()
    cam._start_stats_poller = MagicMock()
    cam.active.PixelFormat.GetValue.side_effect = genicam.TimeoutException('usb stalled')

    cam._start_grabbing()

    cam.active.StartGrabbing.assert_called_once()
    cam.active.PixelFormat.GetValue.assert_not_called()
    cam._start_stats_poller.assert_called_once()
    assert not [r for r in caplog.records if 'FAILED' in r.getMessage()]


# --- Pylon against Basler's camera emulator ---------------------------------------

_EMULATOR_GETTERS = """
import json, logging, sys
sys.path.insert(0, 'src')
logging.disable(logging.CRITICAL)
try:
    import pypylon  # noqa: F401
except ImportError:
    print(json.dumps({'sdk': False})); sys.exit(0)
from _lumascope.drivers.pyloncamera import PylonCamera
getters = json.loads(sys.argv[1])

def answers(cam):
    out = {}
    for name in getters:
        try:
            out[name] = repr(getattr(cam, name)())
        except Exception as e:
            out[name] = f'raised {type(e).__name__}: {e}'
    return out

cam = PylonCamera()
if not cam.active and not cam.connect():
    print(json.dumps({'sdk': True, 'connected': False})); sys.exit(0)
try:
    connected = answers(cam)
finally:
    cam.disconnect()
gone = answers(cam)
print(json.dumps({'sdk': True, 'connected': True, 'connected_answers': connected, 'gone': gone}))
"""


def test_the_pylon_getters_answer_basler_emulators_values_and_not_connected_after():
    """The production driver against Basler's emulator (``PYLON_CAMEMU``), in a
    child process since the suite's stand-in replaces pypylon here:
    connected, each getter answers a value; disconnected, each answers its
    not-connected value and raises nothing. A pin: the SDK's own answers,
    which the stubs above stand in for, unchanged by the raise."""
    getters = list(_NOT_CONNECTED_PYLON)
    repo = pathlib.Path(__file__).resolve().parents[1]
    done = subprocess.run(
        [sys.executable, '-I', '-c', _EMULATOR_GETTERS, json.dumps(getters)],
        cwd=str(repo),
        env={**os.environ, 'PYLON_CAMEMU': '1'},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    answer = json.loads(done.stdout.strip().splitlines()[-1])
    if not answer['sdk']:
        pytest.skip('pypylon is not installed on this host')
    if not answer['connected']:
        pytest.skip("pypylon's camera emulator did not open on this host")

    connected = answer['connected_answers']
    assert {g: a for g, a in connected.items() if a.startswith('raised')} == {}
    assert all(connected[g] != repr(v) for g, v in _NOT_CONNECTED_PYLON.items()), connected
    assert answer['gone'] == {g: repr(v) for g, v in _NOT_CONNECTED_PYLON.items()}


# getter: its answer when the camera is not connected
_NOT_CONNECTED_PYLON = {
    'get_gain': -1,
    'get_exposure_t': -1,
    'get_pixel_format': None,
    'get_supported_pixel_formats': (),
    'get_max_frame_size': {},
    'get_min_frame_size': {},
    'get_frame_size': None,
}
