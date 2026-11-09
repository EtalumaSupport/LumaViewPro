# Copyright Etaluma, Inc.
"""A pylon getter's failed node read raises ``CameraReadFailedError`` and does
NOT tear the camera down; not connected, it answers its not-connected value.

The old behavior latched the device-removed flag (no reset short of a full
reconnect) from a caught genicam.RuntimeException inside value getters, so a
single flaky USB node read silently killed the live stream. Disconnect
authority now lives only with the definitive-evidence owners (the SDK removal
callback, the DEVICE_NOT_FOUND grab path, the consecutive-failure cascade).
A failed read is told as a failure, never as a value a caller could take for
the camera's: the raise carries the SDK's words, a timeout's too, and the
getter logs nothing, so the one reporter where the flight stops logs it once.
The feature probes raise the same way.
"""

from pypylon import genicam

from _lumascope.drivers.exceptions import CameraReadFailedError
from tests.camera_fakes import bare_pylon_camera


def _boom(*_args, **_kwargs):
    raise genicam.RuntimeException('transient node read failure')


def _timeout(*_args, **_kwargs):
    raise genicam.TimeoutException('transient node read failure: timed out')


def _induce_frame_size(cam):
    cam.active.Width.GetValue.side_effect = _boom


def _induce_min_frame_size(cam):
    cam.active.Width.GetMin.side_effect = _boom


def _induce_max_frame_size(cam):
    cam.active.Width.GetMax.side_effect = _boom


def _induce_binning(cam):
    cam.active.BinningVertical.GetValue.side_effect = _boom


def _induce_gain(cam):
    cam.active.Gain.GetValue.side_effect = _boom


def _absent(name):
    raise genicam.LogicalErrorException(f'Node not existing: {name}')


def _induce_exposure_nodes_absent(cam):
    cam.active.GetNodeMap.return_value.GetNode.side_effect = _absent


def _induce_exposure_read_raises(cam):
    cam.active.GetNodeMap.return_value.GetNode.return_value.GetValue.side_effect = _boom


def _induce_pixel_format(cam):
    cam._pixel_format_cache = None
    cam.active.PixelFormat.GetValue.side_effect = _boom


def _induce_supported_formats(cam):
    cam.active.PixelFormat.GetSymbolics.side_effect = _boom


def _induce_nodemap(cam):
    cam.active.GetNodeMap.side_effect = _boom


# getter, how its read is failed, the words its raise carries
_GETTERS = (
    ('get_frame_size', _induce_frame_size, 'transient node read failure'),
    ('get_min_frame_size', _induce_min_frame_size, 'transient node read failure'),
    ('get_max_frame_size', _induce_max_frame_size, 'transient node read failure'),
    ('get_binning_size', _induce_binning, 'transient node read failure'),
    ('get_gain', _induce_gain, 'transient node read failure'),
    ('get_exposure_t', _induce_exposure_nodes_absent, 'neither BslEffectiveExposureTime'),
    ('get_exposure_t', _induce_exposure_read_raises, 'transient node read failure'),
    ('get_pixel_format', _induce_pixel_format, 'transient node read failure'),
    ('get_supported_pixel_formats', _induce_supported_formats, 'transient node read failure'),
    ('_frame_grid', _induce_max_frame_size, 'transient node read failure'),
)

# getter: its answer when the camera is not connected
_NOT_CONNECTED = {
    'get_frame_size': None,
    'get_min_frame_size': {},
    'get_max_frame_size': {},
    'get_binning_size': 1,
    'get_gain': -1,
    'get_exposure_t': -1,
    'get_pixel_format': None,
    'get_supported_pixel_formats': (),
    '_frame_grid': None,
}


def _as_timeouts(cam) -> None:
    """Every read ``_boom`` was set on times out instead."""
    for node in ('Width', 'BinningVertical', 'Gain', 'PixelFormat'):
        for accessor in ('GetValue', 'GetMin', 'GetMax', 'GetSymbolics', 'GetInc'):
            mock = getattr(getattr(cam.active, node), accessor)
            if mock.side_effect is _boom:
                mock.side_effect = _timeout
    read = cam.active.GetNodeMap.return_value.GetNode.return_value.GetValue
    if read.side_effect is _boom:
        read.side_effect = _timeout


def test_a_failed_read_raises_with_the_sdks_words_logs_nothing_and_keeps_the_camera(caplog):
    offenders = []
    for timed_out in (False, True):
        for getter, induce, words in _GETTERS:
            cam = bare_pylon_camera()
            induce(cam)
            if timed_out:
                _as_timeouts(cam)
            caplog.clear()
            try:
                answer = getattr(cam, getter)()
            except CameraReadFailedError as e:
                if words not in str(e):
                    offenders.append(f'{getter}: raised without the words {words!r}: {e}')
            else:
                offenders.append(f'{getter} (timed out: {timed_out}): answered {answer!r}')
            logged = [r.getMessage() for r in caplog.records if r.levelno >= 30]
            if logged:
                offenders.append(f'{getter}: logged {logged}')
            if cam._mark_disconnected.called:
                offenders.append(f'{getter}: marked the camera removed')
    assert not offenders, '\n'.join(offenders)


def test_not_connected_answers_each_getters_value():
    offenders = []
    for never_connected in (None, False):
        for getter, expected in _NOT_CONNECTED.items():
            cam = bare_pylon_camera()
            cam.active = never_connected
            answer = getattr(cam, getter)()
            if answer != expected:
                offenders.append(f'{getter} with active={never_connected!r}: {answer!r}')
    assert not offenders, '\n'.join(offenders)


def test_a_failed_feature_probe_raises_and_keeps_the_camera():
    offenders = []
    for probe in (
        'supports_conversion_gain_mode',
        'supports_line_noise_reduction',
        'supports_black_level',
    ):
        cam = bare_pylon_camera()
        _induce_nodemap(cam)
        try:
            answer = getattr(cam, probe)()
        except CameraReadFailedError as e:
            if 'transient node read failure' not in str(e):
                offenders.append(f'{probe}: raised without the SDK words: {e}')
        else:
            offenders.append(f'{probe}: answered {answer!r}')
        if cam._mark_disconnected.called:
            offenders.append(f'{probe}: marked the camera removed')
    assert not offenders, '\n'.join(offenders)
