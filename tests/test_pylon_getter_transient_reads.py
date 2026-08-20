# Copyright Etaluma, Inc.
"""Regression net: a transient node-read failure in a pylon pure getter
returns the getter's documented sentinel and does NOT tear the camera down.

The old behavior latched the device-removed flag (no reset short of a full
reconnect) from a caught genicam.RuntimeException inside value getters, so a
single flaky USB node read silently killed the live stream. Disconnect
authority now lives only with the definitive-evidence owners (the SDK removal
callback, the DEVICE_NOT_FOUND grab path, the consecutive-failure cascade).

One parametrized test covers every pure getter: induce a RuntimeException on
the getter's node accessor, assert (a) the documented sentinel comes back and
(b) _mark_disconnected was never called.
"""

import pytest

from pypylon import genicam

from tests.camera_fakes import bare_pylon_camera


def _boom(*_args, **_kwargs):
    raise genicam.RuntimeException('transient node read failure')


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


def _induce_exposure_nodes_unreadable(cam):
    cam._node_attr_get = lambda *a, **k: None


def _induce_exposure_read_raises(cam):
    cam._node_attr_get = _boom


def _induce_pixel_format(cam):
    cam._pixel_format_cache = None
    cam.active.PixelFormat.GetValue.side_effect = _boom


def _induce_supported_formats(cam):
    cam.active.PixelFormat.GetSymbolics.side_effect = _boom


def _induce_nodemap(cam):
    cam.active.GetNodeMap.side_effect = _boom


GETTER_CASES = [
    pytest.param('get_frame_size', _induce_frame_size, None, id='get_frame_size'),
    pytest.param('get_min_frame_size', _induce_min_frame_size, {}, id='get_min_frame_size'),
    pytest.param('get_max_frame_size', _induce_max_frame_size, {}, id='get_max_frame_size'),
    pytest.param('get_binning_size', _induce_binning, -1, id='get_binning_size'),
    pytest.param('get_gain', _induce_gain, -1, id='get_gain'),
    pytest.param(
        'get_exposure_t',
        _induce_exposure_nodes_unreadable,
        -1,
        id='get_exposure_t-nodes-unreadable',
    ),
    pytest.param(
        'get_exposure_t', _induce_exposure_read_raises, -1, id='get_exposure_t-read-raises'
    ),
    pytest.param('get_pixel_format', _induce_pixel_format, None, id='get_pixel_format'),
    pytest.param(
        'get_supported_pixel_formats', _induce_supported_formats, (), id='supported_pixel_formats'
    ),
    pytest.param(
        'supports_conversion_gain_mode', _induce_nodemap, False, id='supports_conversion_gain'
    ),
    pytest.param('supports_line_noise_reduction', _induce_nodemap, False, id='supports_line_noise'),
    pytest.param('get_all_temperatures', _induce_nodemap, {}, id='get_all_temperatures'),
]


@pytest.mark.parametrize(('getter', 'induce', 'sentinel'), GETTER_CASES)
def test_transient_read_returns_sentinel_and_keeps_camera(getter, induce, sentinel):
    cam = bare_pylon_camera()
    induce(cam)
    result = getattr(cam, getter)()
    assert result == sentinel
    cam._mark_disconnected.assert_not_called()
