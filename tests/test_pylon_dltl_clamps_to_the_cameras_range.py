# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The Pylon DLTL setter clamps a value to the camera's stated range, and its
docstring records both of Basler's range warnings."""

from __future__ import annotations

from _lumascope.drivers import pyloncamera
from tests.camera_fakes import bare_pylon_camera, nodemaps_answer_attributes
from tests.log_capture import records_of


def _bare_pylon_camera():
    return nodemaps_answer_attributes(bare_pylon_camera())


def _pylon_cam_warnings(caplog):
    """The WARNING lines the Pylon driver wrote to its camera log."""
    return [
        r.getMessage()
        for r in records_of(caplog, pyloncamera)
        if r.name == pyloncamera._cam_log.name and r.levelname == 'WARNING'
    ]


class TestPylonDltlClampAndDocWarnings:
    """set_device_link_throughput_limit must clamp out-of-range
    values via DeviceLinkThroughputLimit.GetMin() / GetMax() rather
    than letting the SDK raise OutOfRangeException, and the docstring
    must record both Basler doc warnings: rolling-shutter distortion
    if too low, corrupt/dropped frames if too high.

    Per per-camera spec pages a2a3536-31umbas.html and
    daa3840-45um.html: both production cameras are rolling-shutter,
    so both warnings apply on both. Per
    network-bandwidth-control-(blaze).md the DLTL throttle
    mechanism (pause-insertion between packets) is identical across
    transports.
    """

    def _camera_with_dltl_range(self, lo=1000, hi=5000):
        cam = _bare_pylon_camera()
        cam.active.DeviceLinkThroughputLimit.GetMin.return_value = lo
        cam.active.DeviceLinkThroughputLimit.GetMax.return_value = hi
        return cam

    def _set_dltl(self, caplog, cam, value_bps):
        with caplog.at_level('WARNING'):
            result = cam.set_device_link_throughput_limit('On', value_bps=value_bps)
        return result, _pylon_cam_warnings(caplog)

    def test_below_minimum_clamps_to_min_and_warns(self, caplog):
        cam = self._camera_with_dltl_range()
        result, warnings = self._set_dltl(caplog, cam, 500)
        assert result is True
        cam.active.DeviceLinkThroughputLimit.SetValue.assert_called_once_with(1000)
        assert any('clamping' in m for m in warnings), (
            f'A clamped low DLTL value must warn the operator; got {warnings!r}'
        )

    def test_above_maximum_clamps_to_max_and_warns(self, caplog):
        cam = self._camera_with_dltl_range()
        result, warnings = self._set_dltl(caplog, cam, 9000)
        assert result is True
        cam.active.DeviceLinkThroughputLimit.SetValue.assert_called_once_with(5000)
        assert any('clamping' in m for m in warnings)

    def test_in_range_value_passes_unclamped(self, caplog):
        cam = self._camera_with_dltl_range()
        result, warnings = self._set_dltl(caplog, cam, 3000)
        assert result is True
        cam.active.DeviceLinkThroughputLimit.SetValue.assert_called_once_with(3000)
        assert not warnings

    def test_minmax_query_failure_passes_value_through(self, caplog):
        """Best-effort contract: if the range query fails, the value is
        written unchanged (the SDK's own OutOfRangeException then
        surfaces through the RuntimeException branch)."""
        cam = _bare_pylon_camera()
        cam.active.DeviceLinkThroughputLimit.GetMin.side_effect = RuntimeError('no node')
        result, _warnings = self._set_dltl(caplog, cam, 12345)
        assert result is True
        cam.active.DeviceLinkThroughputLimit.SetValue.assert_called_once_with(12345)

    def test_docstring_records_both_basler_range_warnings(self):
        # pin-justified: the docstring is the documented operator
        # contract for picking a DLTL value (per-camera spec pages name
        # rolling-shutter distortion when too low, corrupt/dropped
        # frames when too high); the warning text itself is the artifact.
        from _lumascope.drivers.pyloncamera import PylonCamera

        docstring = (PylonCamera.set_device_link_throughput_limit.__doc__ or '').lower()
        assert 'rolling shutter' in docstring or 'rolling-shutter' in docstring, (
            'DLTL setter docstring must record the rolling-shutter '
            'distortion warning per per-camera spec pages.'
        )
        assert 'corrupt' in docstring or 'dropped' in docstring, (
            'DLTL setter docstring must record the too-high warning '
            '(corrupt or dropped frames) per per-camera spec pages.'
        )
