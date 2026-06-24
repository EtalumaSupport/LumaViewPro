# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Stub-testable units for the rebuilt IDS driver (no SDK / no camera).

Covers the Mac-authorable pieces of the native-depth rebuild:
  - significant_bits derived from the wire format name (the depth that pairs
    with each delivered frame, replacing the old pinned-8 behavior)
  - the IPL conversion target that unpacks each wire format to native depth
  - the frame-rate crash-stop cap (soft AcquisitionFrameRateTarget), which
    replaced the USB-saturating maximize path

The live grab loop, recovery, and real conversion are bench-gated.
"""

from __future__ import annotations

import ids_peak_ipl
import pytest

from drivers.camera_profiles import CameraProfile
from drivers.idscamera import IDSCamera, _ids_ipl_target, ids_significant_bits
from tests.camera_fakes import bare_ids_camera


class _RecordingNode:
    """Node that records writes and serves configured min/max/value, for
    asserting SetValue calls and exercising range-dependent logic."""

    _UNSET = object()

    def __init__(self, value=_UNSET, minimum=None, maximum=None):
        self.value = value
        self._min = minimum
        self._max = maximum
        self.entry = None

    def SetValue(self, v):
        self.value = v

    def Value(self):
        return self.value

    def Minimum(self):
        return self._min

    def Maximum(self):
        return self._max

    def SetCurrentEntry(self, entry):
        self.entry = entry


class _RecordingNodemap:
    """Minimal nodemap: distinct recording node per name (MagicMock collapses
    them all to one return_value, so a real fake is needed to tell them apart).
    Pre-seed specific nodes via `preset`; unknown names auto-create."""

    def __init__(self, preset=None):
        self.nodes: dict[str, _RecordingNode] = dict(preset or {})

    def FindNode(self, name):
        return self.nodes.setdefault(name, _RecordingNode())


class _RecordingDataStream:
    """Records the timeout passed to WaitForFinishedBuffer, then raises to
    short-circuit the rest of the grab (we only assert the timeout arg)."""

    def __init__(self):
        self.timeout_arg = None

    def WaitForFinishedBuffer(self, timeout):
        self.timeout_arg = timeout
        raise RuntimeError('short-circuit after recording the timeout')


class TestSignificantBitsFromFormat:
    @pytest.mark.parametrize(
        'wire,expected',
        [
            ('Mono12g24IDS', 12),
            ('Mono12p', 12),
            ('Mono12', 12),
            ('Mono10g40IDS', 10),
            ('Mono10p', 10),
            ('Mono10', 10),
            ('Mono8', 8),
            ('Mono8g', 8),
            ('Mono16', 16),  # leading-bit-count fallback for an unprefixed name
            ('BayerRG8', 8),  # no Mono token -> safe 8-bit default
            ('', 8),
        ],
    )
    def test_significant_bits(self, wire, expected):
        assert ids_significant_bits(wire) == expected


class TestIplTarget:
    def test_mono12_unpacks_to_mono12(self):
        assert _ids_ipl_target('Mono12g24IDS') is ids_peak_ipl.PixelFormatName_Mono12

    def test_mono10_unpacks_to_mono10(self):
        assert _ids_ipl_target('Mono10g40IDS') is ids_peak_ipl.PixelFormatName_Mono10

    def test_mono8_stays_mono8(self):
        assert _ids_ipl_target('Mono8') is ids_peak_ipl.PixelFormatName_Mono8


class TestDepthContract:
    """The driver no longer pins depth to 8; the container is the inherited
    16-bit width and the payload depth is derived from the active format."""

    def test_native_bit_depth_is_container_width_16(self):
        assert IDSCamera.native_bit_depth == 16

    @pytest.mark.parametrize(
        'wire,expected',
        [('Mono12g24IDS', 12), ('Mono10g40IDS', 10), ('Mono8', 8)],
    )
    def test_significant_bits_property_tracks_format(self, wire, expected):
        cam = bare_ids_camera()
        cam._pixel_format_cache = wire
        assert cam.significant_bits == expected


class TestFrameRateCap:
    """The crash-stop cap: enable + set AcquisitionFrameRateTarget, never
    maximize the rate (which saturated USB3 and exhausted the buffer pool)."""

    def test_cap_constant_is_below_sustained_rate(self):
        # Sustained host-unpack rate is ~18 fps; the static cap sits under it.
        assert 0 < IDSCamera._FPS_CAP <= 18

    def test_enable_writes_target_and_enable(self):
        cam = bare_ids_camera()
        cam.remote_nodemap = _RecordingNodemap()
        cam.set_max_acquisition_frame_rate(True, 16.0)
        nodes = cam.remote_nodemap.nodes
        assert nodes['AcquisitionFrameRateTargetEnable'].value is True
        assert nodes['AcquisitionFrameRateTarget'].value == 16.0

    def test_disable_skips_target_write(self):
        cam = bare_ids_camera()
        cam.remote_nodemap = _RecordingNodemap()
        cam.set_max_acquisition_frame_rate(False)
        nodes = cam.remote_nodemap.nodes
        assert nodes['AcquisitionFrameRateTargetEnable'].value is False
        # Target is only written when enabled.
        assert 'AcquisitionFrameRateTarget' not in nodes

    def test_inactive_camera_is_a_noop(self):
        cam = bare_ids_camera()
        cam.active = False
        cam.remote_nodemap = _RecordingNodemap()
        cam.set_max_acquisition_frame_rate(True, 16.0)
        assert cam.remote_nodemap.nodes == {}


class TestGainDbConversion:
    """The IDS Gain node is a linear multiplier; LVP drives gain in dB. The
    driver converts dB <-> factor so 0 dB maps to the node's 1.0x unity floor
    (the previous unconverted 0.0 write was rejected as out-of-range)."""

    def _cam_with_gain_node(self, value=1.0, minimum=1.0, maximum=31.62):
        cam = bare_ids_camera()
        cam.remote_nodemap = _RecordingNodemap(
            {
                'Gain': _RecordingNode(value=value, minimum=minimum, maximum=maximum),
                'GainSelector': _RecordingNode(),
            }
        )
        return cam

    def test_zero_db_maps_to_unity_factor(self):
        cam = self._cam_with_gain_node()
        assert cam.gain(0.0) is True
        assert cam.remote_nodemap.nodes['Gain'].value == pytest.approx(1.0)

    def test_twenty_db_maps_to_ten_x(self):
        cam = self._cam_with_gain_node()
        assert cam.gain(20.0) is True
        assert cam.remote_nodemap.nodes['Gain'].value == pytest.approx(10.0)

    def test_thirty_db_maps_to_full_scale_factor(self):
        cam = self._cam_with_gain_node()
        assert cam.gain(30.0) is True
        assert cam.remote_nodemap.nodes['Gain'].value == pytest.approx(31.62, abs=0.05)

    def test_selects_analog_all(self):
        cam = self._cam_with_gain_node()
        cam.gain(6.0)
        assert cam.remote_nodemap.nodes['GainSelector'].entry == 'AnalogAll'

    def test_get_gain_returns_db(self):
        cam = self._cam_with_gain_node(value=10.0)
        assert cam.get_gain() == pytest.approx(20.0)  # 20*log10(10)

    def test_capability_range_reported_in_db(self):
        cam = bare_ids_camera()
        cam.profile = CameraProfile()
        cam.remote_nodemap = _RecordingNodemap(
            {
                'Gain': _RecordingNode(value=1.0, minimum=1.0, maximum=31.62),
                'ExposureTime': _RecordingNode(value=1e4, minimum=20.0, maximum=2e6),
            }
        )
        cam._query_dynamic_capabilities()
        assert cam.profile.gain.total_min_db == pytest.approx(0.0, abs=0.01)
        assert cam.profile.gain.total_max_db == pytest.approx(30.0, abs=0.1)


class TestGrabNewCaptureTimeout:
    """grab_new_capture takes float seconds but WaitForFinishedBuffer wants an
    integer millisecond timeout -- passing the float made every capture-path
    grab fail with a SWIG type error."""

    def test_passes_integer_millisecond_timeout(self):
        cam = bare_ids_camera()
        cam.cam_image_handler = object()  # only needs to be non-None
        cam.data_stream = _RecordingDataStream()
        ok, _ts = cam.grab_new_capture(3.0)
        assert ok is False  # the fake raises after recording
        assert cam.data_stream.timeout_arg == 3000
        assert isinstance(cam.data_stream.timeout_arg, int)
