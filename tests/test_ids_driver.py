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

from drivers.idscamera import IDSCamera, _ids_ipl_target, ids_significant_bits
from tests.camera_fakes import bare_ids_camera


class _RecordingNode:
    """Node that records the last value written, for asserting SetValue calls."""

    _UNSET = object()

    def __init__(self):
        self.value = self._UNSET

    def SetValue(self, v):
        self.value = v


class _RecordingNodemap:
    """Minimal nodemap: distinct recording node per name (MagicMock collapses
    them all to one return_value, so a real fake is needed to tell them apart)."""

    def __init__(self):
        self.nodes: dict[str, _RecordingNode] = {}

    def FindNode(self, name):
        return self.nodes.setdefault(name, _RecordingNode())


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
