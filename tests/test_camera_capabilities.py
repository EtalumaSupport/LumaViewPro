"""Camera capability flags -- ``is_color_native`` and ``native_bit_depth``.

Phase 1c.1 of the color audit / mono-native restructure. The flags
surface through ``scope.capabilities`` so downstream allocators size
buffers to the actual driver-delivered shape rather than assuming
3-channel uint16 / Bayer-decoded color.

Tests are class-attribute checks (no driver instantiation) plus an
end-to-end check that ``ScopeCapabilities.from_drivers`` reads the
flags off the camera instance.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from drivers.camera import Camera
from drivers.fx2driver import FX2Camera
from drivers.idscamera import IDSCamera
from drivers.pyloncamera import PylonCamera
from drivers.simulated_camera import SimulatedCamera
from modules.scope_capabilities import ScopeCapabilities


class TestBaseCameraDefaults:
    """The Camera ABC declares the default contract:

    - mono cameras (the LVP shipping fleet) report ``is_color_native=False``
    - 16-bit container default covers all Pylon Mono10 / Mono12 / Mono16
      sensors and the simulator (which packs into uint16).
    """

    def test_is_color_native_default_is_false(self):
        assert Camera.is_color_native is False

    def test_native_bit_depth_default_is_16(self):
        assert Camera.native_bit_depth == 16


class TestPylonDefaults:
    """Pylon drivers (LS850 / LS820 / LS720 / dart M / ace 2) report the
    16-bit container width even when the wire-level payload is Mono10 or
    Mono12. They are not color-native."""

    def test_pylon_inherits_mono_default(self):
        assert PylonCamera.is_color_native is False

    def test_pylon_inherits_16bit_container(self):
        assert PylonCamera.native_bit_depth == 16


class TestIDSOverride:
    """IDS U3-34L0XCP-M (IMX676) reports only Mono10 / Mono12 packed
    formats natively. The driver forces Mono8 at the SDK boundary so
    downstream code handles 8-bit consistently -- the capability flag
    reflects the actual delivered container."""

    def test_ids_native_bit_depth_is_8(self):
        assert IDSCamera.native_bit_depth == 8

    def test_ids_is_not_color_native(self):
        assert IDSCamera.is_color_native is False


class TestFX2Override:
    """The FX2 (Lumascope Classic, MT9P031) delivers Mono8 only -- the
    driver accepts no other pixel format and builds uint8 buffers. The
    capability flag must report 8-bit, not the inherited 16-bit default."""

    def test_fx2_native_bit_depth_is_8(self):
        assert FX2Camera.native_bit_depth == 8

    def test_fx2_is_not_color_native(self):
        assert FX2Camera.is_color_native is False


class TestSimulatedDefaults:
    """Simulator defaults to the 16-bit container so tests exercising the
    Pylon-equivalent path see the same shape."""

    def test_sim_inherits_mono_default(self):
        assert SimulatedCamera.is_color_native is False

    def test_sim_inherits_16bit_container(self):
        assert SimulatedCamera.native_bit_depth == 16


class TestScopeCapabilitiesIntegration:
    """``ScopeCapabilities.from_drivers`` reads the two flags off the
    camera instance and surfaces them on the immutable capability snapshot.
    Built without driver instantiation so unit tests don't need hardware."""

    def _stub_motion(self):
        motion = MagicMock()
        motion.detect_present_axes.return_value = ('X', 'Y', 'Z')
        motion.get_microscope_model.return_value = 'TEST-MODEL'
        motion.motorconfig = None
        return motion

    def _stub_led(self):
        led = MagicMock()
        led.available_channels.return_value = ('Blue', 'Green', 'Red')
        led.available_colors.return_value = ('Blue', 'Green', 'Red')
        led.supports_firmware_stim.return_value = False
        return led

    def _stub_camera(self, is_color_native=False, native_bit_depth=16):
        cam = SimpleNamespace(
            is_color_native=is_color_native,
            native_bit_depth=native_bit_depth,
            profile=SimpleNamespace(
                model_name='STUB',
                has_auto_gain=False,
                has_auto_exposure=False,
                pixel_formats=('Mono8',),
                binning_sizes=(1,),
                exposure_max_us=10000,
            ),
            get_max_frame_size=lambda: {'width': 1024, 'height': 768},
        )
        return cam

    def test_mono_camera_surfaces_defaults(self):
        caps = ScopeCapabilities.from_drivers(
            motion=self._stub_motion(),
            led=self._stub_led(),
            camera=self._stub_camera(),
        )
        assert caps.is_color_native is False
        assert caps.native_bit_depth == 16

    def test_ids_shape_surfaces_8bit(self):
        caps = ScopeCapabilities.from_drivers(
            motion=self._stub_motion(),
            led=self._stub_led(),
            camera=self._stub_camera(native_bit_depth=8),
        )
        assert caps.native_bit_depth == 8

    def test_color_camera_surfaces_true(self):
        """Phase 2 activation path: a hypothetical color-native camera
        reports is_color_native=True. Phase 1 ships the flag plumbing
        so Phase 2 can flip the flag without touching capabilities."""
        caps = ScopeCapabilities.from_drivers(
            motion=self._stub_motion(),
            led=self._stub_led(),
            camera=self._stub_camera(is_color_native=True),
        )
        assert caps.is_color_native is True

    def test_no_camera_safe_defaults(self):
        """``camera=None`` (headless / disconnected) yields mono / 16-bit
        defaults rather than raising. Matches the empty-default
        contract on the rest of ScopeCapabilities."""
        caps = ScopeCapabilities.from_drivers(
            motion=self._stub_motion(),
            led=self._stub_led(),
            camera=None,
        )
        assert caps.is_color_native is False
        assert caps.native_bit_depth == 16


@pytest.mark.parametrize(
    'driver_cls,expected_color,expected_depth',
    [
        (Camera, False, 16),
        (PylonCamera, False, 16),
        (IDSCamera, False, 8),
        (SimulatedCamera, False, 16),
    ],
)
def test_driver_class_attributes_match_contract(driver_cls, expected_color, expected_depth):
    """One-shot parametrized class-attribute check across all camera drivers."""
    assert driver_cls.is_color_native is expected_color
    assert driver_cls.native_bit_depth == expected_depth
