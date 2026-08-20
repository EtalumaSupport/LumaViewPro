# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""LAYER-L: cache-only-property regression tests.

Locks down the invariant that certain Lumascope properties / accessor
methods never trigger driver-side I/O (serial command, Pylon SDK call).
The audit (`AUDIT_LAYER_VIOLATIONS_2026-05-01.md` LV-11, LV-45) verified
these by reading the source; this test pins it so a future maintainer
who changes a property to do real I/O sees a failure rather than
silently introducing a per-frame / per-tick MainThread blocker.

The properties under test:
  - Lumascope.frame_size_cached (dict; called per-frame from
    ui/scope_display.py:578-580)
  - Lumascope.pixel_format_cached (str; called per-frame from
    ui/scope_display.py:578-580)
  - Lumascope.get_target_position(axis) (float/dict; called at 10Hz
    from ui/motion_settings.py:228 update_gui)
  - Lumascope.get_current_position(axis) (float/dict; called at
    similar cadence)

Implementation: wraps `self.motion.exchange_command`,
`self.motion.exchange_json`, and the camera's SDK setters with a
MagicMock that raises if invoked. Reads the property; if any of the
spies fire, the test fails with a clear message.
"""

import pytest

import modules.lumascope_api as lumascope_api


def _explode(name):
    """Build a callable that raises if invoked, naming itself in the error."""

    def _raise(*args, **kwargs):
        raise AssertionError(
            f'Cache-only invariant broken: {name} was called during a '
            f'property read that the audit declared cache-only. See '
            f'docs/AUDIT_LAYER_VIOLATIONS_2026-05-01.md LV-11 / LV-45.'
        )

    return _raise


@pytest.fixture
def scope_with_io_traps():
    """Build a real Lumascope(simulate=True) with serial-equivalent and
    SDK-equivalent methods replaced by exploding stubs.

    Reading any property declared cache-only must NOT touch the trapped
    methods.

    register_atexit=False -- the trapped exchange_command would otherwise
    raise AssertionError when LVP-A-7's atexit-registered
    _emergency_shutdown -> disconnect -> stop_motion fires at pytest
    interpreter exit. The trap is the whole point of the fixture; we
    just want it scoped to the test, not to interpreter teardown.
    """
    scope = lumascope_api.Lumascope(simulate=True, register_atexit=False)

    # Trap motor-board serial-equivalent methods.
    if hasattr(scope._motion_driver, 'exchange_command'):
        scope._motion_driver.exchange_command = _explode('motion.exchange_command')
    if hasattr(scope._motion_driver, 'exchange_json'):
        scope._motion_driver.exchange_json = _explode('motion.exchange_json')
    if hasattr(scope._motion_driver, 'exchange_multiline'):
        scope._motion_driver.exchange_multiline = _explode('motion.exchange_multiline')

    # Trap camera SDK-equivalent methods that would indicate live I/O.
    cam = scope._camera_driver
    if cam is not None:
        for attr in (
            'set_pixel_format',
            'set_binning_size',
            'set_frame_size',
            'start_grabbing',
            'stop_grabbing',
            'update_camera_config',
        ):
            if hasattr(cam, attr):
                setattr(cam, attr, _explode(f'camera.{attr}'))

    return scope


class TestCameraPropertiesCacheOnly:
    """LV-11: frame_size_cached and pixel_format_cached are read per-frame
    from the scope_display thread; they must be cache reads.
    """

    def test_camera_frame_size_no_io(self, scope_with_io_traps):
        """Reading frame_size_cached must not call any driver-side SDK
        method. If this fails, scope_display's per-frame readout has
        become a per-frame Pylon SDK call."""
        result = scope_with_io_traps.imaging.frame_size_cached
        assert isinstance(result, dict)

    def test_camera_pixel_format_no_io(self, scope_with_io_traps):
        """Reading pixel_format_cached must not call any driver-side SDK
        method. Same risk as frame_size_cached."""
        result = scope_with_io_traps.imaging.pixel_format_cached
        assert isinstance(result, str)

    def test_camera_frame_size_repeated_reads(self, scope_with_io_traps):
        """Repeated reads must remain cache-only -- no lazy refresh that
        flips to SDK after first call."""
        for _ in range(5):
            scope_with_io_traps.imaging.frame_size_cached  # noqa: B018 -- bare attribute access is the test, exercising the getter to assert no driver IO fires


class TestPositionAccessorsCacheOnly:
    """LV-45: get_target_position and get_current_position are called at
    ~10Hz from the UI on MainThread; they must be cache reads.
    """

    def test_get_target_position_single_axis_no_io(self, scope_with_io_traps):
        """LV-45 invariant: get_target_position('Z') is cache-only."""
        scope_with_io_traps.motion.get_target_position('Z')

    def test_get_target_position_all_axes_no_io(self, scope_with_io_traps):
        """get_target_position(None) returns a dict snapshot -- cache-only."""
        result = scope_with_io_traps.motion.get_target_position()
        assert isinstance(result, dict)

    def test_get_current_position_single_axis_no_io(self, scope_with_io_traps):
        """get_current_position uses the predicted-position cache during
        MOVING and the position cache during IDLE -- both cache-only."""
        scope_with_io_traps.motion.get_current_position('X')

    def test_get_current_position_all_axes_no_io(self, scope_with_io_traps):
        """get_current_position(None) returns a dict snapshot -- cache-only."""
        result = scope_with_io_traps.motion.get_current_position()
        assert isinstance(result, dict)

    def test_position_repeated_reads_no_io(self, scope_with_io_traps):
        """10Hz cadence simulation: repeated reads stay cache-only."""
        for _ in range(20):
            scope_with_io_traps.motion.get_target_position('Z')
            scope_with_io_traps.motion.get_current_position('Z')
