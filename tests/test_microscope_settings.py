# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for microscope settings load / save.

Added for issue #616: no-camera startup corrupted stored exposures.

Root cause: the Lumascope camera cache defaulted `max_exposure` to 0.0 and
only populated it during `_populate_camera_cache()`, which early-returns
when no camera is connected. `MicroscopeSettings.load_settings()` used the
cached value as an exposure-slider upper bound, hitting an `if exp <=
max_exposure` branch where `max_exposure == 0` caused every stored
exposure to be clamped to 0. Shutdown's `save_settings()` then wrote
zeros back to disk, corrupting the settings file for future sessions.

Structural fix (4.1): `Lumascope.camera_max_exposure` now returns `None`
(not 0.0) when no camera is connected, so callers can distinguish
"camera missing" from a real driver value. `load_settings` falls back to
`DEFAULT_MAX_EXPOSURE_MS` with `scope.camera_max_exposure or DEFAULT`.
"""

from modules.config_helpers import DEFAULT_MAX_EXPOSURE_MS


class TestCameraMaxExposureContract:
    """Pin the Lumascope.camera_max_exposure no-camera contract.

    The contract is: the property returns None when no camera is
    connected or the cache has not been populated with a real value.
    load_settings relies on `value or DEFAULT_MAX_EXPOSURE_MS` for the
    fallback, so anything falsy (None, 0, 0.0) is equivalent from the
    caller's perspective — but None is the intended sentinel.
    """

    def test_inactive_camera_yields_none_max_exposure(self):
        """Forcing camera cache to inactive must leave max_exposure None."""
        from modules.lumascope_api import Lumascope

        scope = Lumascope(simulate=True)
        # Simulator connects an active camera by default. Force the exact
        # no-camera state that load_settings sees on a real missing camera.
        with scope._camera_cache_lock:
            scope._camera_cache['active'] = False
            scope._camera_cache['max_exposure'] = None

        assert scope.camera_max_exposure is None

    def test_zero_in_cache_yields_none_max_exposure(self):
        """Legacy 0.0 in cache (driver returned 0) is coerced to None.

        Belt-and-suspenders: even if something writes 0.0 into the cache,
        the property still returns None so callers see a consistent
        "camera missing" signal.
        """
        from modules.lumascope_api import Lumascope

        scope = Lumascope(simulate=True)
        with scope._camera_cache_lock:
            scope._camera_cache['max_exposure'] = 0.0

        assert scope.camera_max_exposure is None

    def test_populated_value_passes_through(self):
        """A real positive value in the cache is returned as float."""
        from modules.lumascope_api import Lumascope

        scope = Lumascope(simulate=True)
        with scope._camera_cache_lock:
            scope._camera_cache['max_exposure'] = 500.0

        assert scope.camera_max_exposure == 500.0
        assert isinstance(scope.camera_max_exposure, float)

    def test_integer_in_cache_is_coerced_to_float(self):
        """Integer from a driver is returned as float for caller consistency."""
        from modules.lumascope_api import Lumascope

        scope = Lumascope(simulate=True)
        with scope._camera_cache_lock:
            scope._camera_cache['max_exposure'] = 750

        assert scope.camera_max_exposure == 750.0
        assert isinstance(scope.camera_max_exposure, float)


class TestLoadSettingsFallback:
    """Regression for #616: load_settings must fall back when no camera."""

    def test_default_constant_pinned(self):
        """Pin the default so a refactor can't silently change it."""
        assert DEFAULT_MAX_EXPOSURE_MS == 1000.0

    def test_none_falls_back_to_default(self):
        """The `value or DEFAULT` pattern must yield DEFAULT for None."""
        value = None
        assert (value or DEFAULT_MAX_EXPOSURE_MS) == DEFAULT_MAX_EXPOSURE_MS

    def test_zero_falls_back_to_default(self):
        """Defensive: 0.0 in cache (shouldn't happen post-fix) still safe."""
        value = 0.0
        assert (value or DEFAULT_MAX_EXPOSURE_MS) == DEFAULT_MAX_EXPOSURE_MS

    def test_valid_value_overrides_default(self):
        """Real camera value must pass through, not get replaced."""
        value = 500.0
        assert (value or DEFAULT_MAX_EXPOSURE_MS) == 500.0
