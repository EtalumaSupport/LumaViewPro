# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for microscope settings load / save.

Added for issue #616: no-camera startup corrupted stored exposures.

Root cause: the Lumascope camera cache defaults `max_exposure` to 0.0 and
only populates it during `_populate_camera_cache()`, which early-returns
when no camera is connected. `MicroscopeSettings.load_settings()` used the
cached value as an exposure-slider upper bound, hitting an `if exp <=
max_exposure` branch where `max_exposure == 0` caused every stored
exposure to be clamped to 0. Shutdown's `save_settings()` then wrote zeros
back to disk, corrupting the settings file for future sessions.

Fix: `modules.config_helpers.get_safe_max_exposure(scope)` collapses both
the exception path and the sentinel 0.0 path into a single safe default of
`DEFAULT_MAX_EXPOSURE_MS` (1000 ms). load_settings now routes through this
helper instead of reading the property directly.
"""

import pytest

from modules.config_helpers import DEFAULT_MAX_EXPOSURE_MS, get_safe_max_exposure


class _Scope:
    """Minimal scope stub exposing only camera_max_exposure."""
    def __init__(self, value):
        self._v = value

    @property
    def camera_max_exposure(self):
        if isinstance(self._v, Exception):
            raise self._v
        return self._v


class TestGetSafeMaxExposure:
    """Regression for #616 — no-camera startup must not clamp exposures to 0."""

    def test_zero_returns_default(self):
        """Cache default of 0.0 (no camera) must fall through to DEFAULT."""
        assert get_safe_max_exposure(_Scope(0.0)) == DEFAULT_MAX_EXPOSURE_MS

    def test_negative_returns_default(self):
        """Defensive: any non-positive sentinel must fall through."""
        assert get_safe_max_exposure(_Scope(-1.0)) == DEFAULT_MAX_EXPOSURE_MS

    def test_none_returns_default(self):
        """None would happen if the cache were wiped; must fall through."""
        assert get_safe_max_exposure(_Scope(None)) == DEFAULT_MAX_EXPOSURE_MS

    def test_exception_returns_default(self):
        """Exception from property access (e.g. driver gone) falls through."""
        assert get_safe_max_exposure(_Scope(RuntimeError("camera gone"))) == DEFAULT_MAX_EXPOSURE_MS

    def test_valid_value_passes_through(self):
        """Normal camera value is returned unchanged."""
        assert get_safe_max_exposure(_Scope(500.0)) == 500.0

    def test_valid_int_is_coerced_to_float(self):
        """Integer from a different camera driver is returned as float."""
        result = get_safe_max_exposure(_Scope(750))
        assert result == 750.0
        assert isinstance(result, float)

    def test_very_large_value_passes_through(self):
        """Multi-second exposures (astronomy-style sensors) must not be clamped."""
        assert get_safe_max_exposure(_Scope(60000.0)) == 60000.0

    def test_default_constant_matches_documented_value(self):
        """Pin the constant so a refactor can't silently change it."""
        assert DEFAULT_MAX_EXPOSURE_MS == 1000.0


class TestLumascopeMaxExposureContract:
    """Document the Lumascope contract: camera_max_exposure is 0.0 when no camera.

    This is the sentinel that #616 depends on detecting. If Lumascope ever
    changes to raise instead, get_safe_max_exposure's exception path still
    covers it — both tests will pass. If Lumascope changes to return None,
    same story. The test pins the current behavior so anyone modifying the
    camera cache sees the dependency.
    """

    def test_inactive_camera_yields_zero_max_exposure(self):
        """Forcing camera cache to inactive should leave max_exposure at 0.0."""
        from modules.lumascope_api import Lumascope

        scope = Lumascope(simulate=True)
        # Simulator connects an active camera by default. Force the exact
        # no-camera state that load_settings sees on a real missing camera.
        with scope._camera_cache_lock:
            scope._camera_cache['active'] = False
            scope._camera_cache['max_exposure'] = 0.0

        assert scope.camera_max_exposure == 0.0
        # And the helper handles it correctly
        assert get_safe_max_exposure(scope) == DEFAULT_MAX_EXPOSURE_MS
