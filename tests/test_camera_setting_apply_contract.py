# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression net: the camera-setting apply contract (cluster 7 commit A).

The three geometry/format setters (set_frame_size, set_binning_size,
set_pixel_format) observe success by VALUE and failure by RAISE:

  - success returns the applied value (frame size returns the DELIVERED
    geometry, which may differ from the request);
  - a LIVE driver rejecting the apply (False return) or raising from it
    raises CameraSettingRejected -- after logging and firing exactly one
    notifications.error -- so a caller that drops the return cannot
    record a rejected apply as current;
  - an absent / inactive camera stays a quiet sentinel (None / False)
    per the missing-hardware contract, with the deduped absent
    notification, and never raises;
  - the cache keeps the prior hardware truth through every failure shape.

Also pins Lumascope.initialize's persisted-binning reconciliation: a
persisted factor the connected camera does not support is replaced by the
camera-reported factor BEFORE the apply (the settings-file-vs-swapped-
camera case), and a supported factor passes through unchanged.

Harness: the ScriptedCameraDriver / _build_imaging helpers from
tests/test_camera_getter_sentinel_containment.py (imported, not copied),
extended with scriptable apply results.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from modules.exceptions import CameraSettingRejected
from tests.test_camera_getter_sentinel_containment import (
    GOOD_ROUND,
    ScriptedCameraDriver,
    _build_imaging,
)


class _RecordingNotifications:
    """Stand-in for modules.lumascope_api.imaging.notifications."""

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.criticals = []

    def error(self, category, title, message, **kw):
        self.errors.append((category, title, message))

    def warning(self, category, title, message, **kw):
        self.warnings.append((category, title, message))

    def critical(self, category, title, message, **kw):
        self.criticals.append((category, title, message))


class ApplyDriver(ScriptedCameraDriver):
    """ScriptedCameraDriver plus scriptable apply results for the three
    state-changing setters. Defaults apply cleanly (frame size delivers
    the request); tests override the apply_* callables per failure shape."""

    def __init__(self, scripts: dict, active: bool = True):
        super().__init__(scripts, active)
        self.apply_frame_size = lambda w, h: {'width': w, 'height': h}
        self.apply_binning = lambda size: True
        self.apply_pixel_format = lambda fmt: True

    def set_frame_size(self, w, h):
        return self.apply_frame_size(w, h)

    def set_binning_size(self, size):
        return self.apply_binning(size)

    def set_pixel_format(self, pixel_format):
        return self.apply_pixel_format(pixel_format)


def apply_driver(active: bool = True) -> ApplyDriver:
    """Steady-good reads (populate caches 1936x1216 / Mono12 / binning 2)
    with clean default applies."""
    return ApplyDriver({name: [value] for name, value in GOOD_ROUND.items()}, active=active)


@pytest.fixture
def notes(monkeypatch) -> _RecordingNotifications:
    recorder = _RecordingNotifications()
    monkeypatch.setattr('modules.lumascope_api.imaging.notifications', recorder)
    return recorder


# --- A. Rejection is loud ----------------------------------------------------


def test_set_frame_size_rejection_raises_and_preserves_cache(notes):
    driver = apply_driver()
    imaging = _build_imaging(driver)  # populate caches 1936x1216
    driver.apply_frame_size = lambda w, h: False

    with pytest.raises(CameraSettingRejected) as excinfo:
        imaging.set_frame_size(1900, 1900)

    assert excinfo.value.setting == 'frame_size'
    assert excinfo.value.requested == {'width': 1900, 'height': 1900}
    assert len(notes.errors) == 1, notes.errors
    assert imaging.camera_frame_size == {'width': 1936, 'height': 1216}, (
        'a rejected resize must leave the cache at the geometry the hardware still holds'
    )


def test_set_binning_size_rejection_raises_and_preserves_cache(notes):
    driver = apply_driver()
    imaging = _build_imaging(driver)  # populate caches binning 2
    driver.apply_binning = lambda size: False

    with pytest.raises(CameraSettingRejected) as excinfo:
        imaging.set_binning_size(4)

    assert excinfo.value.setting == 'binning'
    assert excinfo.value.requested == 4
    assert len(notes.errors) == 1, notes.errors
    assert imaging._binning_size == 2, (
        'a rejected binning must not commit the requested factor -- '
        'scale-bar / FOV math reads this value'
    )


def test_set_pixel_format_rejection_raises_and_preserves_cache(notes):
    driver = apply_driver()
    imaging = _build_imaging(driver)  # populate caches 'Mono12'
    driver.apply_pixel_format = lambda fmt: False

    with pytest.raises(CameraSettingRejected) as excinfo:
        imaging.set_pixel_format('Mono8')

    assert excinfo.value.setting == 'pixel_format'
    assert excinfo.value.requested == 'Mono8'
    assert len(notes.errors) == 1, notes.errors
    assert imaging.camera_pixel_format == 'Mono12'


# --- B. Delivered geometry returned -------------------------------------------


def test_set_frame_size_returns_delivered_geometry_and_caches_it(notes):
    # The driver clamps/snaps the request to its legal grid; the caller
    # receives the geometry actually in effect, and the cache matches it.
    driver = apply_driver()
    imaging = _build_imaging(driver)
    driver.apply_frame_size = lambda w, h: {'width': 1896, 'height': 1900}

    delivered = imaging.set_frame_size(1900, 1900)

    assert delivered == {'width': 1896, 'height': 1900}
    assert imaging.camera_frame_size == {'width': 1896, 'height': 1900}
    assert notes.errors == []


# --- C. Driver-raise paths -----------------------------------------------------


def test_set_binning_size_driver_raise_becomes_typed_rejection(notes):
    driver = apply_driver()
    imaging = _build_imaging(driver)

    def _boom(size):
        raise RuntimeError('SDK sulked')

    driver.apply_binning = _boom

    with pytest.raises(CameraSettingRejected) as excinfo:
        imaging.set_binning_size(4)

    assert isinstance(excinfo.value.__cause__, RuntimeError), (
        'the driver exception must be chained onto the typed rejection'
    )
    assert len(notes.errors) == 1, notes.errors
    assert imaging._binning_size == 2  # prior factor intact


def test_set_pixel_format_driver_raise_becomes_typed_rejection(notes):
    driver = apply_driver()
    imaging = _build_imaging(driver)

    def _boom(fmt):
        raise RuntimeError('SDK sulked')

    driver.apply_pixel_format = _boom

    with pytest.raises(CameraSettingRejected) as excinfo:
        imaging.set_pixel_format('Mono8')

    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert len(notes.errors) == 1, notes.errors
    assert imaging.camera_pixel_format == 'Mono12'


# --- D. Absent / inactive camera stays a quiet sentinel -------------------------


def test_absent_camera_setters_return_sentinels_and_notify(notes):
    imaging = _build_imaging(None)

    assert imaging.set_frame_size(1900, 1900) is None
    assert len(notes.warnings) == 1

    assert imaging.set_binning_size(2) is False
    assert len(notes.warnings) == 2

    assert imaging.set_pixel_format('Mono8') is False
    assert len(notes.warnings) == 3

    assert notes.errors == []  # absent is the quiet shape, not the loud one
    # Nothing recorded in the cache: every entry still holds its seed.
    assert imaging.camera_frame_size == {'width': 0, 'height': 0}
    assert imaging._binning_size == 1
    assert imaging.camera_pixel_format is None


def test_inactive_driver_setters_return_sentinels_without_reaching_driver(notes):
    # The absent guard checks driver.active too: an inactive driver used to
    # fall through to the driver's own False (and set_binning_size then
    # treated it as a live rejection).
    driver = apply_driver(active=False)
    reached = []
    driver.apply_binning = lambda size: reached.append(size) or True
    driver.apply_frame_size = lambda w, h: reached.append((w, h)) or {'width': w, 'height': h}
    driver.apply_pixel_format = lambda fmt: reached.append(fmt) or True
    imaging = _build_imaging(driver)

    assert imaging.set_frame_size(1900, 1900) is None
    assert imaging.set_binning_size(2) is False
    assert imaging.set_pixel_format('Mono8') is False
    assert reached == [], 'an inactive driver must never receive the apply'
    assert len(notes.warnings) == 3
    assert notes.errors == []


# --- F. initialize persisted-binning reconciliation ------------------------------


def _init_config(binning_size: int, frame_width: int = 1900, frame_height: int = 1900):
    from modules.scope_init_config import ScopeInitConfig

    return ScopeInitConfig(
        labware=None,
        objective_id='4x',
        turret_config=None,
        binning_size=binning_size,
        frame_width=frame_width,
        frame_height=frame_height,
        acceleration_pct=100,
        stage_offset={'x': 0, 'y': 0},
        scale_bar_enabled=False,
        capture_depth=8,
    )


def _drive_initialize(config, monkeypatch, *, no_camera: bool = False, prepare=None):
    """Full Lumascope(simulate=True).initialize with spies on
    set_binning_size / set_frame_size (delegating to the real setters), a
    recorder on the _lumascope logger, and an accel-limit spy marking that
    bring-up reached its final step.

    Returns (applied_binnings, applied_frames, logged_errors, reached_end).
    """
    from modules.lumascope_api import Lumascope

    scope = Lumascope(simulate=True)
    saved_driver = scope._camera_driver
    try:
        applied_binnings = []
        real_set_binning = scope.imaging.set_binning_size
        monkeypatch.setattr(
            scope.imaging,
            'set_binning_size',
            lambda size: applied_binnings.append(size) or real_set_binning(size),
        )
        applied_frames = []
        real_set_frame = scope.imaging.set_frame_size
        monkeypatch.setattr(
            scope.imaging,
            'set_frame_size',
            lambda w, h: applied_frames.append((w, h)) or real_set_frame(w, h),
        )
        reached_end = []
        real_accel = scope.motion.set_acceleration_limit
        monkeypatch.setattr(
            scope.motion,
            'set_acceleration_limit',
            lambda val_pct: reached_end.append(val_pct) or real_accel(val_pct=val_pct),
        )
        errors = []
        monkeypatch.setattr(
            'modules.lumascope_api._lumascope.logger',
            SimpleNamespace(
                error=lambda msg, *a, **kw: errors.append(str(msg)),
                warning=lambda *a, **kw: None,
                info=lambda *a, **kw: None,
                debug=lambda *a, **kw: None,
                exception=lambda *a, **kw: None,
            ),
        )
        if prepare is not None:
            prepare(scope)
        if no_camera:
            scope._camera_driver = None
        scope.initialize(config)
        return applied_binnings, applied_frames, errors, bool(reached_end)
    finally:
        scope._camera_driver = saved_driver
        scope.disconnect()


def test_initialize_reconciles_unsupported_persisted_binning(monkeypatch):
    # A settings file written against a different camera persists a factor
    # this camera does not support (sim supports [1, 2, 4]); initialize must
    # apply the camera-reported factor instead, and say so.
    applied, _frames, errors, _ = _drive_initialize(_init_config(8), monkeypatch)
    assert applied == [1], (
        f'unsupported persisted binning must fall back to the '
        f'camera-reported factor; applied {applied}'
    )
    assert any('persisted binning' in e for e in errors), errors


def test_initialize_passes_supported_persisted_binning_through(monkeypatch):
    applied, frames, errors, _ = _drive_initialize(_init_config(2), monkeypatch)
    assert applied == [2]
    assert frames == [(1900, 1900)]  # supported factor: frame passes through as-is
    assert not any('persisted binning' in e for e in errors), errors


def test_initialize_refits_persisted_frame_at_reconciled_binning(monkeypatch):
    # The persisted frame is a DISPLAYED size at the persisted factor: 484x304
    # persisted at 8x describes a 3872x2432 native intent. Reconciled to the
    # camera-reported 1x, the frame must be refit from that native intent
    # (capped at the sim's 1920x1200 native, aligned to its 48x4 grid ->
    # 1920x1200), NOT applied as a tiny 484x304 ROI at 1x.
    applied, frames, errors, _ = _drive_initialize(
        _init_config(8, frame_width=484, frame_height=304), monkeypatch
    )
    assert applied == [1]
    assert frames != [(484, 304)], 'the persisted displayed size must be refit, not reused'
    assert frames == [(1920, 1200)], frames
    assert any('persisted binning' in e for e in errors), errors


def test_initialize_reconciliation_fires_exactly_one_user_warning(monkeypatch):
    # The reconciliation is user-visible, not just a log line: the saved
    # binning silently coming up different needs a popup naming the fix
    # (pick a binning in Microscope Settings to update the saved value).
    recorder = _RecordingNotifications()
    monkeypatch.setattr('modules.lumascope_api._lumascope.notifications', recorder)
    _drive_initialize(_init_config(8), monkeypatch)
    saved_binning_warnings = [w for w in recorder.warnings if w[1] == 'Saved binning not supported']
    assert len(saved_binning_warnings) == 1, recorder.warnings


def test_initialize_supported_binning_fires_no_reconciliation_warning(monkeypatch):
    recorder = _RecordingNotifications()
    monkeypatch.setattr('modules.lumascope_api._lumascope.notifications', recorder)
    _drive_initialize(_init_config(2), monkeypatch)
    assert not any(w[1] == 'Saved binning not supported' for w in recorder.warnings), (
        recorder.warnings
    )


def test_initialize_without_camera_skips_reconciliation_quietly(monkeypatch):
    # No camera: the applies are quiet no-ops and reconciliation must not run
    # at all -- the absent-fallback capability values must not masquerade as
    # a camera's answer and fire a false 'not supported' ERROR.
    applied, _frames, errors, reached_end = _drive_initialize(
        _init_config(8, frame_width=484, frame_height=304),
        monkeypatch,
        no_camera=True,
    )
    assert applied == [8], 'no reconciliation without a camera: the persisted factor passes'
    assert errors == [], f'no reconciliation/rejection ERROR may fire without a camera: {errors}'
    assert reached_end, 'initialize must complete without a camera'


# --- initialize containment (a mid-bring-up rejection must not abort) ------------


def test_initialize_contains_frame_size_rejection_and_completes(monkeypatch):
    # A live driver rejecting the frame-size apply mid-initialize: the typed
    # rejection is logged and bring-up CONTINUES (a propagated raise once
    # crashed the app build via load_settings' re-raise).
    def _reject_frame(scope):
        scope._camera_driver.set_frame_size = lambda w, h: False

    _applied, frames, errors, reached_end = _drive_initialize(
        _init_config(2), monkeypatch, prepare=_reject_frame
    )
    assert frames == [(1900, 1900)]  # the apply was attempted...
    assert any('frame size apply rejected' in e for e in errors), errors
    assert reached_end, (
        'initialize must run to completion (stage offset / scale bar / '
        'acceleration) despite the contained rejection'
    )
