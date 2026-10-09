# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression net: the camera-setting apply contract (cluster 7 commit A).

The three geometry/format setters (set_frame_size, set_binning_size,
set_pixel_format) observe success by VALUE and failure by RAISE:

  - success returns the applied value (frame size returns the DELIVERED
    geometry, which may differ from the request);
  - a LIVE driver rejecting the apply (False return) or raising from it
    raises CameraSettingRejected, carrying the title and sentence the one
    reporter shows -- the setter itself neither logs nor notifies -- so a
    caller that drops the return cannot record a rejected apply as current;
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

from modules.exceptions import CameraSettingRejected, HardwareCommandRefusedError, MissingPart
from modules.notification_center import Severity
from tests.test_camera_getter_sentinel_containment import (
    GOOD_ROUND,
    ScriptedCameraDriver,
    _build_imaging,
)
from tests.scope_fakes import bind_settings_like_a_session, build_scope


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


def _posted(centre_posts, severity):
    """The (category, title, message) of each post at ``severity``."""
    return [(n.category, n.title, n.message) for n in centre_posts if n.severity == severity]


def _assert_carries_its_words(rejected, title, in_sentence, centre_posts):
    """The rejection is shown once, by its reporter, in its own words: the
    setter shows nothing, and the exception carries the title and the
    person's sentence the reporter shows."""
    shown = _posted(centre_posts, Severity.ERROR) + _posted(centre_posts, Severity.WARNING)
    assert shown == [], 'the setter must not show the rejection itself; the reporter shows it once'
    assert rejected.title == title
    assert in_sentence in str(rejected), str(rejected)


# --- A. Rejection is loud ----------------------------------------------------


def test_set_frame_size_rejection_raises_and_preserves_cache(centre_posts):
    driver = apply_driver()
    imaging = _build_imaging(driver)  # populate caches 1936x1216
    driver.apply_frame_size = lambda w, h: False

    with pytest.raises(CameraSettingRejected) as excinfo:
        imaging.set_frame_size(1900, 1900)

    assert excinfo.value.setting == 'frame_size'
    assert excinfo.value.requested == {'width': 1900, 'height': 1900}
    _assert_carries_its_words(excinfo.value, 'Frame size change failed', '1900x1900', centre_posts)
    assert imaging.frame_size_cached == {'width': 1936, 'height': 1216}, (
        'a rejected resize must leave the cache at the geometry the hardware still holds'
    )


def test_set_binning_size_rejection_raises_and_preserves_cache(centre_posts):
    driver = apply_driver()
    imaging = _build_imaging(driver)  # populate caches binning 2
    driver.apply_binning = lambda size: False

    with pytest.raises(CameraSettingRejected) as excinfo:
        imaging.set_binning_size(4)

    assert excinfo.value.setting == 'binning'
    assert excinfo.value.requested == 4
    _assert_carries_its_words(excinfo.value, 'Binning change failed', '4x4', centre_posts)
    assert imaging._binning_size == 2, (
        'a rejected binning must not commit the requested factor -- '
        'scale-bar / FOV math reads this value'
    )


def test_set_pixel_format_rejection_raises_and_preserves_cache(centre_posts):
    driver = apply_driver()
    imaging = _build_imaging(driver)  # populate caches 'Mono12'
    driver.apply_pixel_format = lambda fmt: False

    with pytest.raises(CameraSettingRejected) as excinfo:
        imaging.set_pixel_format('Mono8')

    assert excinfo.value.setting == 'pixel_format'
    assert excinfo.value.requested == 'Mono8'
    _assert_carries_its_words(excinfo.value, 'Pixel format change failed', 'Mono8', centre_posts)
    assert imaging.pixel_format_cached == 'Mono12'


def test_a_reported_rejection_is_shown_once_in_its_own_words(centre_posts):
    # The end of the flight: the reporter shows the setter's rejection once,
    # as a fault under the setter's title and in its sentence -- not the
    # generic sentence an untyped fault gets -- and nothing else shows it.
    from modules.notification_center import NotificationCenter

    driver = apply_driver()
    imaging = _build_imaging(driver)
    driver.apply_binning = lambda size: False
    with pytest.raises(CameraSettingRejected) as excinfo:
        imaging.set_binning_size(4)

    centre = NotificationCenter(dedup_window_s=10.0)
    shown = []
    centre.add_listener(shown.append, min_severity=Severity.INFO)
    centre.report_outcome(excinfo.value, solicited=True, category='BINNING')
    centre.report_outcome(excinfo.value, solicited=True, category='BINNING')

    assert [(n.severity, n.title, n.message) for n in shown] == [
        (Severity.ERROR, 'Binning change failed', str(excinfo.value))
    ]
    assert '4x4' in shown[0].message
    assert _posted(centre_posts, Severity.ERROR) == [], 'the setter showed it as well'


# --- B. Delivered geometry returned -------------------------------------------


def test_set_frame_size_returns_delivered_geometry_and_caches_it(centre_posts):
    # The driver clamps/snaps the request to its legal grid; the caller
    # receives the geometry actually in effect, and the cache matches it.
    driver = apply_driver()
    imaging = _build_imaging(driver)
    driver.apply_frame_size = lambda w, h: {'width': 1896, 'height': 1900}

    delivered = imaging.set_frame_size(1900, 1900)

    assert delivered == {'width': 1896, 'height': 1900}
    assert imaging.frame_size_cached == {'width': 1896, 'height': 1900}
    assert _posted(centre_posts, Severity.ERROR) == []


# --- C. Driver-raise paths -----------------------------------------------------


def test_set_binning_size_driver_raise_becomes_typed_rejection(centre_posts):
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
    _assert_carries_its_words(excinfo.value, 'Binning change failed', 'SDK sulked', centre_posts)
    assert imaging._binning_size == 2  # prior factor intact


def test_set_pixel_format_driver_raise_becomes_typed_rejection(centre_posts):
    driver = apply_driver()
    imaging = _build_imaging(driver)

    def _boom(fmt):
        raise RuntimeError('SDK sulked')

    driver.apply_pixel_format = _boom

    with pytest.raises(CameraSettingRejected) as excinfo:
        imaging.set_pixel_format('Mono8')

    assert isinstance(excinfo.value.__cause__, RuntimeError)
    _assert_carries_its_words(
        excinfo.value, 'Pixel format change failed', 'SDK sulked', centre_posts
    )
    assert imaging.pixel_format_cached == 'Mono12'


# --- D. Absent / inactive camera is refused, naming it ---------------------------


def _refused_for_the_camera(apply) -> None:
    with pytest.raises(HardwareCommandRefusedError) as exc:
        apply()
    assert exc.value.missing == MissingPart.CAMERA


def test_absent_camera_setters_are_refused_and_post_nothing(centre_posts):
    imaging = _build_imaging(None)

    _refused_for_the_camera(lambda: imaging.set_frame_size(1900, 1900))
    _refused_for_the_camera(lambda: imaging.set_binning_size(2))
    _refused_for_the_camera(lambda: imaging.set_pixel_format('Mono8'))

    # Reporting the refusal is the caller's: the API posts nothing.
    assert _posted(centre_posts, Severity.WARNING) == []
    # Nothing recorded in the cache: every entry still holds its seed.
    assert imaging.frame_size_cached == {'width': 0, 'height': 0}
    assert imaging._binning_size == 1
    assert imaging.pixel_format_cached is None


def test_inactive_driver_setters_are_refused_without_reaching_driver(centre_posts):
    # The presence question reads driver.active too: an inactive driver used
    # to fall through to the driver's own False (and set_binning_size then
    # treated it as a live rejection).
    driver = apply_driver(active=False)
    reached = []
    driver.apply_binning = lambda size: reached.append(size) or True
    driver.apply_frame_size = lambda w, h: reached.append((w, h)) or {'width': w, 'height': h}
    driver.apply_pixel_format = lambda fmt: reached.append(fmt) or True
    imaging = _build_imaging(driver)

    _refused_for_the_camera(lambda: imaging.set_frame_size(1900, 1900))
    _refused_for_the_camera(lambda: imaging.set_binning_size(2))
    _refused_for_the_camera(lambda: imaging.set_pixel_format('Mono8'))
    assert reached == [], 'an inactive driver must never receive the apply'
    assert _posted(centre_posts, Severity.WARNING) == []


# --- F. initialize persisted-binning reconciliation ------------------------------


def _init_config(binning_size: int, frame_width: int = 1900, frame_height: int = 1900):
    from modules.scope_init_config import ScopeInitConfig

    return ScopeInitConfig(
        turreted=False,
        preferred_turret_slot=None,
        binning_size=binning_size,
        frame_width=frame_width,
        frame_height=frame_height,
        acceleration_pct=100,
        image_mode='8bit',
    )


def _drive_initialize(config, monkeypatch, *, no_camera: bool = False, prepare=None):
    """Full Lumascope(simulate=True).initialize with spies on
    set_binning_size / set_frame_size (delegating to the real setters), a
    recorder on the _lumascope logger, and an accel-limit spy marking that
    bring-up reached its final step.

    Returns (applied_binnings, applied_frames, logged_errors, reached_end).
    """

    scope = build_scope(simulate=True)
    # The objective bring-up checks is the settings' one.
    bind_settings_like_a_session(scope, objective_id='4x Oly')
    saved_driver = scope._camera_driver
    try:
        # initialize is bring-up and binds the impl seams (it runs before
        # executor registration by design), so the spies sit there.
        applied_binnings = []
        real_set_binning = scope.imaging._set_binning_size_impl
        monkeypatch.setattr(
            scope.imaging,
            '_set_binning_size_impl',
            lambda size: applied_binnings.append(size) or real_set_binning(size),
        )
        applied_frames = []
        real_set_frame = scope.imaging._set_frame_size_impl
        monkeypatch.setattr(
            scope.imaging,
            '_set_frame_size_impl',
            lambda w, h: applied_frames.append((w, h)) or real_set_frame(w, h),
        )
        reached_end = []
        real_accel = scope.motion._set_acceleration_limit_impl
        monkeypatch.setattr(
            scope.motion,
            '_set_acceleration_limit_impl',
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
    applied, _frames, _errors, _ = _drive_initialize(_init_config(8), monkeypatch)
    assert applied == [1], (
        f'unsupported persisted binning must fall back to the '
        f'camera-reported factor; applied {applied}'
    )


def test_initialize_passes_supported_persisted_binning_through(monkeypatch):
    # A frame the simulated 1920 x 1200 sensor delivers at 2x.
    applied, frames, errors, _ = _drive_initialize(
        _init_config(2, frame_width=900, frame_height=600), monkeypatch
    )
    assert applied == [2]
    assert frames == [(900, 600)]  # supported factor: frame passes through as-is
    assert not any('persisted binning' in e for e in errors), errors


def test_initialize_refits_persisted_frame_at_reconciled_binning(monkeypatch):
    # The persisted frame is a DISPLAYED size at the persisted factor: 484x304
    # persisted at 8x describes a 3872x2432 native intent. Reconciled to the
    # camera-reported 1x, the frame must be refit from that native intent
    # (capped at the sim's 3840x2160 native -> 3840x2160), NOT applied as a
    # tiny 484x304 ROI at 1x.
    applied, frames, _errors, _ = _drive_initialize(
        _init_config(8, frame_width=484, frame_height=304), monkeypatch
    )
    assert applied == [1]
    assert frames != [(484, 304)], 'the persisted displayed size must be refit, not reused'
    assert frames == [(3840, 2160)], frames


def test_initialize_reconciliation_fires_exactly_one_user_warning(monkeypatch, centre_posts):
    # The reconciliation is user-visible, not just a log line: the saved
    # binning silently coming up different needs a popup naming the fix
    # (pick a binning in Microscope Settings to update the saved value).
    from modules.exceptions import BinningSubstitutedNotice

    _drive_initialize(_init_config(8), monkeypatch)
    substituted = [n for n in centre_posts if n.reason == BinningSubstitutedNotice.reason]
    assert len(substituted) == 1, centre_posts


def test_initialize_supported_binning_fires_no_reconciliation_warning(monkeypatch, centre_posts):
    from modules.exceptions import BinningSubstitutedNotice

    _drive_initialize(_init_config(2), monkeypatch)
    assert not any(n.reason == BinningSubstitutedNotice.reason for n in centre_posts), centre_posts


def test_initialize_without_camera_skips_reconciliation_quietly(monkeypatch):
    # No camera: nothing is applied and reconciliation must not run at all --
    # the absent-fallback capability values must not masquerade as a
    # camera's answer and fire a false 'not supported' ERROR.
    applied, frames, errors, reached_end = _drive_initialize(
        _init_config(8, frame_width=484, frame_height=304),
        monkeypatch,
        no_camera=True,
    )
    assert (applied, frames) == ([], []), 'with no camera there is nothing to apply'
    assert errors == [], f'no reconciliation/rejection ERROR may fire without a camera: {errors}'
    assert reached_end, 'initialize must complete without a camera'


# --- initialize containment (a mid-bring-up rejection must not abort) ------------


def test_initialize_contains_frame_size_rejection_and_completes(monkeypatch):
    # A live driver rejecting the frame-size apply mid-initialize: the typed
    # rejection is reported once, where bring-up stops its flight, and
    # bring-up CONTINUES (a propagated raise once crashed the app build via
    # load_settings' re-raise).
    from modules.notification_center import notifications

    reported = []
    monkeypatch.setattr(
        notifications, 'report_outcome', lambda exc, **kw: reported.append((exc, kw))
    )

    def _reject_frame(scope):
        scope._camera_driver.set_frame_size = lambda w, h: False

    _applied, frames, _errors, reached_end = _drive_initialize(
        _init_config(2, frame_width=900, frame_height=600), monkeypatch, prepare=_reject_frame
    )
    assert frames == [(900, 600)]  # the apply was attempted...
    rejections = [exc for exc, _kw in reported if isinstance(exc, CameraSettingRejected)]
    assert [exc.setting for exc in rejections] == ['frame_size'], reported
    assert reached_end, (
        'initialize must run to completion (stage offset / scale bar / '
        'acceleration) despite the contained rejection'
    )


def _a_camera_with_both_toggles(answer):
    """A ``prepare`` giving the simulated camera both toggles, each answering ``answer``."""

    def prepare(scope):
        import dataclasses

        scope.capabilities = dataclasses.replace(
            scope.capabilities,
            camera_supports_conversion_gain_mode=True,
            camera_supports_line_noise_reduction=True,
        )
        scope._camera_driver.set_conversion_gain_mode = lambda mode: answer()
        scope._camera_driver.set_line_noise_reduction = lambda enabled: answer()

    return prepare


def _refuses():
    return False


def _raises():
    raise RuntimeError('node write failed')


@pytest.mark.parametrize('answer', [_refuses, _raises], ids=['refuses', 'raises'])
def test_initialize_contains_a_toggle_rejection_and_completes(monkeypatch, answer):
    # A camera that has both toggles but refuses (or raises from) each at
    # bring-up: each is reported once, unsolicited, and bring-up runs on.
    import dataclasses

    from modules.notification_center import notifications

    reported = []
    monkeypatch.setattr(
        notifications, 'report_outcome', lambda exc, **kw: reported.append((exc, kw))
    )
    config = dataclasses.replace(
        _init_config(1, frame_width=900, frame_height=600),
        high_conversion_gain=True,
        line_noise_reduction=True,
    )

    _applied, _frames, _errors, reached_end = _drive_initialize(
        config, monkeypatch, prepare=_a_camera_with_both_toggles(answer)
    )

    rejections = [
        (exc.setting, kw['solicited'])
        for exc, kw in reported
        if isinstance(exc, CameraSettingRejected)
    ]
    assert rejections == [('conversion_gain_mode', False), ('line_noise_reduction', False)]
    assert reached_end, 'bring-up runs to completion past a refused toggle'
