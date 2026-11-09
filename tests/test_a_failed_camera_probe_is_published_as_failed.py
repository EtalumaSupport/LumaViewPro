# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A camera feature probe that fails as the camera comes up is published as
failed, never as "the camera lacks it".

The five boolean camera capabilities (``camera_supports_conversion_gain_mode``,
``camera_supports_line_noise_reduction``, ``camera_supports_black_level``,
``camera_has_clock_latch``, ``camera_reports_temperature``) are ``bool | None``:
``None`` is a probe that failed, and whether the camera has the feature is not
known (``scope_capabilities._probe``). What reads them acts on that:

- bring-up applies neither toggle whose probe failed, keeps its saved setting,
  and keeps the failure on the bring-up record's camera part; it reports
  nothing more, ``_probe``'s WARNING being the fault's one log line;
- a setter on ``None`` probes the camera live: it writes, refuses as absent,
  or raises ``CameraReadFailedError`` with nothing written;
- the camera-state snapshot reads the black level live on ``None``;
- REST publishes ``null``;
- the GUI offers a toggle unless the camera lacks it.

The simulated camera has neither toggle (a simulator gap, Eric 2026-10-11),
and none of its probes can fail: each test that needs one gives the simulated
camera a probe whose answer it sets.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import _lumascope.modules.app_context as _app_ctx
from _lumascope.drivers.exceptions import CameraReadFailedError
from _lumascope.drivers.simulated_camera import SimulatedCamera
from _lumascope.modules.exceptions import CameraSettingRejected
from _lumascope.modules import scope_capabilities
from _lumascope.modules.lumascope_api.bring_up import Part
from _lumascope.modules.notification_center import notifications
from _lumascope.modules.scope_session import ScopeSession
from _lumascope.rest.app import build_app
from tests.settings_fixtures import complete_settings
from ui.advanced_settings import AdvancedSettings


class _Probe:
    """A probe the test answers: ``answer`` is returned, or raised when an exception.

    a stand-in by design: the simulated camera's probes cannot fail (and it
    has no conversion gain node), a simulator gap.
    """

    def __init__(self, answer):
        self.answer = answer

    def __call__(self, *_camera):
        if isinstance(self.answer, BaseException):
            raise self.answer
        return self.answer


def _failing():
    return CameraReadFailedError('BslConversionGainMode probe failed: usb gone')


@pytest.fixture
def toggles(monkeypatch):
    """The simulated camera's conversion gain probe fails as it comes up, and
    the writes that reach the camera are recorded; saved on before bring-up."""
    probe = _Probe(_failing())
    written = []
    monkeypatch.setattr(SimulatedCamera, 'supports_conversion_gain_mode', probe, raising=False)
    monkeypatch.setattr(
        SimulatedCamera,
        'set_conversion_gain_mode',
        lambda _camera, mode: written.append(mode) or True,
        raising=False,
    )
    reported = []
    real_report = notifications.report_outcome

    def report(exc, **kwargs):
        reported.append(exc)
        return real_report(exc, **kwargs)

    monkeypatch.setattr(notifications, 'report_outcome', report)
    session = ScopeSession.create(
        complete_settings(camera={'high_conversion_gain': True}),
        simulate=True,
        warn_pre_release=False,
    )
    yield SimpleNamespace(session=session, probe=probe, written=written, reported=reported)
    session.shutdown()


def test_as_shipped_a_camera_without_the_toggles_publishes_false_not_none():
    session = ScopeSession.create(complete_settings(), simulate=True, warn_pre_release=False)
    try:
        caps = session.scope.capabilities
        assert caps.camera_supports_conversion_gain_mode is False
        assert caps.camera_supports_line_noise_reduction is False
        assert caps.camera_supports_black_level is True
    finally:
        session.shutdown()


def test_a_failed_toggle_probe_keeps_the_saved_setting_and_is_on_the_record(toggles):
    scope = toggles.session.scope
    assert scope.capabilities.camera_supports_conversion_gain_mode is None
    assert scope.read_setting('camera.high_conversion_gain') is True
    assert scope.camera.high_conversion_gain is None, 'what the camera holds is not known'
    assert toggles.written == []

    camera_part = next(p for p in toggles.session.bring_up_record().parts if p.part == Part.CAMERA)
    assert camera_part.cause == 'settings_not_applied'
    assert (
        "CameraReadFailedError: The camera's conversion gain probe failed when it came up; "
        'the saved setting is kept and was not applied'
    ) in camera_part.detail


def test_a_failed_toggle_probe_is_told_once_in_the_log_and_never_reported(monkeypatch, caplog):
    monkeypatch.setattr(
        SimulatedCamera, 'supports_conversion_gain_mode', _Probe(_failing()), raising=False
    )
    reported = []
    monkeypatch.setattr(notifications, 'report_outcome', lambda exc, **kw: reported.append(exc))
    with caplog.at_level(logging.WARNING):
        session = ScopeSession.create(
            complete_settings(camera={'high_conversion_gain': True}),
            simulate=True,
            warn_pre_release=False,
        )
    try:
        told = [r.getMessage() for r in caplog.records if 'conversion' in r.getMessage().lower()]
        assert len(told) == 1, told
        assert 'usb gone' in told[0]
        assert not [e for e in reported if 'conversion' in str(e).lower()]
    finally:
        session.shutdown()


def test_a_write_on_a_failed_probe_probes_the_camera_live(toggles):
    camera = toggles.session.scope.camera
    for enabled in (True, False):
        with pytest.raises(CameraSettingRejected) as refused:
            camera.set_high_conversion_gain(enabled)
        assert refused.value.reason == 'camera_read_failed'
        assert 'usb gone' in str(refused.value.__cause__)
    assert toggles.written == []
    assert toggles.session.scope.read_setting('camera.high_conversion_gain') is True

    toggles.probe.answer = True
    camera.set_high_conversion_gain(True)
    assert toggles.written == ['High']
    assert camera.high_conversion_gain is True
    assert toggles.session.scope.capabilities.camera_supports_conversion_gain_mode is None, (
        "bring-up's record, which the setter's own probe does not rewrite"
    )


def test_rest_publishes_a_failed_probe_as_null(toggles):
    with TestClient(build_app(toggles.session)) as client:
        answer = client.get('/api/v1/scope/capabilities')
    assert answer.status_code == 200
    body = answer.json()
    assert body['camera_supports_conversion_gain_mode'] is None
    assert body['camera_supports_line_noise_reduction'] is False


@pytest.fixture
def black_level(monkeypatch):
    """The simulated camera's black level probe fails as it comes up."""
    probe = _Probe(CameraReadFailedError('BlackLevel probe failed: usb gone'))
    monkeypatch.setattr(SimulatedCamera, 'supports_black_level', probe)
    session = ScopeSession.create(complete_settings(), simulate=True, warn_pre_release=False)
    yield SimpleNamespace(session=session, probe=probe)
    session.shutdown()


def test_the_black_level_on_a_failed_probe_is_set_and_saved_live(black_level):
    scope = black_level.session.scope
    assert scope.capabilities.camera_supports_black_level is None

    black_level.probe.answer = True
    assert scope.camera.set_black_level(3.0) == 3.0

    snapshot = scope.camera.save_camera_state('test')
    assert snapshot['black_level'] == 3.0
    scope.camera.set_black_level(5.0)
    scope.camera.restore_camera_state(snapshot)
    assert scope.camera.read_black_level() == 3.0


def test_a_restore_whose_live_black_level_probe_fails_reports_it_and_goes_on(
    black_level, monkeypatch
):
    """The black level is not sent when its probe fails: that refusal is
    reported once, as a refused black level write is, and the restore goes on."""
    scope = black_level.session.scope
    black_level.probe.answer = True
    scope.camera.set_black_level(3.0)
    snapshot = scope.camera.save_camera_state('test')
    scope.camera.set_black_level(5.0)
    reported = []
    monkeypatch.setattr(
        '_lumascope.modules.notification_center.notifications.report_outcome',
        lambda exc, **kw: reported.append(exc),
    )

    black_level.probe.answer = CameraReadFailedError('BlackLevel probe failed: usb gone')
    scope.camera.restore_camera_state(snapshot)

    assert [(r.reason, r.setting) for r in reported] == [('camera_read_failed', 'black_level')]
    assert scope.camera.read_black_level() == 5.0


def test_probe_answers_absent_for_a_missing_member_and_failed_for_a_hardware_fault():
    def missing():
        raise AttributeError('no such member')

    def faulted():
        raise CameraReadFailedError('usb gone')

    def buggy():
        raise TypeError('a bug')

    assert scope_capabilities._probe('x', missing, False, failed=None) is False
    assert scope_capabilities._probe('x', faulted, False, failed=None) is None
    with pytest.raises(TypeError):
        scope_capabilities._probe('x', buggy, False, failed=None)


def test_advanced_settings_offers_a_toggle_unless_the_camera_lacks_it(monkeypatch):
    """The GUI shows the toggle for True and for a failed probe (None), and
    hides it only for a camera without it; a press then goes to the API."""
    shown = {}
    for supported in (True, None, False):
        caps = SimpleNamespace(
            camera_supports_conversion_gain_mode=supported,
            camera_supports_line_noise_reduction=supported,
            has_xy_stage=True,
            has_firmware_stim=False,
        )
        scope = SimpleNamespace(capabilities=caps)
        monkeypatch.setattr(
            _app_ctx,
            'ctx',
            SimpleNamespace(
                settings=complete_settings(), lumaview=SimpleNamespace(scope=scope), scope=scope
            ),
        )
        # a stand-in by design: the subject is show_settings' reading of the
        # capabilities; its widget tree is Kivy's, which the suite does not run.
        panel = SimpleNamespace(
            ids=defaultdict(MagicMock),
            _FPS_LIMITS=(),
            _show_camera_modes=lambda: None,
            _show_fps_limit=lambda _slider_id: None,
            load_scopes=lambda: None,
        )
        AdvancedSettings.show_settings(panel)
        shown[supported] = (panel.conversion_gain_supported, panel.line_noise_reduction_supported)
    assert shown == {True: (True, True), None: (True, True), False: (False, False)}
