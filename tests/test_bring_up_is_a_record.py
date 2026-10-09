# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Bring-up is a record the session holds, and what it reports comes from that record.

A client that subscribed at creation hears each bring-up outcome once,
typed; a client that connects later reads the same facts from
``ScopeSession.bring_up_record()``: which parts came up, the cause for each
that did not, what bring-up substituted for a saved setting the camera
could not take, and the settings file set aside. Nothing came up is one
notice. A camera's heading is its own, not doubled with its category.

Real-mode scopes, with the registries answering for the hardware this
machine does not have.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

import modules.lumascope_api._lumascope as lumascope_module
import modules.settings_init as settings_init
from drivers.null_ledboard import NullLEDBoard
from drivers.null_motorboard import NullMotionBoard
from drivers.registry import DriverFallback
from drivers.simulated_camera import SimulatedCamera
from modules.exceptions import (
    BinningSubstitutedNotice,
    CameraNotAvailableError,
    LedBoardUnavailableError,
    LedSafetyOffNotTakenError,
    NoHardwareDetectedNotice,
    PartialHardwareError,
)
from modules.lumascope_api.bring_up import CAMERA, LED, MOTOR, SettingsSetAside
from modules.notification_center import NotificationCenter, OutcomeKind, Severity
from modules.scope_session import ScopeSession
from tests.ast_seams import find_def
from tests.settings_fixtures import complete_settings


class _SafetyOffFailedBoard(NullLEDBoard):
    """A board that connected and reported its connect-time LEDS_OFF did not complete."""

    last_safety_off_error = 'TimeoutError: no reply'


@pytest.fixture
def heard(monkeypatch):
    """A fresh centre under the scope, no dedup, every post recorded."""
    centre = NotificationCenter(dedup_window_s=0)
    posts = []
    centre.add_listener(posts.append, min_severity=Severity.DEBUG)
    monkeypatch.setattr(lumascope_module, 'notifications', centre)
    return posts


def _bring_up(monkeypatch, tmp_path, *, motor=None, led=None, camera=None, **settings):
    """A real-mode session whose boards and camera are what the test says.

    ``motor`` / ``led``: a ``DriverFallback`` (the part is absent for that
    cause), a board instance (present), or None (present, a null stand-in
    that the registry reports as real). ``camera``: an exception to raise
    while connecting, or None for the simulated camera standing in for a
    real one.
    """

    def _board(part, null_cls):
        if isinstance(part, DriverFallback):
            return null_cls(), part
        return (null_cls() if part is None else part), None

    monkeypatch.setattr(
        lumascope_module.motor_registry,
        'create_with_fallback',
        lambda name='auto', **kw: _board(motor, NullMotionBoard),
    )
    monkeypatch.setattr(
        lumascope_module.led_registry,
        'create_with_fallback',
        lambda name='auto', **kw: _board(led, NullLEDBoard),
    )
    real_camera_create = lumascope_module.camera_registry.create

    def _camera(name='auto', **kw):
        if camera is not None:
            raise camera
        return real_camera_create('sim', **kw)

    monkeypatch.setattr(lumascope_module.camera_registry, 'create', _camera)
    return ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), **settings), warn_pre_release=False
    )


def _outcomes(posts, kind):
    return [n for n in posts if n.kind == kind]


class TestEachPartNamesItsCause:
    def test_a_motor_board_whose_port_is_held_is_on_the_record_and_in_the_one_report(
        self, monkeypatch, tmp_path, heard
    ):
        s = _bring_up(
            monkeypatch,
            tmp_path,
            motor=DriverFallback('port_in_use', ('MotorBoard',)),
            microscope='LS850T',
        )
        try:
            motor = s.bring_up_record().part(MOTOR)
            assert (motor.up, motor.expected, motor.cause) == (False, True, 'port_in_use')
            assert motor.missing
            faults = _outcomes(heard, OutcomeKind.FAULT)
            assert [n.reason for n in faults] == [PartialHardwareError('').reason]
            assert 'Motor Controller (port in use)' in faults[0].message
            assert faults[0].shown
        finally:
            s.shutdown()
            s.scope.disconnect()

    def test_a_manual_scope_is_not_missing_the_motor_board_it_never_had(
        self, monkeypatch, tmp_path, heard
    ):
        s = _bring_up(
            monkeypatch,
            tmp_path,
            motor=DriverFallback('not_detected', ('MotorBoard',)),
            microscope='LS620',
        )
        try:
            motor = s.bring_up_record().part(MOTOR)
            assert (motor.up, motor.expected, motor.cause) == (False, False, 'not_detected')
            assert not motor.missing
            assert s.bring_up_record().missing == ()
            assert _outcomes(heard, OutcomeKind.FAULT) == []
        finally:
            s.shutdown()
            s.scope.disconnect()

    def test_an_absent_led_board_is_said_once_with_the_advice_its_cause_needs(
        self, monkeypatch, tmp_path, heard
    ):
        s = _bring_up(
            monkeypatch,
            tmp_path,
            led=DriverFallback('not_responding', ('LEDBoard',)),
            microscope='LS850T',
        )
        try:
            led = s.bring_up_record().part(LED)
            assert (led.up, led.cause) == (False, 'not_responding')
            faults = _outcomes(heard, OutcomeKind.FAULT)
            by_reason = {n.reason: n for n in faults}
            assert set(by_reason) == {'not_responding', 'partial_hardware'}
            unavailable = by_reason['not_responding']
            assert unavailable.title == LedBoardUnavailableError.title
            assert 'Power-cycle' in unavailable.message
            assert 'LED Board (not responding)' in by_reason['partial_hardware'].message
        finally:
            s.shutdown()
            s.scope.disconnect()

    def test_a_camera_that_raised_keeps_its_own_heading_and_its_cause(
        self, monkeypatch, tmp_path, heard
    ):
        s = _bring_up(
            monkeypatch,
            tmp_path,
            camera=PermissionError('COM7 access denied'),
            microscope='LS850T',
        )
        try:
            camera = s.bring_up_record().part(CAMERA)
            assert (camera.up, camera.cause) == (False, 'camera_port_in_use')
            assert 'PermissionError' in camera.detail
            faults = _outcomes(heard, OutcomeKind.FAULT)
            by_reason = {n.reason: n for n in faults}
            assert by_reason['camera_port_in_use'].title == 'Camera port in use'
            assert by_reason['camera_port_in_use'].category == 'Camera'
            assert 'Camera (port in use)' in by_reason['partial_hardware'].message
        finally:
            s.shutdown()
            s.scope.disconnect()

    def test_the_camera_report_carries_the_backend_s_own_error(self, monkeypatch, tmp_path):
        reported = []
        monkeypatch.setattr(
            lumascope_module.notifications,
            'report_outcome',
            lambda exc, **kw: reported.append(exc),
        )
        backend_error = FileNotFoundError('/dev/video0 missing')
        s = _bring_up(monkeypatch, tmp_path, camera=backend_error, microscope='LS850T')
        try:
            failed = [e for e in reported if isinstance(e, CameraNotAvailableError)]
            assert len(failed) == 1
            assert failed[0].reason == 'camera_not_detected'
            assert failed[0].__cause__ is backend_error
        finally:
            s.shutdown()
            s.scope.disconnect()


class TestASimulatedScopeReportsNothing:
    def test_a_simulated_scope_s_parts_are_all_up_and_nothing_is_reported(self, tmp_path, heard):
        s = ScopeSession.create(
            complete_settings(live_folder=str(tmp_path)), simulate=True, warn_pre_release=False
        )
        try:
            record = s.bring_up_record()
            assert [p.up for p in record.parts] == [True, True, True]
            assert record.missing == ()
            assert heard == []
        finally:
            s.shutdown()


class TestASimulatedManualScope:
    def test_its_motor_board_is_neither_up_nor_missing(self, tmp_path, heard):
        s = ScopeSession.create(
            complete_settings(live_folder=str(tmp_path), microscope='LS620'),
            simulate=True,
            warn_pre_release=False,
        )
        try:
            motor = s.bring_up_record().part(MOTOR)
            assert (motor.up, motor.expected, motor.cause) == (False, False, None)
            assert s.bring_up_record().missing == ()
            assert heard == []
        finally:
            s.shutdown()


class TestNothingCameUp:
    def test_one_notice_and_no_fault(self, monkeypatch, tmp_path, heard):
        s = _bring_up(
            monkeypatch,
            tmp_path,
            motor=DriverFallback('not_detected', ('MotorBoard',)),
            led=DriverFallback('not_detected', ('LEDBoard',)),
            camera=FileNotFoundError('no camera'),
            microscope='LS850T',
        )
        try:
            assert s.scope.no_hardware
            record = s.bring_up_record()
            assert [p.up for p in record.parts] == [False, False, False]
            assert [n.reason for n in heard] == [NoHardwareDetectedNotice.reason]
            assert heard[0].kind == OutcomeKind.NOTICE
            assert heard[0].shown
        finally:
            s.shutdown()
            s.scope.disconnect()


class TestAProblemOnAPartThatCameUp:
    def test_a_refused_led_safety_off_is_recorded_and_reported_once(
        self, monkeypatch, tmp_path, heard
    ):
        s = _bring_up(monkeypatch, tmp_path, led=_SafetyOffFailedBoard(), microscope='LS850T')
        try:
            led = s.bring_up_record().part(LED)
            assert (led.up, led.cause) == (True, 'safety_off_failed')
            assert led.detail == 'TimeoutError: no reply'
            assert not led.missing
            safety = [n for n in heard if n.reason == 'safety_off_failed']
            assert len(safety) == 1
            assert safety[0].title == LedSafetyOffNotTakenError.title
            assert 'TimeoutError: no reply' in safety[0].message
        finally:
            s.shutdown()
            s.scope.disconnect()


class TestSubstitutions:
    def test_a_saved_binning_the_camera_lacks_is_recorded_and_reported_once(
        self, monkeypatch, tmp_path, heard
    ):
        s = _bring_up(monkeypatch, tmp_path, microscope='LS850T', binning={'size': '8x8'})
        try:
            sub = s.bring_up_record().substitution('binning')
            assert (sub.saved, sub.used) == (8, 1)
            notices = [n for n in heard if n.reason == BinningSubstitutedNotice.reason]
            assert len(notices) == 1
            assert notices[0].kind == OutcomeKind.NOTICE
            assert '8x8' in notices[0].message and '1x1' in notices[0].message
            # The notice tells the person what became of the saved value,
            # and what it says is what the store did.
            assert 'now the saved binning' in notices[0].message
            assert s.settings['binning']['size'] == '1x1', 'the delivered binning is stored'
        finally:
            s.shutdown()
            s.scope.disconnect()

    def test_a_saved_full_depth_mode_on_an_8_bit_camera_runs_as_saved(
        self, monkeypatch, tmp_path, heard
    ):
        # Every mode is a save policy every camera honours: an 8-bit camera
        # keeps the depth it delivers, so nothing is substituted or reported.
        monkeypatch.setattr(SimulatedCamera, 'get_supported_pixel_formats', lambda self: ('Mono8',))
        s = _bring_up(monkeypatch, tmp_path, microscope='LS850T', image_mode='12bit_scientific')
        try:
            assert s.bring_up_record().substitution('image_mode') is None
            assert _outcomes(heard, OutcomeKind.NOTICE) == []
            assert s.settings['image_mode'] == '12bit_scientific'
        finally:
            s.shutdown()
            s.scope.disconnect()

    def test_a_camera_that_takes_the_saved_values_substitutes_nothing(
        self, monkeypatch, tmp_path, heard
    ):
        # A frame the simulated 1920 x 1200 sensor delivers at 2x2.
        s = _bring_up(
            monkeypatch,
            tmp_path,
            microscope='LS850T',
            binning={'size': '2x2'},
            frame={'width': 900, 'height': 600},
        )
        try:
            assert s.bring_up_record().substitutions == ()
            assert _outcomes(heard, OutcomeKind.NOTICE) == []
        finally:
            s.shutdown()
            s.scope.disconnect()

    def test_a_saved_frame_larger_than_the_scope_delivers_is_refitted_and_reported_once(
        self, monkeypatch, tmp_path, heard
    ):
        # The LS560's lens images 1700 of the sensor (data/scopes.json
        # MaxFrame); the simulated sensor is 3840 x 2160.
        s = _bring_up(
            monkeypatch,
            tmp_path,
            microscope='LS560',
            binning={'size': '1x1'},
            frame={'width': 1900, 'height': 1100},
        )
        try:
            sub = s.bring_up_record().substitution('frame')
            assert (sub.saved, sub.used) == ((1900, 1100), (1700, 1100))
            assert s.scope.imaging.frame_size_cached == {'width': 1700, 'height': 1100}
            assert s.settings['frame']['width'] == 1700, 'the frame that ran is stored'
            notices = [n for n in heard if n.reason == 'frame_refitted']
            assert len(notices) == 1
            assert notices[0].kind == OutcomeKind.NOTICE
            assert notices[0].title == 'Saved frame too large'
            assert '1900x1100' in notices[0].message and '1700x1100' in notices[0].message
            # What the notice says became of the saved frame is what the
            # store did.
            assert 'now the saved frame' in notices[0].message
        finally:
            s.shutdown()
            s.scope.disconnect()

    def test_the_refitted_frame_is_not_reported_again_at_the_next_bring_up(
        self, monkeypatch, tmp_path, heard
    ):
        first = _bring_up(
            monkeypatch,
            tmp_path,
            microscope='LS560',
            binning={'size': '1x1'},
            frame={'width': 1900, 'height': 1100},
        )
        try:
            stored = {k: dict(first.settings[k]) for k in ('binning', 'frame')}
        finally:
            first.shutdown()
            first.scope.disconnect()
        heard.clear()
        again = _bring_up(monkeypatch, tmp_path, microscope='LS560', **stored)
        try:
            assert again.bring_up_record().substitutions == ()
            assert [n for n in heard if n.reason == 'frame_refitted'] == []
        finally:
            again.shutdown()
            again.scope.disconnect()


class TestTheSettingsFileSetAside:
    def test_the_record_carries_the_rejected_file_and_its_reason(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings_init, 'rejected_current_json', None)
        session = ScopeSession.create(
            complete_settings(live_folder=str(tmp_path)), simulate=True, warn_pre_release=False
        )
        try:
            assert session.bring_up_record().settings_set_aside is None

            monkeypatch.setattr(
                settings_init, 'rejected_current_json', ('/x/data/current.json', 'not valid JSON')
            )
            assert session.bring_up_record().settings_set_aside == SettingsSetAside(
                pathlib.Path('/x/data/current.json'), 'not valid JSON'
            )
        finally:
            session.shutdown()

    def test_the_host_s_question_reads_the_reason_from_the_record(self):
        fn = find_def('lumaviewpro.py', '_ask_about_rejected_settings', class_name='LumaViewProApp')
        source = ast.unparse(fn)
        assert 'bring_up_record().settings_set_aside.reason' in source
        assert 'rejected_current_json' not in source


class TestAFailedStartIsRecordedOnce:
    def test_the_host_does_not_log_a_composition_failure_the_crash_hook_records(self):
        fn = find_def('lumaviewpro.py', 'build', class_name='LumaViewProApp')
        handlers = [node for node in ast.walk(fn) if isinstance(node, ast.ExceptHandler)]
        logging_handlers = [
            h
            for h in handlers
            if any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr in ('exception', 'error', 'critical')
                for n in ast.walk(h)
            )
        ]
        assert logging_handlers == [], (
            'a handler in build() that logs and re-raises writes a second record of a '
            'failure the crash hook already records; the late settings rejection is '
            'logged once, by the fallback, with its cause'
        )

    def test_no_hardware_is_the_api_s_notice_not_a_popup_the_host_opens(self):
        fn = find_def('lumaviewpro.py', 'on_start', class_name='LumaViewProApp')
        source = ast.unparse(fn)
        assert 'no_hardware' not in source
        assert 'No hardware detected' not in source
