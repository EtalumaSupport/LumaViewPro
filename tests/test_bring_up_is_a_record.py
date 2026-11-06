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

import _lumascope.modules.lumascope_api._lumascope as lumascope_module
import _lumascope.modules.settings_init as settings_init
from _lumascope.drivers.exceptions import BoardNoReplyError
from _lumascope.drivers.null_ledboard import NullLEDBoard
from _lumascope.drivers.null_motorboard import NullMotionBoard
from _lumascope.drivers.registry import DeclinedDriver, DriverFallback, NoLiveDriverError
from _lumascope.drivers.simulated_camera import SimulatedCamera
from _lumascope.drivers.simulated_motorboard import SimulatedMotorBoard
from _lumascope.modules.exceptions import (
    BinningSubstitutedNotice,
    CameraNotAvailableError,
    LedBoardUnavailableError,
    LedSafetyOffNotTakenError,
    MotorBoardUnavailableError,
)
from _lumascope.modules.lumascope_api.bring_up import (
    CAUSE_PHRASES,
    Part,
    PartStatus,
    SettingsSetAside,
)
from _lumascope.modules.notification_center import OutcomeKind
from _lumascope.modules.scope_session import ScopeSession
from tests.ast_seams import find_def
from tests.motorconfig_fixtures import SHIPPED_MOTOR_DEFAULTS
from tests.settings_fixtures import complete_settings, installation_of_its_own


class _SafetyOffFailedBoard(NullLEDBoard):
    """A board that connected and reported its connect-time LEDS_OFF did not complete."""

    last_safety_off_error = 'TimeoutError: no reply'


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
    # On an installation of its own: a real scope takes its lock.
    return ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), **settings),
        source_path=str(installation_of_its_own()),
        warn_pre_release=False,
    )


def _status(record, part):
    """``part``'s status, read from the record's ``parts``."""
    (status,) = [status for status in record.parts if status.part == part]
    return status


def _substitution(record, setting):
    """What bring-up used for ``setting`` instead of the saved value, or None."""
    return next((sub for sub in record.substitutions if sub.setting == setting), None)


def _outcomes(posts, kind):
    return [n for n in posts if n.kind == kind]


class TestEachPartNamesItsCause:
    def test_every_camera_cause_has_its_words_in_a_list_of_parts(self):
        # A part's label reads its cause's phrase; a camera reason without
        # one would raise where bring-up reports the camera that failed.
        missing = sorted(set(CameraNotAvailableError._WORDS) - set(CAUSE_PHRASES))
        assert missing == []
        assert PartStatus(Part.CAMERA, up=False, cause='camera_firmware_missing').describe() == (
            'Camera (firmware file missing)'
        )

    def test_a_motor_board_whose_port_is_held_is_on_the_record_and_told_once_with_its_remedy(
        self, monkeypatch, tmp_path, centre_posts
    ):
        s = _bring_up(
            monkeypatch,
            tmp_path,
            motor=DriverFallback('port_in_use', ('MotorBoard',)),
            microscope='LS850T',
        )
        try:
            motor = _status(s.bring_up_record(), Part.MOTOR)
            assert (motor.up, motor.expected, motor.cause) == (False, True, 'port_in_use')
            assert motor.missing
            faults = _outcomes(centre_posts, OutcomeKind.FAULT)
            assert [(n.title, n.reason, n.category) for n in faults] == [
                (MotorBoardUnavailableError.title, 'port_in_use', 'Motion')
            ]
            assert 'Close other programs holding the port' in faults[0].message
            assert faults[0].shown
        finally:
            s.shutdown()
            s.scope.disconnect()

    def test_each_board_is_told_once_for_every_cause_with_that_cause_s_advice(
        self, monkeypatch, tmp_path, centre_posts
    ):
        # One walk over the registry's five causes, the motor's table naming
        # exactly those, for both boards: an offender is a board and cause
        # told other than once, by its own type, in its own words.
        boards = (
            ('led', LedBoardUnavailableError, 'LEDBoard'),
            ('motor', MotorBoardUnavailableError, 'MotorBoard'),
        )
        offenders = []
        for part, notice, driver in boards:
            for cause in sorted(MotorBoardUnavailableError.causes):
                centre_posts.clear()
                s = _bring_up(
                    monkeypatch,
                    tmp_path,
                    microscope='LS850T',
                    **{part: DriverFallback(cause, (driver,))},
                )
                try:
                    told = [
                        (n.title, n.reason, n.message)
                        for n in _outcomes(centre_posts, OutcomeKind.FAULT)
                    ]
                finally:
                    s.shutdown()
                    s.scope.disconnect()
                if told != [(notice.title, cause, str(notice(cause)))]:
                    offenders.append((part, cause, told))
        assert offenders == []

    def test_two_boards_missing_are_two_notices_one_each(self, monkeypatch, tmp_path, centre_posts):
        s = _bring_up(
            monkeypatch,
            tmp_path,
            led=DriverFallback('port_in_use', ('LEDBoard',)),
            motor=DriverFallback('port_in_use', ('MotorBoard',)),
            microscope='LS850T',
        )
        try:
            faults = _outcomes(centre_posts, OutcomeKind.FAULT)
            assert sorted((n.title, n.category) for n in faults) == sorted(
                [
                    (LedBoardUnavailableError.title, 'Illumination'),
                    (MotorBoardUnavailableError.title, 'Motion'),
                ]
            )
        finally:
            s.shutdown()
            s.scope.disconnect()

    def test_a_manual_scope_is_not_missing_the_motor_board_it_never_had(
        self, monkeypatch, tmp_path, centre_posts
    ):
        s = _bring_up(
            monkeypatch,
            tmp_path,
            motor=DriverFallback('not_detected', ('MotorBoard',)),
            microscope='LS620',
        )
        try:
            motor = _status(s.bring_up_record(), Part.MOTOR)
            assert (motor.up, motor.expected, motor.cause) == (False, False, 'not_detected')
            assert not motor.missing
            assert s.bring_up_record().missing == ()
            assert _outcomes(centre_posts, OutcomeKind.FAULT) == []
        finally:
            s.shutdown()
            s.scope.disconnect()

    def test_a_scope_whose_model_the_catalogue_lacks_is_missing_its_led_board(
        self, monkeypatch, tmp_path, centre_posts
    ):
        # Only a resolved identity can say a scope has no LED board: one
        # whose board reports a model this release does not list has no
        # layers to read, and its held LED board is missing, not unexpected.
        # A stand-in by design: no simulated board reports an unlisted model.
        monkeypatch.setattr(NullMotionBoard, 'get_microscope_model', lambda self: 'LS999X')
        s = _bring_up(
            monkeypatch,
            tmp_path,
            led=DriverFallback('port_in_use', ('LEDBoard',)),
            microscope='LS850T',
        )
        try:
            assert s.scope.channel_identity.channels == ()
            led = _status(s.bring_up_record(), Part.LED)
            assert (led.up, led.expected, led.cause) == (False, True, 'port_in_use')
            assert led in s.bring_up_record().missing
            faults = _outcomes(centre_posts, OutcomeKind.FAULT)
            assert [(n.title, n.reason) for n in faults] == [
                (LedBoardUnavailableError.title, 'port_in_use')
            ]
        finally:
            s.shutdown()
            s.scope.disconnect()

    def test_an_absent_led_board_is_said_once_with_the_advice_its_cause_needs(
        self, monkeypatch, tmp_path, centre_posts
    ):
        s = _bring_up(
            monkeypatch,
            tmp_path,
            led=DriverFallback('not_responding', ('LEDBoard',)),
            microscope='LS850T',
        )
        try:
            led = _status(s.bring_up_record(), Part.LED)
            assert (led.up, led.cause) == (False, 'not_responding')
            faults = _outcomes(centre_posts, OutcomeKind.FAULT)
            assert [(n.title, n.reason) for n in faults] == [
                (LedBoardUnavailableError.title, 'not_responding')
            ]
            assert 'Power-cycle' in faults[0].message
        finally:
            s.shutdown()
            s.scope.disconnect()

    def test_a_camera_that_raised_is_told_once_under_its_own_heading_and_its_cause(
        self, monkeypatch, tmp_path, centre_posts
    ):
        s = _bring_up(
            monkeypatch,
            tmp_path,
            camera=PermissionError('COM7 access denied'),
            microscope='LS850T',
        )
        try:
            camera = _status(s.bring_up_record(), Part.CAMERA)
            assert (camera.up, camera.cause) == (False, 'port_in_use')
            assert 'PermissionError' in camera.detail
            faults = _outcomes(centre_posts, OutcomeKind.FAULT)
            assert [(n.title, n.reason, n.category) for n in faults] == [
                ('Camera port in use', 'port_in_use', 'Camera')
            ]
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
        backend_error = NoLiveDriverError(
            'camera', DriverFallback('not_detected', ('PylonCamera', 'IDSCamera'))
        )
        s = _bring_up(monkeypatch, tmp_path, camera=backend_error, microscope='LS850T')
        try:
            failed = [e for e in reported if isinstance(e, CameraNotAvailableError)]
            assert len(failed) == 1
            assert failed[0].reason == 'not_detected'
            assert failed[0].__cause__ is backend_error
        finally:
            s.shutdown()
            s.scope.disconnect()


class TestASimulatedScopeReportsNothing:
    def test_a_simulated_scope_s_parts_are_all_up_and_nothing_is_reported(
        self, tmp_path, centre_posts
    ):
        s = ScopeSession.create(
            complete_settings(live_folder=str(tmp_path)), simulate=True, warn_pre_release=False
        )
        try:
            record = s.bring_up_record()
            assert [p.up for p in record.parts] == [True, True, True]
            assert record.missing == ()
            assert centre_posts == []
        finally:
            s.shutdown()


class TestASimulatedManualScope:
    def test_its_motor_board_is_neither_up_nor_missing(self, tmp_path, centre_posts):
        s = ScopeSession.create(
            complete_settings(live_folder=str(tmp_path), microscope='LS620'),
            simulate=True,
            warn_pre_release=False,
        )
        try:
            motor = _status(s.bring_up_record(), Part.MOTOR)
            assert (motor.up, motor.expected, motor.cause) == (False, False, None)
            assert s.bring_up_record().missing == ()
            assert centre_posts == []
        finally:
            s.shutdown()


class TestNothingCameUp:
    def test_one_notice_and_no_fault(self, monkeypatch, tmp_path, centre_posts):
        s = _bring_up(
            monkeypatch,
            tmp_path,
            motor=DriverFallback('not_detected', ('MotorBoard',)),
            led=DriverFallback('not_detected', ('LEDBoard',)),
            camera=NoLiveDriverError('camera', DriverFallback('not_detected', ('PylonCamera',))),
            microscope='LS850T',
        )
        try:
            assert s.scope.no_hardware
            record = s.bring_up_record()
            assert [p.up for p in record.parts] == [False, False, False]
            assert [n.reason for n in centre_posts] == ['no_hardware']
            assert centre_posts[0].kind == OutcomeKind.NOTICE
            assert centre_posts[0].shown
        finally:
            s.shutdown()
            s.scope.disconnect()


# The FX2 driver this installation did not load, as both registries carry it.
_FX2_DECLINED = (DeclinedDriver('fx2', 'pyusb is not installed'),)


def _no_camera_with_the_fx2_declined():
    return NoLiveDriverError(
        'camera', DriverFallback('not_detected', ('PylonCamera',), declined=_FX2_DECLINED)
    )


def _told(posts):
    """Each fault told, as (category, reason)."""
    return sorted((n.category, n.reason) for n in _outcomes(posts, OutcomeKind.FAULT))


class TestADeclinedFx2IsToldOnlyWhereItCouldBeTheBoard:
    """A declined FX2 driver is told where the scope may be an FX2 scope (LS560,
    LS620, LS720), and never where the hardware has shown it is not: an
    EL-0940 LED board found on USB, or a camera that came up."""

    def test_an_el0940_with_its_led_board_up_is_told_its_camera_is_not_found(
        self, monkeypatch, tmp_path, centre_posts
    ):
        s = _bring_up(
            monkeypatch, tmp_path, camera=_no_camera_with_the_fx2_declined(), microscope='LS850T'
        )
        try:
            camera = _status(s.bring_up_record(), Part.CAMERA)
            assert camera.cause == 'not_detected'
            # The record still names the driver that did not load.
            assert 'fx2: pyusb is not installed' in camera.detail
            assert _told(centre_posts) == [('Camera', 'not_detected')]
        finally:
            s.shutdown()
            s.scope.disconnect()

    def test_a_scope_where_nothing_came_up_is_told_once_that_a_driver_did_not_load(
        self, monkeypatch, tmp_path, centre_posts
    ):
        # An LS620 or LS560: every part is on the FX2.
        s = _bring_up(
            monkeypatch,
            tmp_path,
            motor=DriverFallback('not_detected', ('MotorBoard',)),
            led=DriverFallback('not_detected', ('LEDBoard',), declined=_FX2_DECLINED),
            camera=_no_camera_with_the_fx2_declined(),
            microscope='LS620',
        )
        try:
            assert [n.reason for n in centre_posts] == ['no_hardware_driver_unavailable']
            assert 'A driver did not load on this installation' in centre_posts[0].message
        finally:
            s.shutdown()
            s.scope.disconnect()

    def test_a_scope_whose_motor_came_up_is_told_a_driver_did_not_load_for_camera_and_leds(
        self, monkeypatch, tmp_path, centre_posts
    ):
        # An LS720: its motor controller is its own board, its LEDs and camera the FX2.
        s = _bring_up(
            monkeypatch,
            tmp_path,
            led=DriverFallback('not_detected', ('LEDBoard',), declined=_FX2_DECLINED),
            camera=_no_camera_with_the_fx2_declined(),
            microscope='LS720',
        )
        try:
            record = s.bring_up_record()
            assert _status(record, Part.CAMERA).cause == 'camera_driver_unavailable'
            assert _status(record, Part.LED).cause == 'led_driver_unavailable'
            assert _told(centre_posts) == [
                ('Camera', 'camera_driver_unavailable'),
                ('Illumination', 'led_driver_unavailable'),
            ]
        finally:
            s.shutdown()
            s.scope.disconnect()

    def test_a_camera_that_came_up_rules_the_fx2_out_for_a_missing_led_board(
        self, monkeypatch, tmp_path, centre_posts
    ):
        # No FX2 scope has a camera that can come up with the FX2 declined.
        s = _bring_up(
            monkeypatch,
            tmp_path,
            led=DriverFallback('not_detected', ('LEDBoard',), declined=_FX2_DECLINED),
            microscope='LS850T',
        )
        try:
            assert _status(s.bring_up_record(), Part.LED).cause == 'not_detected'
            assert _told(centre_posts) == [('Illumination', 'not_detected')]
        finally:
            s.shutdown()
            s.scope.disconnect()

    def test_an_led_board_found_on_usb_rules_the_fx2_out_for_a_missing_camera(
        self, monkeypatch, tmp_path, centre_posts
    ):
        # A board the port search did not find is sent nothing, so one whose
        # port is held was found: the EL-0940's.
        s = _bring_up(
            monkeypatch,
            tmp_path,
            led=DriverFallback('port_in_use', ('LEDBoard',), declined=_FX2_DECLINED),
            camera=_no_camera_with_the_fx2_declined(),
            microscope='LS850T',
        )
        try:
            record = s.bring_up_record()
            assert _status(record, Part.LED).cause == 'port_in_use'
            assert _status(record, Part.CAMERA).cause == 'not_detected'
            assert _told(centre_posts) == [
                ('Camera', 'not_detected'),
                ('Illumination', 'port_in_use'),
            ]
        finally:
            s.shutdown()
            s.scope.disconnect()


class TestAProblemOnAPartThatCameUp:
    def test_a_refused_led_safety_off_is_recorded_and_reported_once(
        self, monkeypatch, tmp_path, centre_posts
    ):
        s = _bring_up(monkeypatch, tmp_path, led=_SafetyOffFailedBoard(), microscope='LS850T')
        try:
            led = _status(s.bring_up_record(), Part.LED)
            assert (led.up, led.cause) == (True, 'safety_off_failed')
            assert led.detail == 'TimeoutError: no reply'
            assert not led.missing
            safety = [n for n in centre_posts if n.reason == 'safety_off_failed']
            assert len(safety) == 1
            assert safety[0].title == LedSafetyOffNotTakenError.title
            assert 'TimeoutError: no reply' in safety[0].message
        finally:
            s.shutdown()
            s.scope.disconnect()

    def test_a_motor_board_that_does_not_take_the_acceleration_limit_is_up_and_says_so(
        self, monkeypatch, tmp_path, centre_posts
    ):
        board = SimulatedMotorBoard(
            motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, timing='instant', fail_on={'AMAXX'}
        )
        s = _bring_up(monkeypatch, tmp_path, motor=board, microscope='LS850T')
        try:
            motor = _status(s.bring_up_record(), Part.MOTOR)
            assert (motor.up, motor.cause) == (True, 'acceleration_not_set')
            assert motor.detail == 'BoardNoReplyError: AMAXX: no reply from the board'
            assert not motor.missing
            told = [n for n in centre_posts if n.reason == BoardNoReplyError.reason]
            assert len(told) == 1, 'told once'
            assert (told[0].title, told[0].category) == (BoardNoReplyError.title, 'Motion')
            assert s.scope.motor_connected, 'the scope came up with its motor board'
        finally:
            s.shutdown()
            s.scope.disconnect()


class TestSubstitutions:
    def test_a_saved_binning_the_camera_lacks_is_recorded_and_reported_once(
        self, monkeypatch, tmp_path, centre_posts
    ):
        s = _bring_up(monkeypatch, tmp_path, microscope='LS850T', binning={'size': '8x8'})
        try:
            sub = _substitution(s.bring_up_record(), 'binning')
            assert (sub.saved, sub.used) == (8, 1)
            notices = [n for n in centre_posts if n.reason == BinningSubstitutedNotice.reason]
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
        self, monkeypatch, tmp_path, centre_posts
    ):
        # Every mode is a save policy every camera honours: an 8-bit camera
        # keeps the depth it delivers, so nothing is substituted or reported.
        monkeypatch.setattr(SimulatedCamera, 'get_supported_pixel_formats', lambda self: ('Mono8',))
        s = _bring_up(monkeypatch, tmp_path, microscope='LS850T', image_mode='12bit_scientific')
        try:
            assert _substitution(s.bring_up_record(), 'image_mode') is None
            assert _outcomes(centre_posts, OutcomeKind.NOTICE) == []
            assert s.settings['image_mode'] == '12bit_scientific'
        finally:
            s.shutdown()
            s.scope.disconnect()

    def test_a_camera_that_takes_the_saved_values_substitutes_nothing(
        self, monkeypatch, tmp_path, centre_posts
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
            assert _outcomes(centre_posts, OutcomeKind.NOTICE) == []
        finally:
            s.shutdown()
            s.scope.disconnect()

    def test_a_saved_frame_larger_than_the_scope_delivers_is_refitted_and_reported_once(
        self, monkeypatch, tmp_path, centre_posts
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
            sub = _substitution(s.bring_up_record(), 'frame')
            assert (sub.saved, sub.used) == ((1900, 1100), (1700, 1100))
            assert s.scope.camera.frame_size.to_dict() == {'width': 1700, 'height': 1100}
            assert s.settings['frame']['width'] == 1700, 'the frame that ran is stored'
            notices = [n for n in centre_posts if n.reason == 'frame_refitted']
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
        self, monkeypatch, tmp_path, centre_posts
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
        centre_posts.clear()
        again = _bring_up(monkeypatch, tmp_path, microscope='LS560', **stored)
        try:
            assert again.bring_up_record().substitutions == ()
            assert [n for n in centre_posts if n.reason == 'frame_refitted'] == []
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
        # The bring-up thread and the compose it runs: what either raises is
        # raised again by the hand-over on the Kivy thread.
        handlers = [
            node
            for name in ('_bring_up_on_its_thread', '_bring_up_the_session')
            for node in ast.walk(find_def('lumaviewpro.py', name, class_name='LumaViewProApp'))
            if isinstance(node, ast.ExceptHandler)
        ]
        assert len(handlers) == 2, 'the bring-up thread and the template fallback each have one'
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
            'a handler in the bring-up that logs and re-raises writes a second record of a '
            'failure the crash hook already records; the late settings rejection is '
            'logged once, by the fallback, with its cause'
        )

    def test_no_hardware_is_the_api_s_notice_not_a_popup_the_host_opens(self):
        fn = find_def('lumaviewpro.py', '_fit', class_name='LumaViewProApp')
        source = ast.unparse(fn)
        assert 'no_hardware' not in source
        assert 'No hardware detected' not in source
