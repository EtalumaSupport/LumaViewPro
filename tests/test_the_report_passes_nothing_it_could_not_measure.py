# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The support report passes nothing it could not measure.

The report's LED leakage check parsed a raw LEDREADS dump and marked a
channel PASS when its line could not be parsed; its serial-latency step
timed the channel's own "Board not connected" as a round trip; its LED
selftest, I2C scan and readings asked every LED board for text commands an
FX2 scope's LED peripheral cannot carry; and a command-line report whose
scope could not be built came out with every hardware file blaming a cable.

Now a check that was not possible says so, a channel that was not read is
NOT MEASURED, only a board's reply is timed, the LED board's command set
(none, legacy INFO, v2) decides what is asked, and a scope that could not
be built is named, with its cause, in every hardware file.
"""

import pytest

from drivers import fx2driver
from drivers.simulated_fx2 import SimulatedFX2
from modules.lumascope_api import Lumascope
from modules.lumascope_api.diagnostics import (
    ERROR_PREFIX,
    LED_COMMANDS_LEGACY,
    LED_COMMANDS_V2,
    NO_REPLY,
    NO_RESPONSE,
    NOT_CONNECTED,
    is_board_reply,
)
from modules.scope_session import ScopeSession
from modules.tech_support_report import FirmwareDiagnostics, TechSupportReport
from tests.scope_fakes import spec_scope
from tests.settings_fixtures import complete_settings

NOT_ON_THIS_BOARD = 'Not supported on this LED board'


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path / 'live'), microscope='LS850'), simulate=True
    )
    yield s
    s.shutdown()


@pytest.fixture
def fx2_led():
    """An FX2 LED controller on a simulated FX2."""
    return fx2driver.FX2LEDController(connection=SimulatedFX2().connection)


def test_the_channels_stand_ins_are_not_replies():
    for stand_in in (NOT_CONNECTED, NO_REPLY, NO_RESPONSE, f'{ERROR_PREFIX}timeout', '', None):
        assert not is_board_reply(stand_in), stand_in
    assert is_board_reply('Etaluma LED Controller v2.0.1')
    assert is_board_reply(['line one', 'line two'])


def test_a_v2_board_carries_every_led_command(session):
    assert session.scope.diagnostics.get_led_info()['command_set'] == LED_COMMANDS_V2


def test_a_pre_v2_board_keeps_its_info_and_says_what_it_cannot_do(session, monkeypatch):
    monkeypatch.setattr(session.scope._led_driver, 'firmware_version', '1.4')
    assert session.scope.diagnostics.get_led_info()['command_set'] == LED_COMMANDS_LEGACY

    diag = FirmwareDiagnostics(scope=session.scope)
    info = diag.get_led_info()
    assert is_board_reply(info) and 'Not supported' not in str(info), info
    for answer in (diag.run_led_selftest(), diag.get_i2c_scan(), diag.get_led_readings()):
        assert 'needs v2' in answer, answer
    leakage = diag.check_led_leakage()
    assert leakage['passed'] is False and 'needs v2' in leakage['not_applicable']


def test_an_fx2_led_board_is_not_asked_for_text_commands(session, monkeypatch, fx2_led, tmp_path):
    monkeypatch.setattr(session.scope, '_led_driver', fx2_led)
    try:
        assert session.scope.diagnostics.get_led_info()['command_set'] is None
        diag = FirmwareDiagnostics(scope=session.scope)
        for answer in (
            diag.get_led_info(),
            diag.run_led_selftest(),
            diag.get_i2c_scan(),
            diag.get_led_readings(),
        ):
            assert answer.startswith(NOT_ON_THIS_BOARD), answer
        leakage = diag.check_led_leakage()
        assert leakage['passed'] is False
        assert leakage['not_applicable'].startswith(NOT_ON_THIS_BOARD)

        report = TechSupportReport(scope=session.scope)
        report._step_serial_latency(tmp_path)
        latency = (tmp_path / 'hardware_checks' / 'serial_latency.txt').read_text()
        led_block = latency.split('--- LED Board')[1].split('--- Motor Board')[0]
        assert NOT_ON_THIS_BOARD in led_block and 'Mean' not in led_block, latency
    finally:
        monkeypatch.undo()


def test_every_channel_read_and_dark_is_a_pass(session, tmp_path):
    TechSupportReport(scope=session.scope)._step_led_checks(tmp_path)
    text = (tmp_path / 'hardware_checks' / 'led_leakage.txt').read_text()
    assert 'Overall: PASS' in text, text


def test_a_channel_that_was_not_read_is_not_a_pass(session, monkeypatch, tmp_path):
    driver = session.scope._led_driver
    real = driver.read_led_current
    monkeypatch.setattr(driver, 'read_led_current', lambda ch: None if ch == 2 else real(ch))

    leakage = FirmwareDiagnostics(scope=session.scope).check_led_leakage()

    assert leakage['passed'] is False
    assert leakage['channels']['CH2'] == {'i_sens_mA': None, 'status': 'NOT MEASURED'}
    TechSupportReport(scope=session.scope)._step_led_checks(tmp_path)
    text = (tmp_path / 'hardware_checks' / 'led_leakage.txt').read_text()
    assert 'CH2: not measured' in text and 'Overall: INCOMPLETE' in text, text


def test_a_leaking_channel_warns(session, monkeypatch):
    monkeypatch.setattr(
        session.scope._led_driver, 'read_led_current', lambda ch: 5.0 if ch == 1 else 0.0
    )
    leakage = FirmwareDiagnostics(scope=session.scope).check_led_leakage()
    assert leakage['passed'] is False
    assert leakage['channels']['CH1']['status'] == 'WARN'


def test_a_board_that_does_not_answer_is_not_timed(session, monkeypatch):
    # The API answers at once for a board it cannot reach; that is no round trip.
    monkeypatch.setattr(session.scope._led_driver, 'found', False)
    latency = FirmwareDiagnostics(scope=session.scope).measure_serial_latency('led', 'INFO')
    assert 'min_ms' not in latency and 'All' in latency['error'], latency


def test_a_fullinfo_that_timed_out_is_not_a_serial_number():
    scope = spec_scope()
    scope.diagnostics.send_diagnostic_command.return_value = NO_REPLY
    assert FirmwareDiagnostics(scope=scope).get_serial_number() == 'UNKNOWN'


CAUSE = 'data/motorconfig_defaults.json is missing'


def _cannot_build(cls):
    raise RuntimeError(CAUSE)


def test_a_scope_that_cannot_be_built_is_named_in_every_hardware_file(monkeypatch, tmp_path):
    monkeypatch.setattr(Lumascope, 'create_diagnostic', classmethod(_cannot_build))
    report = TechSupportReport()
    report.diag.connect_standalone()

    sn = report._run_scope_steps(tmp_path, lambda pct, msg: None)

    assert sn == 'UNKNOWN'
    for rel in (
        'firmware_info/firmware_info.txt',
        'firmware_configs/config_backup.txt',
        'firmware_tests/led_selftest.txt',
        'hardware_checks/led_leakage.txt',
        'hardware_checks/serial_latency.txt',
        'motion_tests/homing_test.txt',
        'camera_info/camera_info.txt',
    ):
        text = (tmp_path / rel).read_text()
        assert 'could not be built' in text and CAUSE in text, (rel, text)


def test_the_command_line_says_why_and_does_not_send_the_user_to_the_cable(monkeypatch, caplog):
    from modules import tech_support_report

    monkeypatch.setattr(Lumascope, 'create_diagnostic', classmethod(_cannot_build))
    prompts = []
    monkeypatch.setattr('builtins.input', prompts.append)
    monkeypatch.setattr(tech_support_report.TechSupportReport, 'generate', lambda *a, **k: None)
    monkeypatch.setattr('sys.argv', ['tech_support_report'])

    with caplog.at_level('INFO', logger=tech_support_report.logger.name):
        tech_support_report.main()

    assert CAUSE in caplog.text
    assert prompts == [], 'the power-cycle prompt was shown for a scope that could not be built'
    assert 'USB cable' not in caplog.text


def test_a_board_that_does_not_enter_engineering_mode_is_not_measured(session, monkeypatch):
    monkeypatch.setattr(
        session.scope.diagnostics, 'enter_led_engineering_mode', lambda timeout_s=5: False
    )
    diag = FirmwareDiagnostics(scope=session.scope)
    leakage = diag.check_led_leakage()
    assert leakage['passed'] is False and 'engineering mode' in leakage['error']
    assert 'SELFTEST was not run' in diag.run_led_selftest()


def test_with_no_led_board_the_api_reads_nothing_and_names_no_command_set(session, monkeypatch):
    monkeypatch.setattr(type(session.scope), 'led_connected', property(lambda self: False))
    monkeypatch.setattr(session.scope._led_driver, 'is_connected', lambda: False)
    assert session.scope.diagnostics.read_led_currents_ma() == {}
    info = session.scope.diagnostics.get_led_info()
    assert info['connected'] is False and info['command_set'] is None


def test_a_retry_that_builds_the_scope_clears_the_failure(monkeypatch):
    built = object()
    answers = iter([RuntimeError(CAUSE), built])

    def create(cls):
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(Lumascope, 'create_diagnostic', classmethod(create))
    diag = FirmwareDiagnostics()
    diag.connect_standalone()
    assert diag.build_failure is not None and diag.scope is None
    diag.connect_standalone()
    assert diag.build_failure is None and diag.scope is built
