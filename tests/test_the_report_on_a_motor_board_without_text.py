# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The support report on a motor board with no text command channel says so, and sends it none.

Every motor section of the report sent the board the EL-0940 firmware's
text commands -- INFO, FULLINFO, the position reads, the SPI register dump,
the latency loop, the homing test's register cross-check -- on any connected
board. The LS720's TMCM-6110 speaks binary datagrams, so on it every section
would have been a page of non-answers. Now the motor info names the board's
command set, and each section that sends text answers ``not applicable`` on
a board without one; the homing and fan tests still run, through the API.
"""

import json

import pytest

from drivers.simulated_motorboard import SimulatedMotorBoard
from modules.lumascope_api.diagnostics import MOTOR_COMMANDS_TEXT
from modules.scope_session import ScopeSession
from modules.tech_support_report import NO_MOTOR_TEXT, TechSupportReport
from tests.settings_fixtures import complete_settings

TEXT_SECTIONS = (
    'firmware_info/motor_info.txt',
    'firmware_info/motor_status.txt',
    'hardware_checks/tmc5072_registers.txt',
)


@pytest.fixture
def session(tmp_path):
    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path / 'live'), microscope='LS850'), simulate=True
    )
    yield session
    session.shutdown()
    session.scope.disconnect()


def _without_text(monkeypatch, session):
    """The connected simulated board, as a board with no text channel; its text sends recorded."""
    monkeypatch.delattr(SimulatedMotorBoard, 'exchange_multiline')
    diagnostics = session.scope.diagnostics
    sent = []
    single = diagnostics.send_diagnostic_command
    multi = diagnostics.send_diagnostic_command_multiline

    def record_single(target, command, **kwargs):
        if target == 'motor':
            sent.append(command)
        return single(target, command, **kwargs)

    def record_multi(target, command, **kwargs):
        if target == 'motor':
            sent.append(command)
        return multi(target, command, **kwargs)

    monkeypatch.setattr(diagnostics, 'send_diagnostic_command', record_single)
    monkeypatch.setattr(diagnostics, 'send_diagnostic_command_multiline', record_multi)
    return sent


def _report(session, tmp_path):
    out = tmp_path / 'report'
    out.mkdir()
    TechSupportReport(session=session)._run_scope_steps(out, lambda pct, msg: None)
    return out


class TestTheMotorInfoNamesTheCommandSet:
    def test_a_text_board_names_its_set(self, session):
        assert session.scope.diagnostics.get_motor_info()['command_set'] == MOTOR_COMMANDS_TEXT

    def test_a_board_without_text_names_none(self, session, monkeypatch):
        monkeypatch.delattr(SimulatedMotorBoard, 'exchange_multiline')

        assert session.scope.diagnostics.get_motor_info()['command_set'] is None

    def test_no_board_names_none(self, session, monkeypatch):
        monkeypatch.setattr(type(session.scope), 'motor_connected', property(lambda self: False))

        assert session.scope.diagnostics.get_motor_info()['command_set'] is None


def test_every_text_section_says_not_applicable_and_the_board_is_sent_no_text(
    session, monkeypatch, tmp_path
):
    sent = _without_text(monkeypatch, session)

    out = _report(session, tmp_path)

    assert sent == [], f'the report sent a board without text: {sent}'
    for rel in TEXT_SECTIONS:
        assert NO_MOTOR_TEXT in (out / rel).read_text(), rel
    latency = (out / 'hardware_checks' / 'serial_latency.txt').read_text()
    assert NO_MOTOR_TEXT in latency.split('--- Motor Board')[1]
    homing = json.loads((out / 'motion_tests' / 'homing_test.json').read_text())
    for axis, row in homing['axes'].items():
        assert row['home_response'] == 'OK', axis
        assert row['actual_after'] == row['target_after'] == NO_MOTOR_TEXT, axis
    assert 'Fan Test' in (out / 'hardware_checks' / 'fan_test.txt').read_text()


def test_the_typed_reads_a_board_without_text_lacks_say_not_applicable(
    session, monkeypatch, tmp_path
):
    # The 6110 has no TMC5072 and no fan tachometer: its driver offers
    # neither read, so the API answers None, which the report states.
    _without_text(monkeypatch, session)
    monkeypatch.delattr(SimulatedMotorBoard, 'read_drv_status')
    monkeypatch.delattr(SimulatedMotorBoard, 'read_fanspeed')

    out = _report(session, tmp_path)

    status = (out / 'firmware_info' / 'motor_status.txt').read_text()
    assert NO_MOTOR_TEXT in status.split('TMC5072 Driver Status:')[1]
    assert 'None' not in status
    peripherals = (out / 'firmware_info' / 'peripherals.txt').read_text()
    assert peripherals.startswith('Fan: Not applicable: this motor board reports no fan speed')


def test_the_serial_number_is_the_drivers_or_unknown(session, monkeypatch, tmp_path):
    _without_text(monkeypatch, session)
    monkeypatch.setattr(session.scope._motion_driver, 'get_serial_number', lambda: None)

    report = TechSupportReport(session=session)

    assert report.diag.get_serial_number() == 'UNKNOWN'


def test_a_text_board_is_still_asked(session, tmp_path):
    out = _report(session, tmp_path)

    assert NO_MOTOR_TEXT not in '\n'.join(p.read_text() for p in out.rglob('*.txt'))
