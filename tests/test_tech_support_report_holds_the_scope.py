# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The support report's hardware steps hold the scope, or skip and say so.

The report's LED selftest, LED leakage check, fan sweep and homing test
drive the hardware. Nothing arbitrated them: the report button stays live
during runs, and a report pressed during a recording homed the stage under
it. The steps now run under the session's diagnostic claim. When another
activity holds the scope the claim is refused: every step that sends a
board command records that it was skipped and why, the firmware-info step
writes what the drivers cached at connect, and the camera step still runs.
"""

import pytest

from modules.caller import Caller, calling
from modules.tech_support_report import TechSupportReport
from tests.scope_fakes import session_on_a_simulated_scope

WRITING_STEPS = ('_step_firmware_tests', '_step_led_checks', '_step_fan_test', '_step_homing_test')
BOARD_QUERY_STEPS = ('_step_configbackup', '_step_tmc_registers', '_step_serial_latency')
ALWAYS_STEPS = ('_step_firmware_info', '_step_camera_diagnostics')
READING_STEPS = BOARD_QUERY_STEPS + ALWAYS_STEPS
SKIPPED_FILES = (
    'firmware_tests/led_selftest.txt',
    'hardware_checks/led_leakage.txt',
    'hardware_checks/fan_test.txt',
    'motion_tests/homing_test.txt',
    'firmware_configs/config_backup.txt',
    'hardware_checks/tmc5072_registers.txt',
    'hardware_checks/serial_latency.txt',
)


@pytest.fixture
def report_and_calls(monkeypatch):
    """A report over a session whose nine scope steps are spies.

    Each spy records the scope's holder at the moment it ran, which is the
    fact under test: the hardware itself is not.
    """
    session = session_on_a_simulated_scope()
    report = TechSupportReport(session=session)
    calls = []
    for name in WRITING_STEPS + READING_STEPS:

        def _spy(tmp, *_args, _name=name):
            calls.append((_name, session.activity_claim.owner))
            return 'SN0000' if _name == '_step_firmware_info' else None

        monkeypatch.setattr(report, name, _spy)
    return session, report, calls


def test_the_writing_steps_run_while_the_report_holds_the_scope(report_and_calls, tmp_path):
    session, report, calls = report_and_calls

    sn = report._run_scope_steps(tmp_path, lambda pct, msg: None, include_bw=False)

    assert sn == 'SN0000'
    held_during = dict(calls)
    for name in WRITING_STEPS:
        assert held_during[name] == 'diagnostic', f'{name} ran without the scope held'
    assert session.activity_claim.owner is None, 'the report left the scope held'


def test_during_a_recording_the_board_steps_are_skipped_and_say_so(report_and_calls, tmp_path):
    session, report, calls = report_and_calls
    recording = session.activity_claim.try_claim('recording')
    try:
        report._run_scope_steps(tmp_path, lambda pct, msg: None, include_bw=False)
    finally:
        recording.release()

    ran = {name for name, _ in calls}
    board_steps = set(WRITING_STEPS) | set(BOARD_QUERY_STEPS)
    assert ran.isdisjoint(board_steps), (
        f'board-command steps ran during a recording: {sorted(ran & board_steps)}'
    )
    assert set(ALWAYS_STEPS) <= ran, (
        'the firmware-info (cached) and camera steps must still run when the scope is in use'
    )
    for rel in SKIPPED_FILES:
        text = (tmp_path / rel).read_text()
        assert 'SKIPPED' in text and 'recording' in text, (rel, text)


def test_under_a_held_control_session_the_board_steps_are_skipped_and_say_so(
    report_and_calls, tmp_path
):
    """A control session held by someone else refuses the report's claim as
    another activity does: the board steps are skipped and say why, and the
    report goes on to save."""
    session, report, calls = report_and_calls
    with calling(Caller.REST):
        controller = session.activity_claim.take_control('plate reader', finishing=lambda: False)
    try:
        report._run_scope_steps(tmp_path, lambda pct, msg: None, include_bw=False)
    finally:
        session.activity_claim.end_control(controller.id)

    ran = {name for name, _ in calls}
    board_steps = set(WRITING_STEPS) | set(BOARD_QUERY_STEPS)
    assert ran.isdisjoint(board_steps), (
        f'board-command steps ran under a held session: {sorted(ran & board_steps)}'
    )
    assert set(ALWAYS_STEPS) <= ran
    for rel in SKIPPED_FILES:
        text = (tmp_path / rel).read_text()
        assert 'SKIPPED' in text and 'plate reader' in text, (rel, text)


def test_the_session_supplies_the_scope(tmp_path):
    session = session_on_a_simulated_scope()
    report = TechSupportReport(session=session)
    assert report.scope is session.scope


def test_under_another_hold_the_bandwidth_test_is_skipped_and_the_report_saved(
    tmp_path, monkeypatch
):
    """The bandwidth test reaches the camera through its lane, which refuses
    a bystander while another activity holds the scope; run outside the
    report's own claim it was refused at its first frame and the whole report
    was lost. It runs under that claim with the other hardware steps, and is
    skipped and says so when the claim is refused."""
    import threading
    import zipfile

    from modules import tech_support_report
    from modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    # The host's USB inventory lists this machine's real serial ports, which
    # no test may touch; it is the one host read stood in for.
    monkeypatch.setattr(tech_support_report, '_collect_usb_devices', lambda: [])
    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path / 'live')), simulate=True
    )
    held, done = threading.Event(), threading.Event()

    def _holder():
        with session.diagnostic_claim():
            held.set()
            done.wait(60.0)

    holder = threading.Thread(target=_holder)
    holder.start()
    try:
        assert held.wait(10.0)
        saved = session.make_support_report(
            output_dir=tmp_path / 'reports', include_bandwidth_test=True
        )
    finally:
        done.set()
        holder.join(10.0)
        session.shutdown()

    with zipfile.ZipFile(saved.path) as report:
        summary = next(n for n in report.namelist() if n.endswith('bandwidth_test/summary.txt'))
        text = report.read(summary).decode()
    assert 'SKIPPED' in text and 'diagnostic' in text, text
