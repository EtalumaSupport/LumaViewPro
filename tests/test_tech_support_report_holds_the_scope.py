# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The support report's hardware steps hold the scope, or skip and say so.

The report's LED selftest, LED leakage check, fan sweep and homing test
drive the hardware. Nothing arbitrated them: the report button stays live
during runs, and a report pressed during a recording homed the stage under
it. The steps now run under the session's diagnostic claim. When another
activity holds the scope the claim is refused, the reads still run, and
each writing step records that it was skipped and why.
"""

from unittest.mock import MagicMock

import pytest

from modules.tech_support_report import TechSupportReport
from tests.scope_fakes import spec_scope

WRITING_STEPS = ('_step_firmware_tests', '_step_led_checks', '_step_fan_test', '_step_homing_test')
READING_STEPS = (
    '_step_firmware_info',
    '_step_configbackup',
    '_step_tmc_registers',
    '_step_serial_latency',
    '_step_camera_diagnostics',
)
SKIPPED_FILES = (
    'firmware_tests/led_selftest.txt',
    'hardware_checks/led_leakage.txt',
    'hardware_checks/fan_test.txt',
    'motion_tests/homing_test.txt',
)


def _make_session():
    from modules.scope_session import ScopeSession

    scope = spec_scope()
    file_io_executor = MagicMock()
    file_io_executor.is_protocol_queue_active.return_value = False
    return ScopeSession(
        settings={},
        scope=scope,
        io_executor=MagicMock(),
        camera_executor=MagicMock(),
        file_io_executor=file_io_executor,
    )


@pytest.fixture
def report_and_calls(monkeypatch):
    """A report over a session whose nine scope steps are spies.

    Each spy records the scope's holder at the moment it ran, which is the
    fact under test: the hardware itself is not.
    """
    session = _make_session()
    report = TechSupportReport(session=session)
    calls = []
    for name in WRITING_STEPS + READING_STEPS:

        def _spy(tmp, _name=name):
            calls.append((_name, session.exclusive_activity))
            return 'SN0000' if _name == '_step_firmware_info' else None

        monkeypatch.setattr(report, name, _spy)
    return session, report, calls


def test_the_writing_steps_run_while_the_report_holds_the_scope(report_and_calls, tmp_path):
    session, report, calls = report_and_calls

    sn = report._run_scope_steps(tmp_path, lambda pct, msg: None)

    assert sn == 'SN0000'
    held_during = dict(calls)
    for name in WRITING_STEPS:
        assert held_during[name] == 'diagnostic', f'{name} ran without the scope held'
    assert session.exclusive_activity is None, 'the report left the scope held'


def test_during_a_recording_the_writing_steps_are_skipped_and_say_so(report_and_calls, tmp_path):
    session, report, calls = report_and_calls
    recording = session.activity_claim.try_claim('recording')
    try:
        report._run_scope_steps(tmp_path, lambda pct, msg: None)
    finally:
        recording.release()

    ran = {name for name, _ in calls}
    assert ran.isdisjoint(WRITING_STEPS), (
        f'hardware-writing steps ran during a recording: {sorted(ran & set(WRITING_STEPS))}'
    )
    assert set(READING_STEPS) <= ran, 'the reads must still run when the scope is in use'
    for rel in SKIPPED_FILES:
        text = (tmp_path / rel).read_text()
        assert 'SKIPPED' in text and 'recording' in text, (rel, text)


def test_the_session_supplies_the_scope(tmp_path):
    session = _make_session()
    report = TechSupportReport(session=session)
    assert report.scope is session.scope
