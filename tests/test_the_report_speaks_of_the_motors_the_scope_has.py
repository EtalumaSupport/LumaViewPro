# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The support report speaks about the motors the scope has.

An LS560 or LS620 is built without a motor board, and runs on the null
board exactly as an LS850T with its board unplugged does. The report asked
only "is a motor board connected?", so the manual scope's report said
"Motor board not connected" in every motor file and sent support after a
cable that does not exist. And the homing test homed Z, T and the stage on
every scope, writing a T row marked OK on a scope with no turret: a pass for
a check it could not run.

The report now tells three answers apart -- connected, missing, and none on
this model -- and homes and reports only the axes the scope has.
"""

import pytest

from _lumascope.modules.scope_session import ScopeSession
from _lumascope.modules.tech_support_report import TechSupportReport
from tests.settings_fixtures import complete_settings

MOTOR_FILES = (
    'firmware_info/motor_info.txt',
    'firmware_info/motor_status.txt',
    'firmware_configs/motor_config.txt',
    'hardware_checks/motor_driver_registers.txt',
    'hardware_checks/fan_test.txt',
    'motion_tests/homing_test.txt',
)


def _report(tmp_path, model, monkeypatch=None, unplug=False, held=False, tier='fast'):
    session = ScopeSession.create(
        complete_settings(
            live_folder=str(tmp_path / 'live'), microscope=model, simulator_tier=tier
        ),
        simulate=True,
    )
    if unplug:
        monkeypatch.setattr(type(session.scope), 'motor_connected', property(lambda self: False))
    out = tmp_path / 'report'
    out.mkdir()
    claim = session.activity_claim.try_claim('diagnostic') if held else None
    try:
        TechSupportReport(session=session)._run_scope_steps(
            out, lambda pct, msg: None, include_bw=False
        )
    finally:
        if claim is not None:
            claim.release()
        session.shutdown()
        session.scope.disconnect()
    return out


def _all_text(out):
    return '\n'.join(p.read_text() for p in out.rglob('*.txt'))


@pytest.mark.parametrize('model', ['LS620', 'LS560'])
def test_a_manual_scope_says_it_has_no_motor_board(tmp_path, model):
    out = _report(tmp_path, model)

    for rel in MOTOR_FILES:
        assert 'No motor board on this model' in (out / rel).read_text(), rel
    latency = (out / 'hardware_checks' / 'serial_latency.txt').read_text()
    assert 'No motor board on this model' in latency.split('--- Motor Board')[1]
    everything = _all_text(out)
    assert 'not connected' not in everything.lower()
    assert 'Home response' not in everything


def test_an_unplugged_board_on_a_motorised_scope_is_still_reported_missing(tmp_path, monkeypatch):
    out = _report(tmp_path, 'LS850T', monkeypatch, unplug=True)

    for rel in (
        'hardware_checks/motor_driver_registers.txt',
        'hardware_checks/fan_test.txt',
        'motion_tests/homing_test.txt',
    ):
        assert 'Motor board not connected' in (out / rel).read_text(), rel
    assert 'No motor board on this model' not in _all_text(out)


@pytest.mark.slow
@pytest.mark.parametrize(
    ('model', 'tier'),
    [('LS620', 'fast'), ('LS720', 'fast'), ('LS850T', 'firmware')],
    ids=['no_motor_board', 'tmcm6110', 'el0940_field_firmware'],
)
def test_no_motor_section_names_a_part_the_scope_may_lack_or_writes_none(tmp_path, model, tier):
    # The motor driver sections were headed TMC5072, the EL-0940's part, on
    # every model, and a read the firmware does not offer was written as
    # None. The LS720 runs the production 6110 driver; the LS850T's field
    # firmware offers neither the driver status nor the fan speed.
    out = _report(tmp_path, model, tier=tier)

    assert not [p.name for p in out.rglob('*') if 'tmc5072' in p.name.lower()]
    texts = {
        rel: (out / rel).read_text()
        for rel in (*MOTOR_FILES, 'firmware_info/peripherals.txt')
        if (out / rel).exists()
    }
    assert 'firmware_info/motor_status.txt' in texts
    for rel, text in texts.items():
        assert 'TMC5072' not in text and 'None' not in text, (rel, text)


@pytest.mark.slow
def test_a_turretless_scope_homes_no_turret(tmp_path):
    homing = (_report(tmp_path, 'LS850') / 'motion_tests' / 'homing_test.txt').read_text()

    assert 'Overall: PASS' in homing
    assert [axis for axis in 'ZTXY' if f'{axis} axis:' in homing] == ['Z', 'X', 'Y']


@pytest.mark.slow
def test_a_focus_only_scope_homes_only_its_focus(tmp_path):
    homing = (_report(tmp_path, 'LS820') / 'motion_tests' / 'homing_test.txt').read_text()

    assert 'Overall: PASS' in homing
    assert [axis for axis in 'ZTXY' if f'{axis} axis:' in homing] == ['Z']


def test_a_manual_scope_held_by_another_activity_still_says_it_has_no_motor_board(tmp_path):
    # The board queries are skipped while the scope is held; the motor files
    # are written from what is known without them.
    out = _report(tmp_path, 'LS620', held=True)

    for rel in ('firmware_info/motor_info.txt', 'firmware_info/motor_status.txt'):
        assert 'No motor board on this model' in (out / rel).read_text(), rel
    assert (
        'Fan: No motor board on this model'
        in (out / 'firmware_info' / 'peripherals.txt').read_text()
    )


def test_a_board_that_reports_no_axes_is_not_a_homing_pass(tmp_path, monkeypatch):
    import dataclasses

    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), microscope='LS850'), simulate=True
    )
    try:
        caps = session.scope.capabilities
        monkeypatch.setattr(
            session.scope,
            'capabilities',
            dataclasses.replace(
                caps, axes=(), has_focus=False, has_xy_stage=False, has_turret=False
            ),
        )
        result = TechSupportReport(session=session).diag.run_homing_test()
    finally:
        session.shutdown()
        session.scope.disconnect()

    assert 'passed' not in result
    assert 'no axes' in result['error']
