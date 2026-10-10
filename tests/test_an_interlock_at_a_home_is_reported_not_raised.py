# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A home the stage's interlock refuses is reported where it was asked for, as no board is.

The startup home and the support report's homing test answered a refused
home only when the refusal was ``'not_connected'``, and re-raised every
other as a caller's defect. A stage whose lid is open or whose power is off
refuses the home for the hardware's state, which is the person's to fix at
the scope: bring-up shows it once and skips the turret move, and the report
records it against the axis, as both do for a missing motor controller.
"""

from __future__ import annotations

import pytest

from drivers.exceptions import MotionInterlockError
from modules.exceptions import HardwareCommandRefusedError
from modules.notification_center import Severity
from modules.scope_session import ScopeSession
from modules.tech_support_report import TechSupportReport
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings

REASONS = {
    'lid_open': ('Lid Open', "The microscope's lid is open. Close it to move or home the stage."),
    'stage_unpowered': (
        'No Stage Power',
        "The stage has no power. Check the stage's power supply, then home.",
    ),
}


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), microscope='LS850T'), simulate=True
    )
    try:
        yield s
    finally:
        s.shutdown()


@pytest.mark.parametrize('reason', sorted(REASONS))
def test_bring_up_shows_it_once_and_leaves_the_turret_alone(session, centre_posts, reason):
    turret = []

    def refused(axis):
        raise HardwareCommandRefusedError(reason, 'home')

    session.start_application_session(home_fn=refused, turret_fn=turret.append)

    title, words = REASONS[reason]
    shown = [n for n in centre_posts if n.severity >= Severity.INFO]
    assert [(n.severity, n.title, n.message) for n in shown] == [(Severity.WARNING, title, words)]
    assert turret == []


@pytest.mark.parametrize('reason', sorted(REASONS))
def test_the_report_records_it_against_the_axis(session, monkeypatch, reason):
    home_sim_scope(session.scope)

    def refused():
        raise MotionInterlockError(reason, moved=False, stopped=False)

    monkeypatch.setattr(session.scope._motion_driver, 'zhome', refused)

    result = TechSupportReport(scope=session.scope).diag.run_homing_test()

    assert result['passed'] is False
    assert result['axes']['Z']['status'] == 'FAIL'
    assert result['axes']['Z']['home_response'] == f'Error: {REASONS[reason][1]}'
