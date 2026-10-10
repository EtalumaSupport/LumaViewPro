# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The support report's homing test records why a home failed.

A home that fails raises, naming why. The report's homing test is where
that outcome ends, so it records the raised words against the axis and
marks the test failed, instead of the bare "home failed" a bool left it.
"""

from __future__ import annotations

import pytest

from modules.scope_session import ScopeSession
from modules.tech_support_report import TechSupportReport
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), microscope='LS850T'), simulate=True
    )
    try:
        home_sim_scope(s.scope)
        yield s
    finally:
        s.shutdown()


@pytest.mark.slow
def test_a_failed_z_home_is_recorded_in_its_own_words(session, monkeypatch):
    monkeypatch.setattr(session.scope._motion_driver, 'zhome', lambda: False)

    result = TechSupportReport(scope=session.scope).diag.run_homing_test()

    assert result['passed'] is False
    assert result['axes']['Z']['status'] == 'FAIL'
    assert result['axes']['Z']['home_response'] == (
        'Error: Z axis homing failed. Position is unknown.'
    )
