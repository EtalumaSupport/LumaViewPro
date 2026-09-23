# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The active objective cannot change while a run holds the scope.

A run reads the active objective at every capture. A slot assignment, a
slot clear or a selection mid-run would stamp a different scale into the
rest of the run's files than the objective its steps were built for. The
Session's objective writers refuse while a protocol-class run holds the
activity claim, with the reason every run lockout uses; a write that would
change nothing stays a silent no-op, because the GUI re-writes the current
value on programmatic widget updates.
"""

import pytest

from modules.exceptions import HardwareCommandRefusedError
from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(
        complete_settings(
            live_folder=str(tmp_path),
            microscope='LS850T',
            objective_confirmed=True,
            turret_objectives={1: '10x Oly', 2: '4x Oly', 3: None, 4: None},
        ),
        simulate=True,
    )
    try:
        home_sim_scope(s.scope)
        s.scope.motion.move_turret(1)
        yield s
    finally:
        s.shutdown()


@pytest.fixture
def under_a_run(session):
    assert session.activity_claim.try_claim('protocol', run_trigger_source='test')
    yield session
    session.activity_claim.release('protocol')


@pytest.mark.parametrize(
    'write',
    [
        lambda s: s.select_objective('20x Oly'),
        lambda s: s.assign_turret_objective(1, '20x Oly'),
        lambda s: s.assign_turret_objective(3, '20x Oly'),
        lambda s: s.clear_turret_objective(2),
    ],
    ids=['select', 'assign_live_slot', 'assign_other_slot', 'clear'],
)
def test_a_change_is_refused_and_writes_nothing(under_a_run, write):
    session = under_a_run
    before = dict(session.settings['turret_objectives'])

    with pytest.raises(HardwareCommandRefusedError) as excinfo:
        write(session)

    assert excinfo.value.reason == 'exclusive_activity_running'
    assert session.settings['turret_objectives'] == before
    assert session.scope.runtime_state.get_current_objective_id() == '10x Oly'


@pytest.mark.parametrize(
    'write',
    [
        lambda s: s.select_objective('10x Oly'),
        lambda s: s.assign_turret_objective(1, '10x Oly'),
        lambda s: s.clear_turret_objective(3),
    ],
    ids=['select_current', 'assign_same', 'clear_empty'],
)
def test_a_write_that_changes_nothing_is_not_refused(under_a_run, write):
    write(under_a_run)


def test_after_the_run_the_change_is_accepted(under_a_run):
    session = under_a_run
    session.activity_claim.release('protocol')
    try:
        session.assign_turret_objective(1, '20x Oly')
        assert session.scope.runtime_state.get_current_objective_id() == '20x Oly'
    finally:
        assert session.activity_claim.try_claim('protocol', run_trigger_source='test')
