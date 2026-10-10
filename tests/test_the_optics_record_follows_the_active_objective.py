# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The optics behind image scale are recorded each time the active objective changes.

On a turret scope the active objective changes when the turret moves or a
slot is assigned, and neither passes through a selection member. The record
used to be written only by selection, so a support bundle could not show
which scale a capture after a turret move used. It is now written by the
one derivation, once per change, however the objective changed.
"""

import pytest

import modules.config_helpers as config_helpers
from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings


@pytest.fixture
def recorded(monkeypatch):
    seen = []
    monkeypatch.setattr(
        config_helpers,
        'log_resolved_optics',
        lambda objective_id, focal_length, binning_size, *, capabilities: seen.append(objective_id),
    )
    return seen


@pytest.fixture
def turret_session(tmp_path):
    session = ScopeSession.create(
        complete_settings(
            live_folder=str(tmp_path),
            microscope='LS850T',
            objective_confirmed=True,
            turret_objectives={'1': '4x Oly', '2': '10x Oly', '3': None, '4': None},
        ),
        simulate=True,
    )
    home_sim_scope(session.scope)
    yield session
    session.shutdown()


def _read(session):
    return session.scope.runtime_state.get_current_objective_id()


def test_a_turret_move_records_the_new_objective_once(turret_session, recorded):
    turret_session.scope.motion.move_turret(1)
    assert _read(turret_session) == '4x Oly'
    turret_session.scope.motion.move_turret(2)
    assert _read(turret_session) == '10x Oly'
    assert _read(turret_session) == '10x Oly'

    assert recorded == ['4x Oly', '10x Oly']


def test_answering_the_question_records_the_answer(turret_session, recorded):
    turret_session.scope.motion.move_turret(3)
    question = turret_session.objective_question()
    assert question is not None and question.turret_position == 3

    turret_session.confirm_objective('20x Oly', turret_position=question.turret_position)

    assert _read(turret_session) == '20x Oly'
    assert recorded == ['20x Oly']
