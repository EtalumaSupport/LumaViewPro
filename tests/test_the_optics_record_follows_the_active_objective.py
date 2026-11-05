# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The optics behind image scale are recorded each time the active objective changes.

On a turret scope the active objective changes when the turret moves or a
slot is assigned, and neither passes through a selection member. The record
used to be written only by selection, so a support bundle could not show
which scale a capture after a turret move used. It is now written by the
one derivation, once per change, however the objective changed.
"""

import logging

import pytest

import _lumascope.modules.config_helpers as config_helpers
from _lumascope.modules.lumascope_api.simulation import SimulatedScope
from _lumascope.modules.scope_session import ScopeSession
from tests.log_capture import messages, records_of
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings


@pytest.fixture
def recorded(turret_session, monkeypatch):
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
    # The home put slot 1 in the light path, a change whose record waits for
    # the first reader -- a test's read, or any frame the camera starts. Read
    # here, so the change a test watches is its own.
    assert _read(session) == '4x Oly'
    yield session
    session.shutdown()


def _read(session):
    return session.scope.runtime_state.get_current_objective_id()


def test_a_turret_move_records_the_new_objective_once(turret_session, recorded):
    turret_session.scope.motion.move_turret(2)
    assert _read(turret_session) == '10x Oly'
    assert _read(turret_session) == '10x Oly'
    turret_session.scope.motion.move_turret(1)
    assert _read(turret_session) == '4x Oly'

    assert recorded == ['10x Oly', '4x Oly']


def test_answering_the_question_records_the_answer(turret_session, recorded):
    turret_session.scope.motion.move_turret(3)
    question = turret_session.objective_question()
    assert question is not None and question.turret_slot == 3

    turret_session.confirm_objective('20x Oly', slot=question.turret_slot)

    assert _read(turret_session) == '20x Oly'
    assert recorded == ['20x Oly']


def test_a_scope_with_no_camera_records_no_image_scale(caplog):
    """The record explains the scale of the images a scope takes; a scope
    whose camera is not on the bus takes none, so reading its objective
    writes no optics line -- not a warning that its images carry no
    PhysicalSizeX, which the error log a support bundle is read from would
    carry on every camera-less launch."""
    session = ScopeSession.create(
        complete_settings(microscope='LS850'),
        simulate=True,
        warn_pre_release=False,
        simulated=SimulatedScope(camera_present=False),
    )
    try:
        assert not session.scope.camera_connected
        with caplog.at_level(logging.DEBUG):
            assert _read(session) is not None
        assert [m for m in messages(records_of(caplog, config_helpers)) if '[Optics' in m] == []
    finally:
        session.shutdown()
