# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An empty protocol needs no objective; a protocol with steps refuses without one.

The Protocol panel builds an empty protocol on the first frame, before the
startup question is asked. It used to assemble a full run config to do it,
and on a turret scope whose slot has no assignment that config refuses an
unknown objective -- so the app exited at startup on every fresh install of
a turret model. An empty protocol has no step to stamp an objective into
and one tile needs no field of view, so it is built with none.
"""

import pytest

from modules.exceptions import ConfigError, ObjectiveUnknownError
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


@pytest.fixture
def unassigned_turret_session(tmp_path):
    """A fresh LS850T install: no slot assigned, the turret not yet homed."""
    session = ScopeSession.create(
        complete_settings(
            live_folder=str(tmp_path),
            microscope='LS850T',
            turret_objectives={'1': None, '2': None, '3': None, '4': None},
        ),
        simulate=True,
    )
    yield session
    session.shutdown()


def test_an_empty_protocol_is_built_while_the_objective_is_unknown(unassigned_turret_session):
    session = unassigned_turret_session
    assert session.scope.runtime_state.get_current_objective_id() is None

    protocol = session.create_empty_protocol()

    assert protocol.num_steps() == 0


def test_a_run_config_still_refuses_the_unknown_objective(unassigned_turret_session):
    with pytest.raises(ObjectiveUnknownError):
        unassigned_turret_session.get_sequenced_capture_config()


def test_steps_are_never_built_without_an_objective(tmp_path):
    # A run config whose objective is missing must not become steps stamped
    # with nothing: the builder refuses rather than record no objective.
    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), microscope='LS850'), simulate=True
    )
    try:
        session.settings['BF']['acquire'] = 'image'
        config = session.get_sequenced_capture_config()
        config['objective_id'] = None
        with pytest.raises(ConfigError, match='no objective'):
            session.scope.protocols.create_protocol(input_config=config)
    finally:
        session.shutdown()
