# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A model saved for the next start is the Session's answer, not the GUI's comparison.

Before, Advanced Settings compared the saved model with the running one
itself and then raised its notice at info, which the popup bridge never
shows, so choosing an LS560 on a running LS620 said nothing on screen. Now
the Session says which model waits for the next start, and the GUI shows
that answer.

The answer names only a start that happens. A scope whose board names its
model (every EL-0940 answering FULLINFO, the LS720, every OEM unit) runs as
the board's model at every start, so nothing waits there, and the selection
made on it survives its starts for the scope that takes its model from it
(a Classic FX2). The simulated scopes built as the selection (the LS620
fixture) are the scopes whose model the selection decides.
"""

from __future__ import annotations

import copy

import pytest

from _lumascope.modules.lumascope_api.simulation import SimulatedScope
from _lumascope.modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings
from tests.test_a_simulated_scope_comes_up_as_a_units_own_config import OEM_NAME, unit_config


@pytest.fixture
def ls620_session(tmp_path):
    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path / 'live'), microscope='LS620'), simulate=True
    )
    yield session
    session.shutdown()


def test_nothing_waits_while_the_saved_model_is_the_one_running(ls620_session):
    assert ls620_session.model_at_next_start is None


def test_a_different_model_waits_for_the_next_start_until_the_running_one_is_chosen_again(
    ls620_session,
):
    ls620_session.select_model('LS560')
    assert ls620_session.model_at_next_start == 'LS560'
    assert ls620_session.scope.channel_identity.model == 'LS620'

    ls620_session.select_model('LS620')
    assert ls620_session.model_at_next_start is None


@pytest.fixture
def heard(ls620_session):
    """What a client subscribed to the session's outcomes hears."""
    notes = []
    ls620_session.add_outcome_listener(notes.append)
    return notes


def test_another_model_is_reported_once_as_waiting_for_the_next_start(ls620_session, heard):
    ls620_session.select_model('LS560')

    deferred = [n for n in heard if n.title == 'Scope model saved']
    assert len(deferred) == 1
    assert deferred[0].reason == 'scope_model_deferred'
    assert deferred[0].solicited is True
    assert deferred[0].category == 'Microscope'
    assert 'LS560' in deferred[0].message and 'next time' in deferred[0].message


def test_the_running_model_is_reported_as_nothing(ls620_session, heard):
    ls620_session.select_model('LS620')

    assert [n for n in heard if n.title == 'Scope model saved'] == []


def test_an_unknown_model_is_refused_and_reported_as_nothing(ls620_session, heard):
    from _lumascope.modules.exceptions import ScopeModelUnknownError

    with pytest.raises(ScopeModelUnknownError):
        ls620_session.select_model('LS9999')

    assert [n for n in heard if n.title == 'Scope model saved'] == []


_SAVED_REASONS = {'scope_model_deferred', 'scope_model_for_another_scope'}


@pytest.fixture
def started(tmp_path):
    """Start a simulated session as ``simulated``, from ``settings`` or the template's."""
    sessions = []

    def _start(simulated, settings=None, **overrides):
        if settings is None:
            settings = complete_settings(live_folder=str(tmp_path / 'live'), **overrides)
        session = ScopeSession.create(
            settings, simulate=True, simulated=simulated, warn_pre_release=False
        )
        sessions.append(session)
        return session

    yield _start
    for session in sessions:
        session.shutdown()


def test_a_model_chosen_on_a_scope_that_names_its_own_is_saved_for_another_scope(started):
    session = started(SimulatedScope(model='LS850'), microscope='LS850')
    heard = []
    session.add_outcome_listener(heard.append)

    session.select_model('LS560')

    assert session.selected_model == 'LS560'
    assert session.model_at_next_start is None
    saved = [n for n in heard if getattr(n, 'reason', None) in _SAVED_REASONS]
    assert [n.reason for n in saved] == ['scope_model_for_another_scope']
    assert 'LS560' in saved[0].message and 'LS850' in saved[0].message


def test_the_selection_survives_a_start_on_a_scope_that_names_its_own(started):
    first = started(SimulatedScope(model='LS850'), microscope='LS850')
    first.select_model('LS560')

    again = started(SimulatedScope(model='LS850'), settings=copy.deepcopy(first.settings))
    assert again.scope.channel_identity.model == 'LS850'
    assert again.settings['microscope'] == 'LS560'

    # The scope with no board to ask, connected next, comes up as the selection.
    classic = started(SimulatedScope(), settings=copy.deepcopy(again.settings))
    assert classic.scope.channel_identity.model == 'LS560'


def test_an_oem_unit_names_no_model_for_its_next_start(started):
    session = started(
        SimulatedScope(unit_config=unit_config('XYZ')),
        microscope='LS850',
        simulator_tier='firmware',
    )

    assert session.scope.channel_identity.model == OEM_NAME
    assert session.model_at_next_start is None
    assert session.settings['microscope'] == 'LS850'


def test_whether_the_board_named_the_model_is_the_identitys_answer(started):
    cases = (
        # (the scope, its stored model, whether its board named the model)
        ('an LS850 board', SimulatedScope(model='LS850'), 'LS850', True),
        ("the LS720's board", SimulatedScope(model='LS720'), 'LS720', True),
        ('an LS620 built as the selection', SimulatedScope(), 'LS620', False),
        ('a Classic LS560', SimulatedScope(), 'LS560', False),
    )
    wrong = []
    for name, simulated, stored, named in cases:
        identity = started(simulated, microscope=stored).scope.channel_identity
        if identity.model_named_by_board is not named:
            wrong.append(f'{name}: model_named_by_board={identity.model_named_by_board!r}')
    assert not wrong, '\n'.join(wrong)
