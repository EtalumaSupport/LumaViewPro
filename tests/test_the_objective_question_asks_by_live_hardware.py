# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Whether a scope has a turret is answered by the board when it can be.

The startup objective question decided by the DECLARED model -- the
settings' `microscope` string looked up in scopes.json. That is right for
exactly one case, a motorboard that is not talking: it reports no axes,
and that is precisely when a stale stored objective must not be adopted
unasked. It is wrong for every other case, and the shipped template makes
the wrong case the common one: `data/settings.json` declares LS850, whose
catalogue entry has no turret, so a real LS850T arriving with shipped
settings was never asked for the objective at its current slot.

So: the board when the board is connected, the declaration only when it
is not. One helper answers it, used by the two sites that asked the
declaration directly.

This cannot be observed in the simulator -- the simulated board's model
IS the declared model (`ScopeSession` passes `settings['microscope']` as
`configured_model`), so live and declared can never disagree there. The
disagreement is therefore built directly, the way the refusal funnel's
own test builds one.
"""

from __future__ import annotations

import dataclasses

import pytest

from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    yield s
    s.shutdown()


def _declare(session, model: str) -> None:
    session.settings['microscope'] = model


def _board(session, *, connected: bool, has_turret: bool) -> None:
    """Set what the LIVE hardware reports, independent of the declaration."""
    session.scope.capabilities = dataclasses.replace(
        session.scope.capabilities, has_turret=has_turret
    )
    session.scope.__dict__['_probe_motor_connected'] = connected


class TestTheTurretFactPrefersTheBoard:
    def test_a_connected_board_outranks_a_declaration_that_denies_it(self, session, monkeypatch):
        """The shipped-template case: LS850 declared, a real turret present."""
        _declare(session, 'LS850')
        monkeypatch.setattr(type(session.scope), 'motor_connected', property(lambda self: True))
        _board(session, connected=True, has_turret=True)

        assert session.scope_has_turret() is True

    def test_a_connected_board_outranks_a_declaration_that_invents_one(self, session, monkeypatch):
        _declare(session, 'LS850T')
        monkeypatch.setattr(type(session.scope), 'motor_connected', property(lambda self: True))
        _board(session, connected=True, has_turret=False)

        assert session.scope_has_turret() is False

    def test_a_disconnected_board_falls_back_to_the_declaration(self, tmp_path, monkeypatch):
        """The case the declaration was chosen for, and it still holds.

        A dead board reports no axes. Believing it would say "no turret",
        and a stale stored objective would then be adopted without anyone
        being asked -- which is the failure the declared model was there
        to prevent. The model is declared when the scope is brought up,
        the one way a declaration reaches a running scope.
        """
        session = ScopeSession.create(
            complete_settings(live_folder=str(tmp_path), microscope='LS850T'), simulate=True
        )
        try:
            monkeypatch.setattr(
                type(session.scope), 'motor_connected', property(lambda self: False)
            )
            _board(session, connected=False, has_turret=False)

            assert session.scope_has_turret() is True
        finally:
            session.shutdown()

    def test_a_disconnected_board_on_a_turretless_declaration_says_no(self, session, monkeypatch):
        _declare(session, 'LS850')
        monkeypatch.setattr(type(session.scope), 'motor_connected', property(lambda self: False))
        _board(session, connected=False, has_turret=False)

        assert session.scope_has_turret() is False


class TestTheQuestionUsesIt:
    def test_the_question_asks_for_a_slot_when_the_board_reports_a_turret(
        self, session, monkeypatch
    ):
        """LS850 declared, real turret: the question must name a position."""
        _declare(session, 'LS850')
        session.settings['objective_confirmed'] = False
        session.settings['turret_objectives'] = {1: None, 2: None, 3: None, 4: None}
        monkeypatch.setattr(type(session.scope), 'motor_connected', property(lambda self: True))
        _board(session, connected=True, has_turret=True)
        # The question names the live slot, the one motion reports.
        monkeypatch.setattr(session.scope.motion, 'get_turret_slot', lambda: 1)
        monkeypatch.setattr(session, 'settings_are_provisional', lambda: False)
        monkeypatch.setattr(type(session.scope), 'no_hardware', property(lambda self: False))

        question = session.objective_question()

        assert question is not None, 'a turreted scope with an empty slot must be asked'
        assert question.turret_position == 1, (
            'the question must name the slot the answer will be assigned to'
        )

    def test_no_slot_is_named_when_the_board_reports_no_turret(self, session, monkeypatch):
        _declare(session, 'LS850T')
        session.settings['objective_confirmed'] = False
        monkeypatch.setattr(type(session.scope), 'motor_connected', property(lambda self: True))
        _board(session, connected=True, has_turret=False)
        monkeypatch.setattr(session, 'settings_are_provisional', lambda: False)
        monkeypatch.setattr(type(session.scope), 'no_hardware', property(lambda self: False))

        question = session.objective_question()

        assert question is not None
        assert question.turret_position is None, (
            'a scope with no turret has no slot to assign the answer to'
        )


def _brought_up(tmp_path, model: str, **stored) -> ScopeSession:
    """A session on ``model`` from a file holding ``stored``, its turret homed to slot 1."""
    from tests.scope_fakes import TEST_TURRET_OBJECTIVES, home_sim_scope

    session = ScopeSession.create(
        complete_settings(
            live_folder=str(tmp_path),
            microscope=model,
            turret_objectives=dict(TEST_TURRET_OBJECTIVES),
            **stored,
        ),
        simulate=True,
    )
    if session.scope_has_turret():
        home_sim_scope(session.scope)
        session.scope.motion.move_turret(1)
    return session


class TestAChangeOfTopologyAsksAgain:
    """The confirmation records whether the scope it was given on had a turret.

    A scope brought up from the same file with the other answer owes the
    question again: the confirmed objective was confirmed for the other
    light path. Keyed on ``scope_has_turret()``, so a model chosen in the
    GUI and a board that reports another model at bring-up are one case.
    """

    @pytest.mark.parametrize(
        'confirmed_on, brought_up_on',
        [('LS850', 'LS850T'), ('LS850T', 'LS850')],
    )  # parametrize-justified: each direction needs its own two bring-ups
    def test_a_confirmation_given_on_the_other_topology_owes_the_question(
        self, tmp_path, confirmed_on, brought_up_on
    ):
        first = _brought_up(tmp_path, confirmed_on, objective_confirmed=None)
        try:
            first.confirm_objective(
                '10x Oly', turret_position=1 if first.scope_has_turret() else None
            )
            stored = first.get_setting('objective_confirmed')
            assert first.objective_question() is None
        finally:
            first.shutdown()

        second = _brought_up(tmp_path, brought_up_on, objective_confirmed=stored)
        try:
            assert second.objective_question() is not None, (
                f'confirmed on {confirmed_on}, brought up on {brought_up_on}: not asked'
            )
        finally:
            second.shutdown()

    def test_a_turretless_question_proposes_the_objective_the_turret_period_left(self, tmp_path):
        session = _brought_up(
            tmp_path, 'LS850', objective_confirmed='turret', objective_id='40x Phase'
        )
        try:
            assert session.objective_question().proposed == '40x Phase'
        finally:
            session.shutdown()

    @pytest.mark.parametrize('model, recorded', [('LS850', 'turretless'), ('LS850T', 'turret')])
    # parametrize-justified: each topology is its own bring-up
    def test_a_confirmation_from_before_the_record_takes_the_bring_ups_topology(
        self, tmp_path, model, recorded
    ):
        """A stored ``true`` predates the record; the upgrade asks nothing."""
        session = _brought_up(tmp_path, model, objective_confirmed=True)
        try:
            assert session.get_setting('objective_confirmed') == recorded
            assert session.objective_question() is None
        finally:
            session.shutdown()

    def test_an_unconfirmed_file_from_before_the_record_is_still_unconfirmed(self, tmp_path):
        session = _brought_up(tmp_path, 'LS850', objective_confirmed=False)
        try:
            assert session.get_setting('objective_confirmed') is None
            assert session.objective_question() is not None
        finally:
            session.shutdown()


class TestTheDeclarationIsNoLongerReadDirectly:
    def test_neither_site_calls_model_has_turret_itself(self):
        """Two readers, one helper: a third spelling is how they diverge."""
        import ast

        from tests.ast_seams import find_def

        for name in ('objective_question', 'configure_scope'):
            fn = find_def('modules/scope_session.py', name)
            assert fn is not None, f'{name} not found'
            calls = [
                node.func.attr
                for node in ast.walk(fn)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            ]
            assert 'model_has_turret' not in calls, (
                f'{name} reads the declared model directly again; ask scope_has_turret()'
            )
