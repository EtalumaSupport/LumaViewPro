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


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(ScopeSession.load_user_settings('.'), simulate=True)
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

    def test_a_disconnected_board_falls_back_to_the_declaration(self, session, monkeypatch):
        """The case the declaration was chosen for, and it still holds.

        A dead board reports no axes. Believing it would say "no turret",
        and a stale stored objective would then be adopted without anyone
        being asked -- which is the failure the declared model was there
        to prevent.
        """
        _declare(session, 'LS850T')
        monkeypatch.setattr(type(session.scope), 'motor_connected', property(lambda self: False))
        _board(session, connected=False, has_turret=False)

        assert session.scope_has_turret() is True

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
