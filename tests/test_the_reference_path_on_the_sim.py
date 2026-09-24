# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The objective and jog gestures show the API's outcome as it is typed, on a simulated scope.

Each gesture hands its Session or API call to the GUI boundary: a refusal is
one warning under the API's title in the API's sentence, logged once with no
traceback, and the display shows what the API says afterwards. The widgets
write no popup and no log line of their own.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

import modules.app_context as _app_ctx
import ui.vertical_control as vc
from modules.notification_center import Severity
from tests.scope_fakes import home_sim_scope
from tests.shown_outcomes import capture_shown
from tests.test_the_gui_displays_the_turret import _Stand, session, stand  # noqa: F401 -- fixtures
from ui import ui_helpers

OUTCOMES = 'LVP.outcomes'
NOTIFICATIONS = 'LVP.notifications'


def _records(caplog):
    return [
        (r.name, r.levelno, bool(r.exc_info))
        for r in caplog.records
        if r.name in (OUTCOMES, NOTIFICATIONS) and r.levelno >= logging.WARNING
    ]


@pytest.fixture
def run_holds_the_scope(session):
    held = session.activity_claim.try_claim('protocol')
    assert held is not None
    yield
    held.release()


class TestWhileARunHoldsTheScope:
    def test_picking_another_objective_is_one_microscope_busy_warning(
        self, stand, session, run_holds_the_scope, monkeypatch, caplog
    ):
        home_sim_scope(session.scope)
        shown = capture_shown(monkeypatch)
        with caplog.at_level(logging.DEBUG):
            stand.pick_objective('4x Oly')

        assert [(n.severity, n.title) for n in shown] == [(Severity.WARNING, 'Microscope Busy')]
        assert _records(caplog) == [(NOTIFICATIONS, logging.WARNING, False)]
        assert stand.popups == []
        # The display shows the objective still mounted in slot 1.
        assert stand.ids['objective_spinner2'].text == '10x Oly'

    def test_picking_the_mounted_objective_is_no_outcome(
        self, stand, session, run_holds_the_scope, monkeypatch
    ):
        home_sim_scope(session.scope)
        shown = capture_shown(monkeypatch)
        stand.pick_objective('10x Oly')
        assert shown == []


def test_resetting_an_unknown_slot_is_one_objective_unknown_warning(
    stand, session, monkeypatch, caplog
):
    shown = capture_shown(monkeypatch)
    with caplog.at_level(logging.DEBUG):
        stand.reset_turret_objective()

    assert [(n.severity, n.title) for n in shown] == [(Severity.WARNING, 'Objective Unknown')]
    assert 'home the turret' in shown[0].message
    assert _records(caplog) == [(NOTIFICATIONS, logging.WARNING, False)]
    assert stand.popups == []


def test_a_refused_answer_is_shown_and_the_startup_question_still_resolves(
    stand, session, run_holds_the_scope, monkeypatch
):
    home_sim_scope(session.scope)
    shown = capture_shown(monkeypatch)
    resolved = []
    stand._resolve_objective = lambda on_resolved: resolved.append(on_resolved)
    vc.VerticalControl._apply_objective_answer(stand, '4x Oly', 1, on_resolved='startup')

    assert [n.title for n in shown] == ['Microscope Busy']
    assert resolved == ['startup']
    assert stand.popups == []


@pytest.mark.parametrize('axis', ['Z', 'X'])
def test_a_jog_with_the_objective_unknown_is_one_objective_unknown_warning(
    session, monkeypatch, axis
):
    shown = capture_shown(monkeypatch)
    moved = []
    monkeypatch.setattr(
        _app_ctx,
        'ctx',
        SimpleNamespace(scope=session.scope, session=SimpleNamespace(controls_locked=False)),
    )
    monkeypatch.setattr(ui_helpers, 'move_relative', lambda *a, **k: moved.append(a))
    # Slot 3 has no objective assigned, so the step cannot be sized.
    home_sim_scope(session.scope)
    session.scope.motion.move_turret(3)

    if axis == 'Z':
        monkeypatch.setattr(vc, 'move_relative', ui_helpers.move_relative)
        vc.VerticalControl._z_jog(SimpleNamespace(), +1, coarse=False)
    else:
        from ui import motion_settings

        monkeypatch.setattr(motion_settings, 'move_relative', ui_helpers.move_relative)
        motion_settings.XYStageControl._xy_jog(SimpleNamespace(), 'X', +1, coarse=False)

    assert moved == []
    assert [(n.severity, n.title) for n in shown] == [(Severity.WARNING, 'Objective Unknown')]
