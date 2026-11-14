# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The objective and jog gestures show the API's outcome as it is typed, on a simulated scope.

Each gesture hands its Session or API call to the GUI boundary: a refusal is
one warning under the API's title in the API's sentence, logged once with no
traceback, and the display shows what the API says afterwards. The widgets
write no popup and no log line of their own.
"""

from __future__ import annotations

import logging
import sys
from types import SimpleNamespace

import pytest

import _lumascope.modules.app_context as _app_ctx
import ui.vertical_control as vc
from _lumascope.modules.notification_center import Severity
from tests.scope_fakes import home_sim_scope
from tests.test_the_gui_displays_the_turret import _Stand, session, stand  # noqa: F401 -- fixtures
from tests.protocol_drives import run_identity

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
    held = session._activity_claim.try_claim('protocol', run=run_identity())
    assert held is not None
    yield
    held.release()


class TestWhileARunHoldsTheScope:
    def test_pressing_another_slot_is_one_microscope_busy_warning(
        self, stand, session, run_holds_the_scope, caplog, centre_posts
    ):
        home_sim_scope(session.scope)
        with caplog.at_level(logging.DEBUG):
            stand.turret_select(2)

        assert [(n.severity, n.title) for n in centre_posts] == [
            (Severity.WARNING, 'Microscope Busy')
        ]
        assert _records(caplog) == [(NOTIFICATIONS, logging.WARNING, False)]
        assert stand.popups == []
        # The display shows the turret still on slot 1.
        assert stand.ids['turret_pos_1_btn'].state == 'down'


def test_resetting_an_unknown_slot_is_one_objective_unknown_warning(
    stand, session, caplog, centre_posts
):
    with caplog.at_level(logging.DEBUG):
        stand.reset_turret_objective()

    assert [(n.severity, n.title) for n in centre_posts] == [
        (Severity.WARNING, 'Objective Unknown')
    ]
    assert 'home the scope' in centre_posts[0].message
    assert _records(caplog) == [(NOTIFICATIONS, logging.WARNING, False)]
    assert stand.popups == []


def test_a_refused_answer_is_shown_and_the_startup_question_still_resolves(
    stand, session, run_holds_the_scope, centre_posts
):
    home_sim_scope(session.scope)
    resolved = []
    stand._resolve_objective = lambda on_resolved: resolved.append(on_resolved)
    vc.VerticalControl._apply_objective_answer(stand, '4x Oly', 1, on_resolved='startup')

    assert [n.title for n in centre_posts] == ['Microscope Busy']
    assert resolved == ['startup']
    assert stand.popups == []


def _jog_from_the_gui(session, monkeypatch, axis):
    """Press a fine jog button for ``axis`` and wait until the lane has answered it."""
    from _lumascope.modules.sequential_io_executor import IOTask
    from ui import motion_settings

    ctx = _app_ctx.AppContext.over_session(session, lumaview=SimpleNamespace(scope=session.scope))
    # The panels the jog's redraw reads; the Kivy tree never runs here.
    ctx.motion_settings = SimpleNamespace(
        ids={'verticalcontrol_id': SimpleNamespace(update_gui=lambda **_: None)},
        update_xy_stage_control_gui=lambda: None,
    )
    monkeypatch.setattr(_app_ctx, 'ctx', ctx)
    # No running app: the controls are not locked, as on a scope nothing holds.
    monkeypatch.setattr(sys.modules['kivy.app'].App, 'get_running_app', lambda: None)
    if axis == 'Z':
        vc.VerticalControl._z_jog(SimpleNamespace(), +1, coarse=False)
    else:
        motion_settings.XYStageControl._xy_jog(SimpleNamespace(), axis, +1, coarse=False)
    assert ctx.io_executor.call(IOTask(action=lambda: None), 'settle', 30.0) is None


@pytest.mark.parametrize('axis', ['Z', 'X'])
def test_a_jog_with_the_objective_unknown_is_one_objective_unknown_warning(
    session, monkeypatch, axis, centre_posts
):
    # Slot 3 has no objective assigned, so the step cannot be sized.
    home_sim_scope(session.scope)
    session.scope.motion.move_turret(3)
    before = session.scope.motion.target_positions()

    _jog_from_the_gui(session, monkeypatch, axis)

    assert session.scope.motion.target_positions() == before
    assert [(n.severity, n.title) for n in centre_posts] == [
        (Severity.WARNING, 'Objective Unknown')
    ]
    assert 'turret slot 3 has no objective assigned' in centre_posts[0].message


@pytest.mark.parametrize('axis', ['Z', 'X'])
def test_a_jog_on_an_unhomed_scope_is_one_scope_not_homed_warning(
    session, monkeypatch, axis, centre_posts
):
    """The turret has no slot either, but the gate answers: a home is what helps."""
    before = session.scope.motion.target_positions()

    _jog_from_the_gui(session, monkeypatch, axis)

    assert session.scope.motion.target_positions() == before
    assert [(n.severity, n.title) for n in centre_posts] == [(Severity.WARNING, 'Scope Not Homed')]
