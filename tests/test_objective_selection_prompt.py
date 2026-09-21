# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An unknowable objective gets ASKED about, never assumed silently.

Whether the objective is unknowable is the Session's decision (see
test_session_objective_question.py). This widget only renders the
question the Session returns, hands the choice back through
``confirm_objective`` and shows what happened -- and every failure on
that path becomes a notification, because it runs on Clock callbacks,
where a raise exits the app.
"""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import modules.app_context as _app_ctx
import ui.notification_popup as notification_popup
import ui.vertical_control as vc
from modules.exceptions import ConfigError
from modules.scope_session import ObjectiveQuestion
from tests.ast_seams import parse_module
from ui.vertical_control import VerticalControl

REPO_ROOT = Path(__file__).resolve().parents[1]

CHOICES = ('4x Oly', '10x Oly', '20x Oly')


class _ScriptedSession:
    """The Session as the renderer sees it: a question, and an answer."""

    def __init__(self, question=None, *, changed=True, provisional=False):
        self.question = question
        self.changed = changed
        self.provisional = provisional
        self.confirmed = []
        self.cleared = []
        self.is_protocol_running = False

    def objective_question(self):
        if isinstance(self.question, Exception):
            raise self.question
        return self.question

    def settings_are_provisional(self):
        # The renderer asks this before running a startup continuation on a
        # question that is not owed: while settings are provisional the host
        # re-asks later, so nothing may be hung on this pass.
        return self.provisional

    def confirm_objective(self, objective_id, turret_position=None):
        if isinstance(self.changed, Exception):
            raise self.changed
        self.confirmed.append((objective_id, turret_position))
        return self.changed

    def get_objective_info(self, objective_id):
        return {'magnification': 10, 'focal_length': 18.0}

    def clear_turret_objective(self, position):
        self.cleared.append(position)


class _Stand:
    """The real renderer methods; the widget tree and the FOV refresh stood in."""

    prompt_if_objective_unknown = VerticalControl.prompt_if_objective_unknown
    _render_objective_question = VerticalControl._render_objective_question
    _apply_objective_answer = VerticalControl._apply_objective_answer
    # Borrowed, not stubbed: it decides whether a startup step waiting on
    # the objective runs, and a stub here would answer for that.
    _resolve_objective = VerticalControl._resolve_objective
    reset_turret_objective = VerticalControl.reset_turret_objective
    _selected_turret_position = VerticalControl._selected_turret_position

    def __init__(self):
        self.ids = {'objective_spinner2': SimpleNamespace(text='')}
        for position in range(1, 5):
            self.ids[f'turret_pos_{position}_btn'] = SimpleNamespace(
                text=str(position), state='normal'
            )
        self.fov_refreshes = []
        self.turret_states = []

    def _refresh_fov(self, objective_id):
        self.fov_refreshes.append(objective_id)

    def update_all_turret_btn_states(self, position):
        self.turret_states.append(position)


class _ImmediateClock:
    """Kivy's Clock with the delay taken out, for tests that must see
    what a scheduled callback did."""

    @staticmethod
    def schedule_once(callback, timeout=0):
        callback(0)


class _Harness:
    def __init__(self, monkeypatch, session):
        self.session = session
        self.stand = _Stand()
        self.popups = []
        self.error_popups = []
        self.gui_log = []
        self.logger = MagicMock()
        monkeypatch.setattr(_app_ctx, 'ctx', SimpleNamespace(session=session))
        monkeypatch.setattr(
            notification_popup,
            'show_objective_selection_popup',
            lambda **kw: self.popups.append(kw),
        )
        monkeypatch.setattr(
            notification_popup, 'show_notification_popup', lambda **kw: self.error_popups.append(kw)
        )
        monkeypatch.setattr(
            vc.gui_logger,
            'select',
            lambda kind, value: self.gui_log.append(('SELECT', kind, value)),
        )
        monkeypatch.setattr(vc, 'logger', self.logger)
        monkeypatch.setattr(
            vc.gui_logger,
            'button',
            lambda name: self.gui_log.append(('BUTTON', name)),
        )
        # The reset handler SCHEDULES its follow-up rather than calling
        # it, so a test that never runs the callback cannot tell a live
        # trigger from a deleted one. Run it inline.
        monkeypatch.setattr(vc, 'Clock', _ImmediateClock)

    def prompt(self):
        self.stand.prompt_if_objective_unknown()

    def answer(self, chosen):
        assert len(self.popups) == 1
        self.popups[0]['on_confirm'](chosen)

    def info_lines(self):
        return [str(call.args[0]) for call in self.logger.info.call_args_list]


class TestTheQuestionIsRendered:
    def test_no_question_renders_nothing(self, monkeypatch):
        h = _Harness(monkeypatch, _ScriptedSession(None))
        h.prompt()
        assert h.popups == [] and h.error_popups == []

    def test_a_question_renders_once_with_its_choices_and_proposal(self, monkeypatch):
        question = ObjectiveQuestion(turret_position=2, proposed='10x Oly', choices=CHOICES)
        h = _Harness(monkeypatch, _ScriptedSession(question))
        h.prompt()
        assert len(h.popups) == 1
        popup = h.popups[0]
        assert popup['objectives'] == list(CHOICES)
        assert popup['current_objective_id'] == '10x Oly'
        assert 'turret position 2' in popup['message']

    def test_a_question_without_a_position_says_so(self, monkeypatch):
        question = ObjectiveQuestion(turret_position=None, proposed='4x Oly', choices=CHOICES)
        h = _Harness(monkeypatch, _ScriptedSession(question))
        h.prompt()
        assert 'turret position' not in h.popups[0]['message']

    def test_a_failed_query_is_one_notification_and_no_question(self, monkeypatch):
        h = _Harness(monkeypatch, _ScriptedSession(ConfigError('the objective catalogue is empty')))
        h.prompt()
        assert h.popups == []
        assert len(h.error_popups) == 1
        shown = h.error_popups[0]
        assert 'not confirmed' in shown['title']
        assert 'catalogue is empty' in shown['message'] and 'scale' in shown['message']


class TestTheAnswerReachesTheSession:
    def _question(self, position=2):
        return ObjectiveQuestion(turret_position=position, proposed='10x Oly', choices=CHOICES)

    def test_the_answer_goes_through_confirm_objective(self, monkeypatch):
        session = _ScriptedSession(self._question())
        h = _Harness(monkeypatch, session)
        h.prompt()
        h.answer('20x Oly')
        assert session.confirmed == [('20x Oly', 2)]
        assert h.stand.ids['objective_spinner2'].text == '20x Oly'
        assert h.stand.turret_states == [2]
        assert h.stand.ids['turret_pos_2_btn'].text == '10x'

    def test_a_changed_objective_logs_and_refreshes(self, monkeypatch):
        h = _Harness(monkeypatch, _ScriptedSession(self._question(), changed=True))
        h.prompt()
        h.answer('20x Oly')
        assert ('SELECT', 'OBJECTIVE', '20x Oly') in h.gui_log
        assert ('SELECT', 'TURRET_OBJECTIVE', '20x Oly') in h.gui_log
        assert any('select_objective()' in line for line in h.info_lines())
        assert h.stand.fov_refreshes == ['20x Oly']

    def test_an_unchanged_objective_binds_the_slot_and_nothing_else(self, monkeypatch):
        h = _Harness(monkeypatch, _ScriptedSession(self._question(), changed=False))
        h.prompt()
        h.answer('10x Oly')
        assert ('SELECT', 'OBJECTIVE', '10x Oly') not in h.gui_log
        assert ('SELECT', 'TURRET_OBJECTIVE', '10x Oly') in h.gui_log
        assert not any('select_objective()' in line for line in h.info_lines())
        assert h.stand.fov_refreshes == []

    def test_no_position_means_no_slot_rendering(self, monkeypatch):
        h = _Harness(monkeypatch, _ScriptedSession(self._question(position=None)))
        h.prompt()
        h.answer('20x Oly')
        assert h.stand.turret_states == []
        assert not any(kind == 'TURRET_OBJECTIVE' for _, kind, _ in h.gui_log)

    def test_a_raise_inside_the_answer_is_one_notification(self, monkeypatch):
        session = _ScriptedSession(self._question(), changed=ConfigError("unknown objective 'x'"))
        h = _Harness(monkeypatch, session)
        h.prompt()
        h.answer('x')
        assert len(h.error_popups) == 1
        assert 'unknown objective' in h.error_popups[0]['message']


def _method_calls(rel_path: str, class_name: str, method_name: str) -> set[str]:
    module = parse_module(rel_path)
    cls = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
    )
    names = set()
    for node in ast.walk(method):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.add(node.func.id)
    return names


class TestUnknownObjectiveEventsReachThePrompt:
    """Seam pins: the two mid-session generators of an unknowable
    objective must route into the prompt (behavior verified in the unit
    tests above and in the sim; these lock the wiring so a refactor
    cannot quietly drop a trigger)."""

    def test_turret_select_wires_the_prompt(self):
        calls = _method_calls('ui/vertical_control.py', 'VerticalControl', 'turret_select')
        assert 'prompt_if_objective_unknown' in calls

    def test_reset_turret_objective_does_not_wire_the_prompt(self):
        """The one trigger that must NOT exist.

        Clearing a slot is the user saying the slot is empty; asking
        them straight back which objective is in it, through a modal
        with no cancel path, is a question they have just answered --
        and it re-assigned the slot they had cleared, which made the
        button dead at every position it can reach. The two triggers
        above still ask, so nothing assumes an objective at a position
        the user has not been asked about.
        """
        calls = _method_calls('ui/vertical_control.py', 'VerticalControl', 'reset_turret_objective')
        assert 'prompt_if_objective_unknown' not in calls


class TestResetLeavesTheSlotCleared:
    """The behaviour the seam pin above protects."""

    def _reset_at(self, monkeypatch, position, question):
        h = _Harness(monkeypatch, _ScriptedSession(question))
        h.stand.ids[f'turret_pos_{position}_btn'].state = 'down'
        h.stand.reset_turret_objective()
        return h

    def test_a_reset_clears_the_slot_and_opens_no_popup(self, monkeypatch):
        # The session has a question to ask -- hardware present, settings
        # resolved -- which is exactly when the old trigger fired.
        question = ObjectiveQuestion(turret_position=3, proposed='10x Oly', choices=CHOICES)
        h = self._reset_at(monkeypatch, 3, question)
        assert h.session.cleared == [3]
        assert h.popups == []
        assert h.session.confirmed == []

    def test_the_cleared_slot_is_not_re_assigned(self, monkeypatch):
        question = ObjectiveQuestion(turret_position=2, proposed='4x Oly', choices=CHOICES)
        h = self._reset_at(monkeypatch, 2, question)
        # The re-assignment the dead button performed went through
        # confirm_objective; nothing may reach it from a reset.
        assert h.session.confirmed == []
        assert h.stand.ids['turret_pos_2_btn'].text == '2'

    def test_every_position_behaves_the_same(self, monkeypatch):
        for position in range(1, 5):
            question = ObjectiveQuestion(
                turret_position=position, proposed='10x Oly', choices=CHOICES
            )
            h = self._reset_at(monkeypatch, position, question)
            assert h.session.cleared == [position]
            assert h.popups == []


def test_template_ships_the_unconfirmed_flag():
    # The template default MUST be false: the flag existing with true
    # (or missing, treated as confirmed) would silently skip the one
    # prompt that makes a person confirm the shipped 20x default.
    # pin-justified: reads the shipped settings TEMPLATE (a JSON data
    # contract); no AST seam exists for a data file.
    template = json.loads((REPO_ROOT / 'data' / 'settings.json').read_text())
    assert template.get('objective_confirmed') is False
