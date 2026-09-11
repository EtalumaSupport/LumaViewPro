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

    def __init__(self, question=None, *, changed=True):
        self.question = question
        self.changed = changed
        self.confirmed = []
        self.is_protocol_running = False

    def objective_question(self):
        if isinstance(self.question, Exception):
            raise self.question
        return self.question

    def confirm_objective(self, objective_id, turret_position=None):
        if isinstance(self.changed, Exception):
            raise self.changed
        self.confirmed.append((objective_id, turret_position))
        return self.changed

    def get_objective_info(self, objective_id):
        return {'magnification': 10, 'focal_length': 18.0}


class _Stand:
    """The real renderer methods; the widget tree and the FOV refresh stood in."""

    prompt_if_objective_unknown = VerticalControl.prompt_if_objective_unknown
    _render_objective_question = VerticalControl._render_objective_question
    _apply_objective_answer = VerticalControl._apply_objective_answer

    def __init__(self):
        self.ids = {'objective_spinner2': SimpleNamespace(text='')}
        for position in range(1, 5):
            self.ids[f'turret_pos_{position}_btn'] = SimpleNamespace(text=str(position))
        self.fov_refreshes = []
        self.turret_states = []

    def _refresh_fov(self, objective_id):
        self.fov_refreshes.append(objective_id)

    def update_all_turret_btn_states(self, position):
        self.turret_states.append(position)


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

    def test_reset_turret_objective_wires_the_prompt(self):
        calls = _method_calls('ui/vertical_control.py', 'VerticalControl', 'reset_turret_objective')
        assert 'prompt_if_objective_unknown' in calls


def test_template_ships_the_unconfirmed_flag():
    # The template default MUST be false: the flag existing with true
    # (or missing, treated as confirmed) would silently skip the one
    # prompt that makes a person confirm the shipped 20x default.
    # pin-justified: reads the shipped settings TEMPLATE (a JSON data
    # contract); no AST seam exists for a data file.
    template = json.loads((REPO_ROOT / 'data' / 'settings.json').read_text())
    assert template.get('objective_confirmed') is False
