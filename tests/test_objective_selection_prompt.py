# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An unknowable objective gets ASKED about, never assumed silently.

Two ways the app cannot know what is in the light path: no person has
ever confirmed the objective on this install (the settings template
ships a 20x default), or the light path sits on a turret position with
no assignment. Either way the pixel size stamped into the scale bar and
image metadata would be a fabrication -- the prompt makes the user the
source of truth at exactly the moment truth is missing.
"""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import modules.app_context as _app_ctx
from tests.ast_seams import parse_module
from ui.vertical_control import VerticalControl

REPO_ROOT = Path(__file__).resolve().parents[1]


class _PromptRecorder:
    maybe_prompt_objective_selection = VerticalControl.maybe_prompt_objective_selection

    def __init__(self):
        self.calls = []

    def prompt_objective_selection(self, turret_position=None, first_run=False):
        self.calls.append({'turret_position': turret_position, 'first_run': first_run})


def _ctx_with(settings):
    return SimpleNamespace(settings=settings)


class TestMaybePromptGate:
    def test_unconfirmed_install_prompts_with_position_on_turret_model(self, monkeypatch):
        monkeypatch.setattr(
            _app_ctx,
            'ctx',
            _ctx_with(
                {
                    'objective_confirmed': False,
                    'turret_position': 1,
                    'turret_objectives': {1: '20x Oly'},
                }
            ),
        )
        stand = _PromptRecorder()
        stand.maybe_prompt_objective_selection(model_has_turret=True)
        assert stand.calls == [{'turret_position': 1, 'first_run': True}]

    def test_unconfirmed_install_prompts_without_position_on_non_turret_model(self, monkeypatch):
        monkeypatch.setattr(_app_ctx, 'ctx', _ctx_with({'objective_confirmed': False}))
        stand = _PromptRecorder()
        stand.maybe_prompt_objective_selection(model_has_turret=False)
        assert stand.calls == [{'turret_position': None, 'first_run': True}]

    def test_confirmed_install_with_assigned_slot_stays_quiet(self, monkeypatch):
        monkeypatch.setattr(
            _app_ctx,
            'ctx',
            _ctx_with(
                {
                    'objective_confirmed': True,
                    'turret_position': 2,
                    'turret_objectives': {1: '20x Oly', 2: '4x Oly'},
                }
            ),
        )
        stand = _PromptRecorder()
        stand.maybe_prompt_objective_selection(model_has_turret=True)
        assert stand.calls == []

    def test_confirmed_install_on_unassigned_slot_prompts_for_it(self, monkeypatch):
        monkeypatch.setattr(
            _app_ctx,
            'ctx',
            _ctx_with(
                {
                    'objective_confirmed': True,
                    'turret_position': 3,
                    'turret_objectives': {1: '20x Oly', 3: None},
                }
            ),
        )
        stand = _PromptRecorder()
        stand.maybe_prompt_objective_selection(model_has_turret=True)
        assert stand.calls == [{'turret_position': 3, 'first_run': False}]

    def test_confirmed_non_turret_model_never_prompts(self, monkeypatch):
        monkeypatch.setattr(_app_ctx, 'ctx', _ctx_with({'objective_confirmed': True}))
        stand = _PromptRecorder()
        stand.maybe_prompt_objective_selection(model_has_turret=False)
        assert stand.calls == []


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
        assert 'prompt_objective_selection' in calls

    def test_reset_turret_objective_wires_the_prompt(self):
        calls = _method_calls('ui/vertical_control.py', 'VerticalControl', 'reset_turret_objective')
        assert 'prompt_objective_selection' in calls


def test_template_ships_the_unconfirmed_flag():
    # The template default MUST be false: the flag existing with true
    # (or missing, treated as confirmed) would silently skip the one
    # prompt that makes a person confirm the shipped 20x default.
    # pin-justified: reads the shipped settings TEMPLATE (a JSON data
    # contract); no AST seam exists for a data file.
    template = json.loads((REPO_ROOT / 'data' / 'settings.json').read_text())
    assert template.get('objective_confirmed') is False
