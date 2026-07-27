# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""#734 regression: Save Focus writes ONLY the selected step, never siblings.

Bug class: intent inferred from float equality. Save Focus propagated the
new Z to every step of the layer whose Z matched the previous saved focus
(within 1e-3 um). Because both protocol builders create every step of a
layer at the identical layer focus, equality was the DEFAULT state, not
evidence the user wanted the step updated -- and step navigation re-synced
the layer focus to the viewed step's Z, so the "baseline" always matched
the current step. Net effect (customer log): tuning and saving three steps
one after another collapsed all three to the last saved value.

Fix: the equality inference is deleted outright (no update_layer_focus, no
tolerance constant). Save Focus writes the layer default plus the Z of the
step selected AT CLICK TIME, and nothing else. Sibling steps can only be
changed through the explicit apply-to-all-steps-in-channel action.

Test approach
-------------
Kivy's BoxLayout is MagicMock'd in the test env, so LayerControl's real
method bodies are unreachable via attribute lookup. As in
test_layer_control_type_consistency.py, execute_save_focus is extracted
from the source with ast, compiled standalone, and driven against a
SimpleNamespace self + fake ctx holding a REAL Protocol -- the same
settings-write + step-write path production runs.
"""

from __future__ import annotations

import ast
import pathlib
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd


REPO = pathlib.Path(__file__).resolve().parent.parent
LAYER_CONTROL_SRC = REPO / 'ui' / 'layer_control.py'
PROTOCOL_SRC = REPO / 'modules' / 'protocol.py'


def _make_protocol_with_steps(rows: list[dict]):
    """Build a Protocol instance by direct-wiring the steps DataFrame."""
    import sys

    sys.path.insert(0, str(REPO))
    from modules.protocol import Protocol

    proto = Protocol.__new__(Protocol)
    proto._config = {
        'steps': pd.DataFrame(rows),
        'version': 1,
        'custom_step_count': 0,
    }
    proto._num_steps_cache = None
    return proto


def _extract_method(method_name: str, extra_globals: dict):
    """Compile a LayerControl method into a standalone callable."""
    tree = ast.parse(LAYER_CONTROL_SRC.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'LayerControl':
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    src = ast.unparse(child)
                    ns = dict(extra_globals)
                    exec(compile(src, f'<layer_control::{method_name}>', 'exec'), ns)
                    return ns[method_name]
    raise AssertionError(f'LayerControl.{method_name} not found in source')


def _make_env(proto, z_positions):
    """Fake ctx + extraction globals wired to a real Protocol.

    ``z_positions``: list of Z values get_current_position returns per call.
    Clock.schedule_once runs the callback immediately so the refresh path
    executes inside the test.
    """
    from modules.exceptions import ProtocolError

    ctx = SimpleNamespace(
        settings={
            'BF': {'focus': 7000.0},
            'Blue': {'focus': 7000.0},
        },
        settings_lock=threading.Lock(),
        protocol=proto,
        scope=SimpleNamespace(
            motion=SimpleNamespace(get_current_position=MagicMock(side_effect=z_positions))
        ),
        stage=MagicMock(),
        motion_settings=MagicMock(),
    )
    fake_app_ctx = SimpleNamespace(ctx=ctx)
    clock = SimpleNamespace(schedule_once=lambda cb, dt=0: cb(0))
    fn = _extract_method(
        'execute_save_focus',
        {
            '_app_ctx': fake_app_ctx,
            'logger': MagicMock(),
            'Clock': clock,
            'ProtocolError': ProtocolError,
        },
    )
    return fn, ctx


class TestSaveFocusSelectedStepOnly:
    def test_save_focus_leaves_sibling_steps_alone(self):
        proto = _make_protocol_with_steps(
            [
                {'Name': 'A1_BF', 'Color': 'BF', 'Z': 7000.0, 'X': 1, 'Y': 1},
                {'Name': 'B1_BF', 'Color': 'BF', 'Z': 7000.0, 'X': 1, 'Y': 1},
                {'Name': 'C1_BF', 'Color': 'BF', 'Z': 7000.0, 'X': 1, 'Y': 1},
            ]
        )
        fn, ctx = _make_env(proto, [5000.0])
        # Precondition that made the old inference fire: the saved layer
        # focus equals every sibling's Z, so all three sat "at baseline."
        assert ctx.settings['BF']['focus'] == 7000.0
        fake_self = SimpleNamespace(layer='BF')

        fn(fake_self, selected_step=0)

        steps = proto.steps()
        assert steps.loc[0, 'Z'] == 5000.0, 'selected step takes the new focus'
        assert steps.loc[1, 'Z'] == 7000.0, 'sibling step must NOT inherit the save'
        assert steps.loc[2, 'Z'] == 7000.0, 'sibling step must NOT inherit the save'
        assert ctx.settings['BF']['focus'] == 5000.0

    def test_repro_three_distinct_saves_stay_distinct(self):
        # The customer repro: tune + save each of three steps in turn. The
        # step-navigation sync sets the layer focus to the viewed step's Z
        # before each save; pre-fix that made every earlier save's value the
        # "baseline" and collapsed all steps to the LAST saved Z.
        proto = _make_protocol_with_steps(
            [
                {'Name': 'A1_BF', 'Color': 'BF', 'Z': 7000.0, 'X': 1, 'Y': 1},
                {'Name': 'B1_BF', 'Color': 'BF', 'Z': 7000.0, 'X': 1, 'Y': 1},
                {'Name': 'C1_BF', 'Color': 'BF', 'Z': 7000.0, 'X': 1, 'Y': 1},
            ]
        )
        saves = [4717.0, 4718.0, 4719.399]
        fn, ctx = _make_env(proto, saves)
        fake_self = SimpleNamespace(layer='BF')

        for i in range(3):
            ctx.settings['BF']['focus'] = float(proto.steps().loc[i, 'Z'])
            fn(fake_self, selected_step=i)

        assert list(proto.steps()['Z']) == saves, (
            'each step must keep its own saved focus; pre-fix all three '
            'collapsed to the last saved value'
        )

    def test_save_focus_ignores_selected_step_of_other_color(self):
        proto = _make_protocol_with_steps(
            [
                {'Name': 'A1_Blue', 'Color': 'Blue', 'Z': 7000.0, 'X': 1, 'Y': 1},
                {'Name': 'A1_BF', 'Color': 'BF', 'Z': 7000.0, 'X': 1, 'Y': 1},
            ]
        )
        fn, ctx = _make_env(proto, [5000.0])
        fake_self = SimpleNamespace(layer='BF')

        fn(fake_self, selected_step=0)

        steps = proto.steps()
        assert steps.loc[0, 'Z'] == 7000.0, 'selected Blue step untouched by a BF save'
        assert steps.loc[1, 'Z'] == 7000.0, 'unselected BF step untouched'
        assert ctx.settings['BF']['focus'] == 5000.0, 'layer default still saved'

    def test_stale_selected_step_is_skipped(self):
        # The index is captured at click time; the step can be deleted before
        # the queued task runs. The save must degrade to settings-only.
        proto = _make_protocol_with_steps(
            [
                {'Name': 'A1_BF', 'Color': 'BF', 'Z': 7000.0, 'X': 1, 'Y': 1},
            ]
        )
        fn, ctx = _make_env(proto, [5000.0])
        fake_self = SimpleNamespace(layer='BF')

        fn(fake_self, selected_step=5)

        assert proto.steps().loc[0, 'Z'] == 7000.0
        assert ctx.settings['BF']['focus'] == 5000.0

    def test_no_selection_saves_settings_only(self):
        proto = _make_protocol_with_steps(
            [
                {'Name': 'A1_BF', 'Color': 'BF', 'Z': 7000.0, 'X': 1, 'Y': 1},
            ]
        )
        fn, ctx = _make_env(proto, [5000.0])
        fake_self = SimpleNamespace(layer='BF')

        fn(fake_self, selected_step=-1)

        assert proto.steps().loc[0, 'Z'] == 7000.0
        assert ctx.settings['BF']['focus'] == 5000.0


class TestNoBaselineEqualityInferenceRemains:
    """Absence lock: the equality-inference machinery must stay deleted.

    This is a tripwire, not the guard itself -- the guard is that no
    propagation code exists and the bulk action takes no old-Z parameter,
    so the inference has no representation left. The tripwire catches the
    names coming back under refactoring pressure.
    """

    def test_inference_names_absent_from_protocol_and_layer_control(self):
        for path in (PROTOCOL_SRC, LAYER_CONTROL_SRC):
            src = path.read_text()
            for name in (
                'update_layer_focus',
                'step_at_layer_focus',
                'FOCUS_BASELINE_TOLERANCE_UM',
            ):
                assert name not in src, (
                    f'{name} found in {path.name}: baseline-equality '
                    'propagation must not be reintroduced'
                )
