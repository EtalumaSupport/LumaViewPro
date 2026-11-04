# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression for manual step navigation losing its LED preview (#697 sweep).

Manual nav to a different-layer step lit the preview through the LED
authority, then the accordion expansion primed the drawer reconcile, which
read the STORED enable-button state and queued the led_off that killed the
preview 32-70 ms later (serial-trace-proven). With the preview OFF, that
same reconcile was the only code applying the step's camera settings -- so
suppressing it wholesale was wrong (the rev-1 kill).

The contract now: the Session owns the step's whole outcome. Its go_to_step
lights the preview (all dark when the preview is off) and makes the step's
layer the shown channel; the GUI's part is one settings apply, which shows
the layer through show_channel, and no LED command of its own. The drawer
reconcile that killed the preview is gone: a drawer draws the shown channel,
and only a person's open asks the Session for another.

The navigation test drives the real go_to_step over the #733 fixture's
simulated session.
"""

import ast
import pathlib

import pytest

from modules.lumascope_api.illumination import LedState
from tests.scope_fakes import channel_reading
from tests.test_issue_733_step_nav_preview_led_button import (  # noqa: F401 -- stepnav_env is a fixture
    _run_manual_nav,
    stepnav_env,
)

_UI_DIR = pathlib.Path(__file__).resolve().parents[1] / 'ui'
_NAV_SRC = (_UI_DIR / 'step_navigation.py').read_text()
_NAV_TREE = ast.parse(_NAV_SRC)
_IMG_SRC = (_UI_DIR / 'image_settings.py').read_text()
_IMG_TREE = ast.parse(_IMG_SRC)


def _find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f'{name} not found')


# ---------------------------------------------------------------------------
# Nav path: the Session's preview stands, both preview states
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('preview_on', [True, False])
def test_nav_drives_no_camera_and_no_led_itself(stepnav_env, preview_on):
    """Whatever the preview state, the preview is the Session's, inside its
    go_to_step (tests/test_going_to_a_step_is_the_sessions_move.py): the
    step's channel is lit when the preview is on and every channel dark when
    it is off, as the click leaves it. The GUI reads no LED intent from the
    button, sends no LED command of its own (a led_off of its own killed the
    preview) and makes no settings apply of its own."""
    session = stepnav_env.session
    session.update_settings('protocol_led_on', preview_on)

    _run_manual_nav(stepnav_env)

    lit = [r.name for r in session.scope.illumination.get_led_states() if r.state is LedState.LIT]
    assert lit == (['BF'] if preview_on else [])
    if preview_on:
        bf = channel_reading(session.scope.illumination, 'BF')
        assert bf.illumination_ma == session.get_setting('BF.illumination_ma')
    stepnav_env.layer_obj.apply_settings.assert_not_called()
    assert stepnav_env.layer_obj.ids['enable_led_btn'].state == 'normal'


# ---------------------------------------------------------------------------
# Caller contract: nav callers state their target; no pre-write of curr_step
# ---------------------------------------------------------------------------

_PS_SRC = (_UI_DIR / 'protocol_settings.py').read_text()
_PS_TREE = ast.parse(_PS_SRC)


def _statement_blocks(fn):
    for node in ast.walk(fn):
        for field in ('body', 'orelse', 'finalbody'):
            stmts = getattr(node, field, None)
            if isinstance(stmts, list) and stmts and isinstance(stmts[0], ast.stmt):
                yield stmts


def _assigns_self_curr_step(stmt):
    # Direct assignment at THIS block level only: a write nested inside a
    # compound statement (e.g. prev_step's empty-protocol early return,
    # which cannot fall through to the navigation call) is scanned in its
    # own block by _statement_blocks and is legitimate bookkeeping there.
    if not isinstance(stmt, (ast.Assign, ast.AugAssign)):
        return False
    targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
    return any(
        isinstance(t, ast.Attribute)
        and t.attr == 'curr_step'
        and isinstance(t.value, ast.Name)
        and t.value.id == 'self'
        for t in targets
    )


def _calls_go_to_step(stmt):
    return any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == 'go_to_step'
        for n in ast.walk(stmt)
    )


def test_nav_handlers_never_prewrite_curr_step_before_navigating():
    """The bug shape: assign self.curr_step, then call go_to_step in the
    same statement block. The navigation module detects a real step change
    by comparing the target against curr_step -- a pre-write makes that
    comparison read itself (always equal) and the LED preview never fires.
    Bookkeeping writes in blocks that do NOT reach a go_to_step call (e.g.
    the empty-protocol early return in prev_step) are legitimate."""
    offenders = []
    for name in ('handle_step_ui_input_change', 'prev_step', 'next_step'):
        fn = _find_function(_PS_TREE, name)
        for stmts in _statement_blocks(fn):
            write_seen_at = None
            for idx, stmt in enumerate(stmts):
                if _assigns_self_curr_step(stmt):
                    write_seen_at = idx
                if _calls_go_to_step(stmt) and write_seen_at is not None:
                    offenders.append(
                        f'{name}: curr_step written at block stmt '
                        f'{write_seen_at}, go_to_step called at {idx}'
                    )
    assert offenders == [], (
        'nav handler pre-writes curr_step before calling go_to_step; '
        'pass the target as step_index instead: ' + '; '.join(offenders)
    )


def test_wrapper_go_to_step_requires_an_explicit_target():
    """step_index must be a required parameter of the ProtocolSettings
    wrapper: a defaulted (or absent) target re-opens the silent path where
    a caller relies on a pre-written curr_step and the change comparison
    reads itself."""
    fn = _find_function(_PS_TREE, 'go_to_step')
    arg_names = [a.arg for a in fn.args.args]
    assert arg_names[:2] == ['self', 'step_index'], arg_names
    required_count = len(fn.args.args) - len(fn.args.defaults)
    assert required_count >= 2, 'step_index must have no default value'
