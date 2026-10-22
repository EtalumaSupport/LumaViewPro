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

The navigation tests drive the real go_to_step over the #733 stand.
"""

import ast
import pathlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


from tests.test_issue_733_step_nav_preview_led_button import (  # noqa: F401 -- stepnav_env is a fixture
    _make_step,
    stepnav_env,
)

_UI_DIR = pathlib.Path(__file__).resolve().parents[1] / 'ui'
_NAV_SRC = (_UI_DIR / 'step_navigation.py').read_text()
_NAV_TREE = ast.parse(_NAV_SRC)
_IMG_SRC = (_UI_DIR / 'image_settings.py').read_text()
_IMG_TREE = ast.parse(_IMG_SRC)


# The lanes each submitted LED command was queued on, in order.
_submitted_lanes = []


@pytest.fixture(autouse=True)
def _submits_run_at_once(monkeypatch):
    """The bodies submit their LED commands to the IO lane; run each at once."""

    def _submit_reported(call, redraw, label, *, lane=None):
        _submitted_lanes.append(lane)
        call()

    _submitted_lanes.clear()
    monkeypatch.setattr('ui.ui_helpers.submit_reported', _submit_reported)


def _find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f'{name} not found')


# ---------------------------------------------------------------------------
# Nav path: one authority transition + one direct apply, both preview states
# ---------------------------------------------------------------------------


def _run_nav(env, preview_on):
    """The real go_to_step for a person's click on step 0 of a one-step protocol."""
    import ui.step_navigation as step_navigation

    env.ctx.settings['protocol_led_on'] = preview_on
    env.ctx.motion_settings.ids['protocol_settings_id'].curr_step = 5
    step = {**_make_step(), 'Illumination': 250.0}
    protocol = SimpleNamespace(
        num_steps=MagicMock(return_value=1),
        step_row=MagicMock(return_value=step),
        step_list_revision=0,
    )
    # The panel shows this protocol: a completed move lands only on the
    # protocol and step list it was sent for.
    import modules.app_context as _app_ctx

    _app_ctx.ctx.motion_settings.ids['protocol_settings_id']._protocol = protocol
    env.ctx.scope.illumination.led_off = MagicMock()
    step_navigation.go_to_step(protocol, step_index=0, include_move=True)
    return env.ctx.scope.illumination, env.layer_obj


@pytest.mark.parametrize('preview_on', [True, False])
def test_nav_drives_no_camera_and_no_led_itself(stepnav_env, preview_on):
    """Whatever the preview state, the GUI's part is the one Session call,
    with no LED intent read from the button, no LED command and no settings
    apply of its own: the preview and the step's layer on the camera are the
    Session's, inside its go_to_step
    (tests/test_going_to_a_step_is_the_sessions_move.py)."""
    ill, layer_obj = _run_nav(stepnav_env, preview_on=preview_on)
    assert ill.apply_transition.call_count == 0
    assert ill.led_off.call_count == 0, 'nav must not queue its own led_off'
    assert stepnav_env.ctx.session.start_go_to_step.call_count == 1
    layer_obj.apply_settings.assert_not_called()
    assert layer_obj.ids['enable_led_btn'].state == 'normal'


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
