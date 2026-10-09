# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression for manual step navigation losing its LED preview (#697 sweep).

Manual nav to a different-layer step lit the preview through the LED
authority, then the accordion expansion primed the drawer reconcile, which
read the STORED enable-button state and queued the led_off that killed the
preview 32-70 ms later (serial-trace-proven). With the preview OFF, that
same reconcile was the only code applying the step's camera settings -- so
suppressing it wholesale was wrong (the rev-1 kill).

The contract now: the NAV PATH OWNS ITS ENTIRE OUTCOME. One authority
MANUAL_STEP transition covers both preview states (all-dark target when
the preview is off), fired only on a REAL step change; camera + histogram
are applied directly (protocol=False, update_led=False -- no early-return
leaves them to the reconcile, and no LED intent derives from the enable
button). The accordion reconcile stays for genuine user drawer clicks:
programmatic expansion raises a guard checked at FIRE time, set before the
mutation loop under try/finally, cleared on the next Clock tick.

The navigation tests drive the real go_to_step over the #733 stand. The
image-settings module is unimportable under the conftest kivy mocks, so the
accordion functions are carved out of source via AST and exec'd with
stubbed globals -- the real bodies run, not copies.
"""

import ast
import pathlib
import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, call

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
    monkeypatch.setitem(
        sys.modules, 'ui.ui_helpers', SimpleNamespace(submit_reported=_submit_reported)
    )


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
        step=MagicMock(return_value=step),
        step_list_revision=0,
    )
    # The panel shows this protocol: a completed move lands only on the
    # protocol and step list it was sent for.
    import modules.app_context as _app_ctx

    _app_ctx.ctx.motion_settings.ids['protocol_settings_id']._protocol = protocol
    env.ctx.scope.illumination.led_off = MagicMock()
    step_navigation.go_to_step(protocol, step_idx=0, include_move=True)
    return env.ctx.scope.illumination, env.layer_obj


# The one settings apply carries the no-button-LED contract.
_THE_APPLY = [call(update_led=False)]


@pytest.mark.parametrize('preview_on', [True, False])
def test_nav_applies_the_camera_once_and_drives_no_led_itself(stepnav_env, preview_on):
    """Whatever the preview state, the GUI's part is one settings apply with
    no LED intent read from the button, and no LED command of its own: the
    preview is the Session's, inside its go_to_step
    (tests/test_going_to_a_step_is_the_sessions_move.py)."""
    ill, layer_obj = _run_nav(stepnav_env, preview_on=preview_on)
    assert ill.apply_transition.call_count == 0
    assert ill.led_off.call_count == 0, 'nav must not queue its own led_off'
    assert stepnav_env.ctx.session.start_go_to_step.call_count == 1
    # Camera + histogram no longer depend on the accordion reconcile:
    # protocol=False runs the camera block and histogram sync;
    # update_led=False keeps the enable button out of it.
    assert layer_obj.apply_settings.call_args_list == _THE_APPLY
    assert layer_obj.ids['enable_led_btn'].state == 'normal'


# ---------------------------------------------------------------------------
# Accordion reconcile: fire-time guard for programmatic expansion
# ---------------------------------------------------------------------------


def _load_do_accordion_collapse():
    node = None
    for cls in ast.walk(_IMG_TREE):
        if isinstance(cls, ast.ClassDef) and cls.name == 'ImageSettings':
            for item in cls.body:
                if isinstance(item, ast.FunctionDef) and item.name == '_do_accordion_collapse':
                    node = item
    assert node is not None, 'ImageSettings._do_accordion_collapse not found'
    src = ast.get_source_segment(_IMG_SRC, node)
    # Method source is indented one class level; dedent for exec.
    src = '\n'.join(line[4:] if line.startswith('    ') else line for line in src.splitlines())
    return src


class _FakeAccordionItem:
    def __init__(self, collapse):
        self.collapse = collapse


class _FakeImageSettings:
    """Carries exactly the attributes the real method body reads."""

    def __init__(self, ctx, layers):
        self._ctx = ctx
        self._layers = layers  # {name: (accordion_item, layer_control)}
        self._suppress_reconcile_for_programmatic_expand = False
        self.ids = {'toggle_imagesettings': SimpleNamespace(state='down')}

    def accordion_item_lookup(self, layer):
        return self._layers[layer][0]

    def layer_lookup(self, layer):
        return self._layers[layer][1]


def _reconcile_harness(guard_set):
    ctx = SimpleNamespace(
        initializing=False,
        protocol_running=threading.Event(),
        session=SimpleNamespace(run_lockout=False),
        scope=SimpleNamespace(illumination=MagicMock()),
        io_executor=object(),
    )
    # 'Green' collapsed with its LED enabled (the channel the reconcile
    # would kill); 'Red' open (the layer it would apply).
    ctx.scope.illumination.get_led_state.return_value = {'enabled': True}
    layers = {
        'Green': (_FakeAccordionItem(collapse=True), MagicMock()),
        'Red': (_FakeAccordionItem(collapse=False), MagicMock()),
    }
    fake_self = _FakeImageSettings(ctx, layers)
    fake_self._suppress_reconcile_for_programmatic_expand = guard_set

    namespace = {
        'logger': MagicMock(),
        '_app_ctx': SimpleNamespace(ctx=ctx),
        'common_utils': SimpleNamespace(get_layers=lambda: list(layers)),
    }
    exec(_load_do_accordion_collapse(), namespace)
    do_collapse = namespace['_do_accordion_collapse']
    return do_collapse, fake_self, ctx, layers


def test_guard_set_at_fire_time_suppresses_the_reconcile():
    do_collapse, fake_self, ctx, layers = _reconcile_harness(guard_set=True)
    do_collapse(fake_self)
    assert ctx.scope.illumination.led_off.call_count == 0, (
        'a trigger primed by programmatic expansion must not kill the nav preview'
    )
    assert layers['Red'][1].apply_settings.call_count == 0, (
        'the reconcile apply must defer to the nav path'
    )


def test_guard_clear_runs_the_user_click_reconcile_as_today():
    do_collapse, fake_self, ctx, layers = _reconcile_harness(guard_set=False)
    do_collapse(fake_self)
    ctx.scope.illumination.led_off.assert_called_once_with('Green')
    assert _submitted_lanes == [ctx.io_executor]
    layers['Red'][1].apply_settings.assert_called_once_with()


def test_prime_then_clear_frame_order_suppresses_once_then_rearms():
    """One frame: mutations prime the trigger, THEN the clear is scheduled.
    Running the frame queue in that order must suppress the primed
    reconcile once and leave the next (user) fire live."""
    do_collapse, fake_self, ctx, _layers = _reconcile_harness(guard_set=True)
    frame_queue = [
        lambda: do_collapse(fake_self),  # the trigger primed by the mutations
        lambda: setattr(fake_self, '_suppress_reconcile_for_programmatic_expand', False),
    ]
    for event in frame_queue:
        event()
    assert ctx.scope.illumination.led_off.call_count == 0
    # Next frame: a genuine user click fires the trigger again.
    do_collapse(fake_self)
    ctx.scope.illumination.led_off.assert_called_once_with('Green')


def test_set_expanded_layer_pins_the_guard_ordering():
    """Source-structure pin: guard set BEFORE the mutation loop, mutations
    inside try, clear scheduled inside finally -- an exception mid-loop
    cannot wedge the guard, and the clear cannot precede the priming."""
    node = None
    for cls in ast.walk(_IMG_TREE):
        if isinstance(cls, ast.ClassDef) and cls.name == 'ImageSettings':
            for item in cls.body:
                if isinstance(item, ast.FunctionDef) and item.name == 'set_expanded_layer':
                    node = item
    assert node is not None

    guard_line = None
    try_node = None
    for stmt in ast.walk(node):
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == '_suppress_reconcile_for_programmatic_expand'
                ):
                    guard_line = stmt.lineno
        if isinstance(stmt, ast.Try):
            try_node = stmt
    assert guard_line is not None, 'set_expanded_layer never raises the guard'
    assert try_node is not None, 'the mutation loop is not wrapped in try/finally'
    assert guard_line < try_node.lineno, 'guard must be raised before the mutations'
    assert any(isinstance(s, ast.For) for s in ast.walk(try_node)), (
        'the mutation loop must sit inside the try'
    )
    final_src = '\n'.join(ast.get_source_segment(_IMG_SRC, s) for s in try_node.finalbody)
    assert 'Clock.schedule_once' in final_src, (
        'the guard clear must be SCHEDULED (next tick), not cleared inline -- '
        'an inline clear before the primed trigger fires re-ships the race'
    )


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
        'pass the target as step_idx instead: ' + '; '.join(offenders)
    )


def test_wrapper_go_to_step_requires_an_explicit_target():
    """step_idx must be a required parameter of the ProtocolSettings
    wrapper: a defaulted (or absent) target re-opens the silent path where
    a caller relies on a pre-written curr_step and the change comparison
    reads itself."""
    fn = _find_function(_PS_TREE, 'go_to_step')
    arg_names = [a.arg for a in fn.args.args]
    assert arg_names[:2] == ['self', 'step_idx'], arg_names
    required_count = len(fn.args.args) - len(fn.args.defaults)
    assert required_count >= 2, 'step_idx must have no default value'
