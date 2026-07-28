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

The ui modules are unimportable under the conftest kivy mocks, so the
functions under test are carved out of source via AST and exec'd with
stubbed globals -- the real bodies run, not copies.
"""

import ast
import pathlib
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, call

from modules.lumascope_api.illumination import LedTransition, LedTransitionCtx

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
# Nav path: one authority transition + one direct apply, both preview states
# ---------------------------------------------------------------------------


def _load_nav_outcome():
    node = _find_function(_NAV_TREE, '_apply_manual_nav_outcome')
    namespace = {
        'LedTransition': LedTransition,
        'LedTransitionCtx': LedTransitionCtx,
        '_schedule_ui': lambda fn, _t: fn(0),
    }
    exec(ast.get_source_segment(_NAV_SRC, node), namespace)
    return namespace['_apply_manual_nav_outcome']


def _nav_doubles(preview_on):
    ctx = SimpleNamespace(scope=SimpleNamespace(illumination=MagicMock()))
    ctx.scope.illumination.color2ch.return_value = 7
    settings = {'protocol_led_on': preview_on}
    layer_obj = MagicMock()
    step = {'Illumination': 250.0}
    return ctx, settings, layer_obj, step


def _run_nav(preview_on, step_changed=True):
    apply_nav = _load_nav_outcome()
    ctx, settings, layer_obj, step = _nav_doubles(preview_on)
    apply_nav(
        ctx=ctx,
        settings=settings,
        layer_obj=layer_obj,
        step=step,
        color='Green',
        ignore_auto_gain=False,
        step_changed=step_changed,
    )
    return ctx.scope.illumination, layer_obj


def test_preview_on_nav_fires_one_authority_transition_and_no_button_read():
    ill, layer_obj = _run_nav(preview_on=True)
    assert ill.apply_transition_async.call_count == 1
    transition, led_ctx = ill.apply_transition_async.call_args.args
    assert transition is LedTransition.MANUAL_STEP
    assert led_ctx.preview_on is True
    assert led_ctx.channel == 7
    assert led_ctx.mA == 250.0
    assert ill.led_off_async.call_count == 0, 'nav must not queue its own led_off'
    # The one settings apply carries the no-button-LED contract; nothing
    # else on the layer object is touched (no enable_led_btn read).
    assert layer_obj.method_calls == [
        call.apply_settings(ignore_auto_gain=False, protocol=False, update_led=False)
    ]


def test_preview_off_nav_goes_dark_via_authority_and_still_applies_camera():
    ill, layer_obj = _run_nav(preview_on=False)
    assert ill.apply_transition_async.call_count == 1
    transition, led_ctx = ill.apply_transition_async.call_args.args
    assert transition is LedTransition.MANUAL_STEP
    assert led_ctx.preview_on is False, 'preview OFF must reach the authority as all-dark'
    # Camera + histogram no longer depend on the accordion reconcile:
    # protocol=False runs the camera block and histogram sync;
    # update_led=False keeps the enable button out of it.
    assert layer_obj.method_calls == [
        call.apply_settings(ignore_auto_gain=False, protocol=False, update_led=False)
    ]


def test_same_step_reselection_leaves_the_led_alone():
    ill, layer_obj = _run_nav(preview_on=True, step_changed=False)
    assert ill.apply_transition_async.call_count == 0, (
        're-selecting the current step must not re-drive the LED '
        '(a user-darkened channel stays dark)'
    )
    assert layer_obj.apply_settings.call_count == 1, 'camera settings still apply'


def test_go_to_step_no_longer_gates_the_authority_on_protocol_led_on():
    """Source pin: protocol_led_on decides only preview_on inside the nav
    outcome; it is no longer the gate for whether the authority runs."""
    go_src = ast.get_source_segment(_NAV_SRC, _find_function(_NAV_TREE, 'go_to_step'))
    assert 'protocol_led_on' not in go_src, (
        'go_to_step must not branch the authority call on protocol_led_on'
    )
    nav_src = ast.get_source_segment(
        _NAV_SRC, _find_function(_NAV_TREE, '_apply_manual_nav_outcome')
    )
    assert "preview_on=settings['protocol_led_on']" in nav_src


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
        scope=SimpleNamespace(illumination=MagicMock()),
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
    assert ctx.scope.illumination.led_off_async.call_count == 0, (
        'a trigger primed by programmatic expansion must not kill the nav preview'
    )
    assert layers['Red'][1].apply_settings.call_count == 0, (
        'the reconcile apply must defer to the nav path'
    )


def test_guard_clear_runs_the_user_click_reconcile_as_today():
    do_collapse, fake_self, ctx, layers = _reconcile_harness(guard_set=False)
    do_collapse(fake_self)
    ctx.scope.illumination.led_off_async.assert_called_once_with('Green')
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
    assert ctx.scope.illumination.led_off_async.call_count == 0
    # Next frame: a genuine user click fires the trigger again.
    do_collapse(fake_self)
    ctx.scope.illumination.led_off_async.assert_called_once_with('Green')


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
