"""A typed numeric commit in a layer panel is logged, once, with its value.

Every numeric text box in the layer panel routes through one shared helper,
``LayerControl._validate_and_apply_text_input``. The helper -- not each of the
eight handlers -- owns the log line, so a new text box inherits logging by
routing through it. These tests pin the three properties that make that work:
the helper emits at all, it emits through the debounced text-input path rather
than the slider path, and every caller supplies an id the log name can be
derived from.
"""

import ast

from tests.ast_seams import find_def, parse_module

_MODULE = 'ui/layer_control.py'
_HELPER = '_validate_and_apply_text_input'


def _calls(fn):
    return [n for n in ast.walk(fn) if isinstance(n, ast.Call)]


def _helper():
    fn = find_def(_MODULE, _HELPER, class_name='LayerControl')
    assert fn is not None, f'{_HELPER} is the seam these tests pin; it moved or was renamed'
    return fn


def test_the_helper_emits_a_log_line():
    """Positive pin: the helper reaches the debounced text-input emitter."""
    names = {n.func.id for n in _calls(_helper()) if isinstance(n.func, ast.Name)}
    assert 'text_input_debounced' in names, (
        'the shared helper no longer logs; a typed numeric commit would leave '
        'no line in gui_interactions.log'
    )


def test_the_helper_does_not_emit_through_the_slider_verb():
    """A typed commit and a drag on the same setting must stay distinguishable.

    The slider twin already emits SLIDER <NAME>_<layer>; if the helper used the
    same emitter the bundle could not tell a keystroke from a drag.
    """
    fn = _helper()
    slider_emits = [
        n
        for n in _calls(fn)
        if isinstance(n.func, ast.Attribute)
        and n.func.attr == 'slider'
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == 'gui_logger'
    ]
    assert not slider_emits, 'the helper emits SLIDER for a typed commit'


def test_the_helper_takes_no_optional_log_name():
    """The log name is derived from the widget id, never passed.

    An optional name parameter defaulted to None is what silenced this emitter
    before: every caller omitted it, so the guarded call never ran.
    """
    args = _helper().args
    names = {a.arg for a in list(args.args) + list(args.kwonlyargs)}
    assert 'gui_log_name' not in names, (
        'a log-name parameter is back; an optional one goes unpassed and '
        'silences the line, and a required one can drift from the widget id'
    )


def test_every_caller_supplies_a_derivable_text_id():
    """The derivation contract: each call site's first argument ends in '_text'.

    The helper raises on a non-conforming id, so this test is the build-time
    half of that contract -- it fails in CI rather than under a user's cursor.
    """
    tree = parse_module(_MODULE)
    sites = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == _HELPER
    ]
    assert sites, 'no call sites found; the helper or its callers moved'
    bad = [
        n.args[0].value
        for n in sites
        if n.args
        and isinstance(n.args[0], ast.Constant)
        and isinstance(n.args[0].value, str)
        and not n.args[0].value.endswith('_text')
    ]
    assert not bad, f'call sites whose text_id has no _text suffix: {bad}'
