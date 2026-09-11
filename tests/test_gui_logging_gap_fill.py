"""The thirteen controls that produced no gui_interactions record now produce one.

Each of these controls was operable by a user and left no line naming it in
``gui_interactions.log``. Ten are wired from ``ui/lumaviewpro.kv`` to a small
handler on the owning class; three already had a handler and gained the emitter
inside it.

These tests read the kv as TEXT. The suite mocks Kivy -- ``kivy.lang.parser`` is
not importable here -- so the real parse tree, which is what verified the wiring
when it was built, is out of reach at test time. The text scan is narrower: it
pins that each id's own block carries the expected binding, which is what would
regress if someone moved or dropped a line.

The pairing matters as much as the binding. A kv handler calls ``root.method()``,
where ``root`` is the enclosing rule's class -- so a method living on the wrong
class leaves the binding present and silently dead. Each case below therefore
asserts both halves: the binding is in the widget's block AND the method exists
on the class that owns that block.
"""

import ast
import re

from tests.ast_seams import REPO_ROOT, find_def

_KV = 'ui/lumaviewpro.kv'

# id -> (expected binding events, owning module, owning class, handler method)
_WIRED = {
    'zstack_stepsize_id': (
        ('on_text_validate', 'on_focus'),
        'ui/zstack.py',
        'ZStack',
        'log_step_field',
    ),
    'zstack_range_id': (
        ('on_text_validate', 'on_focus'),
        'ui/zstack.py',
        'ZStack',
        'log_step_field',
    ),
    'graph_title_input': (
        ('on_text_validate', 'on_focus'),
        'ui/post_processing.py',
        'GraphingControls',
        'log_text_commit',
    ),
    'x_axis_label_input': (
        ('on_text_validate', 'on_focus'),
        'ui/post_processing.py',
        'GraphingControls',
        'log_text_commit',
    ),
    'y_axis_label_input': (
        ('on_text_validate', 'on_focus'),
        'ui/post_processing.py',
        'GraphingControls',
        'log_text_commit',
    ),
    'video_gen_fps_id': (
        ('on_text_validate', 'on_focus'),
        'ui/post_processing.py',
        'VideoCreationControls',
        'log_video_gen_fps',
    ),
    'enable_timestamp_overlay_btn': (
        ('on_release',),
        'ui/post_processing.py',
        'VideoCreationControls',
        'log_timestamp_overlay',
    ),
    'zprojection_method_spinner': (
        ('on_text',),
        'ui/post_processing.py',
        'ZProjectionControls',
        'log_zprojection_method',
    ),
    'protocol_disable_image_saving_id': (
        ('on_release',),
        'ui/protocol_settings.py',
        'ProtocolSettings',
        'log_disable_image_saving',
    ),
    'logHistogram_id': (
        ('on_release',),
        'ui/layer_control.py',
        'LayerControl',
        'log_histogram_scale',
    ),
}

# handlers that already existed and gained the emitter inside them
_IN_HANDLER = {
    'exp_text': ('ui/layer_control.py', 'LayerControl', 'exp_text', 'text_input_debounced'),
    'video_recording_format_spinner': (
        'ui/microscope_settings.py',
        'MicroscopeSettings',
        'select_video_recording_format',
        'select',
    ),
    'apply_method_to_preview_image': (
        'ui/post_processing.py',
        'CellCountControls',
        'apply_method_to_preview_image',
        'button',
    ),
}


def _indent_width(line):
    prefix = line[: len(line) - len(line.lstrip(' \t'))]
    return len(prefix.replace('\t', '    '))


def _block_for(control_id):
    """The kv lines belonging to one widget: its id line and deeper siblings."""
    lines = (REPO_ROOT / _KV).read_text().split('\n')
    id_pat = re.compile(r'^[ \t]*id:\s*' + re.escape(control_id) + r'\s*$')
    hits = [i for i, ln in enumerate(lines) if id_pat.match(ln)]
    assert len(hits) == 1, f'{control_id}: expected exactly one id line, found {len(hits)}'
    start = hits[0]
    depth = _indent_width(lines[start])
    block = [lines[start]]
    for ln in lines[start + 1 :]:
        if not ln.strip():
            continue
        if _indent_width(ln) < depth:
            break
        block.append(ln)
    return block


def _called_names(fn):
    names = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                names.add(f.id)
            elif isinstance(f, ast.Attribute):
                names.add(f.attr)
    return names


def test_every_wired_control_binds_its_logging_handler():
    """The kv half: each control's own block carries the expected bindings."""
    for control_id, (events, _mod, _cls, method) in _WIRED.items():
        block = '\n'.join(_block_for(control_id))
        for event in events:
            assert re.search(rf'^[ \t]*{event}:.*{method}\b', block, re.M), (
                f'{control_id} lost its {event} binding to {method}; the control '
                f'would go back to producing no gui_interactions record'
            )


def test_every_wired_handler_lives_on_the_class_that_owns_the_block():
    """The python half: root.<method>() resolves, so the binding is not dead."""
    for control_id, (_events, module, cls, method) in _WIRED.items():
        assert find_def(module, method, class_name=cls) is not None, (
            f'{control_id} binds root.{method}(), where root is {cls}; that method '
            f'is not on {cls}, so the binding is present but silently does nothing'
        )


def test_the_three_existing_handlers_still_reach_their_emitter():
    """The controls whose handler predates this work still log from inside it."""
    for control_id, (module, cls, handler, callee) in _IN_HANDLER.items():
        fn = find_def(module, handler, class_name=cls)
        assert fn is not None, f'{cls}.{handler} moved or was renamed'
        assert callee in _called_names(fn), (
            f'{control_id}: {cls}.{handler} no longer calls {callee}, so operating '
            f'the control leaves no record naming it'
        )


def _record_name_arg(fn, callee):
    """The first positional argument of the call to ``callee`` inside ``fn``."""
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        f = node.func
        name = f.id if isinstance(f, ast.Name) else getattr(f, 'attr', None)
        if name == callee:
            return node.args[0]
    return None


def test_layer_owned_records_carry_the_channel_suffix():
    """A shared record name silently discards a line, so per-channel names differ.

    The debounce table is keyed by record name and cancels any pending line for
    that name, and LayerControl is instantiated once per channel with identical
    child ids. Two channels emitting the same literal name would collide and one
    of the two lines would never be written.

    So the assertion is on the ARGUMENT, not on the function text: the record
    name must be an f-string interpolating ``self.layer``. A test that merely
    looked for the word "layer" somewhere in the handler would pass on a plain
    string literal, which is the bug it is meant to catch.
    """
    for handler, callee in (
        ('exp_text', 'text_input_debounced'),
        ('log_histogram_scale', 'toggle'),
    ):
        fn = find_def('ui/layer_control.py', handler, class_name='LayerControl')
        assert fn is not None, f'LayerControl.{handler} moved or was renamed'

        arg = _record_name_arg(fn, callee)
        assert arg is not None, f'LayerControl.{handler} no longer calls {callee}'
        assert isinstance(arg, ast.JoinedStr), (
            f'LayerControl.{handler} passes a plain string as the record name; '
            f'every channel would emit the same name and collide'
        )
        interpolated = {
            ast.unparse(v.value) for v in arg.values if isinstance(v, ast.FormattedValue)
        }
        assert 'self.layer' in interpolated, (
            f'LayerControl.{handler} builds its record name without self.layer, '
            f'so two channel panels share it and one line is silently dropped; '
            f'found {sorted(interpolated)}'
        )


# --------------------------------------------------------------------------
# A coercing text box records what was typed AND what took effect.
#
# The bare record name means "what the user typed". A companion <NAME>_APPLIED
# line appears only when validation actually changed the value. The two names
# must differ: the debounce table is keyed by record name and cancels any
# pending line for that name, so reusing one name would discard the other.
# --------------------------------------------------------------------------

_HELPER = '_validate_and_apply_text_input'


def _debounced_calls(node):
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == 'text_input_debounced'
        and n.args
    ]


def test_the_shared_helper_records_the_typed_text_not_the_clipped_value():
    """The bare name carries what the user typed; clipping gets its own line."""
    fn = find_def('ui/layer_control.py', _HELPER, class_name='LayerControl')
    assert fn is not None, f'{_HELPER} is the seam these tests pin'

    bare = [c for c in _debounced_calls(fn) if isinstance(c.args[0], ast.Name)]
    assert bare, f'{_HELPER} no longer logs under a plain record name'
    for call in bare:
        assert ast.unparse(call.args[1]) == 'typed_text', (
            'the plain record name must carry the typed text; logging the clipped '
            'value there loses what the user actually entered'
        )


def test_the_shared_helper_reports_a_correction_under_its_own_name():
    """An _APPLIED line exists and is guarded, so unchanged input stays quiet."""
    fn = find_def('ui/layer_control.py', _HELPER, class_name='LayerControl')
    applied = [
        c
        for c in _debounced_calls(fn)
        if isinstance(c.args[0], ast.JoinedStr) and '_APPLIED' in ast.unparse(c.args[0])
    ]
    assert len(applied) == 2, (
        'expected exactly two _APPLIED emissions -- one for a rejected entry, one '
        f'for a clipped one; found {len(applied)}'
    )

    guarded = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.If)
        and any('_APPLIED' in ast.unparse(c.args[0]) for c in _debounced_calls(n) if c.args)
    ]
    assert guarded, (
        'the clipped-value line must be conditional; emitting it unconditionally '
        'would append a correction to every valid keystroke'
    )
    assert any(
        isinstance(g.test, ast.Compare) and ast.unparse(g.test) == 'raw != clipped' for g in guarded
    ), 'the guard must compare the PARSED values -- string compare reports 5 -> 5.0 as a correction'


def test_an_unparseable_entry_is_no_longer_invisible():
    """The reject path used to return without logging anything at all."""
    fn = find_def('ui/layer_control.py', _HELPER, class_name='LayerControl')
    handlers = [n for n in ast.walk(fn) if isinstance(n, ast.ExceptHandler)]
    assert handlers, f'{_HELPER} no longer has a parse-failure path'
    emitted = [c for h in handlers for c in _debounced_calls(h)]
    assert len(emitted) == 2, (
        'a rejected entry must record both the attempt and the value the box was '
        f'reset to; found {len(emitted)} emission(s) on the reject path'
    )


def test_the_zstack_log_binding_runs_before_the_handler_that_coerces():
    """Ordering is load-bearing: set_steps rewrites the box it is reading.

    ``set_steps`` coerces a bad extent to 0 and writes that back into the
    widget. ``log_step_field`` reads the widget to record what was typed, so it
    must be bound FIRST -- kv appends handlers in declaration order. Reversed,
    the "typed" record would carry the coerced 0 and the raw entry would be lost.
    """
    for control_id in ('zstack_stepsize_id', 'zstack_range_id'):
        block = _block_for(control_id)
        log_at = next(i for i, ln in enumerate(block) if 'log_step_field' in ln)
        set_at = next(i for i, ln in enumerate(block) if 'set_steps' in ln)
        assert log_at < set_at, (
            f'{control_id}: set_steps is bound before log_step_field, so the typed '
            f'value is overwritten before it is read'
        )
