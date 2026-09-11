"""Ten controls that were credited with another control's record now emit their own.

These are not the same failure as a control that logs nothing. Each of these
handlers reaches -- or shares a name with something that reaches -- a
``gui_logger`` emitter, so a census asking "does this control reach an emitter?"
counts them as logged. What comes out names a DIFFERENT control: pressing Next
Step wrote an LED toggle, typing an illumination value wrote an LED toggle, and
the two panel toggles were credited to same-named handlers on other classes.

A misattributed record is worse than a missing one. A gap leaves a question
open; a wrong line answers it incorrectly, and nothing downstream can tell it
from the real thing.

Two of the ten are genuine name collisions, and the pins below are written so
that re-introducing the collision fails rather than passes:

- ``toggle_settings`` is defined on MotionSettings AND on ImageSettings
- ``set_position`` is defined on ZStack AND on VerticalControl
"""

import ast

from tests.ast_seams import find_def

_EMITTERS = {'button', 'toggle', 'slider', 'select', 'text_input', 'text_input_debounced'}

# control id -> (module, class, handler)
_SITES = {
    'toggle_motionsettings': ('ui/motion_settings.py', 'MotionSettings', 'toggle_settings'),
    'zstack_spinner': ('ui/zstack.py', 'ZStack', 'set_position'),
    'prev_step_btn': ('ui/protocol_settings.py', 'ProtocolSettings', 'prev_step'),
    'next_step_btn': ('ui/protocol_settings.py', 'ProtocolSettings', 'next_step'),
    'step_number_input': (
        'ui/protocol_settings.py',
        'ProtocolSettings',
        'handle_step_ui_input_change',
    ),
    'tiling_size_apply_id': ('ui/protocol_settings.py', 'ProtocolSettings', 'apply_tiling'),
    'protocol_zstacking_apply_id': (
        'ui/protocol_settings.py',
        'ProtocolSettings',
        'apply_zstacking',
    ),
    'ill_text': ('ui/layer_control.py', 'LayerControl', 'ill_text'),
    'trendline_spinner': ('ui/post_processing.py', 'GraphingControls', 'update_trendline'),
    'text_cell_count_pixels_per_um_id': (
        'ui/post_processing.py',
        'CellCountControls',
        'log_pixels_per_um',
    ),
}


def _emitter_calls(fn):
    out = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.id if isinstance(f, ast.Name) else getattr(f, 'attr', None)
        if name in _EMITTERS:
            out.append(node)
    return out


def test_each_control_emits_a_record_from_its_own_handler():
    """Its OWN handler, not something transitively reached from it."""
    for control_id, (module, cls, handler) in _SITES.items():
        fn = find_def(module, handler, class_name=cls)
        assert fn is not None, f'{cls}.{handler} moved or was renamed'
        assert _emitter_calls(fn), (
            f'{control_id}: {cls}.{handler} emits nothing of its own, so operating '
            f'the control is credited to whatever record it happens to reach'
        )


def test_the_two_name_collisions_emit_distinguishable_records():
    """Same method name on two classes must not produce the same record name."""
    for handler, pairs in (
        (
            'toggle_settings',
            (
                ('ui/motion_settings.py', 'MotionSettings'),
                ('ui/image_settings.py', 'ImageSettings'),
            ),
        ),
        (
            'set_position',
            (('ui/zstack.py', 'ZStack'), ('ui/vertical_control.py', 'VerticalControl')),
        ),
    ):
        names = []
        for module, cls in pairs:
            fn = find_def(module, handler, class_name=cls)
            assert fn is not None, f'{cls}.{handler} moved; the collision pin is stale'
            for call in _emitter_calls(fn):
                if call.args:
                    names.append(ast.unparse(call.args[0]))
        assert len(names) >= 2, (
            f'both definitions of {handler} must emit, or the silent one is still '
            f'credited with the other class record'
        )
        assert len(set(names)) == len(names), (
            f'{handler}: the two classes emit the same record name {names}, so the '
            f'collision this fix exists to remove is still present in the bundle'
        )


def test_a_togglebutton_site_compares_state_rather_than_passing_it():
    """'normal' and 'down' are both truthy; the raw state would log ON forever."""
    fn = find_def('ui/motion_settings.py', 'toggle_settings', class_name='MotionSettings')
    toggles = [c for c in _emitter_calls(fn) if getattr(c.func, 'attr', None) == 'toggle']
    assert toggles, 'MotionSettings.toggle_settings no longer logs its panel'
    for call in toggles:
        assert len(call.args) == 2, f'toggle() needs a name and a state, got {len(call.args)} args'
        assert isinstance(call.args[1], ast.Compare), (
            "the state argument must be a comparison -- passing a ToggleButton's "
            'raw .state logs ON for every gesture including the ones turning it off'
        )


def test_the_trendline_record_is_not_emitted_for_a_programmatic_refresh():
    """set_x_axis/set_y_axis call this with axis=True; the kv spinner does not."""
    fn = find_def('ui/post_processing.py', 'update_trendline', class_name='GraphingControls')
    guarded = [node for node in ast.walk(fn) if isinstance(node, ast.If) and _emitter_calls(node)]
    assert guarded, 'the trendline record is unguarded and fires on axis refreshes'
    assert any(
        isinstance(g.test, ast.UnaryOp)
        and isinstance(g.test.op, ast.Not)
        and ast.unparse(g.test) == 'not axis'
        for g in guarded
    ), 'the guard must be `not axis` -- that is what separates a user selection from a refresh'


def test_step_number_reports_the_value_it_was_clamped_to():
    """Both coercing paths rewrite the box, so both owe an _APPLIED line."""
    fn = find_def(
        'ui/protocol_settings.py', 'handle_step_ui_input_change', class_name='ProtocolSettings'
    )
    applied = [
        c for c in _emitter_calls(fn) if c.args and 'STEP_NUMBER_APPLIED' in ast.unparse(c.args[0])
    ]
    assert len(applied) == 2, (
        'this handler rewrites the box on the unparseable path AND on the clamp '
        f'path; both owe a corrected-value line, found {len(applied)}'
    )
