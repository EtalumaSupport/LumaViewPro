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

_EMITTERS = {'button', 'toggle', 'slider', 'select', 'text_input'}


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


def test_step_number_reports_the_value_it_was_put_back_to():
    """The one path that rewrites the box -- an entry that is not a number,
    put back -- owes an _APPLIED line. A number the protocol has no step for
    is not rewritten: it is the Session's refusal, never a clamp."""
    fn = find_def(
        'ui/protocol_settings.py', 'handle_step_ui_input_change', class_name='ProtocolSettings'
    )
    applied = [
        c for c in _emitter_calls(fn) if c.args and 'STEP_NUMBER_APPLIED' in ast.unparse(c.args[0])
    ]
    assert len(applied) == 1, (
        f'this handler rewrites the box on the unparseable path only, found {len(applied)}'
    )
