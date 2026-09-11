"""``gui_logger.toggle`` refuses a state it cannot render correctly.

The emitter renders its second argument by truthiness. Kivy's two toggle-ish
widgets disagree about what that argument looks like: a ``CheckBox`` exposes
``active`` as a bool, a ``ToggleButton`` exposes ``state`` as ``'normal'`` or
``'down'`` -- and both strings are truthy. A caller passing ``widget.state``
through unconverted logs ``ON`` for every gesture, including the ones turning
the control OFF, and the resulting line is indistinguishable from a real press.

That contract was held by convention at every call site until a new one missed
it. These tests pin the refusal so the next caller is told at the call site
instead of producing a silently wrong record.
"""

import logging

import pytest

from modules import gui_logger


def test_a_bool_renders_on_and_off(caplog):
    """The happy path both directions -- the guard must not eat real calls."""
    with caplog.at_level(logging.INFO, logger='LVP.gui_interactions'):
        gui_logger.toggle('SCALE_BAR', True)
        gui_logger.toggle('SCALE_BAR', False)

    assert [r.getMessage() for r in caplog.records] == [
        'TOGGLE SCALE_BAR ON',
        'TOGGLE SCALE_BAR OFF',
    ]


@pytest.mark.parametrize('state', ['down', 'normal'])
def test_a_togglebutton_state_string_is_refused(state):
    """Both ToggleButton states are truthy, so neither may reach the renderer."""
    with pytest.raises(TypeError) as excinfo:
        gui_logger.toggle('TIMESTAMP_OVERLAY', state)

    message = str(excinfo.value)
    assert "state == 'down'" in message, (
        'the refusal must name the conversion, or the caller it stops has to '
        'go read the emitter to find out what it wanted'
    )
    assert 'TIMESTAMP_OVERLAY' in message, 'the refusal must name the control'


def test_a_refused_call_emits_no_record(caplog):
    """A rejected call must not also leave a half-right line behind."""
    with (
        caplog.at_level(logging.INFO, logger='LVP.gui_interactions'),
        pytest.raises(TypeError),
    ):
        gui_logger.toggle('TIMESTAMP_OVERLAY', 'down')

    assert caplog.records == []


def test_truthy_non_bools_are_refused_too():
    """The guard is about the TYPE, not about this one widget's strings."""
    for state in (1, 0, 'true', [], None):
        with pytest.raises(TypeError):
            gui_logger.toggle('SOME_CONTROL', state)
