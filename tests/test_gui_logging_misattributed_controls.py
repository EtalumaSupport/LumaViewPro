"""Two controls record what the person did with them, not a value that reads like it.

The step-number box puts an entry that is not a number back to the current
step, so its record has to say both what was typed and what the box went back
to; a number the protocol has no step for is not rewritten here (it is the
Session's refusal), so it carries no correction. The motion panel's toggle is
a ``ToggleButton``, whose ``state`` is ``'normal'`` or ``'down'``, both truthy:
its record has to come from the comparison, or closing the panel reads as
opening it.

That each control writes its own record name, rather than a neighbour's, is
the ratchet's roster (``tests/guards/test_gui_logging_ratchet.py``). These
tests drive each handler and read what landed on the ``LVP.gui_interactions``
logger.
"""

import logging
from types import SimpleNamespace

from ui.motion_settings import MotionSettings
from ui.protocol_settings import ProtocolSettings

_GUI_LOG = 'LVP.gui_interactions'


def _records(caplog):
    return [r.getMessage() for r in caplog.records if r.name == _GUI_LOG]


def _step_panel(typed):
    """The protocol panel's step box holding ``typed``, on step 3; the step moves it asks for."""
    steps = []
    box = SimpleNamespace(text=typed)

    def _show_current_step():
        box.text = '3'

    # a stand-in by design: Kivy is stubbed in the test process; the subject is the handler's record
    panel = SimpleNamespace(
        ids={'step_number_input': box},
        curr_step=2,
        update_step_ui_immediate=_show_current_step,
        update_step_ui=lambda: None,
        go_to_step=lambda step_idx: steps.append(step_idx),
    )
    return panel, steps


def test_step_number_reports_the_value_it_was_put_back_to(caplog):
    """'x' is not a number: nothing moves, and the record has the attempt, then the step shown.

    Fails if the unparseable branch drops its ``STEP_NUMBER_APPLIED`` emit.
    """
    caplog.set_level(logging.INFO, logger=_GUI_LOG)
    panel, steps = _step_panel('x')

    ProtocolSettings.handle_step_ui_input_change(panel)

    assert steps == []
    assert _records(caplog) == ['TEXT_INPUT STEP_NUMBER x', 'TEXT_INPUT STEP_NUMBER_APPLIED 3']


def test_a_step_number_the_protocol_lacks_is_handed_on_without_a_correction(caplog):
    """'99' is a number: the box is not rewritten here, so no ``_APPLIED`` line; the move is asked for.

    Fails if a correction is recorded on the number path (a clamp to the
    protocol's last step reintroduced in the widget).
    """
    caplog.set_level(logging.INFO, logger=_GUI_LOG)
    panel, steps = _step_panel('99')

    ProtocolSettings.handle_step_ui_input_change(panel)

    assert steps == [98]
    assert _records(caplog) == ['TEXT_INPUT STEP_NUMBER 99']


def test_a_togglebutton_site_compares_state_rather_than_passing_it(caplog):
    """Closing the motion panel (state 'normal') records OFF.

    Fails if the handler hands the raw ``state`` string to ``gui_logger.toggle``
    (the owner refuses it with ``TypeError``), or compares against the wrong
    state, which records ON.
    """
    caplog.set_level(logging.INFO, logger=_GUI_LOG)
    # a stand-in by design: Kivy is stubbed in the test process; the subject is the handler's record
    panel = SimpleNamespace(
        ids={
            'toggle_motionsettings': SimpleNamespace(state='normal'),
            'verticalcontrol_id': SimpleNamespace(update_gui=lambda: None),
            'protocol_settings_id': SimpleNamespace(select_labware=lambda: None),
        },
        settings_width=300,
        tab_width=30,
        x=0,
    )

    MotionSettings.toggle_settings(panel)

    assert _records(caplog) == ['TOGGLE MOTION_SETTINGS_PANEL OFF']
