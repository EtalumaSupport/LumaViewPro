"""The X and Y boxes show the plate position of the stage position the API knows, or ``--``.

``XYStageControl.update_gui`` is every redraw of the two boxes -- the
position listener's and the 1 Hz poll's alike. While the API does not know X
or Y (``axis_positions`` answers None: unhomed, homing, disconnected, the
board lost) both boxes say so, never a number converted from a position
nobody established. A box the user is typing in holds the write until it is
left; that is ``CommitInput``'s, pinned in tests/test_a_box_commits_an_edit.py.
"""

import pytest

from tests.scope_fakes import home_sim_scope
from ui.motion_settings import XYStageControl


class _Box:
    """A stand-in by design for a position box: the suite stubs Kivy, and
    these pin what the redraw writes, not the box's hold."""

    def __init__(self, text):
        self.text = text


def _control():
    ctrl = XYStageControl.__new__(XYStageControl)
    ctrl.ids = {'x_pos_id': _Box('12.50'), 'y_pos_id': _Box('34.50')}
    return ctrl


def _shown(ctrl):
    return ctrl.ids['x_pos_id'].text, ctrl.ids['y_pos_id'].text


def test_an_unhomed_stage_shows_as_not_known(gui_ctx):
    """Unhomed, X and Y are UNKNOWN and have no target: both boxes show
    ``--``, not a plate position."""
    scope = gui_ctx.scope
    assert scope.capabilities.has_xy_stage
    assert scope.motion.target_positions()['X'] is None
    ctrl = _control()

    ctrl.update_gui()

    assert _shown(ctrl) == ('--', '--')


def test_a_homed_stage_shows_its_plate_position(gui_ctx):
    """Homed, each box shows the plate position of where the stage is."""
    scope = gui_ctx.scope
    home_sim_scope(scope)
    stage = scope.motion.axis_positions()
    plate = scope.runtime_state.stage_to_plate(stage['X'].position_um, stage['Y'].position_um)
    ctrl = _control()

    ctrl.update_gui()

    assert _shown(ctrl) == (format(plate.x_mm, '.2f'), format(plate.y_mm, '.2f'))


def test_a_disconnected_stage_shows_as_not_known(gui_ctx):
    """After a disconnect every axis is UNKNOWN: the boxes stop showing the
    last position, where they used to keep it."""
    scope = gui_ctx.scope
    home_sim_scope(scope)
    ctrl = _control()
    ctrl.update_gui()
    assert '--' not in _shown(ctrl)

    scope.disconnect()
    ctrl.update_gui()

    assert _shown(ctrl) == ('--', '--')


@pytest.mark.parametrize('sim_session', [{'microscope': 'LS820'}], indirect=True)
def test_a_scope_without_an_xy_stage_draws_nothing(gui_ctx):
    """A Z-only scope has no X/Y travel: the boxes are left as they are."""
    assert not gui_ctx.scope.capabilities.has_xy_stage
    ctrl = _control()

    ctrl.update_gui()

    assert _shown(ctrl) == ('12.50', '34.50')
