"""The Z box shows the Z the API knows, through every writer.

Each writer -- a position change, a redraw -- writes the box the same way,
and a Z the API does not know shows as ``--``. A box the user is typing in
holds the write until it is left; that is ``CommitInput``'s, pinned in
tests/test_a_box_commits_an_edit.py.
"""

import pytest

from tests.scope_fakes import home_sim_scope
from ui.vertical_control import VerticalControl


class _Box:
    """A stand-in by design for the Z box: the suite stubs Kivy, and these pin
    what the writers write, not the box's hold."""

    def __init__(self, text=''):
        self.text = text


class _Slider:
    def __init__(self, value=0.0):
        self.value = value
        self.user_interacting = False


@pytest.fixture
def control(gui_ctx):
    """A VerticalControl with only the two widgets these writers touch, over a
    simulated session whose Z was sent to 3774 um."""
    home_sim_scope(gui_ctx.scope)
    gui_ctx.scope.motion.move_z(3774.0)
    ctrl = VerticalControl.__new__(VerticalControl)
    ctrl.ids = {'z_position_id': _Box(), 'obj_position': _Slider()}
    return ctrl


WRITERS = [
    ('show_z', lambda c, p: c.show_z(p)),
    # The redraw writes the Z the API answers, which the fixture put at 3774.
    ('redraw_z', lambda c, p: c.redraw_z()),
]


@pytest.mark.parametrize('name,call', WRITERS, ids=[w[0] for w in WRITERS])
def test_every_writer_shows_the_z(control, name, call):
    control.ids['z_position_id'].text = ''

    call(control, 3774.0)

    assert control.ids['z_position_id'].text == '3774.00', f'{name} did not show the Z'


def test_a_z_with_no_known_position_is_shown_as_not_known(control):
    """While Z homes or is lost the API answers no position; the box says so, not a stale number."""
    control.ids['z_position_id'].text = '3774.00'

    control.show_z(None)

    assert control.ids['z_position_id'].text == '--'
