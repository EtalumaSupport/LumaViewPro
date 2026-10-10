"""The Z read-back box is never written while the user is typing in it.

The box commits its contents on focus loss (`on_focus: if not self.focus:
root.set_position(self.text)` in lumaviewpro.kv), so a read-back written
underneath a part-typed entry is not a display artefact -- it is committed
as a Z move when the user clicks away. Three of the four writers carried a
focus guard and the autofocus writer did not; these tests pin every writer.
"""

from types import SimpleNamespace

import pytest

from modules import app_context as _app_ctx
from ui.vertical_control import VerticalControl


class _Box:
    """Stands in for the Z TextInput: focus plus a settable text."""

    def __init__(self, focus=False, text=''):
        self.focus = focus
        self.text = text


class _Slider:
    def __init__(self, value=0.0):
        self.value = value
        self.user_interacting = False


@pytest.fixture
def control(monkeypatch):
    """A VerticalControl with only the two widgets these writers touch."""
    motion = SimpleNamespace(get_target_position=lambda axis: 3774.0)
    ctx = SimpleNamespace(lumaview=SimpleNamespace(scope=SimpleNamespace(motion=motion)))
    monkeypatch.setattr(_app_ctx, 'ctx', ctx)
    ctrl = VerticalControl.__new__(VerticalControl)
    ctrl.ids = {'z_position_id': _Box(), 'obj_position': _Slider()}
    return ctrl


WRITERS = [
    ('show_z', lambda c, p: c.show_z(p)),
    ('_update_z_position', lambda c, p: c._update_z_position(p)),
    (
        'update_text_only',
        lambda c, p: (setattr(c.ids['obj_position'], 'value', p), c.update_text_only()),
    ),
]


@pytest.mark.parametrize('name,call', WRITERS, ids=[w[0] for w in WRITERS])
def test_writer_does_not_overwrite_a_focused_box(control, name, call):
    """A focused box is mid-entry; its text is the user's, not ours."""
    control.ids['z_position_id'].focus = True
    control.ids['z_position_id'].text = '377'  # partially typed 3774.0

    call(control, 3774.0)

    assert control.ids['z_position_id'].text == '377', (
        f'{name} overwrote a part-typed entry; on focus loss the kv handler '
        f'commits that text as a Z move'
    )


@pytest.mark.parametrize('name,call', WRITERS, ids=[w[0] for w in WRITERS])
def test_writer_updates_an_unfocused_box(control, name, call):
    """The guard must not suppress the ordinary read-back."""
    control.ids['z_position_id'].focus = False
    control.ids['z_position_id'].text = ''

    call(control, 3774.0)

    assert control.ids['z_position_id'].text == '3774.00'


def test_negative_readback_is_floored_not_shown_negative(control):
    """The box has never shown a negative Z; the floor lives in one place now."""
    control.ids['z_position_id'].focus = False

    control._write_z_text(-5.0)

    assert control.ids['z_position_id'].text == '0.00'


def test_redundant_write_leaves_the_text_object_alone(control):
    """Rewriting the same string would churn the enclosing ScrollView."""
    box = control.ids['z_position_id']
    box.focus = False
    box.text = '3774.00'

    writes = []
    type(box).text = property(lambda self: self._t, lambda self, v: writes.append(v))
    box._t = '3774.00'
    try:
        control._write_z_text(3774.0)
    finally:
        del type(box).text

    assert writes == [], 'an identical value was written back to the widget'


def test_a_z_with_no_known_position_is_shown_as_not_known(control):
    """While Z homes or is lost the API answers no position; the box says so, not a stale number."""
    control.ids['z_position_id'].focus = False
    control.ids['z_position_id'].text = '3774.00'

    control.show_z(None)

    assert control.ids['z_position_id'].text == '--'
