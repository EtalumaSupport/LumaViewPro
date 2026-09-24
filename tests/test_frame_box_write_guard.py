# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The frame boxes are never written while the user is typing in one.

The boxes commit their contents on focus loss (`on_focus: if not self.focus:
root.frame_size(...)` in lumaviewpro.kv), so a size written underneath a
part-typed entry is not a display artefact -- it is committed as a framing
change, and pushed to the camera, when the user clicks away.

Four writers touch these boxes: the settings restore at load, the binning
handler's synchronous commit, the delivered-size landing, and the restore of
an entry the handler could not parse. None of them carried a guard. This is
the same hazard `_write_z_text` closes for the Z box, and it is pinned the
same way -- behaviour for the guard, and a structural check that no writer
can grow back around it.
"""

from __future__ import annotations

import ast

import pytest

from tests.ast_seams import find_def, parse_module
from ui.microscope_settings import MicroscopeSettings

_FUNNEL = '_write_frame_text'
_BOXES = ('frame_width_id', 'frame_height_id')


class _Box:
    """Stands in for a frame TextInput: focus plus a settable text."""

    def __init__(self, focus=False, text=''):
        self.focus = focus
        self.text = text


@pytest.fixture
def panel():
    ctrl = MicroscopeSettings.__new__(MicroscopeSettings)
    ctrl.ids = {'frame_width_id': _Box(), 'frame_height_id': _Box()}
    return ctrl


def test_a_focused_box_keeps_what_the_user_is_typing(panel):
    """The box is mid-entry; its text is the user's, not ours."""
    panel.ids['frame_width_id'].focus = True
    panel.ids['frame_width_id'].text = '19'  # partially typed 1900

    panel._write_frame_text(768, 1200)

    assert panel.ids['frame_width_id'].text == '19', (
        'a part-typed frame entry was overwritten; on focus loss the kv '
        'handler commits that text and pushes it to the camera'
    )


def test_the_unfocused_twin_is_still_written(panel):
    """Guarding both boxes on one focus would strand the other.

    The boxes are edited one at a time. If a focused width box suppressed the
    height write too, the height would keep showing a size the camera is not
    at -- trading a commit hazard for a lying readout.
    """
    panel.ids['frame_width_id'].focus = True
    panel.ids['frame_width_id'].text = '19'
    panel.ids['frame_height_id'].focus = False

    panel._write_frame_text(768, 1200)

    assert panel.ids['frame_height_id'].text == '1200'


def test_an_unfocused_pair_takes_the_read_back(panel):
    """The guard must not suppress the ordinary read-back."""
    panel._write_frame_text(768, 1200)

    assert panel.ids['frame_width_id'].text == '768'
    assert panel.ids['frame_height_id'].text == '1200'


def test_a_redundant_write_leaves_the_text_object_alone(panel):
    """Rewriting the same string churns the enclosing ScrollView."""
    panel.ids['frame_width_id'].text = '768'
    panel.ids['frame_height_id'].text = '1200'

    writes = []
    # The property lands on the CLASS, so both boxes take it; seed each one's
    # backing value before the swap or the twin reads a missing attribute.
    for box in panel.ids.values():
        box._t = box.text
    type(panel.ids['frame_width_id']).text = property(
        lambda self: self._t, lambda self, v: writes.append(v)
    )
    try:
        panel._write_frame_text(768, 1200)
    finally:
        del type(panel.ids['frame_width_id']).text

    assert writes == [], f'an identical value was written back to the widget: {writes}'


def test_every_writer_goes_through_the_funnel():
    """A guard present at three sites and missing at the fourth is no guard."""
    tree = parse_module('ui/microscope_settings.py')
    funnel = find_def('ui/microscope_settings.py', _FUNNEL, class_name='MicroscopeSettings')
    assert funnel is not None, f'{_FUNNEL} is the seam these tests pin; it moved or was renamed'
    inside = {id(n) for n in ast.walk(funnel)}

    strays = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or id(node) in inside:
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and target.attr == 'text'
                and any(box in ast.unparse(target) for box in _BOXES)
            ):
                strays.append(f'line {node.lineno}: {ast.unparse(node)}')

    assert not strays, (
        'a frame box is written outside the focus-guarded funnel, so this '
        'writer can commit a size over a part-typed entry:\n  ' + '\n  '.join(strays)
    )


def test_the_funnel_actually_writes_the_boxes():
    """Guards the structural test above against passing because nothing writes."""
    funnel = find_def('ui/microscope_settings.py', _FUNNEL, class_name='MicroscopeSettings')
    writes = [
        node
        for node in ast.walk(funnel)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == 'text' for t in node.targets)
    ]
    assert writes, f'{_FUNNEL} no longer writes anything; the pin above is vacuous'
