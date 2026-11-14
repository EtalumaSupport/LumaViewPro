# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The focus slider spans the scope's own Z travel.

The travel is ``MotionAPI.axis_limits('Z')``; the slider's kv range
is a placeholder, a copy of one model's travel, until the panel asks.
An LS720 travels less than that copy, so a slider left on it offers a
range the API refuses to move into.
"""

from types import SimpleNamespace

import pytest

from ui.vertical_control import VerticalControl


@pytest.mark.parametrize('sim_session', [{'microscope': 'LS720'}], indirect=True)
def test_the_focus_slider_spans_the_scopes_z_travel(gui_ctx):
    travel = gui_ctx.scope.motion.axis_limits('Z')
    panel = VerticalControl.__new__(VerticalControl)
    # A stand-in by design: the Kivy slider, of which only the range is read.
    panel.ids = {'obj_position': SimpleNamespace(min=0.0, max=14000.0)}

    panel.show_z_travel()

    slider = panel.ids['obj_position']
    assert travel.max_um < 14000.0, "the LS720's travel is not the kv's copy"
    assert (slider.min, slider.max) == (travel.min_um, travel.max_um)
