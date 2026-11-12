# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A left click on the plate picture moves the stage there, or as near as it reaches.

The picture is drawn at the plate's size, wider than the stage reaches,
so a click in the strip along an edge is a point past the travel. It was
answered with a refusal popup; the click now asks the API for the nearest
point the stage reaches (``MotionAPI.start_move_to_plate_nearest``) and
the stage goes there. The API's own pin, at every stage offset, is
``tests/guards/test_plate_coordinate_authority.py``; this pins the GUI's
side: the click hands the API the point clicked, through that member.
"""

from types import SimpleNamespace

import _lumascope.modules.app_context as _app_ctx
import ui.ui_helpers as ui_helpers
from _lumascope.modules.coord_transformations import PlateLocation
from ui.stage import Stage


def test_a_click_past_the_travel_moves_the_stage_to_its_nearest_point(
    sim_turreted_session, monkeypatch
):
    """A click at the picture's bottom-left corner, plate (0, 0), is past
    both X's and Y's travel; the move it submits takes the stage to the
    travel's corner, with no refusal."""
    scope = sim_turreted_session.scope
    submitted = []

    def submit_move(label, *, axes, call, lane, on_moved=None):
        # A stand-in by design: the lane task runs here; the reporter it
        # hands an outcome to is submit_reported's, which needs Kivy.
        submitted.append((label, axes, call))

    monkeypatch.setattr(
        _app_ctx,
        'ctx',
        SimpleNamespace(scope=scope, session=sim_turreted_session, io_executor=None),
    )
    monkeypatch.setattr(ui_helpers, 'submit_move', submit_move)
    monkeypatch.setattr('ui.stage.xy_gestures_offered', lambda: True)
    picture = SimpleNamespace(
        x=0.0, y=0.0, width=100.0, height=100.0, collide_point=lambda *p: True
    )
    click = SimpleNamespace(button='left', pos=(0.0, 100.0))

    Stage.on_touch_down(picture, click)

    [(label, axes, call)] = submitted
    assert (label, axes) == ('STAGE_CLICK', ('X', 'Y'))
    went = call().result()
    limits = {axis: scope.motion.get_axis_limits(axis).max_um for axis in ('X', 'Y')}
    targets = scope.motion.get_target_positions()
    assert (targets['X'], targets['Y']) == (limits['X'], limits['Y'])
    on_plate = scope.runtime_state.stage_to_plate(limits['X'], limits['Y'])
    assert went == PlateLocation(x_mm=on_plate.x_mm, y_mm=on_plate.y_mm, z_um=None)
