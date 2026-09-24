# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The stage map shows the crosshair again once the position is known.

A home puts every axis in HOMING, and the map's redraw then hides the
crosshair because the position is unknown. When the home ends where the
crosshair was last drawn -- a Home pressed at the centre ends at the centre
-- the redraw that follows must still draw it: the crosshair it would skip
redrawing is no longer on screen.

The redraw's skip compares the position against what it last drew, so these
drive it with a stub scope and stop it at the first step past the skip (the
labware lookup): reaching that step is "drawn", returning before it is
"skipped".
"""

from types import SimpleNamespace

import pytest

import modules.app_context as _app_ctx
import ui.stage as stage_module
from ui.stage import Stage

CENTRE = (60616.06, 40598.40)


class _DrawnError(Exception):
    """The redraw went past its skip and into the drawing."""


def _refuse_to_draw():
    raise _DrawnError


@pytest.fixture
def stage_and_motion(monkeypatch):
    motion = SimpleNamespace(homed=True)
    motion.has_homed = lambda: motion.homed
    motion.get_target_position = lambda axis: CENTRE['XY'.index(axis)]
    motion.get_current_position = lambda axis: CENTRE['XY'.index(axis)]
    monkeypatch.setattr(
        _app_ctx,
        'ctx',
        SimpleNamespace(
            settings={}, coordinate_transformer=None, scope=SimpleNamespace(motion=motion)
        ),
    )
    monkeypatch.setattr(stage_module, 'get_selected_labware', _refuse_to_draw)
    stage = Stage.__new__(Stage)
    stage._protocol_step_redraw = False
    stage._stage_limits_um = lambda: (1e9, 1e9)
    # The crosshair was last drawn at the centre.
    stage._prev_x_target, stage._prev_y_target = CENTRE
    stage._prev_x_current, stage._prev_y_current = CENTRE
    return stage, motion


def test_an_unchanged_known_position_is_not_redrawn(stage_and_motion):
    stage, _motion = stage_and_motion
    assert stage.draw_labware_io_calculations() is None


def test_after_the_position_was_unknown_the_same_position_is_drawn(stage_and_motion):
    stage, motion = stage_and_motion
    motion.homed = False  # homing: the redraw hides the crosshair
    with pytest.raises(_DrawnError):
        stage.draw_labware_io_calculations()
    motion.homed = True  # the home ended where the crosshair was last drawn
    with pytest.raises(_DrawnError):
        stage.draw_labware_io_calculations()
