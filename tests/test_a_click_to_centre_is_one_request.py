# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A right-click that centres the image asks for one X and Y move: refused, it moves neither.

The click converts its offset from the image centre into a distance on each
axis and asks the motion API for both as one request
(``MotionAPI.start_move_xy_by``). As two moves, X was driven and then Y
refused for its travel, leaving the stage at a place nobody asked for.

Driven through ``ScopeDisplay.touch`` on the simulated scope; the gesture's
lane submit is run inline (see ``_inline_gesture``).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import modules.app_context as _app_ctx
from modules import image_utils
from modules.exceptions import PositionOutOfRangeError
from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings
from ui import ui_helpers
from ui.scope_display import ScopeDisplay

# The displayed image, in widget pixels, and the sensor frame behind it.
_SHOWN = 100.0
_FRAME = 1000


@pytest.fixture
def scope(tmp_path, monkeypatch):
    session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    try:
        home_sim_scope(session.scope)
        monkeypatch.setattr(_app_ctx, 'ctx', SimpleNamespace(scope=session.scope))
        monkeypatch.setattr(ui_helpers, 'xy_gestures_offered', lambda: True)
        monkeypatch.setattr(ui_helpers, 'submit_gesture', _inline_gesture(session.scope))
        yield session.scope
    finally:
        session.shutdown()


def _inline_gesture(scope):
    """A stand-in by design: the gesture's moves run here, under the command
    the lane task would open, instead of on the GUI's lane, which needs Kivy."""

    def submit_gesture(label, *, axes, then, moves, on_moved=None):
        with scope.motion.one_command(axes, then=then):
            moves()

    return submit_gesture


def _click(dx_px: float, dy_px: float) -> None:
    """A right-click ``dx_px``, ``dy_px`` widget pixels from the centre of the shown image."""
    display = SimpleNamespace(
        norm_image_size=(_SHOWN, _SHOWN),
        center_x=_SHOWN / 2,
        center_y=_SHOWN / 2,
        full_resolution_frame_size=lambda: (_FRAME, _FRAME),
    )
    event = SimpleNamespace(
        is_touch=True,
        device='mouse',
        button='right',
        pos=(_SHOWN / 2 + dx_px, _SHOWN / 2 + dy_px),
    )
    ScopeDisplay.touch(display, None, event)


def _distance_um(scope, offset_px: float) -> float:
    pixel_um = scope.runtime_state.get_pixel_size_um()
    return image_utils.click_offset_to_um(_SHOWN / 2 + offset_px, _SHOWN, _FRAME, pixel_um)


def test_a_click_whose_y_is_past_travel_moves_neither(scope):
    motion = scope.motion
    dy = _distance_um(scope, 40.0)
    limits = motion.get_axis_limits('Y')
    # Y half the click's distance short of the end the click moves it past.
    y = limits.max_um - abs(dy) / 2 if dy > 0 else limits.min_um + abs(dy) / 2
    motion.move_xy(60_000.0, y)

    with pytest.raises(PositionOutOfRangeError) as refused:
        _click(10.0, 40.0)

    assert (refused.value.reason, refused.value.axis) == ('position_out_of_range', 'Y')
    motion.wait_until_finished_moving(timeout_s=10.0)
    assert (motion.get_target_positions()['X'], motion.get_target_positions()['Y']) == (60_000.0, y)


def test_a_click_inside_travel_moves_both_by_its_offsets(scope):
    motion = scope.motion
    motion.move_xy(60_000.0, 40_000.0)

    _click(10.0, 20.0)
    motion.wait_until_finished_moving(timeout_s=10.0)

    assert motion.get_target_positions()['X'] == pytest.approx(60_000.0 + _distance_um(scope, 10.0))
    assert motion.get_target_positions()['Y'] == pytest.approx(40_000.0 + _distance_um(scope, 20.0))
