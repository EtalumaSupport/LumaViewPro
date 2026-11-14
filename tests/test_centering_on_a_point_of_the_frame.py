# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Centering on a point of the frame is the API's: ``MotionAPI.start_center_on``.

The right-click on the live image used to read the pixel size itself and,
when the objective was unknown, log a line and do nothing: the stage stayed
and nobody was told why. It also converted the click with its own maths,
and the cursor's plate readout with a second copy. The member takes the
point as fractions of the camera's frame (row 0 at the top, as
``capture_still().image`` and a saved file hold it), refuses as a jog does, and moves X and
Y as one request under one command; ``RuntimeStateAPI.frame_point_distance``
is the one conversion both the move and the readout use, and
``RuntimeState.frame_point_plate`` the readout's plate position: the
position, or the refusal that says why it is unknown, never raised.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

import _lumascope.modules.app_context as _app_ctx
from _lumascope.modules.exceptions import (
    ArgumentRefusedError,
    ObjectiveUnknownError,
    PositionOutOfRangeError,
)
from _lumascope.modules.axes import AxisState
from _lumascope.modules.coord_transformations import PlatePosition
from _lumascope.modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings

# The simulated frame helpers live with the simulator's own image tests.
from tests.test_a_simulated_frame_shows_the_light_and_the_field_it_was_made_under import (
    _image_shift,
    _lit_frame,
    _session,
)
from ui import shader, ui_helpers
from ui.scope_display import ScopeDisplay

_TURRET = {'1': '4x Oly', '2': None, '3': None, '4': None}


@pytest.fixture
def turret_session(tmp_path):
    """A homed LS850T on slot 1, its 4x objective known."""
    session = ScopeSession.create(
        complete_settings(
            live_folder=str(tmp_path), microscope='LS850T', turret_objectives=_TURRET
        ),
        simulate=True,
    )
    try:
        home_sim_scope(session.scope)
        session.scope.motion.move_turret(1)
        session.scope.motion.move_xy(60_000.0, 40_000.0)
        yield session
    finally:
        session.shutdown()


def _xy(motion) -> tuple[float, float]:
    targets = motion.target_positions()
    return targets['X'], targets['Y']


def test_an_unassigned_slot_refuses_with_the_objective_s_reason_and_moves_nothing(turret_session):
    motion = turret_session.scope.motion
    motion.move_turret(2)
    before = _xy(motion)

    with pytest.raises(ObjectiveUnknownError) as refused:
        motion.start_center_on(0.9, 0.9)

    assert refused.value.reason == 'slot_unassigned'
    assert _xy(motion) == before


def test_an_unconfirmed_objective_with_no_turret_refuses_and_moves_nothing(tmp_path):
    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), objective_confirmed=None), simulate=True
    )
    try:
        home_sim_scope(session.scope)
        motion = session.scope.motion
        before = _xy(motion)

        with pytest.raises(ObjectiveUnknownError) as refused:
            motion.start_center_on(0.9, 0.9)

        assert refused.value.reason == 'none_selected'
        assert _xy(motion) == before
    finally:
        session.shutdown()


def test_a_point_moves_each_axis_by_its_fraction_of_the_field_from_the_centre(turret_session):
    scope = turret_session.scope
    field = scope.runtime_state.field_of_view()

    scope.motion.start_center_on(0.5, 0.5).result()
    assert _xy(scope.motion) == pytest.approx((60_000.0, 40_000.0), abs=0.01)

    # A point above the centre, row 0 at the top, is up the stage: +Y.
    scope.motion.start_center_on(0.8, 0.35).result()
    assert _xy(scope.motion) == pytest.approx(
        (60_000.0 + 0.3 * field.width_um, 40_000.0 + 0.15 * field.height_um), abs=0.01
    )


def test_a_point_whose_y_is_past_travel_moves_neither(turret_session):
    """One X and Y request: refused, it moves neither, never X then a refused Y."""
    motion = turret_session.scope.motion
    dy = turret_session.scope.runtime_state.frame_point_distance(0.6, 0.1).dy_um
    limits = motion.axis_limits('Y')
    # Y half the point's distance short of the end it would move past.
    y = limits.max_um - dy / 2
    motion.move_xy(60_000.0, y)

    with pytest.raises(PositionOutOfRangeError) as refused:
        motion.start_center_on(0.6, 0.1)

    assert (refused.value.reason, refused.value.axis) == ('position_out_of_range', 'Y')
    motion.wait_until_finished_moving(timeout_s=10.0)
    assert _xy(motion) == (60_000.0, y)


def test_a_fraction_outside_the_frame_or_not_a_number_is_refused_and_moves_nothing(turret_session):
    motion = turret_session.scope.motion
    before = _xy(motion)

    for x, y, reason, argument in (
        (1.2, 0.5, 'out_of_range', 'x_fraction'),
        (0.5, -0.1, 'out_of_range', 'y_fraction'),
        (math.nan, 0.5, 'not_a_number', 'x_fraction'),
    ):
        with pytest.raises(ArgumentRefusedError) as refused:
            motion.start_center_on(x, y)
        assert (refused.value.reason, refused.value.argument) == (reason, argument)
        with pytest.raises(ArgumentRefusedError):
            turret_session.scope.runtime_state.frame_point_distance(x, y)

    assert _xy(motion) == before


@pytest.mark.slow
def test_centering_on_a_point_of_the_api_s_frame_brings_what_was_there_to_the_centre():
    """Against the simulator's rendering of the stage: the content at (0.7, 0.6)
    of ``capture_still().image`` moves by a fifth of the width and a tenth of
    the height toward lower column and row, to the frame's centre."""
    with _session('LS850', homed=True) as session:
        scope = session.scope
        session.select_objective('10x Oly')
        before = _lit_frame(scope)
        scope.motion.start_center_on(0.7, 0.6).result()
        after = _lit_frame(scope)

    height, width = before.shape
    rows, cols = _image_shift(before, after)
    # Up to two pixels either way is the simulator's drift.
    assert abs(rows + 0.1 * height) <= 2.5 and abs(cols + 0.2 * width) <= 2.5, (rows, cols)


def test_the_plate_position_of_a_point_is_where_centering_on_it_brings_the_stage(turret_session):
    scope = turret_session.scope

    read = scope.runtime_state.frame_point_plate(0.75, 0.3)
    scope.motion.start_center_on(0.75, 0.3).result()
    x, y = _xy(scope.motion)
    reached = scope.runtime_state.stage_to_plate(x, y)

    assert (read.x_mm, read.y_mm) == pytest.approx((reached.x_mm, reached.y_mm), abs=1e-6)


def test_the_cursor_a_quarter_of_the_way_down_the_shown_image_is_that_far_down_the_frame():
    """Kivy's y counts up from the bottom and the frame's row 0 is its top:
    the cursor's fraction is the one ``start_center_on`` takes, and its pixel
    is that row of the frame."""
    # A stand-in by design: the subject is the widget's own conversion of a
    # window point; Kivy's widgets do not run in the test process.
    display = SimpleNamespace(
        texture=object(),
        norm_image_size=(100.0, 100.0),
        center_x=50.0,
        center_y=50.0,
        full_resolution_frame_size=lambda: (1000, 500),
    )
    viewer = SimpleNamespace(ids={'scope_display_id': display}, to_local=lambda x, y: (x, y))

    # Three quarters of the way up the window is a quarter of the way down.
    shader.ShaderViewer._on_mouse_pos(viewer, None, (25.0, 75.0))

    assert viewer._mouse_fraction == pytest.approx((0.25, 0.25))
    assert (viewer._mouse_pixel_x, viewer._mouse_pixel_y) == (250, 125)


def test_a_point_reads_a_plate_position_after_a_binning_change_before_a_new_frame_is_shown(
    turret_session,
):
    """A fraction is the same place on the sample at every binning: the shown
    frame and the camera's may differ, and the cursor's readout neither
    raises nor goes blank."""
    scope = turret_session.scope
    scope.camera.set_binning(2)

    assert isinstance(scope.runtime_state.frame_point_plate(0.99, 0.01), PlatePosition)


def test_the_point_at_the_frame_centre_is_where_the_stage_is(turret_session):
    scope = turret_session.scope
    stage = scope.motion.axis_positions()

    assert scope.runtime_state.frame_point_plate(0.5, 0.5) == scope.runtime_state.stage_to_plate(
        stage['X'].position_um, stage['Y'].position_um
    )


def _refusal(answer):
    """A refusal answered as a value, by its type and reason."""
    return type(answer).__name__, answer.reason


def test_a_point_on_a_stage_whose_x_and_y_are_not_known_answers_their_refusal(sim_session):
    """Unhomed, every axis is UNKNOWN while the position cache still answers
    its seed: the answer names X and Y, never a position converted from a
    stage position nobody established, and is raised to no one."""
    answer = sim_session.scope.runtime_state.frame_point_plate(0.5, 0.5)

    assert _refusal(answer) == ('AxisStateUnknownError', 'position_unknown')
    assert set(answer.axes_without_position) == {'X', 'Y'}


def test_a_point_with_only_z_unknown_still_has_a_plate_position(turret_session):
    scope = turret_session.scope
    # a stand-in by design: a simulator gap, the fast motor board fails no
    # home, so Z alone is put UNKNOWN by the state's own writer.
    scope.motion._set_axis_state('Z', AxisState.UNKNOWN)

    assert isinstance(scope.runtime_state.frame_point_plate(0.5, 0.5), PlatePosition)


# parametrize-justified: each model is its own simulated scope.
@pytest.mark.parametrize('microscope', ['LS820', 'LS620'])
def test_a_point_on_a_scope_with_no_xy_stage_answers_the_missing_part(tmp_path, microscope):
    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), microscope=microscope), simulate=True
    )
    try:
        answer = session.scope.runtime_state.frame_point_plate(0.5, 0.5)
    finally:
        session.shutdown()

    assert _refusal(answer) == ('HardwareCommandRefusedError', 'axis_absent')


def test_a_point_under_an_unknown_objective_answers_the_objective_s_refusal(turret_session):
    scope = turret_session.scope
    scope.motion.move_turret(2)

    answer = scope.runtime_state.frame_point_plate(0.5, 0.5)

    assert _refusal(answer) == ('ObjectiveUnknownError', 'slot_unassigned')


def test_a_fraction_outside_the_frame_is_the_caller_s_error_and_raised(turret_session):
    with pytest.raises(ArgumentRefusedError) as refused:
        turret_session.scope.runtime_state.frame_point_plate(1.5, 0.5)

    assert refused.value.reason == 'out_of_range'


def test_a_right_click_hands_the_api_the_point_and_the_api_s_refusal_reaches_the_reporter(
    turret_session, monkeypatch
):
    """The GUI says only where the click was; the call it submits is the
    member's, and its refusal is what the reporter is handed."""
    scope = turret_session.scope
    scope.motion.move_turret(2)
    submitted = []

    def submit_move(label, *, axes, call, lane, on_moved=None):
        # A stand-in by design: the lane task runs here; the reporter it
        # hands an outcome to is submit_reported's, which needs Kivy.
        submitted.append((label, axes, call))

    monkeypatch.setattr(_app_ctx, 'ctx', SimpleNamespace(scope=scope, io_executor=None))
    monkeypatch.setattr(ui_helpers, 'xy_gestures_offered', lambda: True)
    monkeypatch.setattr(ui_helpers, 'submit_move', submit_move)
    display = SimpleNamespace(norm_image_size=(100.0, 100.0), center_x=50.0, center_y=50.0)
    event = SimpleNamespace(is_touch=True, device='mouse', button='right', pos=(90.0, 75.0))

    ScopeDisplay.touch(display, None, event)

    [(label, axes, call)] = submitted
    assert (label, axes) == ('SCOPE_CLICK_TO_CENTER', ('X', 'Y'))
    with pytest.raises(ObjectiveUnknownError) as refused:
        call()
    assert refused.value.reason == 'slot_unassigned'
