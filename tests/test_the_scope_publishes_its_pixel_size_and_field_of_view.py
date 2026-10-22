# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The scope publishes its effective pixel size and field of view, so no caller recomputes them.

``scope.runtime_state.get_pixel_size_um()`` is the micrometres per pixel at
the current objective and the camera's binning, the scale every saved image
carries; ``get_field_of_view()`` is the area the camera's frame covers at
that scale, a record of width and height in micrometres. Each answers None
when it cannot be known: no objective known, no camera, or a binning or
frame the camera has not reported.

Each is checked against a still saved through the Session, on the simulated
LS850 and on the simulated Lumascope Classic (LS620), whose 2.2 um sensor
pitch is #747's.
"""

from __future__ import annotations

import pytest
import tifffile

from modules import common_utils
from modules.exceptions import CameraSettingRejected
from modules.image_utils import read_pixel_size_um
from modules.scope_capabilities import FieldOfView
from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings


def _session(tmp_path, **settings):
    return ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), **settings), simulate=True
    )


def _saved_still(session):
    (still,) = session.manual_capture.capture(false_color_on=False).result(timeout=30.0)
    return still.file


def _rounded(pixel_size_um):
    return round(pixel_size_um, common_utils.max_decimal_precision('pixel_size'))


# parametrize-justified: each model is its own simulated scope and optics.
@pytest.mark.parametrize('microscope', ['LS850', 'LS620'])
def test_each_answers_the_scale_and_extent_of_a_saved_still(tmp_path, microscope):
    session = _session(tmp_path, microscope=microscope)
    try:
        state = session.scope.runtime_state
        for size in session.scope.capabilities.camera_binning_sizes[:2]:
            session.camera.set_binning(size)
            path = _saved_still(session)
            height, width = tifffile.imread(path).shape[:2]

            pixel_size_um = state.get_pixel_size_um()
            assert _rounded(pixel_size_um) == read_pixel_size_um(path), (microscope, size)
            assert state.get_field_of_view() == FieldOfView(
                width_um=pixel_size_um * width, height_um=pixel_size_um * height
            ), (microscope, size)
    finally:
        session.shutdown()


def test_the_pixel_size_follows_the_binning(tmp_path):
    session = _session(tmp_path, microscope='LS850')
    try:
        at_one = session.scope.runtime_state.get_pixel_size_um()
        session.camera.set_binning(2)
        assert session.scope.runtime_state.get_pixel_size_um() == pytest.approx(2 * at_one)
    finally:
        session.shutdown()


def test_with_no_objective_known_each_is_none(tmp_path):
    # A turret on a slot with no objective assigned: nothing is in the light path.
    session = _session(
        tmp_path,
        microscope='LS850T',
        objective_confirmed=True,
        turret_position=1,
        turret_objectives={'1': '4x Oly', '2': None, '3': None, '4': None},
    )
    try:
        home_sim_scope(session.scope)
        session.scope.motion.move_turret(2)
        state = session.scope.runtime_state
        assert state.get_current_objective_id() is None
        assert (state.get_pixel_size_um(), state.get_field_of_view()) == (None, None)
    finally:
        session.shutdown()


def test_with_the_cameras_binning_or_frame_unknown_each_is_none(tmp_path):
    """The camera's part is its held binning and frame, None with no camera
    and after a write the camera never answered; the scope's optics are known."""
    session = _session(tmp_path, microscope='LS850')
    try:
        state = session.scope.runtime_state
        assert state.get_pixel_size_um() is not None
        # Unanswered twice: the write and the API's one resend.
        session.scope.simulation.refuse_camera_write('frame_size', 'no_answer', count=2)
        with pytest.raises(CameraSettingRejected):
            session.camera.set_frame_size(800, 600)
        assert state.get_pixel_size_um() is not None
        assert state.get_field_of_view() is None

        session.scope.simulation.refuse_camera_write('binning', 'no_answer', count=2)
        with pytest.raises(CameraSettingRejected):
            session.camera.set_binning(2)
        assert (state.get_pixel_size_um(), state.get_field_of_view()) == (None, None)
    finally:
        session.shutdown()
