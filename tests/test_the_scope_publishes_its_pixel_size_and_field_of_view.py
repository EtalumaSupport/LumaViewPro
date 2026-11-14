# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The scope publishes its effective pixel size and field of view, so no caller recomputes them.

``scope.runtime_state.pixel_size_um()`` is the micrometres per pixel at
the current objective and the camera's binning, the scale every saved image
carries; ``field_of_view()`` is the area the camera's frame covers at
that scale, a record of width and height in micrometres. Each is refused,
naming its first cause, when it cannot be known: no objective known
(``ObjectiveUnknownError``), no camera (the camera's ``'not_connected'``), or
optics the scope does not declare, a binning or a frame the camera has not
reported (``PixelSizeUnknownError``); ``frame_point_distance``, the stage
distance to a point of the frame, is refused as the field of view is.
``optics_status()`` answers the objective and the field of view for a
display: each its value, or the refusal that says why it is unknown, never
raised.

Each is checked against a still saved through the Session, on the simulated
LS850 and on the simulated Lumascope Classic (LS620), whose 2.2 um sensor
pitch is #747's.
"""

from __future__ import annotations

import pytest
import tifffile
from fastapi.testclient import TestClient

from _lumascope.modules import common_utils
from _lumascope.modules.exceptions import (
    CameraSettingRejected,
    HardwareCommandRefusedError,
    ObjectiveUnknownError,
    PixelSizeUnknownError,
)
from _lumascope.modules.lumascope_api.simulation import SimulatedScope
from _lumascope.modules.image_utils import read_pixel_size_um
from _lumascope.modules.lumascope_api.runtime_state import OpticsStatus
from _lumascope.modules.scope_capabilities import FieldOfView
from _lumascope.modules.scope_parts import Part
from _lumascope.modules.scope_session import ScopeSession
from _lumascope.rest.app import build_app
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings


def _session(tmp_path, **settings):
    return ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), **settings), simulate=True
    )


def _saved_still(session):
    (still,) = session.manual_capture.start_save_still().result(timeout=30.0)
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
            session.scope.camera.set_binning(size)
            path = _saved_still(session)
            height, width = tifffile.imread(path).shape[:2]

            pixel_size_um = state.pixel_size_um()
            assert _rounded(pixel_size_um) == read_pixel_size_um(path), (microscope, size)
            assert state.field_of_view() == FieldOfView(
                width_um=pixel_size_um * width, height_um=pixel_size_um * height
            ), (microscope, size)
    finally:
        session.shutdown()


def test_the_pixel_size_follows_the_binning(tmp_path):
    session = _session(tmp_path, microscope='LS850')
    try:
        at_one = session.scope.runtime_state.pixel_size_um()
        session.scope.camera.set_binning(2)
        assert session.scope.runtime_state.pixel_size_um() == pytest.approx(2 * at_one)
    finally:
        session.shutdown()


def _refusals(state):
    """Each scale read's refusal, by read: its type and reason."""
    reads = {
        'pixel_size': state.pixel_size_um,
        'field_of_view': state.field_of_view,
        'frame_point_distance': lambda: state.frame_point_distance(0.5, 0.5),
    }
    refused = {}
    for name, read in reads.items():
        with pytest.raises(Exception) as raised:
            read()
        refused[name] = (type(raised.value), raised.value.reason)
    return refused


def test_with_no_objective_known_each_is_refused_naming_it(tmp_path):
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
        assert _refusals(state) == dict.fromkeys(
            ('pixel_size', 'field_of_view', 'frame_point_distance'),
            (ObjectiveUnknownError, 'slot_unassigned'),
        )
    finally:
        session.shutdown()


def test_with_the_cameras_binning_or_frame_unknown_each_is_refused_naming_it(tmp_path):
    """The camera's part is its held binning and frame, unknown after a write
    the camera never answered; the scope's optics are known."""
    session = _session(tmp_path, microscope='LS850')
    try:
        state = session.scope.runtime_state
        assert state.pixel_size_um() is not None
        # Unanswered twice: the write and the API's one resend.
        session.scope.simulation.refuse_camera_write('frame_size', 'no_answer', count=2)
        with pytest.raises(CameraSettingRejected):
            session.scope.camera.set_frame_size(800, 600)
        assert state.pixel_size_um() > 0
        with pytest.raises(PixelSizeUnknownError) as frame_unread:
            state.field_of_view()
        assert frame_unread.value.reason == 'frame_size_unread'

        session.scope.simulation.refuse_camera_write('binning', 'no_answer', count=2)
        with pytest.raises(CameraSettingRejected):
            session.scope.camera.set_binning(2)
        assert _refusals(state) == dict.fromkeys(
            ('pixel_size', 'field_of_view', 'frame_point_distance'),
            (PixelSizeUnknownError, 'binning_unread'),
        )
    finally:
        session.shutdown()


def test_with_no_camera_each_is_refused_naming_it(tmp_path):
    """A scope that came up with no camera has no binning or frame to measure
    by: each read is the camera's refusal, after the objective, which is known."""
    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), microscope='LS850'),
        simulate=True,
        simulated=SimulatedScope(camera_present=False),
    )
    try:
        assert session.objective_id
        assert _refusals(session.scope.runtime_state) == dict.fromkeys(
            ('pixel_size', 'field_of_view', 'frame_point_distance'),
            (HardwareCommandRefusedError, 'not_connected'),
        )
    finally:
        session.shutdown()


# --- optics_status: each value, or the refusal that says why, never raised ---


def _refusal(answer):
    """A refusal answered as a value, by its type and reason."""
    return type(answer).__name__, answer.reason


def test_the_optics_status_answers_the_objective_and_the_field_of_view(tmp_path):
    session = _session(tmp_path, microscope='LS850')
    try:
        state = session.scope.runtime_state
        assert state.optics_status() == OpticsStatus(
            objective=state.get_current_objective_id(), field_of_view=state.field_of_view()
        )
        assert isinstance(state.optics_status().field_of_view, FieldOfView)
    finally:
        session.shutdown()


# parametrize-justified: each is its own scope state, built on its own session.
@pytest.mark.parametrize(
    ('settings', 'turret_slot', 'reason'),
    [
        # An unhomed turret is in no known slot.
        ({'microscope': 'LS850T'}, None, 'slot_unknown'),
        (
            {
                'microscope': 'LS850T',
                'objective_confirmed': True,
                'turret_objectives': {'1': '4x Oly', '2': None, '3': None, '4': None},
            },
            2,
            'slot_unassigned',
        ),
        # No turret, and no person has confirmed the stored objective.
        ({'microscope': 'LS850', 'objective_confirmed': None}, None, 'none_selected'),
    ],
)
def test_an_unknown_objective_is_answered_as_its_refusal_in_both_fields(
    tmp_path, settings, turret_slot, reason
):
    session = _session(tmp_path, **settings)
    try:
        if turret_slot is not None:
            home_sim_scope(session.scope)
            session.scope.motion.move_turret(turret_slot)
        status = session.scope.runtime_state.optics_status()
        assert _refusal(status.objective) == ('ObjectiveUnknownError', reason)
        assert _refusal(status.field_of_view) == ('ObjectiveUnknownError', reason)
    finally:
        session.shutdown()


def test_with_no_camera_the_objective_is_known_and_the_field_of_view_is_the_cameras_refusal(
    tmp_path,
):
    session = _session(tmp_path, microscope='LS850')
    try:
        state = session.scope.runtime_state
        known = state.get_current_objective_id()
        session.scope.simulation.remove_camera()
        status = state.optics_status()
        assert status.objective == known
        assert _refusal(status.field_of_view) == ('HardwareCommandRefusedError', 'not_connected')
        assert status.field_of_view.missing.part == Part.CAMERA
    finally:
        session.shutdown()


def test_a_frame_or_binning_the_camera_has_not_reported_is_answered_as_its_refusal(tmp_path):
    session = _session(tmp_path, microscope='LS850')
    try:
        state = session.scope.runtime_state
        # Unanswered twice: the write and the API's one resend.
        session.scope.simulation.refuse_camera_write('frame_size', 'no_answer', count=2)
        with pytest.raises(CameraSettingRejected):
            session.scope.camera.set_frame_size(800, 600)
        assert _refusal(state.optics_status().field_of_view) == (
            'PixelSizeUnknownError',
            'frame_size_unread',
        )

        session.scope.simulation.refuse_camera_write('binning', 'no_answer', count=2)
        with pytest.raises(CameraSettingRejected):
            session.scope.camera.set_binning(2)
        assert _refusal(state.optics_status().field_of_view) == (
            'PixelSizeUnknownError',
            'binning_unread',
        )
    finally:
        session.shutdown()


def test_a_rest_client_reads_the_refusal_with_its_published_fields(tmp_path):
    session = _session(tmp_path, microscope='LS850')
    try:
        known = session.scope.runtime_state.get_current_objective_id()
        session.scope.simulation.remove_camera()
        with TestClient(build_app(session)) as client:
            answer = client.get('/api/v1/scope/runtime_state/optics_status')
        assert answer.status_code == 200
        body = answer.json()
        assert body['objective'] == known
        refused = body['field_of_view']
        assert (refused['kind'], refused['reason']) == ('refusal', 'not_connected')
        assert refused['missing']['part'] == 'camera'
    finally:
        session.shutdown()
