# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A manual still records where the stage was when its frame was grabbed.

The still passed no position to the save, so every manual capture -- the
GUI's button, REST and scripts alike -- wrote a file with no PositionX,
PositionY or PositionZ although the stage knew all three. A defocus series
of a Ronchi ruling, taken one micrometre apart on the LS850T, could not be
reconstructed from its own files.

Driven through the Session on the simulator, read back through the
production reader: a homed LS850 records its plate X / Y and stage Z, the
overlay copy records the same, an axis that has lost its reference records
nothing for that axis, and an LS820 (Z only) records Z and no plate place.
"""

import pytest

from modules.image_utils import read_postproc_input_metadata
from modules.lumascope_api.motion import AxisState
from tests.test_manual_capture_member import _capture, _open_session, _settings

STAGE_X_UM = 40_000.0
STAGE_Y_UM = 30_000.0
STAGE_Z_UM = 5_000.0


def _session(tmp_path, microscope):
    settings = _settings(tmp_path)
    settings['microscope'] = microscope
    return _open_session(settings)


def _where_the_stage_is(scope):
    positions = scope.motion.axis_positions()
    to_plate = scope.runtime_state.plate_transform()
    x, y, z = (positions[ax].position for ax in ('X', 'Y', 'Z'))
    return to_plate(x, y), z


class TestAnXYZScope:
    @pytest.fixture
    def session(self, tmp_path):
        with _session(tmp_path, 'LS850') as session:
            motion = session.scope.motion
            motion.move_absolute('X', STAGE_X_UM)
            motion.move_absolute('Y', STAGE_Y_UM)
            motion.move_absolute('Z', STAGE_Z_UM)
            yield session

    def test_the_file_records_the_stage_position(self, session):
        (plate_x, plate_y), z = _where_the_stage_is(session.scope)
        (path,) = _capture(session)

        metadata = read_postproc_input_metadata(path)
        assert metadata['plate_pos_mm'] == pytest.approx({'x': plate_x, 'y': plate_y}, abs=1e-3)
        assert metadata['z_pos_um'] == pytest.approx(z, abs=1e-3)

    def test_the_overlay_copy_records_the_same_position(self, session):
        raw, overlay = _capture(session, crosshairs=True)

        raw_metadata = read_postproc_input_metadata(raw)
        overlay_metadata = read_postproc_input_metadata(overlay)
        assert 'z_pos_um' in raw_metadata
        assert overlay_metadata['plate_pos_mm'] == raw_metadata['plate_pos_mm']
        assert overlay_metadata['z_pos_um'] == raw_metadata['z_pos_um']

    def test_an_axis_without_its_reference_records_nothing_for_it(self, session):
        session.scope.motion._set_axis_state('Z', AxisState.UNKNOWN)
        (path,) = _capture(session)

        metadata = read_postproc_input_metadata(path)
        assert 'plate_pos_mm' in metadata
        assert 'z_pos_um' not in metadata


class TestAZOnlyScope:
    def test_the_file_records_z_and_no_plate_place(self, tmp_path):
        with _session(tmp_path, 'LS820') as session:
            session.scope.motion.move_absolute('Z', STAGE_Z_UM)
            z = session.scope.motion.axis_positions()['Z'].position
            (path,) = _capture(session)

        metadata = read_postproc_input_metadata(path)
        assert 'plate_pos_mm' not in metadata
        assert metadata['z_pos_um'] == pytest.approx(z, abs=1e-3)
