"""A saved file records the position the capture was taken at, in one frame.

The protocol's steps hold plate millimetres. The saved file's PositionX /
PositionY are declared in millimetres and are read back as plate coordinates
forever. Between them sat a parameter named `x`, documented as stage
micrometres, which the writer filled with plate millimetres and the metadata
builder then converted a second time through `stage_to_plate`.

A three-step protocol at plate X 15.38 / 24.38 / 33.38 -- nine millimetres
apart, increasing -- recorded 122.2446 / 122.2356 / 122.2266: nine MICRONS
apart, and decreasing. The numbers looked like plate coordinates, which is why
this survived. Z was correct throughout, because a step's Z is already stage
micrometres and took no conversion.

These tests pin the FRAME rather than those numbers: a capture at a known plate
position records that plate position. A future change that reintroduces a
conversion anywhere along the chain fails here, whatever the plate geometry
happens to be.
"""

import numpy as np
import pytest

import modules.image_save as image_save
from modules.labware_loader import WellPlateLoader

PLATE_X_MM = 15.38
PLATE_Y_MM = 11.24
STAGE_Z_UM = 4950.0


@pytest.fixture
def positioned_scope(sim_scope):
    """A scope that can answer the position questions a save asks.

    generate_image_metadata refuses without an objective, labware and stage
    offset -- so a test about coordinates has to supply the frame they are
    expressed in. A non-zero stage offset is deliberate: it is the term that
    made the old double conversion visible, and a zero offset would let a
    half-fixed chain pass.
    """
    sim_scope.runtime_state.set_objective('4x Oly')
    sim_scope.runtime_state.set_labware(WellPlateLoader().get_plate('96 well microplate'))
    sim_scope.runtime_state.set_stage_offset({'x': 1000.0, 'y': 2000.0})
    return sim_scope


class TestThePlatePositionSurvivesTheSave:
    def test_metadata_records_the_plate_position_it_was_given(self, positioned_scope):
        """The defect itself: the value arrived in the frame the file records
        and was converted anyway."""
        metadata = image_save.generate_image_metadata(
            positioned_scope,
            channel='BF',
            plate_x_mm=PLATE_X_MM,
            plate_y_mm=PLATE_Y_MM,
            stage_z_um=STAGE_Z_UM,
            objective_id=positioned_scope.runtime_state.get_current_objective_id(),
        )

        assert metadata['plate_pos_mm']['x'] == pytest.approx(PLATE_X_MM), (
            f'recorded plate X is {metadata["plate_pos_mm"]["x"]}, not {PLATE_X_MM} -- '
            'the position was converted on its way into the file'
        )
        assert metadata['plate_pos_mm']['y'] == pytest.approx(PLATE_Y_MM), (
            f'recorded plate Y is {metadata["plate_pos_mm"]["y"]}, not {PLATE_Y_MM}'
        )

    def test_z_stays_stage_micrometres(self, positioned_scope):
        """Z was never broken, and the fix must not break it: the file's
        PositionZ is declared in micrometres and a step's Z already is."""
        metadata = image_save.generate_image_metadata(
            positioned_scope,
            channel='BF',
            plate_x_mm=PLATE_X_MM,
            plate_y_mm=PLATE_Y_MM,
            stage_z_um=STAGE_Z_UM,
            objective_id=positioned_scope.runtime_state.get_current_objective_id(),
        )

        assert metadata['z_pos_um'] == pytest.approx(STAGE_Z_UM)

    def test_two_captures_a_known_distance_apart_record_that_distance(self, positioned_scope):
        """The property no single reading can show. Nine millimetres apart on
        the plate is nine millimetres apart in the files -- in that direction.
        The old chain recorded the pair nine microns apart and inverted."""
        near = image_save.generate_image_metadata(
            positioned_scope,
            channel='BF',
            plate_x_mm=PLATE_X_MM,
            plate_y_mm=PLATE_Y_MM,
            stage_z_um=STAGE_Z_UM,
            objective_id=positioned_scope.runtime_state.get_current_objective_id(),
        )
        far = image_save.generate_image_metadata(
            positioned_scope,
            channel='BF',
            plate_x_mm=PLATE_X_MM + 9.0,
            plate_y_mm=PLATE_Y_MM,
            stage_z_um=STAGE_Z_UM,
            objective_id=positioned_scope.runtime_state.get_current_objective_id(),
        )

        delta = far['plate_pos_mm']['x'] - near['plate_pos_mm']['x']
        assert delta == pytest.approx(9.0), f'two captures 9 mm apart recorded {delta} mm apart'

    def test_the_duplicate_position_keys_agree_with_the_plate_position(self, positioned_scope):
        """`x_pos` / `y_pos` carry the same fact as `plate_pos_mm`. Two spellings
        of one value can drift apart; while both exist they answer the same."""
        metadata = image_save.generate_image_metadata(
            positioned_scope,
            channel='BF',
            plate_x_mm=PLATE_X_MM,
            plate_y_mm=PLATE_Y_MM,
            stage_z_um=STAGE_Z_UM,
            objective_id=positioned_scope.runtime_state.get_current_objective_id(),
        )

        assert metadata['x_pos'] == metadata['plate_pos_mm']['x']
        assert metadata['y_pos'] == metadata['plate_pos_mm']['y']


class TestTheWholeSaveChainCarriesOneFrame:
    def test_a_written_file_reports_the_plate_position_it_was_saved_at(
        self, positioned_scope, tmp_path
    ):
        """End to end through the real save: what a consumer measures off the
        file is what the caller said the capture was taken at."""
        path = image_save.save_image(
            positioned_scope,
            np.zeros((8, 8), dtype=np.uint8),
            save_folder=str(tmp_path),
            file_root='pos_',
            append='a',
            tail_id_mode=None,
            channel='BF',
            false_color_on=False,
            output_format='TIFF',
            save_encoding='8bit',
            significant_bits=8,
            plate_x_mm=PLATE_X_MM,
            plate_y_mm=PLATE_Y_MM,
            stage_z_um=STAGE_Z_UM,
            objective_id=positioned_scope.runtime_state.get_current_objective_id(),
        )

        plane = _read_plane(path)
        assert plane['PositionX'] == pytest.approx(PLATE_X_MM)
        assert plane['PositionY'] == pytest.approx(PLATE_Y_MM)
        assert plane['PositionZ'] == pytest.approx(STAGE_Z_UM)
        assert plane['PositionXUnit'] == 'mm'
        assert plane['PositionZUnit'] == 'um'


def _read_plane(path) -> dict:
    """The Plane block a consumer reads back out of a written TIFF."""
    import json

    import tifffile

    with tifffile.TiffFile(str(path)) as tf:
        return json.loads(tf.pages[0].tags['ImageDescription'].value)['Plane']
