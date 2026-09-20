# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A capture with no position records no position, rather than inventing one.

Three of the four save_image callers pass no position at all: the manual live
capture and both composite captures. They have nothing to pass -- a live
capture is not a protocol step and has no planned coordinate. The metadata
builder answered that by defaulting the missing value to zero and writing it,
so every one of those files carried a plate coordinate that reads downstream
exactly like a measured one. It is a real point on the plate, in a key whose
siblings are all measurements.

This file is the same contract the neighbouring fields already have.
generate_image_metadata omits pixel size when the scope cannot report a scale,
omits gain and exposure when the camera read failed, and omits the well label
on labware that has no wells -- each with the same reason written beside it: a
stand-in written here is read downstream as a real acquisition property. The
position was the one field that broke the rule its neighbours follow.

Absence has to survive the whole chain to be worth anything, so the reader is
pinned here too. It used to discard a file that was missing PositionX
entirely -- returning None, which sends the post-processing path to a fallback
that invents plate position, z, exposure, gain, illumination AND a pixel size
of 1.0. A file that honestly declines to state its position must not cost the
six facts it does state.
"""

import numpy as np
import pytest

import modules.image_save as image_save
import modules.image_utils as image_utils
from modules.labware_loader import WellPlateLoader

POSITION_KEYS = ('plate_pos_mm', 'x_pos', 'y_pos', 'z_pos_um')


@pytest.fixture
def positioned_scope(sim_scope):
    """A scope configured enough to save: the position is what is absent here,
    not the objective or the labware."""
    sim_scope.runtime_state.set_objective('4x Oly')
    sim_scope.runtime_state.set_labware(WellPlateLoader().get_plate('96 well microplate'))
    sim_scope.runtime_state.set_stage_offset({'x': 1000.0, 'y': 2000.0})
    return sim_scope


def _save(scope, folder, **position) -> str:
    return image_save.save_image(
        scope,
        np.zeros((8, 8), dtype=np.uint8),
        save_folder=str(folder),
        file_root='abs_',
        append='a',
        tail_id_mode=None,
        channel='BF',
        false_color_on=False,
        output_format='TIFF',
        save_encoding='8bit',
        significant_bits=8,
        **position,
    )


class TestNoPositionMeansNoPositionKeys:
    def test_metadata_omits_every_position_key(self, positioned_scope):
        """The defect: the missing value became a zero and the zero was
        written."""
        metadata = image_save.generate_image_metadata(
            positioned_scope,
            channel='BF',
            plate_x_mm=None,
            plate_y_mm=None,
            stage_z_um=None,
        )

        present = [k for k in POSITION_KEYS if k in metadata]
        assert present == [], (
            f'a capture with no position recorded {present} -- '
            f'{ {k: metadata[k] for k in present} }'
        )

    def test_the_rest_of_the_metadata_is_unaffected(self, positioned_scope):
        """Omitting the position costs nothing else: the file still states
        everything it does know."""
        metadata = image_save.generate_image_metadata(
            positioned_scope,
            channel='BF',
            plate_x_mm=None,
            plate_y_mm=None,
            stage_z_um=None,
        )

        assert metadata['channel'] == 'BF'
        assert metadata['pixel_size_um'] is not None
        assert metadata['objective']['magnification'] == 4

    def test_a_partial_position_keeps_what_it_has(self, positioned_scope):
        """Z alone is a real case -- a focus-only capture knows its depth and
        not its plate coordinate. What is known is written; what is not is
        absent. The two halves are independent."""
        metadata = image_save.generate_image_metadata(
            positioned_scope,
            channel='BF',
            plate_x_mm=None,
            plate_y_mm=None,
            stage_z_um=4950.0,
        )

        assert metadata['z_pos_um'] == pytest.approx(4950.0)
        assert 'plate_pos_mm' not in metadata

    def test_a_written_file_carries_no_position_fields(self, positioned_scope, tmp_path):
        """End to end: what a consumer reads off the file is silence, not a
        plate corner."""
        path = _save(positioned_scope, tmp_path)

        plane = _read_plane(path)
        for key in ('PositionX', 'PositionY', 'PositionZ'):
            assert key not in plane, (
                f'{key}={plane[key]} was written for a capture with no position'
            )


class TestAFileWithNoPositionIsStillReadable:
    def test_the_reader_keeps_the_facts_the_file_does_state(self, positioned_scope, tmp_path):
        """The regression this guards against: discarding the whole file for a
        missing position sends post-processing to a fallback that invents a
        pixel size of 1.0 -- a scale, measured off the derived output forever."""
        path = _save(positioned_scope, tmp_path)

        recovered = image_utils.read_postproc_input_metadata(path)

        assert recovered is not None, (
            'a file that honestly states no position was discarded entirely'
        )
        assert recovered['channel'] == 'BF'
        assert recovered['pixel_size_um'] != 1.0
        assert 'plate_pos_mm' not in recovered

    def test_a_file_with_a_position_still_round_trips(self, positioned_scope, tmp_path):
        """The other half of the contract, unchanged: a stated position is
        recovered exactly."""
        path = _save(
            positioned_scope, tmp_path, plate_x_mm=15.38, plate_y_mm=11.24, stage_z_um=4950.0
        )

        recovered = image_utils.read_postproc_input_metadata(path)

        assert recovered['plate_pos_mm']['x'] == pytest.approx(15.38)
        assert recovered['plate_pos_mm']['y'] == pytest.approx(11.24)
        assert recovered['z_pos_um'] == pytest.approx(4950.0)


def _read_plane(path) -> dict:
    """The Plane block a consumer reads back out of a written TIFF."""
    import json

    import tifffile

    with tifffile.TiffFile(str(path)) as tf:
        return json.loads(tf.pages[0].tags['ImageDescription'].value)['Plane']
