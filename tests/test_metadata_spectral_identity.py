# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Spectral identity reaches saved metadata; unknown stays absent.

Two units of one model can carry different filtersets, so the files a
unit produces must say which filterset and excitation produced them --
the model and serial fields alone cannot. The identity trio
(channel_display, excitation_nm, filterset) comes from the resolved
layer record. `channel` itself is untouched: it is a serialisation
surface with production readers.

The absent-if-unknown idiom extends to known-null: a broadband layer
has no excitation, and a record-less channel ('Composite', or an
unresolved identity) has no display name -- the keys are absent, never
null and never fabricated. The same idiom now covers illumination_ma,
which used to write a fabricated 0 mA when the LED was off or the
board absent. led_channel is a per-unit board address that changes
with a rewire or board swap; it must never reach metadata.
"""

from pathlib import Path

import pytest

from modules import image_save, image_utils
from modules.labware_loader import WellPlateLoader
from modules.layer_record import UNRESOLVED, LayerIdentity, LayerRecord

PLATE = '24 well microplate'

_BF = LayerRecord(
    id=0, key_name='BF', display_name='Brightfield', led_channel=(3,), excitation_nm=None
)
_GREEN = LayerRecord(
    id=4, key_name='Green', display_name='Green', led_channel=(1,), excitation_nm=488.0
)

_IDENTITY = LayerIdentity(layers=(_BF, _GREEN), filterset='FS-STOCK', source='motorconfig')
_IDENTITY_NO_FILTERSET = LayerIdentity(layers=(_BF, _GREEN), filterset='', source='scopes')


@pytest.fixture
def metadata_scope(sim_scope):
    loader = WellPlateLoader()
    sim_scope.runtime_state.set_objective('20x Oly')
    sim_scope.runtime_state.set_labware(loader.get_plate(PLATE))
    sim_scope.runtime_state.set_stage_offset({'x': 0.0, 'y': 0.0})
    sim_scope.layer_identity = _IDENTITY
    return sim_scope


def _metadata(scope, channel):
    return image_save.generate_image_metadata(scope, channel, 0, 0, 0)


class TestSpectralTrio:
    def test_fluorescence_capture_carries_the_trio(self, metadata_scope):
        metadata = _metadata(metadata_scope, 'Green')
        assert metadata['channel'] == 'Green'
        assert metadata['channel_display'] == 'Green'
        assert metadata['excitation_nm'] == 488.0
        assert metadata['filterset'] == 'FS-STOCK'

    def test_broadband_layer_omits_excitation(self, metadata_scope):
        metadata = _metadata(metadata_scope, 'BF')
        assert metadata['channel_display'] == 'Brightfield'
        assert 'excitation_nm' not in metadata

    def test_composite_keeps_unit_fact_drops_layer_facts(self, metadata_scope):
        # 'Composite' has no layer record, so the per-layer claims are
        # absent -- but the filterset is a fact about the unit, true of
        # a composite too.
        metadata = _metadata(metadata_scope, 'Composite')
        assert 'channel_display' not in metadata
        assert 'excitation_nm' not in metadata
        assert metadata['filterset'] == 'FS-STOCK'

    def test_unresolved_identity_omits_the_trio(self, metadata_scope):
        metadata_scope.layer_identity = UNRESOLVED
        metadata = _metadata(metadata_scope, 'BF')
        assert 'channel_display' not in metadata
        assert 'excitation_nm' not in metadata
        assert 'filterset' not in metadata

    def test_empty_filterset_is_absent(self, metadata_scope):
        metadata_scope.layer_identity = _IDENTITY_NO_FILTERSET
        metadata = _metadata(metadata_scope, 'Green')
        assert 'filterset' not in metadata
        assert metadata['channel_display'] == 'Green'

    def test_led_channel_never_reaches_metadata(self, metadata_scope):
        metadata = _metadata(metadata_scope, 'Green')
        assert 'led_channel' not in str(metadata)


class TestIlluminationAbsentWhenUnknown:
    def test_unlit_channel_omits_illumination(self, metadata_scope):
        # LED off (never set on a fresh sim scope): unknown stays
        # unknown, no fabricated 0 mA.
        metadata = _metadata(metadata_scope, 'Green')
        assert 'illumination_ma' not in metadata

    def test_lit_channel_records_real_current(self, metadata_scope):
        metadata_scope.illumination.led_on('Green', 123.0)
        try:
            metadata = _metadata(metadata_scope, 'Green')
        finally:
            metadata_scope.illumination.led_off('Green')
        assert metadata['illumination_ma'] == pytest.approx(123.0)

    def test_tiff_write_tolerates_absent_illumination(self, metadata_scope):
        import numpy as np

        metadata = _metadata(metadata_scope, 'Green')
        assert 'illumination_ma' not in metadata
        metadata['significant_bits'] = 12
        tiff_data = image_utils.generate_tiff_data(
            data=np.zeros((8, 8), dtype=np.uint16),
            metadata=metadata,
            image_type='ome',
            color='Green',
        )
        plane = tiff_data['metadata']['Plane']
        assert 'Illumination' not in plane
        assert 'IlluminationUnit' not in plane


class TestWrittenFileRoundTrip:
    """The B-8 check: the read-back normalizers rebuild from fixed key
    lists, so every added key is proven to pass through the written
    file -- or its drop is recorded here knowingly."""

    def _write_tile(self, tmp_path, metadata, *, ome):
        import numpy as np

        # The sim scope carries no measured scale, and the read-back
        # required block cannot parse a scale-less file (a pre-existing
        # defect of the same shape as the Illumination hard-require,
        # recorded separately); scale is not under test here.
        metadata['pixel_size_um'] = 1.0
        path = tmp_path / ('tile_ome.tiff' if ome else 'tile_ij.tiff')
        image_utils.write_tiff(
            data=np.zeros((16, 16), dtype=np.uint16),
            metadata=metadata,
            file_loc=path,
            ome=ome,
            color='Green',
            significant_bits=12,
            save_encoding='right_aligned',
        )
        return path

    def test_16bit_imagej_tile_round_trips_the_trio(self, metadata_scope, tmp_path):
        metadata = _metadata(metadata_scope, 'Green')
        path = self._write_tile(tmp_path, metadata, ome=False)
        back = image_utils.read_postproc_input_metadata(path)
        assert back is not None
        assert back['channel'] == 'Green'
        assert back['channel_display'] == 'Green'
        assert back['excitation_nm'] == 488.0
        assert back['filterset'] == 'FS-STOCK'

    def test_16bit_ome_tile_keeps_excitation_drops_the_rest_knowingly(
        self, metadata_scope, tmp_path
    ):
        # tifffile's auto-OME serializer writes only schema attributes:
        # ExcitationWavelength is one (measured 2026-08-30), DisplayName
        # and FilterSet are not -- they join gain / illumination /
        # instrument in that path's documented losses. The imagej /
        # shaped container above is the primary still path and lossless.
        metadata = _metadata(metadata_scope, 'Green')
        path = self._write_tile(tmp_path, metadata, ome=True)
        back = image_utils.read_postproc_input_metadata(path)
        assert back is not None
        assert back['channel'] == 'Green'
        assert back['excitation_nm'] == 488.0
        assert 'channel_display' not in back
        assert 'filterset' not in back

    def test_16bit_tile_without_illumination_still_reads(self, metadata_scope, tmp_path):
        # The read side must not treat a dark capture as an unreadable
        # file: Illumination is optional on read-back, like exposure and
        # gain. (The OME container's fallback reader keeps its documented
        # sentinel-default contract and is not asserted here.)
        metadata = _metadata(metadata_scope, 'Green')
        assert 'illumination_ma' not in metadata
        path = self._write_tile(tmp_path, metadata, ome=False)
        back = image_utils.read_postproc_input_metadata(path)
        assert back is not None
        assert back['plate_pos_mm'] is not None
        assert 'illumination_ma' not in back

    def test_lit_capture_round_trips_illumination(self, metadata_scope, tmp_path):
        metadata_scope.illumination.led_on('Green', 55.0)
        try:
            metadata = _metadata(metadata_scope, 'Green')
        finally:
            metadata_scope.illumination.led_off('Green')
        path = self._write_tile(tmp_path, metadata, ome=False)
        back = image_utils.read_postproc_input_metadata(path)
        assert back is not None
        assert back['illumination_ma'] == pytest.approx(55.0)


class TestDerivedOutputs:
    def test_same_channel_derived_output_keeps_the_trio(self, metadata_scope, tmp_path):
        import numpy as np

        metadata = _metadata(metadata_scope, 'Green')
        metadata['pixel_size_um'] = 1.0
        path = tmp_path / 'in.tiff'
        image_utils.write_tiff(
            data=np.zeros((16, 16), dtype=np.uint16),
            metadata=metadata,
            file_loc=path,
            ome=False,
            color='Green',
            significant_bits=12,
            save_encoding='right_aligned',
        )
        out = image_utils.build_postproc_output_metadata(Path(path), 'Green', significant_bits=12)
        assert out['channel_display'] == 'Green'
        assert out['excitation_nm'] == 488.0
        assert out['filterset'] == 'FS-STOCK'

    def test_channel_change_drops_layer_claims_keeps_unit_fact(self, metadata_scope, tmp_path):
        # A composite built from a Green input is not a Green image: the
        # per-layer claims would pair wrongly with channel='Composite'.
        # The filterset stays -- the same unit produced the inputs.
        import numpy as np

        metadata = _metadata(metadata_scope, 'Green')
        metadata['pixel_size_um'] = 1.0
        path = tmp_path / 'in.tiff'
        image_utils.write_tiff(
            data=np.zeros((16, 16), dtype=np.uint16),
            metadata=metadata,
            file_loc=path,
            ome=False,
            color='Green',
            significant_bits=12,
            save_encoding='right_aligned',
        )
        out = image_utils.build_postproc_output_metadata(
            Path(path), 'Composite', significant_bits=12
        )
        assert 'channel_display' not in out
        assert 'excitation_nm' not in out
        assert out['filterset'] == 'FS-STOCK'

    def test_hyperstack_builder_knowingly_drops_the_trio(self, metadata_scope, tmp_path):
        # Recorded drop, not an accident: build_hyperstack_output_metadata
        # composes its own OME schema dict from scratch and does not copy
        # arbitrary input keys. Extending the hyperstack schema with
        # per-channel spectral arrays is future work; until then this
        # test is the record that the drop is known.
        import numpy as np

        metadata = _metadata(metadata_scope, 'Green')
        metadata['pixel_size_um'] = 1.0
        path = tmp_path / 'in.tiff'
        image_utils.write_tiff(
            data=np.zeros((16, 16), dtype=np.uint16),
            metadata=metadata,
            file_loc=path,
            ome=False,
            color='Green',
            significant_bits=12,
            save_encoding='right_aligned',
        )
        out = image_utils.build_hyperstack_output_metadata(
            Path(path),
            channel_names=['Green'],
            plane_positions={
                'PositionX': [0.0],
                'PositionY': [0.0],
                'PositionZ': [0.0],
                'DeltaT': None,
            },
            significant_bits=12,
            pixel_size_um=1.0,
        )
        assert 'channel_display' not in str(out)
        assert 'filterset' not in str(out)
