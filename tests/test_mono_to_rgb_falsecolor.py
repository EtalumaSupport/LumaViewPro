"""Tests for ``mono_to_rgb_falsecolor`` -- the Phase 1 boundary helper.

The single mono -> RGB widening entry point for encode boundaries (live
preview, MP4 / AVI). Save pipeline does NOT call this -- mono-native
save keeps 2D + layer metadata.

Per-layer + dtype matrix.
"""

from __future__ import annotations

import datetime
import numpy as np
import pytest
import tifffile

from modules import image_utils
from modules.image_utils import mono_to_rgb_falsecolor
from modules.layer_record import GREY, parse_display_color, release_color
from tests.stock_channels import LED0, LED1, LED2


@pytest.fixture
def mono_uint8():
    return np.full((8, 8), 200, dtype=np.uint8)


@pytest.fixture
def mono_uint16():
    return np.full((8, 8), 42000, dtype=np.uint16)


@pytest.mark.parametrize('dtype_fixture', ['mono_uint8', 'mono_uint16'])
class TestFluorescenceLayers:
    """LED 2 / LED 1 / LED 0 / Lumi each place the source in the RGB index of
    their colour (red / green / blue / blue). Other channels are zero."""

    def test_red_layer_populates_index_0(self, request, dtype_fixture):
        mono = request.getfixturevalue(dtype_fixture)
        rgb = mono_to_rgb_falsecolor(mono, release_color(LED2))
        assert rgb.shape == (8, 8, 3)
        assert rgb.dtype == mono.dtype
        assert (rgb[:, :, 0] == mono).all()
        assert (rgb[:, :, 1] == 0).all()
        assert (rgb[:, :, 2] == 0).all()

    def test_green_layer_populates_index_1(self, request, dtype_fixture):
        mono = request.getfixturevalue(dtype_fixture)
        rgb = mono_to_rgb_falsecolor(mono, release_color(LED1))
        assert (rgb[:, :, 0] == 0).all()
        assert (rgb[:, :, 1] == mono).all()
        assert (rgb[:, :, 2] == 0).all()

    def test_blue_layer_populates_index_2(self, request, dtype_fixture):
        mono = request.getfixturevalue(dtype_fixture)
        rgb = mono_to_rgb_falsecolor(mono, release_color(LED0))
        assert (rgb[:, :, 0] == 0).all()
        assert (rgb[:, :, 1] == 0).all()
        assert (rgb[:, :, 2] == mono).all()

    def test_lumi_layer_falls_into_blue_index(self, request, dtype_fixture):
        mono = request.getfixturevalue(dtype_fixture)
        rgb = mono_to_rgb_falsecolor(mono, release_color('Lumi'))
        assert (rgb[:, :, 2] == mono).all()
        assert (rgb[:, :, 0] == 0).all()
        assert (rgb[:, :, 1] == 0).all()


class TestTransmittedLayers:
    """BF / PC / DF tile mono into all three channels -- grayscale RGB so
    that downstream RGB-only consumers (cv2.VideoWriter) don't see a
    Red-channel-only signal."""

    @pytest.mark.parametrize('layer', ['BF', 'PC', 'DF'])
    def test_layer_tiles_to_grayscale_rgb(self, layer):
        mono = np.full((8, 8), 100, dtype=np.uint8)
        rgb = mono_to_rgb_falsecolor(mono, release_color(layer))
        assert rgb.shape == (8, 8, 3)
        for ch in range(3):
            assert (rgb[:, :, ch] == 100).all(), (
                f'layer {layer}: channel {ch} should tile mono value, got {rgb[:, :, ch].flatten()}'
            )


class TestInputShapeContract:
    def test_3d_input_raises(self):
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match='2D'):
            mono_to_rgb_falsecolor(rgb, release_color(LED2))

    def test_1d_input_raises(self):
        flat = np.zeros(64, dtype=np.uint8)
        with pytest.raises(ValueError, match='2D'):
            mono_to_rgb_falsecolor(flat, release_color(LED2))


class TestSourcePreservation:
    """Function returns a new array; the source mono is not modified."""

    def test_source_not_mutated(self):
        mono = np.full((4, 4), 50, dtype=np.uint16)
        mono_copy = mono.copy()
        _ = mono_to_rgb_falsecolor(mono, release_color(LED2))
        assert (mono == mono_copy).all()


class TestAColourIsAValue:
    """A colour is the weights each writer multiplies by, so a channel need
    not be red, green or blue: a magenta or an orange channel is coloured,
    saved and described in its own colour (W1c part B)."""

    MAGENTA = parse_display_color('#ff00ff')
    ORANGE = parse_display_color('#ff8000')

    def test_false_colour_multiplies_each_plane_by_its_weight(self):
        mono = np.arange(256, dtype=np.uint8).reshape(16, 16)
        rgb = mono_to_rgb_falsecolor(mono, self.ORANGE)
        assert (rgb[:, :, 0] == mono).all()
        assert (rgb[:, :, 1] == np.rint(mono * self.ORANGE[1])).all()
        assert (rgb[:, :, 2] == 0).all()
        magenta = mono_to_rgb_falsecolor(mono, self.MAGENTA)
        assert (magenta[:, :, 0] == mono).all() and (magenta[:, :, 2] == mono).all()
        assert (magenta[:, :, 1] == 0).all()

    def test_a_baked_image_reads_back_as_its_mono_without_its_colour(self):
        """The composite and stack inputs collapse a baked image by its
        brightest plane, so a non-primary channel is not read dim (magenta
        read 83 for 200), and a build needs no record of the colour a file
        was baked in."""
        for mono in (np.full((4, 4), 200, np.uint8), np.full((4, 4), 4000, np.uint16)):
            for colour in (self.MAGENTA, self.ORANGE):
                baked = mono_to_rgb_falsecolor(mono, colour)
                assert (image_utils.rgb_image_to_gray(baked, baked=True) == mono).all(), colour

    def test_an_8_bit_tiff_carries_the_colour_in_its_palette(self, tmp_path):
        path = tmp_path / 'orange.tiff'
        image_utils.write_tiff(
            data=np.full((8, 8), 7, dtype=np.uint8),
            file_loc=path,
            metadata={**_METADATA, 'channel': LED0},
            ome=False,
            color=self.ORANGE,
            significant_bits=8,
            save_encoding='right_aligned',
            white_bits=None,
        )
        with tifffile.TiffFile(path) as written:
            page = written.pages[0]
            assert int(page.photometric) == 3
            assert [int(v) for v in page.colormap[:, -1]] == [65535, 32896, 0]

    def test_a_16_bit_tiff_carries_the_colour_in_its_lut(self, tmp_path):
        path = tmp_path / 'orange16.tiff'
        image_utils.write_tiff(
            data=np.full((8, 8), 700, dtype=np.uint16),
            file_loc=path,
            metadata={**_METADATA, 'channel': LED0},
            ome=False,
            color=self.ORANGE,
            significant_bits=12,
            save_encoding='right_aligned',
            white_bits=None,
        )
        with tifffile.TiffFile(path) as written:
            lut = np.asarray(written.imagej_metadata['LUTs']).reshape(3, 256)
            assert written.imagej_metadata.get('mode') == 'color'
        assert [int(v) for v in lut[:, -1]] == [255, 128, 0]

    def test_ome_names_the_colour(self, tmp_path):
        reference = tmp_path / 'reference.tiff'
        image_utils.write_tiff(
            data=np.full((8, 8), 7, dtype=np.uint8),
            file_loc=reference,
            metadata={**_METADATA, 'channel': 'BF'},
            ome=True,
            color=GREY,
            significant_bits=8,
            save_encoding='right_aligned',
            white_bits=None,
        )
        metadata = image_utils.build_hyperstack_output_metadata(
            reference,
            channel_names=[LED0],
            channel_colors=[self.MAGENTA],
            plane_positions={
                'PositionX': [0.0],
                'PositionY': [0.0],
                'PositionZ': [0.0],
                'DeltaT': None,
            },
            significant_bits=8,
            pixel_size_um=0.5,
        )
        assert metadata['Channel']['Color'] == [(0xFF00FFFF) - (1 << 32)]


_METADATA = {
    'pixel_size_um': 0.5,
    'objective': {'id': '20x Oly'},
    'exposure_time_ms': 2.0,
    'gain_db': 1.0,
    'illumination_ma': 5.0,
    'z_pos_um': 5000.0,
    'plate_pos_mm': {'x': 1.0, 'y': 2.0},
    'datetime': datetime.datetime(2026, 10, 13, 12, 0, 0),
    'well_label': '',
}
