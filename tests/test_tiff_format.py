"""Tests for TIFF and image saving format correctness.

Verifies photometric interpretation, colormap presence, channel count,
bit depth, metadata, and cross-platform compatibility for all save paths.

Key requirements:
- 8-bit BF/PC/DF: MINISBLACK, single channel, no colormap
- 8-bit fluorescence: PALETTE, single channel, has colormap (false color everywhere)
- 16-bit BF/PC/DF: MINISBLACK, single channel, no colormap, ImageJ format
- 16-bit fluorescence (default): MINISBLACK, single channel, no colormap, ImageJ format
- 16-bit fluorescence (false_color_16bit=on): RGB, 3 channels (false color everywhere)
- OME-TIFF: same photometric rules, OME metadata present
- Windows Preview compatibility: no tag 320 (colormap) on uint16 MINISBLACK images
"""

import ast
import datetime
import errno
from unittest import mock

import numpy as np
import pytest
import tifffile as tf

from modules import image_utils
from modules.layer_record import GREY, release_catalogue, release_color
from tests.stock_channels import LED0, LED1, LED2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_metadata():
    """Build minimal metadata dict that write_tiff / generate_tiff_data require."""
    return {
        'pixel_size_um': 0.5,
        'channel': LED1,
        'significant_bits': 8,
        'objective': {'id': '10x'},
        'exposure_time_ms': 50.0,
        'gain_db': 0.0,
        'illumination_ma': 100.0,
        'z_pos_um': 1000.0,
        'plate_pos_mm': {'x': 10.0, 'y': 20.0},
        'datetime': '2026:03:16 12:00:00',
        'well_label': 'A1',
        'well_site': '1',
    }


@pytest.fixture
def img_8bit():
    return np.random.randint(0, 255, (100, 100), dtype=np.uint8)


@pytest.fixture
def img_16bit():
    """Simulates 12-bit camera data scaled to 16-bit."""
    raw = np.random.randint(0, 4095, (100, 100), dtype=np.uint16)
    return (raw * 16).astype(np.uint16)


@pytest.fixture
def img_rgb_8bit():
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def img_rgb_16bit():
    return np.random.randint(0, 65535, (100, 100, 3), dtype=np.uint16)


@pytest.fixture
def metadata():
    return _make_metadata()


@pytest.fixture
def tmp_tiff(tmp_path):
    """Returns a function that gives a fresh temp .tif path."""
    counter = [0]

    def _make():
        counter[0] += 1
        return tmp_path / f'test_{counter[0]}.tif'

    return _make


def _read_tiff(path):
    """Read a TIFF and return useful info dict."""
    with tf.TiffFile(str(path)) as f:
        page = f.pages[0]
        tag_codes = {t.code for t in page.tags.values()}
        info = {
            'photometric': page.photometric,
            'dtype': page.dtype,
            'shape': page.shape,
            'ndim': len(page.shape),
            'has_colormap_tag': 320 in tag_codes,
            'is_imagej': f.is_imagej,
            'is_ome': f.is_ome,
            'imagej_metadata': f.imagej_metadata,
            'software': None,
        }
        if 305 in tag_codes:
            info['software'] = page.tags[305].value
        return info


# ---------------------------------------------------------------------------
# 8-bit TIFF tests
# ---------------------------------------------------------------------------


class TestTiff8BitBrightfield:
    """8-bit BF/PC/DF should be MINISBLACK with no colormap."""

    @pytest.mark.parametrize('color', ['BF', 'PC', 'DF'])
    def test_photometric_minisblack(self, img_8bit, metadata, tmp_tiff, color):
        path = tmp_tiff()
        image_utils.write_tiff(
            data=img_8bit,
            file_loc=path,
            metadata=metadata,
            ome=False,
            color=release_color(color),
            significant_bits=8,
            save_encoding='8bit',
            white_bits=None,
        )
        info = _read_tiff(path)
        assert info['photometric'] == tf.PHOTOMETRIC.MINISBLACK

    @pytest.mark.parametrize('color', ['BF', 'PC', 'DF'])
    def test_no_colormap_tag(self, img_8bit, metadata, tmp_tiff, color):
        path = tmp_tiff()
        image_utils.write_tiff(
            data=img_8bit,
            file_loc=path,
            metadata=metadata,
            ome=False,
            color=release_color(color),
            significant_bits=8,
            save_encoding='8bit',
            white_bits=None,
        )
        info = _read_tiff(path)
        assert not info['has_colormap_tag'], 'BF/PC/DF should not have colormap tag 320'

    @pytest.mark.parametrize('color', ['BF', 'PC', 'DF'])
    def test_single_channel(self, img_8bit, metadata, tmp_tiff, color):
        path = tmp_tiff()
        image_utils.write_tiff(
            data=img_8bit,
            file_loc=path,
            metadata=metadata,
            ome=False,
            color=release_color(color),
            significant_bits=8,
            save_encoding='8bit',
            white_bits=None,
        )
        info = _read_tiff(path)
        assert info['shape'] == (100, 100), f'Expected (100,100), got {info["shape"]}'


class TestTiff8BitFluorescence:
    """8-bit fluorescence should be PALETTE with colormap (false color in Windows + FIJI)."""

    @pytest.mark.parametrize('color', [LED2, LED1, LED0, 'Lumi'])
    def test_photometric_palette(self, img_8bit, metadata, tmp_tiff, color):
        path = tmp_tiff()
        image_utils.write_tiff(
            data=img_8bit,
            file_loc=path,
            metadata=metadata,
            ome=False,
            color=release_color(color),
            significant_bits=8,
            save_encoding='8bit',
            white_bits=None,
        )
        info = _read_tiff(path)
        assert info['photometric'] == tf.PHOTOMETRIC.PALETTE

    @pytest.mark.parametrize('color', [LED2, LED1, LED0, 'Lumi'])
    def test_has_colormap(self, img_8bit, metadata, tmp_tiff, color):
        path = tmp_tiff()
        image_utils.write_tiff(
            data=img_8bit,
            file_loc=path,
            metadata=metadata,
            ome=False,
            color=release_color(color),
            significant_bits=8,
            save_encoding='8bit',
            white_bits=None,
        )
        with tf.TiffFile(str(path)) as f:
            page = f.pages[0]
            # tifffile exposes colormap as page.colormap
            assert page.colormap is not None, '8-bit fluorescence must have colormap'
            assert page.colormap.shape[1] == 256, 'Colormap should have 256 entries'

    @pytest.mark.parametrize('color', [LED2, LED1, LED0, 'Lumi'])
    def test_single_channel(self, img_8bit, metadata, tmp_tiff, color):
        path = tmp_tiff()
        image_utils.write_tiff(
            data=img_8bit,
            file_loc=path,
            metadata=metadata,
            ome=False,
            color=release_color(color),
            significant_bits=8,
            save_encoding='8bit',
            white_bits=None,
        )
        info = _read_tiff(path)
        assert info['shape'] == (100, 100)

    def test_green_colormap_correct(self, img_8bit, metadata, tmp_tiff):
        path = tmp_tiff()
        image_utils.write_tiff(
            data=img_8bit,
            file_loc=path,
            metadata=metadata,
            ome=False,
            color=release_color(LED1),
            significant_bits=8,
            save_encoding='8bit',
            white_bits=None,
        )
        with tf.TiffFile(str(path)) as f:
            cmap = f.pages[0].colormap
            # Green channel should have a ramp, red and blue should be zero
            assert cmap[0].sum() == 0, 'Red channel should be zero for Green colormap'
            assert cmap[1].sum() > 0, 'Green channel should have values'
            assert cmap[2].sum() == 0, 'Blue channel should be zero for Green colormap'

    def test_red_colormap_correct(self, img_8bit, metadata, tmp_tiff):
        path = tmp_tiff()
        image_utils.write_tiff(
            data=img_8bit,
            file_loc=path,
            metadata=metadata,
            ome=False,
            color=release_color(LED2),
            significant_bits=8,
            save_encoding='8bit',
            white_bits=None,
        )
        with tf.TiffFile(str(path)) as f:
            cmap = f.pages[0].colormap
            assert cmap[0].sum() > 0, 'Red channel should have values'
            assert cmap[1].sum() == 0, 'Green channel should be zero'
            assert cmap[2].sum() == 0, 'Blue channel should be zero'


# ---------------------------------------------------------------------------
# 16-bit TIFF tests (default: false_color_16bit off)
# ---------------------------------------------------------------------------


class TestTiff16BitBrightfield:
    """16-bit BF should be MINISBLACK, ImageJ format, no colormap."""

    @pytest.mark.parametrize('color', ['BF', 'PC', 'DF'])
    def test_photometric_minisblack(self, img_16bit, metadata, tmp_tiff, color):
        path = tmp_tiff()
        image_utils.write_tiff(
            data=img_16bit,
            file_loc=path,
            metadata=metadata,
            ome=False,
            color=release_color(color),
            significant_bits=16,
            save_encoding='right_aligned',
            white_bits=None,
        )
        info = _read_tiff(path)
        assert info['photometric'] == tf.PHOTOMETRIC.MINISBLACK

    @pytest.mark.parametrize('color', ['BF', 'PC', 'DF'])
    def test_no_colormap_tag(self, img_16bit, metadata, tmp_tiff, color):
        path = tmp_tiff()
        image_utils.write_tiff(
            data=img_16bit,
            file_loc=path,
            metadata=metadata,
            ome=False,
            color=release_color(color),
            significant_bits=16,
            save_encoding='right_aligned',
            white_bits=None,
        )
        info = _read_tiff(path)
        assert not info['has_colormap_tag']

    @pytest.mark.parametrize('color', ['BF', 'PC', 'DF'])
    def test_imagej_format(self, img_16bit, metadata, tmp_tiff, color):
        path = tmp_tiff()
        image_utils.write_tiff(
            data=img_16bit,
            file_loc=path,
            metadata=metadata,
            ome=False,
            color=release_color(color),
            significant_bits=16,
            save_encoding='right_aligned',
            white_bits=None,
        )
        info = _read_tiff(path)
        assert info['is_imagej'], '16-bit non-OME should be ImageJ format'

    @pytest.mark.parametrize('color', ['BF', 'PC', 'DF'])
    def test_single_channel(self, img_16bit, metadata, tmp_tiff, color):
        path = tmp_tiff()
        image_utils.write_tiff(
            data=img_16bit,
            file_loc=path,
            metadata=metadata,
            ome=False,
            color=release_color(color),
            significant_bits=16,
            save_encoding='right_aligned',
            white_bits=None,
        )
        info = _read_tiff(path)
        assert info['shape'] == (100, 100)


class TestTiff16BitFluorescenceDefault:
    """16-bit fluorescence (default setting): MINISBLACK, single channel, Windows-compatible."""

    @pytest.mark.parametrize('color', [LED2, LED1, LED0, 'Lumi'])
    def test_photometric_minisblack(self, img_16bit, metadata, tmp_tiff, color):
        path = tmp_tiff()
        image_utils.write_tiff(
            data=img_16bit,
            file_loc=path,
            metadata=metadata,
            ome=False,
            color=release_color(color),
            significant_bits=16,
            save_encoding='right_aligned',
            white_bits=None,
        )
        info = _read_tiff(path)
        assert info['photometric'] == tf.PHOTOMETRIC.MINISBLACK

    @pytest.mark.parametrize('color', [LED2, LED1, LED0, 'Lumi'])
    def test_no_colormap_tag(self, img_16bit, metadata, tmp_tiff, color):
        """Windows Preview requires no tag 320 on uint16 images."""
        path = tmp_tiff()
        image_utils.write_tiff(
            data=img_16bit,
            file_loc=path,
            metadata=metadata,
            ome=False,
            color=release_color(color),
            significant_bits=16,
            save_encoding='right_aligned',
            white_bits=None,
        )
        info = _read_tiff(path)
        assert not info['has_colormap_tag'], (
            'uint16 must not have colormap tag 320 (breaks Windows Preview)'
        )

    @pytest.mark.parametrize('color', [LED2, LED1, LED0, 'Lumi'])
    def test_single_channel(self, img_16bit, metadata, tmp_tiff, color):
        path = tmp_tiff()
        image_utils.write_tiff(
            data=img_16bit,
            file_loc=path,
            metadata=metadata,
            ome=False,
            color=release_color(color),
            significant_bits=16,
            save_encoding='right_aligned',
            white_bits=None,
        )
        info = _read_tiff(path)
        assert info['shape'] == (100, 100)

    @pytest.mark.parametrize('color', [LED2, LED1, LED0, 'Lumi'])
    def test_imagej_format(self, img_16bit, metadata, tmp_tiff, color):
        path = tmp_tiff()
        image_utils.write_tiff(
            data=img_16bit,
            file_loc=path,
            metadata=metadata,
            ome=False,
            color=release_color(color),
            significant_bits=16,
            save_encoding='right_aligned',
            white_bits=None,
        )
        info = _read_tiff(path)
        assert info['is_imagej']

    @pytest.mark.parametrize('color', [LED2, LED1, LED0, 'Lumi'])
    def test_imagej_has_lut_metadata(self, img_16bit, metadata, tmp_tiff, color):
        """ImageJ LUT metadata should be present (even though FIJI ignores it for single images)."""
        path = tmp_tiff()
        image_utils.write_tiff(
            data=img_16bit,
            file_loc=path,
            metadata=metadata,
            ome=False,
            color=release_color(color),
            significant_bits=16,
            save_encoding='right_aligned',
            white_bits=None,
        )
        info = _read_tiff(path)
        ij = info['imagej_metadata']
        assert ij is not None, 'ImageJ metadata should be present'
        assert 'LUTs' in ij, 'ImageJ metadata should contain LUTs'


# ---------------------------------------------------------------------------
# 16-bit TIFF with false_color_16bit setting ON
# ---------------------------------------------------------------------------


class TestTiff16BitFalseColorOn:
    """16-bit fluorescence with a false-color image mode: 3-channel RGB."""

    def _mock_settings(self):
        """Mock app_context so write_tiff resolves the false-color RGB mode."""
        mock_ctx = mock.MagicMock()
        mock_ctx.settings = {'image_mode': '12bit_false_color_rgb'}
        return mock.patch('modules.app_context.ctx', mock_ctx)

    @pytest.mark.parametrize('color', [LED2, LED1, LED0, 'Lumi'])
    def test_widens_to_three_channel_rgb(self, img_16bit, metadata, tmp_tiff, color):
        # save_encoding='rgb' drives the false-color widening: 16-bit
        # fluorescence widens to 3-channel RGB for Windows-Preview color.
        path = tmp_tiff()
        with self._mock_settings():
            image_utils.write_tiff(
                data=img_16bit,
                file_loc=path,
                metadata=metadata,
                ome=False,
                color=release_color(color),
                significant_bits=16,
                save_encoding='rgb',
                white_bits=None,
            )
        info = _read_tiff(path)
        assert info['shape'] == (100, 100, 3), (
            f'{color}: false-color mode must widen to RGB, got {info["shape"]}'
        )
        # RGB carries color in the pixels, not a PALETTE colormap tag.
        assert not info['has_colormap_tag']

    def test_bf_not_affected(self, img_16bit, metadata, tmp_tiff):
        """BF should remain single-channel even in a false-color mode."""
        path = tmp_tiff()
        with self._mock_settings():
            image_utils.write_tiff(
                data=img_16bit,
                file_loc=path,
                metadata=metadata,
                ome=False,
                color=release_color('BF'),
                significant_bits=16,
                save_encoding='rgb',
                white_bits=None,
            )
        info = _read_tiff(path)
        assert info['shape'] == (100, 100), 'BF should stay single-channel'


# ---------------------------------------------------------------------------
# OME-TIFF tests
# ---------------------------------------------------------------------------


class TestOmeTiff:
    """OME-TIFF format tests."""

    @pytest.mark.parametrize('color', ['BF', LED1, LED2])
    def test_is_ome(self, img_8bit, metadata, tmp_tiff, color):
        path = tmp_tiff()
        image_utils.write_tiff(
            data=img_8bit,
            file_loc=path,
            metadata=metadata,
            ome=True,
            color=release_color(color),
            significant_bits=8,
            save_encoding='8bit',
            white_bits=None,
        )
        info = _read_tiff(path)
        assert info['is_ome'], 'OME flag should produce OME-TIFF'

    def test_8bit_bf_minisblack(self, img_8bit, metadata, tmp_tiff):
        path = tmp_tiff()
        image_utils.write_tiff(
            data=img_8bit,
            file_loc=path,
            metadata=metadata,
            ome=True,
            color=release_color('BF'),
            significant_bits=8,
            save_encoding='8bit',
            white_bits=None,
        )
        info = _read_tiff(path)
        assert info['photometric'] == tf.PHOTOMETRIC.MINISBLACK

    def test_16bit_bf_minisblack(self, img_16bit, metadata, tmp_tiff):
        path = tmp_tiff()
        image_utils.write_tiff(
            data=img_16bit,
            file_loc=path,
            metadata=metadata,
            ome=True,
            color=release_color('BF'),
            significant_bits=16,
            save_encoding='right_aligned',
            white_bits=None,
        )
        info = _read_tiff(path)
        assert info['photometric'] == tf.PHOTOMETRIC.MINISBLACK

    def test_16bit_fluorescence_minisblack(self, img_16bit, metadata, tmp_tiff):
        path = tmp_tiff()
        image_utils.write_tiff(
            data=img_16bit,
            file_loc=path,
            metadata=metadata,
            ome=True,
            color=release_color(LED1),
            significant_bits=16,
            save_encoding='right_aligned',
            white_bits=None,
        )
        info = _read_tiff(path)
        assert info['photometric'] == tf.PHOTOMETRIC.MINISBLACK

    def test_not_imagej(self, img_16bit, metadata, tmp_tiff):
        path = tmp_tiff()
        image_utils.write_tiff(
            data=img_16bit,
            file_loc=path,
            metadata=metadata,
            ome=True,
            color=release_color(LED1),
            significant_bits=16,
            save_encoding='right_aligned',
            white_bits=None,
        )
        info = _read_tiff(path)
        assert not info['is_imagej'], 'OME-TIFF should not be ImageJ format'


# ---------------------------------------------------------------------------
# RGB image tests (composite, bullseye)
# ---------------------------------------------------------------------------


class TestRgbImages:
    """Pre-existing RGB images (composites, bullseye) should pass through as RGB."""

    def test_8bit_rgb_photometric(self, img_rgb_8bit, metadata, tmp_tiff):
        path = tmp_tiff()
        image_utils.write_tiff(
            data=img_rgb_8bit,
            file_loc=path,
            metadata=metadata,
            ome=False,
            color=release_color('BF'),
            significant_bits=8,
            save_encoding='8bit',
            white_bits=None,
        )
        info = _read_tiff(path)
        assert info['photometric'] == tf.PHOTOMETRIC.RGB

    def test_8bit_rgb_3_channels(self, img_rgb_8bit, metadata, tmp_tiff):
        path = tmp_tiff()
        image_utils.write_tiff(
            data=img_rgb_8bit,
            file_loc=path,
            metadata=metadata,
            ome=False,
            color=release_color('BF'),
            significant_bits=8,
            save_encoding='8bit',
            white_bits=None,
        )
        info = _read_tiff(path)
        assert info['shape'] == (100, 100, 3)

    def test_16bit_rgb_photometric(self, img_rgb_16bit, metadata, tmp_tiff):
        path = tmp_tiff()
        image_utils.write_tiff(
            data=img_rgb_16bit,
            file_loc=path,
            metadata=metadata,
            ome=False,
            color=release_color('BF'),
            significant_bits=16,
            save_encoding='right_aligned',
            white_bits=None,
        )
        info = _read_tiff(path)
        assert info['photometric'] == tf.PHOTOMETRIC.RGB


# ---------------------------------------------------------------------------
# Metadata tests
# ---------------------------------------------------------------------------


class TestTiffMetadata:
    """Verify metadata is written correctly."""

    def test_software_tag(self, img_8bit, metadata, tmp_tiff):
        path = tmp_tiff()
        image_utils.write_tiff(
            data=img_8bit,
            file_loc=path,
            metadata=metadata,
            ome=False,
            color=release_color('BF'),
            significant_bits=8,
            save_encoding='8bit',
            white_bits=None,
        )
        info = _read_tiff(path)
        assert info['software'] is not None
        assert 'LumaViewPro' in info['software']

    def test_data_integrity_8bit(self, metadata, tmp_tiff):
        """Verify pixel data is preserved exactly."""
        data = np.arange(256, dtype=np.uint8).reshape(16, 16)
        path = tmp_tiff()
        image_utils.write_tiff(
            data=data,
            file_loc=path,
            metadata=metadata,
            ome=False,
            color=release_color('BF'),
            significant_bits=8,
            save_encoding='8bit',
            white_bits=None,
        )
        with tf.TiffFile(str(path)) as f:
            read_back = f.pages[0].asarray()
        np.testing.assert_array_equal(data, read_back)

    def test_data_integrity_16bit(self, metadata, tmp_tiff):
        """Verify pixel data is preserved exactly."""
        data = np.arange(10000, dtype=np.uint16).reshape(100, 100)
        path = tmp_tiff()
        image_utils.write_tiff(
            data=data,
            file_loc=path,
            metadata=metadata,
            ome=False,
            color=release_color('BF'),
            significant_bits=16,
            save_encoding='right_aligned',
            white_bits=None,
        )
        with tf.TiffFile(str(path)) as f:
            read_back = f.pages[0].asarray()
        np.testing.assert_array_equal(data, read_back)

    def test_datetime_tag(self, img_8bit, metadata, tmp_tiff):
        path = tmp_tiff()
        image_utils.write_tiff(
            data=img_8bit,
            file_loc=path,
            metadata=metadata,
            ome=False,
            color=release_color('BF'),
            significant_bits=8,
            save_encoding='8bit',
            white_bits=None,
        )
        with tf.TiffFile(str(path)) as f:
            tag_codes = {t.code for t in f.pages[0].tags.values()}
            assert 306 in tag_codes, 'DateTime tag should be present'


class TestTimestampMetadata:
    """Per-frame timestamp fields in the TIFF Plane metadata (issue #633).

    Each timestamp field is optional -- callers that don't capture them
    omit the keys and the corresponding TIFF Plane fields don't appear.
    Callers that do (Lumascope.generate_image_metadata once the camera
    supports ChunkTimestamp) get host wall-clock + camera-tick provenance
    embedded in every saved frame.
    """

    def test_timestamp_iso_emitted_with_the_source_its_producer_states(self, img_8bit, metadata):
        """The file names the clock the producer says its time came from: a
        still's and a recorded frame's come from different clocks."""
        metadata['timestamp_iso'] = '2026-05-09T14:23:01.123456'
        metadata['timestamp_source'] = 'camera_rebased'
        result = image_utils.generate_tiff_data(
            data=img_8bit,
            metadata=metadata,
            image_type='imagej',
            color=release_color('BF'),
            white_bits=None,
        )
        plane = result['metadata']['Plane']
        assert plane['Timestamp'] == '2026-05-09T14:23:01.123456'
        assert plane['TimestampSource'] == 'camera_rebased'

    def test_a_time_without_its_source_is_refused(self, img_8bit, metadata):
        """No source is assumed: a defaulted one would name the wrong clock
        for every producer that forgot to state its own."""
        metadata['timestamp_iso'] = '2026-05-09T14:23:01.123456'
        with pytest.raises(KeyError, match='timestamp_source'):
            image_utils.generate_tiff_data(
                data=img_8bit,
                metadata=metadata,
                image_type='imagej',
                color=release_color('BF'),
                white_bits=None,
            )

    def test_camera_ticks_emitted_when_provided(self, img_8bit, metadata):
        metadata['timestamp_iso'] = '2026-05-09T14:23:01.123456'
        metadata['timestamp_source'] = 'host_wallclock'
        metadata['timestamp_camera_ticks'] = 1234567890123
        metadata['timestamp_camera_tick_hz'] = 1_000_000_000
        result = image_utils.generate_tiff_data(
            data=img_8bit,
            metadata=metadata,
            image_type='imagej',
            color=release_color('BF'),
            white_bits=None,
        )
        plane = result['metadata']['Plane']
        assert plane['TimestampCameraTicks'] == 1234567890123
        assert plane['TimestampCameraTickHz'] == 1_000_000_000

    def test_frame_id_emitted_when_provided(self, img_8bit, metadata):
        metadata['frame_id'] = 42
        result = image_utils.generate_tiff_data(
            data=img_8bit,
            metadata=metadata,
            image_type='imagej',
            color=release_color('BF'),
            white_bits=None,
        )
        plane = result['metadata']['Plane']
        assert plane['FrameID'] == 42

    def test_no_timestamp_fields_when_metadata_lacks_them(self, img_8bit, metadata):
        # Default metadata fixture has no timestamp_* keys; Plane should
        # not gain spurious fields.
        result = image_utils.generate_tiff_data(
            data=img_8bit,
            metadata=metadata,
            image_type='imagej',
            color=release_color('BF'),
            white_bits=None,
        )
        plane = result['metadata']['Plane']
        assert 'Timestamp' not in plane
        assert 'TimestampSource' not in plane
        assert 'TimestampCameraTicks' not in plane
        assert 'TimestampCameraTickHz' not in plane
        assert 'FrameID' not in plane

    def test_partial_timestamp_metadata_emits_only_what_provided(self, img_8bit, metadata):
        # Provide host wall-clock only (camera doesn't support ChunkTimestamp,
        # e.g. IDS today). The host fields appear; the camera fields do not.
        metadata['timestamp_iso'] = '2026-05-09T14:23:01.123456'
        metadata['timestamp_source'] = 'host_wallclock'
        result = image_utils.generate_tiff_data(
            data=img_8bit,
            metadata=metadata,
            image_type='imagej',
            color=release_color('BF'),
            white_bits=None,
        )
        plane = result['metadata']['Plane']
        assert plane['Timestamp'] == '2026-05-09T14:23:01.123456'
        assert 'TimestampCameraTicks' not in plane
        assert 'TimestampCameraTickHz' not in plane
        assert 'FrameID' not in plane


# ---------------------------------------------------------------------------
# Windows Preview compatibility
# ---------------------------------------------------------------------------


class TestWindowsPreviewCompat:
    """Ensure all 16-bit images are compatible with Windows Preview.

    Windows Preview cannot open:
    - PALETTE photometric with uint16 pixels
    - Tag 320 (ColorMap) with uint16 pixels
    These tests ensure we never produce those combinations.
    """

    @staticmethod
    def _written(img_16bit, metadata, tmp_tiff, *, ome):
        """Each catalogue channel's 16-bit file, read back, by channel."""
        written = {}
        for channel in release_catalogue():
            path = tmp_tiff()
            image_utils.write_tiff(
                data=img_16bit,
                file_loc=path,
                metadata=metadata,
                ome=ome,
                color=release_color(channel),
                significant_bits=16,
                save_encoding='right_aligned',
                white_bits=None,
            )
            written[channel] = _read_tiff(path)
        return written

    def test_no_16bit_tiff_uses_a_palette_or_a_colormap_tag(self, img_16bit, metadata, tmp_tiff):
        """No channel's 16-bit TIFF uses PALETTE photometric or carries tag 320."""
        written = self._written(img_16bit, metadata, tmp_tiff, ome=False)
        palette = [
            c for c, info in written.items() if info['photometric'] == tf.PHOTOMETRIC.PALETTE
        ]
        colormap = [c for c, info in written.items() if info['has_colormap_tag']]
        assert (palette, colormap) == ([], []), (
            f'16-bit PALETTE {palette} and tag 320 {colormap} break Windows Preview'
        )

    def test_no_16bit_ome_tiff_carries_a_colormap_tag(self, img_16bit, metadata, tmp_tiff):
        """OME-TIFF 16-bit should also not have tag 320."""
        written = self._written(img_16bit, metadata, tmp_tiff, ome=True)
        assert [c for c, info in written.items() if info['has_colormap_tag']] == []


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestGetTiffColormap:
    def test_green_colormap_shape(self):
        cmap = image_utils.get_tiff_colormap(release_color(LED1), np.uint8)
        assert cmap.shape == (3, 256)

    def test_green_only_green_channel(self):
        cmap = image_utils.get_tiff_colormap(release_color(LED1), np.uint8)
        assert cmap[0].sum() == 0  # Red = 0
        assert cmap[1].sum() > 0  # Green has values
        assert cmap[2].sum() == 0  # Blue = 0

    def test_gray_all_channels_equal(self):
        cmap = image_utils.get_tiff_colormap(GREY, np.uint8)
        np.testing.assert_array_equal(cmap[0], cmap[1])
        np.testing.assert_array_equal(cmap[1], cmap[2])

    def test_rejects_uint16(self):
        with pytest.raises(NotImplementedError):
            image_utils.get_tiff_colormap(release_color(LED1), np.uint16)

    @pytest.mark.parametrize(
        'colormap',
        [
            release_color(LED2),
            release_color(LED1),
            release_color(LED0),
            GREY,
        ],
    )
    def test_colormap_spans_tiff_full_scale(self, colormap):
        # TIFF ColorMap (tag 320) is a uint16 field whose full scale is
        # 65535. A 0-255 ramp stored there renders ~0.4% brightness in
        # compliant readers (ImageJ's native reader shows black); only
        # Bio-Formats' lenient rescaling masked it. 257 = 65535 // 255
        # maps payload 255 to exactly 65535.
        cmap = image_utils.get_tiff_colormap(colormap, np.uint8)
        assert cmap.dtype == np.uint16, f'{colormap}: ColorMap values must be uint16'
        assert cmap.max() == 65535, f'{colormap}: ramp must reach TIFF full scale'

    def test_green_ramp_steps_by_257(self):
        cmap = image_utils.get_tiff_colormap(release_color(LED1), np.uint8)
        np.testing.assert_array_equal(cmap[1], np.arange(256, dtype=np.uint16) * 257)


class TestTiffStripsAndPaletteScale:
    """Written-file invariants: 8-bit output uses strips, and the PALETTE
    colormap tag spans TIFF full scale.

    Tiling forced ImageJ through Bio-Formats (the native reader cannot open
    tiled TIFFs), whose lenient colormap rescaling masked the 0-255-scale
    ColorMap defect while breaking native open. Strips + full-scale ramps
    render correctly in the native reader, and strip writes drain
    measurably faster on 8-bit output.
    """

    def test_8bit_fluorescence_still_uses_strips(self, img_8bit, metadata, tmp_tiff):
        path = tmp_tiff()
        image_utils.write_tiff(
            data=img_8bit,
            file_loc=path,
            metadata=metadata,
            ome=False,
            color=release_color(LED1),
            significant_bits=8,
            save_encoding='8bit',
            white_bits=None,
        )
        with tf.TiffFile(str(path)) as f:
            assert not f.pages[0].is_tiled, '8-bit still output must use strips'

    def test_8bit_fluorescence_colormap_tag_full_scale(self, img_8bit, metadata, tmp_tiff):
        path = tmp_tiff()
        image_utils.write_tiff(
            data=img_8bit,
            file_loc=path,
            metadata=metadata,
            ome=False,
            color=release_color(LED1),
            significant_bits=8,
            save_encoding='8bit',
            white_bits=None,
        )
        with tf.TiffFile(str(path)) as f:
            cmap = f.pages[0].colormap
            assert cmap is not None
            assert int(cmap.max()) == 65535, (
                'The written ColorMap tag must span TIFF full scale; a 0-255 '
                'ramp reads ~0.4% brightness in compliant readers.'
            )


class TestGetImagejLut:
    def test_green_lut_shape(self):
        lut = image_utils.get_imagej_lut(release_color(LED1))
        assert lut.shape == (3, 256)

    def test_green_lut_correct(self):
        lut = image_utils.get_imagej_lut(release_color(LED1))
        assert lut[0].sum() == 0  # Red = 0
        assert lut[1].sum() > 0  # Green has values
        assert lut[2].sum() == 0  # Blue = 0

    def test_gray_lut_all_equal(self):
        lut = image_utils.get_imagej_lut(GREY)
        np.testing.assert_array_equal(lut[0], lut[1])
        np.testing.assert_array_equal(lut[1], lut[2])


class TestAddFalseColor:
    def test_green_produces_3ch(self):
        data = np.ones((10, 10), dtype=np.uint16) * 1000
        result = image_utils.add_false_color(data, release_color(LED1))
        assert result.shape == (10, 10, 3)
        assert result[:, :, 1].sum() > 0
        assert result[:, :, 0].sum() == 0
        assert result[:, :, 2].sum() == 0

    def test_bf_passthrough(self):
        data = np.ones((10, 10), dtype=np.uint16) * 1000
        result = image_utils.add_false_color(data, release_color('BF'))
        assert result.shape == (10, 10), 'BF should not be false-colored'

    def test_already_rgb_passthrough(self):
        data = np.ones((10, 10, 3), dtype=np.uint8) * 128
        result = image_utils.add_false_color(data, release_color(LED1))
        assert result.shape == (10, 10, 3)

    def test_red_indexed_at_0(self):
        """RGB convention: Red channel data lands at index 0."""
        data = np.ones((10, 10), dtype=np.uint16) * 1000
        result = image_utils.add_false_color(data, release_color(LED2))
        assert result.shape == (10, 10, 3)
        assert result[:, :, 0].sum() > 0, 'Red channel (index 0) should have data'
        assert result[:, :, 1].sum() == 0, 'Green channel (index 1) should be zero'
        assert result[:, :, 2].sum() == 0, 'Blue channel (index 2) should be zero'

    def test_blue_indexed_at_2(self):
        """RGB convention: Blue channel data lands at index 2."""
        data = np.ones((10, 10), dtype=np.uint16) * 1000
        result = image_utils.add_false_color(data, release_color(LED0))
        assert result.shape == (10, 10, 3)
        assert result[:, :, 0].sum() == 0, 'Red channel (index 0) should be zero'
        assert result[:, :, 1].sum() == 0, 'Green channel (index 1) should be zero'
        assert result[:, :, 2].sum() > 0, 'Blue channel (index 2) should have data'

    def test_lumi_indexed_at_2(self):
        """RGB convention: Lumi maps to Blue (index 2)."""
        data = np.ones((10, 10), dtype=np.uint16) * 1000
        result = image_utils.add_false_color(data, release_color('Lumi'))
        assert result.shape == (10, 10, 3)
        assert result[:, :, 0].sum() == 0, 'Red channel (index 0) should be zero'
        assert result[:, :, 1].sum() == 0, 'Green channel (index 1) should be zero'
        assert result[:, :, 2].sum() > 0, 'Blue channel (index 2) should have data'


class TestOmeMetadataReadback:
    """read_postproc_input_metadata recovers acquisition context from an
    OME-TIFF input (which carries no shaped_metadata) via the OME-XML
    fallback, instead of returning None and dropping all context on the
    derived output."""

    def test_ome_input_recovers_position_and_exposure(self, img_16bit, metadata, tmp_tiff):
        path = tmp_tiff()
        image_utils.write_tiff(
            data=img_16bit,
            file_loc=path,
            metadata=metadata,
            ome=True,
            color=release_color(LED1),
            significant_bits=16,
            save_encoding='right_aligned',
            white_bits=None,
        )
        with tf.TiffFile(str(path)) as t:
            assert not t.shaped_metadata, 'OME write should carry no shaped_metadata'
            assert t.ome_metadata, 'OME write should carry an OME-XML description'

        recovered = image_utils.read_postproc_input_metadata(path)
        assert recovered is not None, 'OME-TIFF input must recover via the OME fallback'
        assert recovered['plate_pos_mm']['x'] == pytest.approx(metadata['plate_pos_mm']['x'])
        assert recovered['plate_pos_mm']['y'] == pytest.approx(metadata['plate_pos_mm']['y'])
        assert recovered['z_pos_um'] == pytest.approx(metadata['z_pos_um'])
        assert recovered['exposure_time_ms'] == pytest.approx(metadata['exposure_time_ms'])
        assert recovered['pixel_size_um'] == pytest.approx(metadata['pixel_size_um'])


class TestStructuredReadbackOptionalGainExposure:
    """A structured TIFF whose writer omitted the optional ExposureTime /
    Gain plane fields (value genuinely unknown at capture) must still
    recover positions and pixel size on read-back, and must not invent
    stand-in values for the omitted fields on the way to a derived
    output."""

    def _write(self, img, md, path):
        # 8-bit non-OME is the shaped-metadata path (16-bit non-OME writes
        # ImageJ format, which carries no shaped metadata to read back).
        image_utils.write_tiff(
            data=img,
            file_loc=path,
            metadata=md,
            ome=False,
            color=release_color('BF'),
            significant_bits=8,
            save_encoding='8bit',
            white_bits=None,
        )

    def test_missing_gain_exposure_still_recovers_positions(self, img_8bit, metadata, tmp_tiff):
        path = tmp_tiff()
        md = dict(metadata)
        del md['exposure_time_ms']
        del md['gain_db']
        self._write(img_8bit, md, path)

        recovered = image_utils.read_postproc_input_metadata(path)
        assert recovered is not None, (
            'A plane missing only the optional gain/exposure fields must '
            'not be discarded wholesale (positions + pixel size lost)'
        )
        assert recovered['plate_pos_mm']['x'] == pytest.approx(md['plate_pos_mm']['x'])
        assert recovered['plate_pos_mm']['y'] == pytest.approx(md['plate_pos_mm']['y'])
        assert recovered['z_pos_um'] == pytest.approx(md['z_pos_um'])
        assert recovered['pixel_size_um'] == pytest.approx(md['pixel_size_um'])
        assert 'exposure_time_ms' not in recovered, (
            'An omitted (unknown) exposure must stay omitted, not be fabricated on read-back'
        )
        assert 'gain_db' not in recovered, (
            'An omitted (unknown) gain must stay omitted, not be fabricated on read-back'
        )

    def test_present_gain_exposure_round_trip(self, img_8bit, metadata, tmp_tiff):
        path = tmp_tiff()
        self._write(img_8bit, metadata, path)

        recovered = image_utils.read_postproc_input_metadata(path)
        assert recovered is not None
        assert recovered['exposure_time_ms'] == pytest.approx(metadata['exposure_time_ms'])
        assert recovered['gain_db'] == pytest.approx(metadata['gain_db'])

    def test_black_level_round_trips_and_its_absence_stays_absent(
        self, img_8bit, metadata, tmp_tiff
    ):
        stated = tmp_tiff()
        self._write(img_8bit, {**metadata, 'black_level': 4.0}, stated)
        assert image_utils.read_postproc_input_metadata(stated)['black_level'] == 4.0
        unstated = tmp_tiff()
        self._write(img_8bit, metadata, unstated)
        assert 'black_level' not in image_utils.read_postproc_input_metadata(unstated)


# Largest value a signed-32-bit reader (Bioformats) treats as positive. A TIFF
# RATIONAL stores unsigned uint32, but Bioformats reads the numerator as int32,
# so a numerator above this would surface as a negative PhysicalSize.
_INT32_MAX = 2**31 - 1


class TestResolutionRationalOwner:
    """resolution_for_pixel_size is the single int32-safe owner of the TIFF
    resolution tag, so a high-magnification (small) pixel size cannot be read
    back as a negative PhysicalSize and every write path derives from one helper.
    """

    @pytest.mark.parametrize('pixel_size_um', [0.5, 0.1, 0.0646, 0.0323, 0.01, 0.001])
    def test_numerator_within_int32_and_value_preserved(self, pixel_size_um):
        x, y = image_utils.resolution_for_pixel_size(pixel_size_um)
        for num, den in (x, y):
            assert 0 < num <= _INT32_MAX, (
                f'numerator {num} exceeds int32 at pixel_size_um={pixel_size_um}'
            )
            assert num / den == pytest.approx(1e4 / pixel_size_um, rel=1e-4)

    def test_imagej_convention_is_pixels_per_pixel(self):
        x, _ = image_utils.resolution_for_pixel_size(0.5, per_centimeter=False)
        num, den = x
        assert num / den == pytest.approx(1.0 / 0.5, rel=1e-4)

    def test_written_tiff_resolution_is_int32_safe(self, img_16bit, tmp_tiff):
        # Exercise the production OME write path at a high-magnification pixel
        # size and confirm the on-disk RATIONAL numerator stays int32-safe.
        meta = _make_metadata()
        meta['pixel_size_um'] = 0.0646
        path = tmp_tiff()
        image_utils.write_tiff(
            data=img_16bit,
            file_loc=path,
            metadata=meta,
            ome=True,
            color=release_color(LED1),
            significant_bits=16,
            save_encoding='right_aligned',
            white_bits=None,
        )
        with tf.TiffFile(str(path)) as f:
            xnum = f.pages[0].tags['XResolution'].value[0]
            ynum = f.pages[0].tags['YResolution'].value[0]
        assert 0 < xnum <= _INT32_MAX
        assert 0 < ynum <= _INT32_MAX


class TestFalseColorRgbDepth:
    """12-bit false-color RGB must save bright color (the payload filled to the
    container), not a dark right-aligned RGB. The mode exists to render in color
    in plain viewers, so it shares the scaled mode's brightening -- colorizing a
    still-narrow payload would store dark color that no plain viewer can show.
    """

    def test_false_color_rgb_fills_container(self, tmp_tiff):
        # A bright 12-bit LED 1 frame (right-aligned 0..4095) in false-color RGB
        # mode must brighten to fill the 16-bit container (4095 << 4 = 65520),
        # not stay dark at 4095.
        raw = np.full((64, 64), 4095, dtype=np.uint16)
        meta = _make_metadata()
        meta['significant_bits'] = 12
        path = tmp_tiff()
        image_utils.write_tiff(
            data=raw,
            file_loc=path,
            metadata=meta,
            ome=False,
            color=release_color(LED1),
            significant_bits=12,
            save_encoding='rgb',
            white_bits=None,
        )
        with tf.TiffFile(str(path)) as f:
            arr = f.pages[0].asarray()
        assert arr.ndim == 3, 'false-color RGB must be 3-channel'
        green_max = int(arr[:, :, 1].max())
        assert green_max >= 0xF000, (
            f'false-color RGB saved dark (green max {green_max}); the 12-bit '
            'payload was not brightened to fill the container'
        )

    def test_scientific_stays_right_aligned(self, tmp_tiff):
        # Guard the other side: the scientific (right_aligned) mode must NOT be
        # filled -- it stays native 0..4095 so it round-trips as true values.
        raw = np.full((64, 64), 4095, dtype=np.uint16)
        meta = _make_metadata()
        meta['significant_bits'] = 12
        path = tmp_tiff()
        image_utils.write_tiff(
            data=raw,
            file_loc=path,
            metadata=meta,
            ome=False,
            color=release_color(LED1),
            significant_bits=12,
            save_encoding='right_aligned',
            white_bits=None,
        )
        with tf.TiffFile(str(path)) as f:
            arr = f.pages[0].asarray()
        assert int(arr.max()) == 4095, 'scientific must stay native, not filled'


class TestReportedDepthMatchesFile:
    """The depth a caller reports (in a log / bundle manifest) must equal the
    depth the file actually carries. A scaled 12-bit capture is left-justified
    to a 16-bit file, so reporting the pre-scale sensor depth (12) makes a
    correctly-tagged 16-bit file read as if it were mis-tagged.
    """

    @pytest.mark.parametrize(
        'save_encoding,capture_depth,make',
        [
            ('8bit', 8, lambda: np.zeros((4, 4), np.uint8)),
            ('right_aligned', 12, lambda: np.full((4, 4), 4095, np.uint16)),
            ('msb_aligned', 12, lambda: np.full((4, 4), 4095, np.uint16)),
            ('rgb', 12, lambda: np.full((4, 4), 4095, np.uint16)),
        ],
    )
    def test_helper_matches_file_tag(self, metadata, tmp_tiff, save_encoding, capture_depth, make):
        data = make()
        path = tmp_tiff()
        image_utils.write_tiff(
            data=data,
            file_loc=path,
            metadata={**metadata, 'significant_bits': capture_depth},
            ome=False,
            color=release_color(LED1),
            significant_bits=capture_depth,
            save_encoding=save_encoding,
            white_bits=None,
        )
        file_bits = image_utils.read_tiff_significant_bits(path)
        reported = image_utils.written_significant_bits(
            save_encoding, capture_depth, data.dtype, image_utils.is_color_image(data)
        )
        assert reported == file_bits, (
            f'{save_encoding}: helper reported {reported} but file carries {file_bits}'
        )

    def test_scaled_capture_reports_container_depth_not_sensor(self):
        u16 = np.dtype(np.uint16)
        # The exact bug: a scaled 12-bit capture is a 16-bit file.
        assert image_utils.written_significant_bits('msb_aligned', 12, u16, False) == 16
        # Scientific keeps its narrow depth; 8-bit stays 8.
        assert image_utils.written_significant_bits('right_aligned', 12, u16, False) == 12
        assert image_utils.written_significant_bits('8bit', 8, np.dtype(np.uint8), False) == 8


class TestPostprocMetadataRoundTrip:
    """read_postproc_input_metadata must recover acquisition context from every
    container write_tiff can emit -- shaped JSON (8-bit), ImageJ (16-bit mono),
    and OME-XML. A 16-bit non-OME scan tile lands in the ImageJ container; if the
    reader cannot parse that container, pixel_size comes back None and stitching
    silently degrades to an unregistered simple_position montage.
    """

    def _write(self, data, metadata, path, *, ome, significant_bits, save_encoding):
        image_utils.write_tiff(
            data=data,
            file_loc=path,
            metadata=metadata,
            ome=ome,
            color=release_color('BF'),
            significant_bits=significant_bits,
            save_encoding=save_encoding,
            white_bits=None,
        )

    def _assert_full(self, result):
        assert result is not None
        assert result['pixel_size_um'] == 0.5
        assert result['plate_pos_mm'] == {'x': 10.0, 'y': 20.0}
        assert result['channel'] == LED1
        assert result['objective'] == {'id': '10x'}

    def test_shaped_8bit(self, img_8bit, metadata, tmp_tiff):
        path = tmp_tiff()
        self._write(img_8bit, metadata, path, ome=False, significant_bits=8, save_encoding='8bit')
        self._assert_full(image_utils.read_postproc_input_metadata(path))

    def test_imagej_16bit(self, img_16bit, metadata, tmp_tiff):
        # 16-bit mono non-OME routes through tifffile's ImageJ container -- the
        # regression case that read the tiles as pixel_size None.
        path = tmp_tiff()
        self._write(
            img_16bit, metadata, path, ome=False, significant_bits=16, save_encoding='right_aligned'
        )
        self._assert_full(image_utils.read_postproc_input_metadata(path))

    def test_imagej_16bit_numpy_positions(self, img_16bit, metadata, tmp_tiff):
        # Real scan tiles keep stage positions as numpy scalars, which the ImageJ
        # writer reprs as np.float64(N) -- a call expression literal_eval rejects.
        # The synthetic plain-float case above does not exercise this; the bench
        # tiles did, and returned None until the adapter unwrapped the reprs.
        md = {
            **metadata,
            'plate_pos_mm': {'x': np.float64(10.0), 'y': np.float64(20.0)},
            'z_pos_um': np.float64(1000.0),
        }
        path = tmp_tiff()
        self._write(
            img_16bit, md, path, ome=False, significant_bits=16, save_encoding='right_aligned'
        )
        self._assert_full(image_utils.read_postproc_input_metadata(path))

    def test_ome_16bit(self, img_16bit, metadata, tmp_tiff):
        path = tmp_tiff()
        self._write(
            img_16bit, metadata, path, ome=True, significant_bits=16, save_encoding='right_aligned'
        )
        result = image_utils.read_postproc_input_metadata(path)
        # tifffile's auto-OME serializer keeps pixel size, positions, and channel;
        # objective is dropped (documented OME lossiness, out of scope for this fix).
        assert result is not None
        assert result['pixel_size_um'] == 0.5
        assert result['plate_pos_mm'] == {'x': 10.0, 'y': 20.0}
        assert result['channel'] == LED1

    def test_timestamp_imagej_16bit(self, img_16bit, metadata, tmp_tiff):
        # The per-frame timestamp reader shared the same ImageJ blindness.
        md = {
            **metadata,
            'timestamp_iso': '2026-03-16T12:00:00',
            'timestamp_source': 'host_wallclock',
        }
        path = tmp_tiff()
        self._write(
            img_16bit, md, path, ome=False, significant_bits=16, save_encoding='right_aligned'
        )
        ts = image_utils.read_frame_timestamp(path)
        assert ts == datetime.datetime.fromisoformat('2026-03-16T12:00:00')


# ---------------------------------------------------------------------------
# Every key a save records reads back from the file
# ---------------------------------------------------------------------------

# Recorded keys the file carries in another form, so the reader returns them
# under no key of their own; each names where it is.
_IN_THE_FILE_AS = {
    'significant_bits': 'tag 65123',
}


# What an OME file does not state until the OME plan writes its schema blocks:
# tifffile keeps only the schema's Image, Pixels, Channel and Plane, and drops
# the rest at write.
_NOT_IN_AN_OME_FILE = {'instrument', 'plate', 'well_label', 'filterset', 'channel_display'}

# A layer's own settings: a derived file on another channel states none of them.
_PER_CHANNEL = (
    'channel_display',
    'excitation_nm',
    'illumination_ma',
    'illumination_by_channel_ma',
    'illumination_timing',
    'exposure_time_ms',
    'gain_db',
    'black_level',
    'frames_summed',
)


def _a_full_still(scope, tmp_path, *, save_encoding, significant_bits, dtype, output_format):
    """A BF still with every field of its record set, lit by BF's LED: the
    record the save built, and the file it wrote."""
    from modules import image_save
    from modules.layer_record import release_colors
    from tests.frame_records import frame_record, plate
    from tests.scope_fakes import bind_settings_like_a_session

    bind_settings_like_a_session(
        scope, objective_id='4x Oly', protocol={'labware': '96 well microplate'}
    )
    bf_address = scope.channel_identity.find('BF').led_addresses[0]
    record = frame_record(
        black_level=0.0,
        illumination_ma={bf_address: 50.0},
        frame_significant_bits=significant_bits,
        camera_timestamp_ticks=123,
        camera_tick_hz=1_000_000_000,
        frame_id=7,
        binning_size=2,
        camera_model='daA3840-45um',
        saturated_fraction=0.25,
        dark=False,
        held_channel='BF',
    )
    where = {'plate_x_mm': 10.0, 'plate_y_mm': 20.0, 'stage_z_um': 5000.0}
    common = {
        'objective_id': '4x Oly',
        'frame_record': record,
        'labware': plate(),
        'well_label': 'A1',
        'fallback_focus': 'no focus peak',
    }
    recorded = image_save.generate_image_metadata(scope, 'BF', **where, **common)
    path = image_save.save_image(
        scope,
        np.zeros((8, 8), dtype=dtype),
        save_folder=str(tmp_path),
        file_root='rt_',
        append='a',
        tail_id_mode=None,
        channel='BF',
        false_color_on=False,
        output_format=output_format,
        save_encoding=save_encoding,
        significant_bits=significant_bits,
        white_bits=None,
        channel_colors=release_colors(),
        **where,
        **common,
    )
    return recorded, path


@pytest.mark.parametrize(
    ('save_encoding', 'significant_bits', 'dtype', 'output_format'),
    # parametrize-justified: the shaped-JSON, ImageJ and OME containers are
    # three writers with three readers.
    [
        ('8bit', 8, np.uint8, 'TIFF'),
        ('right_aligned', 12, np.uint16, 'TIFF'),
        ('right_aligned', 12, np.uint16, 'OME-TIFF'),
    ],
)
def test_every_key_a_save_records_reads_back_from_the_file(
    sim_scope, tmp_path, save_encoding, significant_bits, dtype, output_format
):
    """The writer copies the record into the file key by key, so a key the save
    records and the writer does not copy is lost without a word (the held
    layer and the binning were). Every key of a record with every field set
    is read back, in every container, and an OME file loses only what its
    container cannot yet hold. Each recorded key has its row in the one
    table the writer and the readers read."""
    recorded, path = _a_full_still(
        sim_scope,
        tmp_path,
        save_encoding=save_encoding,
        significant_bits=significant_bits,
        dtype=dtype,
        output_format=output_format,
    )

    assert sorted(set(recorded) - set(image_utils.FILE_FACTS)) == [], 'a recorded key has no row'
    read = image_utils.read_postproc_input_metadata(path)
    not_held = _NOT_IN_AN_OME_FILE if output_format == 'OME-TIFF' else set()
    lost = sorted(set(recorded) - set(read) - set(_IN_THE_FILE_AS) - not_held)
    assert lost == [], f'recorded but not in the file: {lost}'
    assert (read['binning_size'], read['camera_held_a_channel'], read['camera_held_channel']) == (
        2,
        True,
        'BF',
    )


def test_every_key_a_derived_file_keeps_reads_back_and_a_composite_keeps_no_layers_setting(
    sim_scope, tmp_path
):
    """A stitch and a composite are written from a still's record; every key
    each keeps reads back from its own file, and the composite, whose channel
    is no input's, keeps none of a layer's settings."""
    _, still = _a_full_still(
        sim_scope,
        tmp_path,
        save_encoding='right_aligned',
        significant_bits=12,
        dtype=np.uint16,
        output_format='TIFF',
    )
    stitch = image_utils.build_postproc_output_metadata(
        still, 'BF', significant_bits=12, adds_frames=False, algorithm='registered'
    )
    composite = image_utils.build_postproc_output_metadata(
        still, 'Composite', significant_bits=8, adds_frames=False
    )

    assert [key for key in _PER_CHANNEL if key in composite] == []
    for name, metadata, data in (
        ('stitch', stitch, np.zeros((8, 8), dtype=np.uint16)),
        ('composite', composite, np.zeros((8, 8, 3), dtype=np.uint8)),
    ):
        path = tmp_path / f'{name}.tiff'
        image_utils.write_tiff(
            data=data,
            file_loc=path,
            metadata=metadata,
            ome=False,
            color=GREY,
            significant_bits=metadata['significant_bits'],
            save_encoding='right_aligned',
            white_bits=None,
        )
        read = image_utils.read_postproc_input_metadata(path)
        lost = sorted(set(metadata) - set(read) - set(_IN_THE_FILE_AS))
        assert lost == [], f'the {name} kept but its file does not state: {lost}'


def test_a_stills_objective_names_its_id_and_units_and_an_older_file_still_reads(
    sim_scope, tmp_path
):
    """The objective's one home in a still is Plane.Objective, under the
    objective's own field names, each naming its unit; Instrument.Objective is
    a rendering of it with the working distance's unit stated. A file written
    under the catalogue's keys reads back under the field names."""
    _, still = _a_full_still(
        sim_scope,
        tmp_path,
        save_encoding='8bit',
        significant_bits=8,
        dtype=np.uint8,
        output_format='OME-TIFF',
    )
    catalogue = sim_scope.objectives['4x Oly']
    stated = image_utils.read_postproc_input_metadata(still)['objective']
    assert (stated['id'], stated['working_distance_mm']) == (
        '4x Oly',
        catalogue.working_distance_mm,
    )
    rendered = image_utils.generate_tiff_data(
        data=np.zeros((8, 8), dtype=np.uint8),
        metadata={**image_utils.read_postproc_input_metadata(still), 'significant_bits': 8},
        image_type='ome',
        color=GREY,
        white_bits=None,
    )['metadata']['Instrument']['Objective']
    assert (rendered['WorkingDistance'], rendered['WorkingDistanceUnit']) == (
        catalogue.working_distance_mm,
        'mm',
    )

    older = tmp_path / 'older.tiff'
    image_utils.write_tiff(
        data=np.zeros((8, 8), dtype=np.uint8),
        file_loc=older,
        metadata={
            'pixel_size_um': 0.5,
            'channel': 'BF',
            'datetime': '2026:03:16 12:00:00',
            'objective': {'aperture': 0.16, 'working_distance': 13.0, 'DOF': 40.0},
        },
        ome=False,
        color=GREY,
        significant_bits=8,
        save_encoding='8bit',
        white_bits=None,
    )
    assert image_utils.read_postproc_input_metadata(older)['objective'] == {
        'numerical_aperture': 0.16,
        'working_distance_mm': 13.0,
        'depth_of_field_um': 40.0,
    }


def test_each_fact_has_one_key_and_an_imagej_document_renders_the_instrument(sim_scope, tmp_path):
    """The scope's maker and model live in ``instrument``, the position in
    ``plate_pos_mm``, the depth in tag 65123: a still's record holds no second
    name for any of them, and an ImageJ file's Document, Fiji's view of the
    scope, is rendered from ``instrument`` and ``plate``."""
    recorded, still = _a_full_still(
        sim_scope,
        tmp_path,
        save_encoding='right_aligned',
        significant_bits=12,
        dtype=np.uint16,
        output_format='TIFF',
    )
    second_names = ('camera_make', 'microscope', 'microscope_model', 'x_pos', 'y_pos')
    assert [key for key in second_names if key in recorded] == []
    with tf.TiffFile(str(still)) as tif:
        description = tif.imagej_metadata
    assert 'significantbits' not in description
    document = ast.literal_eval(description['document'])
    assert (document['Model'], document['PlateName']) == (
        recorded['instrument']['model'],
        recorded['plate']['name'],
    )


# ---------------------------------------------------------------------------
# A write cut short
# ---------------------------------------------------------------------------


def _write_16bit(path, img_16bit, metadata):
    image_utils.write_tiff(
        data=img_16bit,
        file_loc=path,
        metadata=metadata,
        ome=False,
        color=release_color('BF'),
        significant_bits=12,
        save_encoding='right_aligned',
        white_bits=None,
    )


def test_a_write_cut_short_leaves_nothing_under_the_name(img_16bit, metadata, tmp_tiff):
    """A truncated file under the real name reads later as an unreadable frame."""
    path = tmp_tiff()

    def full_disk(self, *args, **kwargs):
        # A stand-in by design: a full disk the suite cannot make. The file is
        # open, as it is when the disk fills, and the write raises.
        raise OSError(errno.ENOSPC, 'No space left on device')

    with mock.patch.object(tf.TiffWriter, 'write', full_disk), pytest.raises(OSError):
        _write_16bit(path, img_16bit, metadata)

    assert list(path.parent.iterdir()) == []


def test_a_whole_write_leaves_only_the_file(img_16bit, metadata, tmp_tiff):
    path = tmp_tiff()

    _write_16bit(path, img_16bit, metadata)

    assert list(path.parent.iterdir()) == [path]
    assert _read_tiff(path)['shape'] == img_16bit.shape
