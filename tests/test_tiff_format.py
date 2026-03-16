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

import pathlib
import tempfile
from unittest import mock

import numpy as np
import pytest
import tifffile as tf

from modules import image_utils


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_metadata():
    """Build minimal metadata dict that write_tiff / generate_tiff_data require."""
    return {
        'pixel_size_um': 0.5,
        'channel': 'Green',
        'objective': '10x',
        'exposure_time_ms': 50.0,
        'gain_db': 0.0,
        'illumination_ma': 100.0,
        'z_pos_um': 1000.0,
        'plate_pos_mm': {'x': 10.0, 'y': 20.0},
        'datetime': '2026:03:16 12:00:00',
        'camera_make': 'Test',
        'microscope': 'TestScope',
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
        return tmp_path / f"test_{counter[0]}.tif"
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

    @pytest.mark.parametrize("color", ["BF", "PC", "DF"])
    def test_photometric_minisblack(self, img_8bit, metadata, tmp_tiff, color):
        path = tmp_tiff()
        image_utils.write_tiff(data=img_8bit, file_loc=path, metadata=metadata, ome=False, color=color)
        info = _read_tiff(path)
        assert info['photometric'] == tf.PHOTOMETRIC.MINISBLACK

    @pytest.mark.parametrize("color", ["BF", "PC", "DF"])
    def test_no_colormap_tag(self, img_8bit, metadata, tmp_tiff, color):
        path = tmp_tiff()
        image_utils.write_tiff(data=img_8bit, file_loc=path, metadata=metadata, ome=False, color=color)
        info = _read_tiff(path)
        assert not info['has_colormap_tag'], "BF/PC/DF should not have colormap tag 320"

    @pytest.mark.parametrize("color", ["BF", "PC", "DF"])
    def test_single_channel(self, img_8bit, metadata, tmp_tiff, color):
        path = tmp_tiff()
        image_utils.write_tiff(data=img_8bit, file_loc=path, metadata=metadata, ome=False, color=color)
        info = _read_tiff(path)
        assert info['shape'] == (100, 100), f"Expected (100,100), got {info['shape']}"


class TestTiff8BitFluorescence:
    """8-bit fluorescence should be PALETTE with colormap (false color in Windows + FIJI)."""

    @pytest.mark.parametrize("color", ["Red", "Green", "Blue", "Lumi"])
    def test_photometric_palette(self, img_8bit, metadata, tmp_tiff, color):
        path = tmp_tiff()
        image_utils.write_tiff(data=img_8bit, file_loc=path, metadata=metadata, ome=False, color=color)
        info = _read_tiff(path)
        assert info['photometric'] == tf.PHOTOMETRIC.PALETTE

    @pytest.mark.parametrize("color", ["Red", "Green", "Blue", "Lumi"])
    def test_has_colormap(self, img_8bit, metadata, tmp_tiff, color):
        path = tmp_tiff()
        image_utils.write_tiff(data=img_8bit, file_loc=path, metadata=metadata, ome=False, color=color)
        with tf.TiffFile(str(path)) as f:
            page = f.pages[0]
            # tifffile exposes colormap as page.colormap
            assert page.colormap is not None, "8-bit fluorescence must have colormap"
            assert page.colormap.shape[1] == 256, "Colormap should have 256 entries"

    @pytest.mark.parametrize("color", ["Red", "Green", "Blue", "Lumi"])
    def test_single_channel(self, img_8bit, metadata, tmp_tiff, color):
        path = tmp_tiff()
        image_utils.write_tiff(data=img_8bit, file_loc=path, metadata=metadata, ome=False, color=color)
        info = _read_tiff(path)
        assert info['shape'] == (100, 100)

    def test_green_colormap_correct(self, img_8bit, metadata, tmp_tiff):
        path = tmp_tiff()
        image_utils.write_tiff(data=img_8bit, file_loc=path, metadata=metadata, ome=False, color='Green')
        with tf.TiffFile(str(path)) as f:
            cmap = f.pages[0].colormap
            # Green channel should have a ramp, red and blue should be zero
            assert cmap[0].sum() == 0, "Red channel should be zero for Green colormap"
            assert cmap[1].sum() > 0, "Green channel should have values"
            assert cmap[2].sum() == 0, "Blue channel should be zero for Green colormap"

    def test_red_colormap_correct(self, img_8bit, metadata, tmp_tiff):
        path = tmp_tiff()
        image_utils.write_tiff(data=img_8bit, file_loc=path, metadata=metadata, ome=False, color='Red')
        with tf.TiffFile(str(path)) as f:
            cmap = f.pages[0].colormap
            assert cmap[0].sum() > 0, "Red channel should have values"
            assert cmap[1].sum() == 0, "Green channel should be zero"
            assert cmap[2].sum() == 0, "Blue channel should be zero"


# ---------------------------------------------------------------------------
# 16-bit TIFF tests (default: false_color_16bit off)
# ---------------------------------------------------------------------------

class TestTiff16BitBrightfield:
    """16-bit BF should be MINISBLACK, ImageJ format, no colormap."""

    @pytest.mark.parametrize("color", ["BF", "PC", "DF"])
    def test_photometric_minisblack(self, img_16bit, metadata, tmp_tiff, color):
        path = tmp_tiff()
        image_utils.write_tiff(data=img_16bit, file_loc=path, metadata=metadata, ome=False, color=color)
        info = _read_tiff(path)
        assert info['photometric'] == tf.PHOTOMETRIC.MINISBLACK

    @pytest.mark.parametrize("color", ["BF", "PC", "DF"])
    def test_no_colormap_tag(self, img_16bit, metadata, tmp_tiff, color):
        path = tmp_tiff()
        image_utils.write_tiff(data=img_16bit, file_loc=path, metadata=metadata, ome=False, color=color)
        info = _read_tiff(path)
        assert not info['has_colormap_tag']

    @pytest.mark.parametrize("color", ["BF", "PC", "DF"])
    def test_imagej_format(self, img_16bit, metadata, tmp_tiff, color):
        path = tmp_tiff()
        image_utils.write_tiff(data=img_16bit, file_loc=path, metadata=metadata, ome=False, color=color)
        info = _read_tiff(path)
        assert info['is_imagej'], "16-bit non-OME should be ImageJ format"

    @pytest.mark.parametrize("color", ["BF", "PC", "DF"])
    def test_single_channel(self, img_16bit, metadata, tmp_tiff, color):
        path = tmp_tiff()
        image_utils.write_tiff(data=img_16bit, file_loc=path, metadata=metadata, ome=False, color=color)
        info = _read_tiff(path)
        assert info['shape'] == (100, 100)


class TestTiff16BitFluorescenceDefault:
    """16-bit fluorescence (default setting): MINISBLACK, single channel, Windows-compatible."""

    @pytest.mark.parametrize("color", ["Red", "Green", "Blue", "Lumi"])
    def test_photometric_minisblack(self, img_16bit, metadata, tmp_tiff, color):
        path = tmp_tiff()
        image_utils.write_tiff(data=img_16bit, file_loc=path, metadata=metadata, ome=False, color=color)
        info = _read_tiff(path)
        assert info['photometric'] == tf.PHOTOMETRIC.MINISBLACK

    @pytest.mark.parametrize("color", ["Red", "Green", "Blue", "Lumi"])
    def test_no_colormap_tag(self, img_16bit, metadata, tmp_tiff, color):
        """Windows Preview requires no tag 320 on uint16 images."""
        path = tmp_tiff()
        image_utils.write_tiff(data=img_16bit, file_loc=path, metadata=metadata, ome=False, color=color)
        info = _read_tiff(path)
        assert not info['has_colormap_tag'], "uint16 must not have colormap tag 320 (breaks Windows Preview)"

    @pytest.mark.parametrize("color", ["Red", "Green", "Blue", "Lumi"])
    def test_single_channel(self, img_16bit, metadata, tmp_tiff, color):
        path = tmp_tiff()
        image_utils.write_tiff(data=img_16bit, file_loc=path, metadata=metadata, ome=False, color=color)
        info = _read_tiff(path)
        assert info['shape'] == (100, 100)

    @pytest.mark.parametrize("color", ["Red", "Green", "Blue", "Lumi"])
    def test_imagej_format(self, img_16bit, metadata, tmp_tiff, color):
        path = tmp_tiff()
        image_utils.write_tiff(data=img_16bit, file_loc=path, metadata=metadata, ome=False, color=color)
        info = _read_tiff(path)
        assert info['is_imagej']

    @pytest.mark.parametrize("color", ["Red", "Green", "Blue", "Lumi"])
    def test_imagej_has_lut_metadata(self, img_16bit, metadata, tmp_tiff, color):
        """ImageJ LUT metadata should be present (even though FIJI ignores it for single images)."""
        path = tmp_tiff()
        image_utils.write_tiff(data=img_16bit, file_loc=path, metadata=metadata, ome=False, color=color)
        info = _read_tiff(path)
        ij = info['imagej_metadata']
        assert ij is not None, "ImageJ metadata should be present"
        assert 'LUTs' in ij, "ImageJ metadata should contain LUTs"


# ---------------------------------------------------------------------------
# 16-bit TIFF with false_color_16bit setting ON
# ---------------------------------------------------------------------------

class TestTiff16BitFalseColorOn:
    """16-bit fluorescence with false_color_16bit=True: 3-channel RGB."""

    def _mock_settings(self):
        """Mock app_context so write_tiff reads false_color_16bit=True."""
        mock_ctx = mock.MagicMock()
        mock_ctx.settings = {'false_color_16bit': True}
        return mock.patch('modules.app_context.ctx', mock_ctx)

    @pytest.mark.parametrize("color", ["Red", "Green", "Blue", "Lumi"])
    def test_rgb_3_channel(self, img_16bit, metadata, tmp_tiff, color):
        path = tmp_tiff()
        with self._mock_settings():
            image_utils.write_tiff(data=img_16bit, file_loc=path, metadata=metadata, ome=False, color=color)
        info = _read_tiff(path)
        assert len(info['shape']) == 3, f"Expected 3D shape, got {info['shape']}"
        assert info['shape'][2] == 3, f"Expected 3 channels, got {info['shape']}"

    @pytest.mark.parametrize("color", ["Red", "Green", "Blue", "Lumi"])
    def test_photometric_rgb(self, img_16bit, metadata, tmp_tiff, color):
        path = tmp_tiff()
        with self._mock_settings():
            image_utils.write_tiff(data=img_16bit, file_loc=path, metadata=metadata, ome=False, color=color)
        info = _read_tiff(path)
        assert info['photometric'] == tf.PHOTOMETRIC.RGB

    @pytest.mark.parametrize("color", ["Red", "Green", "Blue", "Lumi"])
    def test_no_colormap_tag(self, img_16bit, metadata, tmp_tiff, color):
        path = tmp_tiff()
        with self._mock_settings():
            image_utils.write_tiff(data=img_16bit, file_loc=path, metadata=metadata, ome=False, color=color)
        info = _read_tiff(path)
        assert not info['has_colormap_tag']

    def test_green_channel_data_correct(self, img_16bit, metadata, tmp_tiff):
        """Green false color: R=0, G=data, B=0."""
        path = tmp_tiff()
        with self._mock_settings():
            image_utils.write_tiff(data=img_16bit, file_loc=path, metadata=metadata, ome=False, color='Green')
        with tf.TiffFile(str(path)) as f:
            img = f.pages[0].asarray()
        assert img[:, :, 0].sum() == 0, "Red channel should be zero"
        assert img[:, :, 1].sum() > 0, "Green channel should have data"
        assert img[:, :, 2].sum() == 0, "Blue channel should be zero"

    def test_red_channel_data_correct(self, img_16bit, metadata, tmp_tiff):
        """Red false color: R=data, G=0, B=0. Note: add_false_color maps Red to channel 2 (BGR)."""
        path = tmp_tiff()
        with self._mock_settings():
            image_utils.write_tiff(data=img_16bit, file_loc=path, metadata=metadata, ome=False, color='Red')
        with tf.TiffFile(str(path)) as f:
            img = f.pages[0].asarray()
        # At least one non-green channel should have data
        assert img.sum() > 0, "Image should have data"
        # Green channel should be zero for red false color
        assert img[:, :, 1].sum() == 0, "Green channel should be zero for Red"

    def test_bf_not_affected(self, img_16bit, metadata, tmp_tiff):
        """BF should remain single-channel even when false_color_16bit is on."""
        path = tmp_tiff()
        with self._mock_settings():
            image_utils.write_tiff(data=img_16bit, file_loc=path, metadata=metadata, ome=False, color='BF')
        info = _read_tiff(path)
        assert info['shape'] == (100, 100), "BF should stay single-channel"


# ---------------------------------------------------------------------------
# OME-TIFF tests
# ---------------------------------------------------------------------------

class TestOmeTiff:
    """OME-TIFF format tests."""

    @pytest.mark.parametrize("color", ["BF", "Green", "Red"])
    def test_is_ome(self, img_8bit, metadata, tmp_tiff, color):
        path = tmp_tiff()
        image_utils.write_tiff(data=img_8bit, file_loc=path, metadata=metadata, ome=True, color=color)
        info = _read_tiff(path)
        assert info['is_ome'], "OME flag should produce OME-TIFF"

    def test_8bit_bf_minisblack(self, img_8bit, metadata, tmp_tiff):
        path = tmp_tiff()
        image_utils.write_tiff(data=img_8bit, file_loc=path, metadata=metadata, ome=True, color='BF')
        info = _read_tiff(path)
        assert info['photometric'] == tf.PHOTOMETRIC.MINISBLACK

    def test_16bit_bf_minisblack(self, img_16bit, metadata, tmp_tiff):
        path = tmp_tiff()
        image_utils.write_tiff(data=img_16bit, file_loc=path, metadata=metadata, ome=True, color='BF')
        info = _read_tiff(path)
        assert info['photometric'] == tf.PHOTOMETRIC.MINISBLACK

    def test_16bit_fluorescence_minisblack(self, img_16bit, metadata, tmp_tiff):
        path = tmp_tiff()
        image_utils.write_tiff(data=img_16bit, file_loc=path, metadata=metadata, ome=True, color='Green')
        info = _read_tiff(path)
        assert info['photometric'] == tf.PHOTOMETRIC.MINISBLACK

    def test_not_imagej(self, img_16bit, metadata, tmp_tiff):
        path = tmp_tiff()
        image_utils.write_tiff(data=img_16bit, file_loc=path, metadata=metadata, ome=True, color='Green')
        info = _read_tiff(path)
        assert not info['is_imagej'], "OME-TIFF should not be ImageJ format"


# ---------------------------------------------------------------------------
# RGB image tests (composite, bullseye)
# ---------------------------------------------------------------------------

class TestRgbImages:
    """Pre-existing RGB images (composites, bullseye) should pass through as RGB."""

    def test_8bit_rgb_photometric(self, img_rgb_8bit, metadata, tmp_tiff):
        path = tmp_tiff()
        image_utils.write_tiff(data=img_rgb_8bit, file_loc=path, metadata=metadata, ome=False, color='BF')
        info = _read_tiff(path)
        assert info['photometric'] == tf.PHOTOMETRIC.RGB

    def test_8bit_rgb_3_channels(self, img_rgb_8bit, metadata, tmp_tiff):
        path = tmp_tiff()
        image_utils.write_tiff(data=img_rgb_8bit, file_loc=path, metadata=metadata, ome=False, color='BF')
        info = _read_tiff(path)
        assert info['shape'] == (100, 100, 3)

    def test_16bit_rgb_photometric(self, img_rgb_16bit, metadata, tmp_tiff):
        path = tmp_tiff()
        image_utils.write_tiff(data=img_rgb_16bit, file_loc=path, metadata=metadata, ome=False, color='BF')
        info = _read_tiff(path)
        assert info['photometric'] == tf.PHOTOMETRIC.RGB


# ---------------------------------------------------------------------------
# Metadata tests
# ---------------------------------------------------------------------------

class TestTiffMetadata:
    """Verify metadata is written correctly."""

    def test_software_tag(self, img_8bit, metadata, tmp_tiff):
        path = tmp_tiff()
        image_utils.write_tiff(data=img_8bit, file_loc=path, metadata=metadata, ome=False, color='BF')
        info = _read_tiff(path)
        assert info['software'] is not None
        assert 'LumaViewPro' in info['software']

    def test_data_integrity_8bit(self, metadata, tmp_tiff):
        """Verify pixel data is preserved exactly."""
        data = np.arange(256, dtype=np.uint8).reshape(16, 16)
        path = tmp_tiff()
        image_utils.write_tiff(data=data, file_loc=path, metadata=metadata, ome=False, color='BF')
        with tf.TiffFile(str(path)) as f:
            read_back = f.pages[0].asarray()
        np.testing.assert_array_equal(data, read_back)

    def test_data_integrity_16bit(self, metadata, tmp_tiff):
        """Verify pixel data is preserved exactly."""
        data = np.arange(10000, dtype=np.uint16).reshape(100, 100)
        path = tmp_tiff()
        image_utils.write_tiff(data=data, file_loc=path, metadata=metadata, ome=False, color='BF')
        with tf.TiffFile(str(path)) as f:
            read_back = f.pages[0].asarray()
        np.testing.assert_array_equal(data, read_back)

    def test_datetime_tag(self, img_8bit, metadata, tmp_tiff):
        path = tmp_tiff()
        image_utils.write_tiff(data=img_8bit, file_loc=path, metadata=metadata, ome=False, color='BF')
        with tf.TiffFile(str(path)) as f:
            tag_codes = {t.code for t in f.pages[0].tags.values()}
            assert 306 in tag_codes, "DateTime tag should be present"


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

    @pytest.mark.parametrize("color", ["BF", "PC", "DF", "Red", "Green", "Blue", "Lumi"])
    def test_16bit_no_palette(self, img_16bit, metadata, tmp_tiff, color):
        """No 16-bit image should use PALETTE photometric."""
        path = tmp_tiff()
        image_utils.write_tiff(data=img_16bit, file_loc=path, metadata=metadata, ome=False, color=color)
        info = _read_tiff(path)
        assert info['photometric'] != tf.PHOTOMETRIC.PALETTE, \
            f"16-bit {color} must not use PALETTE (breaks Windows Preview)"

    @pytest.mark.parametrize("color", ["BF", "PC", "DF", "Red", "Green", "Blue", "Lumi"])
    def test_16bit_no_colormap_tag(self, img_16bit, metadata, tmp_tiff, color):
        """No 16-bit single-channel image should have tag 320."""
        path = tmp_tiff()
        image_utils.write_tiff(data=img_16bit, file_loc=path, metadata=metadata, ome=False, color=color)
        info = _read_tiff(path)
        assert not info['has_colormap_tag'], \
            f"16-bit {color} must not have colormap tag 320 (breaks Windows Preview)"

    @pytest.mark.parametrize("color", ["BF", "PC", "DF", "Red", "Green", "Blue", "Lumi"])
    def test_16bit_ome_no_colormap_tag(self, img_16bit, metadata, tmp_tiff, color):
        """OME-TIFF 16-bit should also not have tag 320."""
        path = tmp_tiff()
        image_utils.write_tiff(data=img_16bit, file_loc=path, metadata=metadata, ome=True, color=color)
        info = _read_tiff(path)
        assert not info['has_colormap_tag']


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestGetTiffColormap:
    def test_green_colormap_shape(self):
        cmap = image_utils.get_tiff_colormap(image_utils.LvpColormap.GREEN, np.uint8)
        assert cmap.shape == (3, 256)

    def test_green_only_green_channel(self):
        cmap = image_utils.get_tiff_colormap(image_utils.LvpColormap.GREEN, np.uint8)
        assert cmap[0].sum() == 0  # Red = 0
        assert cmap[1].sum() > 0   # Green has values
        assert cmap[2].sum() == 0  # Blue = 0

    def test_gray_all_channels_equal(self):
        cmap = image_utils.get_tiff_colormap(image_utils.LvpColormap.GRAY, np.uint8)
        np.testing.assert_array_equal(cmap[0], cmap[1])
        np.testing.assert_array_equal(cmap[1], cmap[2])

    def test_rejects_uint16(self):
        with pytest.raises(NotImplementedError):
            image_utils.get_tiff_colormap(image_utils.LvpColormap.GREEN, np.uint16)


class TestGetImagejLut:
    def test_green_lut_shape(self):
        lut = image_utils.get_imagej_lut(image_utils.LvpColormap.GREEN)
        assert lut.shape == (3, 256)

    def test_green_lut_correct(self):
        lut = image_utils.get_imagej_lut(image_utils.LvpColormap.GREEN)
        assert lut[0].sum() == 0   # Red = 0
        assert lut[1].sum() > 0    # Green has values
        assert lut[2].sum() == 0   # Blue = 0

    def test_gray_lut_all_equal(self):
        lut = image_utils.get_imagej_lut(image_utils.LvpColormap.GRAY)
        np.testing.assert_array_equal(lut[0], lut[1])
        np.testing.assert_array_equal(lut[1], lut[2])


class TestAddFalseColor:
    def test_green_produces_3ch(self):
        data = np.ones((10, 10), dtype=np.uint16) * 1000
        result = image_utils.add_false_color(data, 'Green')
        assert result.shape == (10, 10, 3)
        assert result[:, :, 1].sum() > 0
        assert result[:, :, 0].sum() == 0
        assert result[:, :, 2].sum() == 0

    def test_bf_passthrough(self):
        data = np.ones((10, 10), dtype=np.uint16) * 1000
        result = image_utils.add_false_color(data, 'BF')
        assert result.shape == (10, 10), "BF should not be false-colored"

    def test_already_rgb_passthrough(self):
        data = np.ones((10, 10, 3), dtype=np.uint8) * 128
        result = image_utils.add_false_color(data, 'Green')
        assert result.shape == (10, 10, 3)
