"""Regression test for 8-bit fluorescence false-color save (#669).

Bug shape: 8-bit fluorescence z-projection images were saved as
grayscale even with the false-color toggle on. The 16-bit equivalents
were correctly false-colored.

Root cause: ``modules/image_utils.py::write_tiff`` gated the
add_false_color path on ``data.dtype == np.uint16`` only. uint8 data
fell through and saved as grayscale. The setting key is
``false_color_16bit`` -- legacy name, but the user-facing intent is
"save fluorescence in false color regardless of bit depth."

The bug also fired for z-projection outputs because ``zprojector.py``
reads source images via ``tifffile.imread`` and writes the output
verbatim -- if the sources are grayscale (because the 8-bit z-stack
saved them grayscale), the projection is grayscale.

Fix: relax the gate to ``data.dtype in (np.uint8, np.uint16)``.
``protocol_image_writer.py`` buffer prep also relaxed so the
per-run false-color buffer is allocated for both bit depths.

Tests below exercise ``write_tiff`` directly for both bit depths.
"""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest
import tifffile as tf

from modules.image_utils import write_tiff


def _grayscale_image(dtype):
    """Build a small fluorescence-shaped grayscale array."""
    shape = (8, 8)
    if dtype is np.uint8:
        return np.full(shape, 128, dtype=np.uint8)
    elif dtype is np.uint16:
        return np.full(shape, 30000, dtype=np.uint16)
    raise ValueError(dtype)


def _minimal_metadata(path, channel='Red'):
    """Minimal metadata dict matching what write_tiff's generate_tiff_data
    expects. The plane metadata is keyed off these fields."""
    return {
        'file_loc': str(path),
        'datetime': '2026-05-17T11:00:00',
        'plate_pos_mm': {'x': 0.0, 'y': 0.0},
        'z_pos_um': 0.0,
        'objective': 'test',
        'exposure_time_ms': 1.0,
        'gain_db': 0.0,
        'illumination_ma': 0.0,
        'pixel_size_um': 1.0,
        'channel': channel,
    }


def _read_back(path):
    return tf.imread(str(path))


class TestWriteTiffFalseColorAppliesForUint8:
    """8-bit fluorescence with false-color toggle on must save as
    3-channel RGB, not grayscale (#669)."""

    @pytest.fixture
    def tmp_tiff(self, tmp_path):
        return tmp_path / "out.tiff"

    @pytest.mark.parametrize("color,expected_channel", [
        ('Red', 0),
        ('Green', 1),
        ('Blue', 2),
    ])
    def test_uint8_fluorescence_saved_as_rgb_with_false_color_on(
            self, tmp_tiff, color, expected_channel):
        data = _grayscale_image(np.uint8)
        write_tiff(
            data=data,
            file_loc=tmp_tiff,
            metadata=_minimal_metadata(tmp_tiff),
            ome=False,
            color=color,
            use_false_color_16bit=True,
        )
        result = _read_back(tmp_tiff)
        assert result.ndim == 3, (
            f"8-bit fluorescence with false-color on must save as "
            f"3-channel RGB, got shape {result.shape}. Pre-fix: gate "
            f"on data.dtype == np.uint16 excluded uint8 and saved "
            f"grayscale."
        )
        assert result.shape[2] == 3
        # The named channel should carry the data; the others should be zero.
        for ch in range(3):
            if ch == expected_channel:
                assert (result[..., ch] == 128).all(), (
                    f"{color} channel (index {ch}) should carry 128 from "
                    f"the source grayscale, got {result[..., ch].flatten()}"
                )
            else:
                assert (result[..., ch] == 0).all(), (
                    f"Non-{color} channel (index {ch}) should be zero, "
                    f"got {result[..., ch].flatten()}"
                )

    def test_uint16_fluorescence_still_saved_as_rgb(self, tmp_tiff):
        # Regression guard: the 16-bit path must keep working.
        data = _grayscale_image(np.uint16)
        write_tiff(
            data=data,
            file_loc=tmp_tiff,
            metadata=_minimal_metadata(tmp_tiff),
            ome=False,
            color='Green',
            use_false_color_16bit=True,
        )
        result = _read_back(tmp_tiff)
        assert result.ndim == 3, "16-bit false-color path must still produce RGB"
        assert result.shape[2] == 3

    def test_uint8_fluorescence_grayscale_when_false_color_off(
            self, tmp_tiff):
        # The fix must not change behavior when the toggle is off.
        data = _grayscale_image(np.uint8)
        write_tiff(
            data=data,
            file_loc=tmp_tiff,
            metadata=_minimal_metadata(tmp_tiff),
            ome=False,
            color='Red',
            use_false_color_16bit=False,
        )
        result = _read_back(tmp_tiff)
        assert result.ndim == 2, (
            "False-color off must still produce single-channel grayscale, "
            "regardless of bit depth."
        )

    def test_uint8_transmitted_layer_not_false_colored(self, tmp_tiff):
        # BF / PC / DF are transmitted, not in get_image_layers(); even
        # with the toggle on they must stay grayscale (false-color is
        # only meaningful for fluorescence and luminescence).
        data = _grayscale_image(np.uint8)
        write_tiff(
            data=data,
            file_loc=tmp_tiff,
            metadata=_minimal_metadata(tmp_tiff),
            ome=False,
            color='BF',
            use_false_color_16bit=True,
        )
        result = _read_back(tmp_tiff)
        assert result.ndim == 2, (
            "BF (transmitted layer) must stay grayscale regardless of "
            "the false-color toggle -- it's a single grayscale channel "
            "and false color has no semantic meaning."
        )
