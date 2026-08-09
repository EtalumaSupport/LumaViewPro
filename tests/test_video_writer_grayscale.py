# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Grayscale is ONE expression at the encode boundary: color None and any
gray-rendering layer label (transmitted light -- no false-color map)
take the same mono encode path, and only chromatic labels widen mono
input to RGB. Previously a transmitted label tiled gray into a
3-channel stream, so identical pixels encoded differently depending on
which of three expressions the caller happened to use.
"""

import numpy as np
import pytest

from modules import image_utils
from modules.video_writer import VideoWriter


class TestLayerRendersGrayscale:
    @pytest.mark.parametrize('layer', ['BF', 'PC', 'DF', 'SomethingNew', None])
    def test_gray_rendering_labels(self, layer):
        assert image_utils.layer_renders_grayscale(layer)

    @pytest.mark.parametrize('layer', ['Red', 'Green', 'Blue', 'Lumi'])
    def test_chromatic_labels(self, layer):
        assert not image_utils.layer_renders_grayscale(layer)


def _write_one_mono_frame(tmp_path, color):
    writer = VideoWriter(
        output_path=tmp_path / 'out.mp4',
        fps=10,
        width=32,
        height=24,
        color=color,
    )
    writer.add_frame(image=np.full((24, 32), 128, dtype=np.uint8))
    writer.close()
    return writer


class TestGrayEncodeUniformity:
    def test_transmitted_label_takes_the_mono_encode_path(self, tmp_path):
        writer = _write_one_mono_frame(tmp_path, color='BF')
        assert writer._is_color is False
        assert writer.output_path.exists()

    def test_none_color_takes_the_mono_encode_path(self, tmp_path):
        # Preservation guard: the None expression is unchanged; the fix
        # makes the transmitted label converge on it.
        writer = _write_one_mono_frame(tmp_path, color=None)
        assert writer._is_color is False

    def test_chromatic_label_still_widens_and_colors(self, tmp_path):
        writer = _write_one_mono_frame(tmp_path, color='Red')
        assert writer._is_color is True
        assert writer.output_path.exists()
