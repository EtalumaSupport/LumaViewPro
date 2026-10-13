# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Grayscale is ONE expression at the encode boundary: grey -- a writer given
no colour, and a transmitted channel's white -- takes the mono encode
path, and only a colour widens mono input to RGB. Previously a
transmitted label tiled gray into a 3-channel stream, so identical pixels
encoded differently depending on which of three expressions the caller
happened to use.
"""

import numpy as np

from modules.video_writer import VideoWriter
from modules.layer_record import release_color


def _write_one_mono_frame(tmp_path, **color):
    writer = VideoWriter(
        output_path=tmp_path / 'out.mp4',
        fps=10,
        width=32,
        height=24,
        **color,
    )
    writer.add_frame(image=np.full((24, 32), 128, dtype=np.uint8))
    writer.close()
    return writer


class TestGrayEncodeUniformity:
    def test_transmitted_label_takes_the_mono_encode_path(self, tmp_path):
        writer = _write_one_mono_frame(tmp_path, color=release_color('BF'))
        assert writer._is_color is False
        assert writer.output_path.exists()

    def test_no_color_takes_the_mono_encode_path(self, tmp_path):
        # A writer given no colour is grey, the same path as the
        # transmitted channel's white.
        writer = _write_one_mono_frame(tmp_path)
        assert writer._is_color is False

    def test_chromatic_label_still_widens_and_colors(self, tmp_path):
        writer = _write_one_mono_frame(tmp_path, color=release_color('Red'))
        assert writer._is_color is True
        assert writer.output_path.exists()
