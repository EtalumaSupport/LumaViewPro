# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A colour image in memory is red-green-blue, whatever file it came from.

``load_pixels`` is the one read of a saved picture: a TIFF and a PNG of the
same picture read to the same array, ``rgb_image_to_gray`` reads its
luminance as red-green-blue, and ``image_to_texture``, the display edge, hands
Kivy's blue-green-red texture the picture's own colours.
"""

from __future__ import annotations

import cv2
import numpy as np
import tifffile

from _lumascope.modules import image_utils
from ui import image_utils_kivy

# One picture: a red-orange left half and a blue right half, each plane
# different, so a swap of red and blue shows in every pixel.
_PICTURE = np.zeros((4, 6, 3), dtype=np.uint8)
_PICTURE[:, :3] = (200, 50, 0)
_PICTURE[:, 3:] = (0, 40, 220)


def _save_both(tmp_path, picture: np.ndarray):
    tiff = tmp_path / 'picture.tiff'
    png = tmp_path / 'picture.png'
    tifffile.imwrite(tiff, picture, photometric='rgb')
    # cv2 writes the planes it is given as blue-green-red, so the PNG on disk
    # holds the same picture only when it is handed the planes reversed.
    order = cv2.COLOR_RGBA2BGRA if picture.shape[2] == 4 else cv2.COLOR_RGB2BGR
    assert cv2.imwrite(str(png), cv2.cvtColor(picture, order))
    return tiff, png


def test_a_tiff_and_a_png_of_one_picture_read_to_the_same_red_green_blue_array(tmp_path):
    tiff, png = _save_both(tmp_path, _PICTURE)

    from_tiff, _bits, _white = image_utils.load_pixels(tiff, collapse_legacy_false_color=False)
    from_png, _bits, _white = image_utils.load_pixels(png, collapse_legacy_false_color=False)

    assert (from_tiff == _PICTURE).all()
    assert (from_png == _PICTURE).all(), 'the PNG read with red and blue swapped'


def test_a_png_with_alpha_reads_red_green_blue_alpha(tmp_path):
    picture = np.dstack([_PICTURE, np.full(_PICTURE.shape[:2], 255, np.uint8)])
    _tiff, png = _save_both(tmp_path, picture)

    from_png, _bits, _white = image_utils.load_pixels(png, collapse_legacy_false_color=False)

    assert (from_png == picture).all(), 'the PNG read with red and blue swapped'


def test_a_colour_image_is_grey_by_its_red_green_blue_luminance():
    gray = image_utils.rgb_image_to_gray(_PICTURE)

    # 0.299 R + 0.587 G + 0.114 B: (200, 50, 0) is 89, (0, 40, 220) is 49.
    assert gray[0, 0] == 89
    assert gray[0, 5] == 49


def test_the_display_edge_hands_kivy_the_pictures_own_colours(monkeypatch):
    blitted = []

    # a stand-in by design: Kivy does not run in the test process; the
    # subject is the bytes the edge hands the texture.
    class _Texture:
        size = None

        @classmethod
        def create(cls, size, colorfmt):
            texture = cls()
            texture.size = size
            texture.colorfmt = colorfmt
            return texture

        def flip_vertical(self):
            pass

        def blit_buffer(self, buf, colorfmt, bufferfmt):
            blitted.append((bytes(buf), colorfmt))

    monkeypatch.setattr(image_utils_kivy, 'Texture', _Texture)

    image_utils_kivy.image_to_texture(_PICTURE)

    ((buf, colorfmt),) = blitted
    assert colorfmt == 'bgr'
    assert buf == np.ascontiguousarray(_PICTURE[..., ::-1]).tobytes(), (
        'the picture was drawn with red and blue swapped'
    )
