# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""JPG export: encode_display_jpg bakes the displayed channel color into
8-bit JPEG pixels (the "save what I see" convenience format), and
save_image routes the JPG output format to that encoder.

JPEG is 8-bit and cannot carry the mono-plus-color-metadata form the
TIFF path uses, so the channel color is rendered into the pixels. The
critical correctness risk is the RGB->BGR axis: add_false_color emits
RGB while cv2 (encode_image) expects BGR, so a missed swap would save
red as blue and vice versa. These tests pin the colors and the
extension routing.
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np


REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from modules import image_utils


def _bright_mono(value: int = 200) -> np.ndarray:
    img = np.zeros((32, 32), dtype=np.uint8)
    img[8:24, 8:24] = value
    return img


def _decode_bgr(jpg_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)  # 3-channel BGR


def _region_means(bgr: np.ndarray):
    region = bgr[8:24, 8:24]
    # cv2 channel order is BGR: index 0=Blue, 1=Green, 2=Red.
    return {
        'blue': region[:, :, 0].mean(),
        'green': region[:, :, 1].mean(),
        'red': region[:, :, 2].mean(),
    }


def test_green_channel_bakes_green():
    jpg = image_utils.encode_display_jpg(_bright_mono(), 'Green', jpeg_quality=95)
    m = _region_means(_decode_bgr(jpg))
    assert m['green'] > m['blue'] + 40
    assert m['green'] > m['red'] + 40


def test_red_channel_bakes_red_not_blue():
    # The RGB->BGR swap guard: a wrong swap would surface as blue here.
    jpg = image_utils.encode_display_jpg(_bright_mono(), 'Red', jpeg_quality=95)
    m = _region_means(_decode_bgr(jpg))
    assert m['red'] > m['blue'] + 40
    assert m['red'] > m['green'] + 40


def test_blue_channel_bakes_blue_not_red():
    jpg = image_utils.encode_display_jpg(_bright_mono(), 'Blue', jpeg_quality=95)
    m = _region_means(_decode_bgr(jpg))
    assert m['blue'] > m['red'] + 40
    assert m['blue'] > m['green'] + 40


def test_bf_is_grayscale():
    jpg = image_utils.encode_display_jpg(_bright_mono(), 'BF', jpeg_quality=95)
    m = _region_means(_decode_bgr(jpg))
    # Grayscale: the three channels are approximately equal.
    assert abs(m['blue'] - m['green']) < 12
    assert abs(m['green'] - m['red']) < 12


def test_quality_affects_file_size():
    rng = np.random.default_rng(0)
    noisy = (rng.random((128, 128)) * 255).astype(np.uint8)
    hi = image_utils.encode_display_jpg(noisy, 'BF', jpeg_quality=95)
    lo = image_utils.encode_display_jpg(noisy, 'BF', jpeg_quality=10)
    assert len(lo) < len(hi)


def test_save_image_routes_jpg_to_encoder(tmp_path):
    # The save path must route the JPG format to the display encoder and
    # resolve a .jpg extension -- proven by saving and decoding the file.
    from types import SimpleNamespace

    from modules import image_save

    path = image_save.save_image(
        SimpleNamespace(),
        _bright_mono(),
        save_folder=str(tmp_path),
        file_root='snap_',
        append='BF',
        color='BF',
        tail_id_mode=None,
        output_format='JPG',
        jpeg_quality=95,
    )
    saved = pathlib.Path(path)
    assert saved.suffix == '.jpg', 'JPG format must resolve a .jpg extension'
    assert saved.exists(), 'the JPG file must land on disk'
    bgr = _decode_bgr(saved.read_bytes())
    assert bgr is not None and bgr.shape[:2] == (32, 32), (
        'the saved file must be a decodable JPEG of the input frame'
    )
