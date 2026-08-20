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
    jpg = image_utils.encode_display_jpg(
        _bright_mono(), 'Green', significant_bits=8, jpeg_quality=95
    )
    m = _region_means(_decode_bgr(jpg))
    assert m['green'] > m['blue'] + 40
    assert m['green'] > m['red'] + 40


def test_red_channel_bakes_red_not_blue():
    # The RGB->BGR swap guard: a wrong swap would surface as blue here.
    jpg = image_utils.encode_display_jpg(_bright_mono(), 'Red', significant_bits=8, jpeg_quality=95)
    m = _region_means(_decode_bgr(jpg))
    assert m['red'] > m['blue'] + 40
    assert m['red'] > m['green'] + 40


def test_blue_channel_bakes_blue_not_red():
    jpg = image_utils.encode_display_jpg(
        _bright_mono(), 'Blue', significant_bits=8, jpeg_quality=95
    )
    m = _region_means(_decode_bgr(jpg))
    assert m['blue'] > m['red'] + 40
    assert m['blue'] > m['green'] + 40


def test_bf_is_grayscale():
    jpg = image_utils.encode_display_jpg(_bright_mono(), 'BF', significant_bits=8, jpeg_quality=95)
    m = _region_means(_decode_bgr(jpg))
    # Grayscale: the three channels are approximately equal.
    assert abs(m['blue'] - m['green']) < 12
    assert abs(m['green'] - m['red']) < 12


def test_quality_affects_file_size():
    rng = np.random.default_rng(0)
    noisy = (rng.random((128, 128)) * 255).astype(np.uint8)
    hi = image_utils.encode_display_jpg(noisy, 'BF', significant_bits=8, jpeg_quality=95)
    lo = image_utils.encode_display_jpg(noisy, 'BF', significant_bits=8, jpeg_quality=10)
    assert len(lo) < len(hi)


def test_save_image_routes_jpg_to_encoder(tmp_path):
    # The save path must route the JPG format to the display encoder and
    # resolve a .jpg extension -- proven by saving and decoding the file.
    from modules import image_save

    path = image_save.save_image(
        _scope_with_depth(),
        _bright_mono(),
        save_folder=str(tmp_path),
        file_root='snap_',
        append='BF',
        color='BF',
        tail_id_mode=None,
        output_format='JPG',
        jpeg_quality=95,
        save_encoding='8bit',
        significant_bits=8,
    )
    saved = pathlib.Path(path)
    assert saved.suffix == '.jpg', 'JPG format must resolve a .jpg extension'
    assert saved.exists(), 'the JPG file must land on disk'
    bgr = _decode_bgr(saved.read_bytes())
    assert bgr is not None and bgr.shape[:2] == (32, 32), (
        'the saved file must be a decodable JPEG of the input frame'
    )


# --- Default significant-bits rule: one helper, three save paths ---
#
# save_live_image resolves the payload depth once at capture (via the shared
# capture_frame_depth rule) and hands it down; a distinctive value (not
# 8 / 12 / 16) proves the camera's reported depth flows through unchanged into
# the post-save log line, rather than any hand-written constant.
_CAMERA_DEPTH = 11


def _wide_frame() -> np.ndarray:
    return np.zeros((16, 16), dtype=np.uint16)


def _scope_with_depth(significant_bits: int = _CAMERA_DEPTH, frame=None):
    from types import SimpleNamespace

    from modules.lumascope_api.imaging import ImagingAPI

    imaging = SimpleNamespace(
        significant_bits=significant_bits,
        last_significant_bits=significant_bits,
        _binning_size=1,
        _capture_and_wait_impl=lambda **kwargs: frame,
    )
    # The REAL shared depth rule, bound to this stub -- the tests pin that
    # every save path resolves through one rule, so the stub must not
    # re-implement it.
    imaging.capture_frame_depth = lambda array, sum_count=1: ImagingAPI.capture_frame_depth(
        imaging, array, sum_count
    )
    return SimpleNamespace(imaging=imaging)


def test_save_live_log_reports_captured_depth(monkeypatch):
    # The post-save log line reports the depth that was written; for a single
    # (non-summed) wide frame it must log the depth resolved at capture.
    # Stub save_image (real write needs a configured scope) and capture the log.
    from modules import image_save

    monkeypatch.setattr(image_save, 'save_image', lambda *a, **k: 'frame.tiff')

    messages = []
    monkeypatch.setattr(image_save.logger, 'info', lambda msg: messages.append(msg))

    scope = _scope_with_depth(frame=_wide_frame())
    image_save.save_live_image(
        scope,
        save_folder='.',
        file_root='img_',
        append='ms',
        color='BF',
        tail_id_mode=None,
        save_encoding='8bit',
        dark_floor_check=False,
    )
    assert any(f'significant_bits={_CAMERA_DEPTH}' in m for m in messages), messages
