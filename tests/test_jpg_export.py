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
        save_encoding='8bit',
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
# When no caller states a payload depth, three save paths fall back to the
# same default: 8 for an already-8-bit frame, else the camera's native depth.
# The TIFF-metadata path (prepare_image_for_saving), the JPG downconversion
# branch, and the post-save log line in save_live_image each used to hand-copy
# that rule. A change to the rule could then update one copy and leave the
# others recording / downconverting / logging against a different depth -- the
# JPG would bake against one depth while the TIFF tag and the log claimed
# another. These pin that all three resolve through the single
# _default_significant_bits helper, so they cannot drift apart.

# A distinctive value (not 8 / 12 / 16) proves the camera's reported depth
# flows through unchanged, rather than any hand-written constant.
_CAMERA_DEPTH = 11


def _wide_frame() -> np.ndarray:
    return np.zeros((16, 16), dtype=np.uint16)


def _scope_with_depth(significant_bits: int = _CAMERA_DEPTH, frame=None):
    from types import SimpleNamespace

    imaging = SimpleNamespace(
        significant_bits=significant_bits,
        _binning_size=1,
        capture_and_wait=lambda **kwargs: frame,
    )
    return SimpleNamespace(imaging=imaging)


def test_default_helper_uint8_is_eight():
    from modules import image_save

    scope = _scope_with_depth()
    assert image_save._default_significant_bits(scope, _bright_mono()) == 8


def test_default_helper_wide_frame_is_camera_depth():
    from modules import image_save

    scope = _scope_with_depth()
    assert image_save._default_significant_bits(scope, _wide_frame()) == _CAMERA_DEPTH


def test_tiff_metadata_depth_resolves_via_helper(monkeypatch):
    # prepare_image_for_saving stamps the SignificantBits tag; with no caller
    # depth it must record the helper's value. generate_image_metadata /
    # generate_image_save_path need a fully configured scope, so stub them --
    # only the depth derivation is under test here.
    from modules import image_save

    monkeypatch.setattr(image_save, 'generate_image_metadata', lambda *a, **k: {})
    monkeypatch.setattr(image_save, 'generate_image_save_path', lambda *a, **k: 'frame.tiff')
    scope = _scope_with_depth()

    out = image_save.prepare_image_for_saving(
        scope,
        array=_wide_frame(),
        save_folder='.',
        file_root='r',
        append='a',
        color='BF',
        tail_id_mode=None,
        output_format='TIFF',
        true_color='BF',
        x=None,
        y=None,
        z=None,
        significant_bits=None,
    )
    assert out['metadata']['significant_bits'] == _CAMERA_DEPTH


def test_jpg_downconvert_depth_resolves_via_helper(tmp_path, monkeypatch):
    # The JPG branch passes the resolved depth to encode_display_jpg; capture
    # that argument and confirm it is the helper's value, not a constant.
    from modules import image_save

    captured = {}

    def _spy_encode(array, color, significant_bits, jpeg_quality=90):
        captured['significant_bits'] = significant_bits
        return b'\xff\xd8\xff\xd9'  # minimal JPEG sentinel; not decoded here

    monkeypatch.setattr(image_save.image_utils, 'encode_display_jpg', _spy_encode)
    scope = _scope_with_depth()

    image_save.save_image(
        scope,
        _wide_frame(),
        save_folder=str(tmp_path),
        file_root='snap_',
        append='BF',
        color='BF',
        tail_id_mode=None,
        output_format='JPG',
        jpeg_quality=95,
        save_encoding='8bit',
        significant_bits=None,
    )
    assert captured['significant_bits'] == _CAMERA_DEPTH


def test_save_live_log_depth_resolves_via_helper(monkeypatch):
    # The post-save log line reports the depth that was written; with no caller
    # depth and a single (non-summed) frame it must log the helper's value.
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
    )
    assert any(f'significant_bits={_CAMERA_DEPTH}' in m for m in messages), messages


def test_three_save_paths_agree_on_default_depth(tmp_path, monkeypatch):
    # The consolidated invariant: for one (scope, wide frame), the TIFF tag,
    # the JPG downconversion, and the live-save log line all resolve the SAME
    # default depth -- the value the single helper returns.
    from modules import image_save

    frame = _wide_frame()
    scope = _scope_with_depth(frame=frame)
    expected = image_save._default_significant_bits(scope, frame)
    assert expected == _CAMERA_DEPTH

    monkeypatch.setattr(image_save, 'generate_image_metadata', lambda *a, **k: {})
    monkeypatch.setattr(image_save, 'generate_image_save_path', lambda *a, **k: 'frame.tiff')
    tiff = image_save.prepare_image_for_saving(
        scope,
        array=frame,
        save_folder='.',
        file_root='r',
        append='a',
        color='BF',
        tail_id_mode=None,
        output_format='TIFF',
        true_color='BF',
        x=None,
        y=None,
        z=None,
        significant_bits=None,
    )
    tiff_depth = tiff['metadata']['significant_bits']

    # Exercise the real JPG branch (through save_image) before stubbing
    # save_image for the live path below, capturing the depth handed to the
    # encoder.
    captured = {}

    def _spy_encode(array, color, significant_bits, jpeg_quality=90):
        captured['d'] = significant_bits
        return b'\xff\xd8\xff\xd9'

    monkeypatch.setattr(image_save.image_utils, 'encode_display_jpg', _spy_encode)
    image_save.save_image(
        scope,
        frame,
        save_folder=str(tmp_path),
        file_root='snap_',
        append='BF',
        color='BF',
        tail_id_mode=None,
        output_format='JPG',
        jpeg_quality=95,
        save_encoding='8bit',
        significant_bits=None,
    )
    jpg_depth = captured['d']

    monkeypatch.setattr(image_save, 'save_image', lambda *a, **k: 'frame.tiff')
    messages = []
    monkeypatch.setattr(image_save.logger, 'info', lambda msg: messages.append(msg))
    image_save.save_live_image(
        scope,
        save_folder='.',
        file_root='img_',
        append='ms',
        color='BF',
        tail_id_mode=None,
        save_encoding='8bit',
    )
    log_depth = next(
        int(m.split('significant_bits=')[1].split(' ')[0])
        for m in messages
        if 'significant_bits=' in m
    )

    assert tiff_depth == jpg_depth == log_depth == expected
