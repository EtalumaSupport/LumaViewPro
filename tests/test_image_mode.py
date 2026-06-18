"""Image-mode consolidation matrix -- the test-first contract.

One global ``image_mode`` selector replaces the ``use_full_pixel_depth``
capture toggle and the ``false_color_16bit`` save toggle. The selector
resolves to two derived facts -- capture depth and save encoding -- that
flow through stills, composites, and video.

These tests describe the TARGET behavior. The ones that exercise surfaces
not yet built (the ``modules.image_mode`` resolver, the ``write_tiff``
``save_encoding`` parameter) carry ``xfail(strict=True)``: they error/fail
today and flip green as the resolver, the migration, and the save-path
rewiring land. ``strict=True`` catches a marker left stale after its
feature ships. Tests that exercise already-correct behavior (the
per-layer mono fallback, VideoWriter's uint16 colorization) are plain
regression guards with no marker.

The four modes and their derived facts:
  8bit                  -> capture 8-bit,  save 8-bit mono
  12bit_scientific      -> capture 12-bit, save right-aligned 0..4095 + SignificantBits
  12bit_scaled          -> capture 12-bit, save MSB-aligned (x16) + SignificantBits
  12bit_false_color_rgb -> capture 12-bit, save 3-channel RGB (per-layer gated)
"""

from __future__ import annotations

import numpy as np
import pytest
import tifffile as tf


RESOLVER_PENDING = (
    'modules.image_mode resolver not yet implemented; flips green when the derivation layer lands.'
)
SAVE_ENCODING_PENDING = (
    'write_tiff save_encoding parameter not yet implemented; flips green when '
    'the save path is rewired off use_false_color_16bit.'
)
VIDEO_FIX_PENDING = (
    '12-bit video frames do not yet honor false color in the manual-record '
    'caller; flips green when the dtype gate is removed.'
)


def _metadata(path, channel='Blue', significant_bits=None):
    """Minimal metadata dict matching write_tiff's generate_tiff_data."""
    meta = {
        'file_loc': str(path),
        'datetime': '2026-06-18T00:00:00',
        'plate_pos_mm': {'x': 0.0, 'y': 0.0},
        'z_pos_um': 0.0,
        'objective': 'test',
        'exposure_time_ms': 1.0,
        'gain_db': 0.0,
        'illumination_ma': 0.0,
        'pixel_size_um': 1.0,
        'channel': channel,
    }
    if significant_bits is not None:
        meta['significant_bits'] = significant_bits
    return meta


# ---------------------------------------------------------------------------
# Resolver: image_mode -> (capture_depth, save_encoding)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason=RESOLVER_PENDING)
@pytest.mark.parametrize(
    ('mode', 'capture_depth', 'save_encoding'),
    [
        ('8bit', 8, '8bit'),
        ('12bit_scientific', 12, 'right_aligned'),
        ('12bit_scaled', 12, 'msb_aligned'),
        ('12bit_false_color_rgb', 12, 'rgb'),
    ],
)
def test_resolve_image_mode(mode, capture_depth, save_encoding):
    """Each mode resolves to its capture depth and save encoding."""
    from modules.image_mode import resolve_image_mode

    resolved = resolve_image_mode(mode)
    assert resolved['capture_depth'] == capture_depth
    assert resolved['save_encoding'] == save_encoding


# ---------------------------------------------------------------------------
# Migration: legacy (use_full_pixel_depth, false_color_16bit) -> image_mode
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason=RESOLVER_PENDING)
@pytest.mark.parametrize(
    ('use_full_pixel_depth', 'false_color_16bit', 'expected_mode'),
    [
        (False, False, '8bit'),
        (False, True, '8bit'),  # 8-bit capture has no 16-bit false color
        (True, False, '12bit_scientific'),
        (True, True, '12bit_false_color_rgb'),
    ],
)
def test_migrate_legacy_settings(use_full_pixel_depth, false_color_16bit, expected_mode):
    """The two retiring keys map onto the new enum. No legacy combo maps to
    12bit_scaled -- it is new and only reachable by explicit selection."""
    from modules.image_mode import migrate_legacy_settings

    assert migrate_legacy_settings(use_full_pixel_depth, false_color_16bit) == expected_mode


# ---------------------------------------------------------------------------
# Save encoding: 12-bit scaled (MSB-aligned, lossless, recoverable)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason=SAVE_ENCODING_PENDING)
def test_save_encoding_scaled_is_msb_aligned_and_recoverable(tmp_path):
    """12bit_scaled left-justifies 0..4095 to 0..65520 (x16) so dumb viewers
    render it bright, while SignificantBits=12 lets a smart reader recover the
    true value. The multiply is lossless: 4095*16 = 65520 < 65536, no clip."""
    from modules.image_utils import write_tiff

    out_path = tmp_path / 'scaled.tiff'
    data = np.full((8, 8), 4095, dtype=np.uint16)  # full-scale 12-bit

    write_tiff(
        data=data,
        file_loc=out_path,
        metadata=_metadata(out_path, channel='BF', significant_bits=12),
        ome=False,
        color='BF',
        save_encoding='msb_aligned',
    )

    with tf.TiffFile(str(out_path)) as t:
        page = t.pages[0]
        arr = page.asarray()
        sig = page.tags.get('SignificantBits') or page.tags.get('SampleFormat')

    assert arr.dtype == np.uint16
    assert arr[0, 0] == 65520, 'full-scale 12-bit must left-justify to 65520 (4095 x 16)'
    assert (arr >> 4 == 4095).all(), 'x16 must be exactly recoverable by >>4'
    assert sig is not None and sig.value == 12, 'SignificantBits tag must carry 12'


@pytest.mark.xfail(strict=True, reason=SAVE_ENCODING_PENDING)
def test_save_encoding_right_aligned_is_raw(tmp_path):
    """12bit_scientific stores the raw right-aligned value with SignificantBits=12
    -- correct quantitative data, the current default behavior under the new
    parameter name."""
    from modules.image_utils import write_tiff

    out_path = tmp_path / 'sci.tiff'
    data = np.full((8, 8), 4095, dtype=np.uint16)

    write_tiff(
        data=data,
        file_loc=out_path,
        metadata=_metadata(out_path, channel='BF', significant_bits=12),
        ome=False,
        color='BF',
        save_encoding='right_aligned',
    )

    with tf.TiffFile(str(out_path)) as t:
        arr = t.pages[0].asarray()
    assert arr[0, 0] == 4095, 'right-aligned must store the raw value, not scale it'


# ---------------------------------------------------------------------------
# Save encoding: RGB false color -- fluorescence widens, transmitted stays mono
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason=SAVE_ENCODING_PENDING)
def test_save_encoding_rgb_widens_fluorescence(tmp_path):
    """rgb encoding bakes the layer color into 3-channel RGB for a
    fluorescence layer. Blue lands at index 2, value preserved, others zero."""
    from modules.image_utils import write_tiff

    out_path = tmp_path / 'blue_rgb.tiff'
    data = np.full((8, 8), 42000, dtype=np.uint16)

    write_tiff(
        data=data,
        file_loc=out_path,
        metadata=_metadata(out_path, channel='Blue'),
        ome=False,
        color='Blue',
        save_encoding='rgb',
    )

    result = tf.imread(str(out_path))
    assert result.shape == (8, 8, 3)
    assert result[0, 0, 2] == 42000 and result[0, 0, 0] == 0 and result[0, 0, 1] == 0


@pytest.mark.xfail(strict=True, reason=SAVE_ENCODING_PENDING)
def test_save_encoding_rgb_keeps_transmitted_mono(tmp_path):
    """The per-layer color gate survives the rewire: rgb encoding does NOT
    force color onto a transmitted (BF) layer -- it stays 2D mono. This is the
    'mono when there is no color to preserve' invariant."""
    from modules.image_utils import write_tiff

    out_path = tmp_path / 'bf_mono.tiff'
    data = np.full((8, 8), 3000, dtype=np.uint16)

    write_tiff(
        data=data,
        file_loc=out_path,
        metadata=_metadata(out_path, channel='BF'),
        ome=False,
        color='BF',
        save_encoding='rgb',
    )

    result = tf.imread(str(out_path))
    assert result.ndim == 2, 'BF must stay mono even under rgb encoding (no color to bake)'


# ---------------------------------------------------------------------------
# Video: 12-bit frame honors false color (the silent-drop fix)
# ---------------------------------------------------------------------------


def test_videowriter_colorizes_uint16_frame():
    """VideoWriter already downconverts a uint16 frame to 8-bit and applies
    false color when a layer color is set -- the capability the manual-record
    fix relies on. Regression guard: this must keep working."""
    from unittest.mock import MagicMock, patch

    from modules.video_writer import VideoWriter

    mono = np.full((8, 8), 50000, dtype=np.uint16)
    captured = {}

    def fake_write(frame):
        captured['frame'] = frame.copy()

    with patch('cv2.VideoWriter') as MockCv2:
        instance = MagicMock()
        instance.write = fake_write
        instance.isOpened.return_value = True
        MockCv2.return_value = instance

        writer = VideoWriter(
            output_path='/tmp/dummy_imgmode.avi', fps=10, width=8, height=8, color='Green'
        )
        writer.add_frame(mono)
        writer.close()

    assert 'frame' in captured, 'a uint16 frame must reach cv2.VideoWriter.write'
    written = captured['frame']
    assert written.shape[-1] == 3, 'uint16 frame must be colorized to 3-channel, not skipped'
    # Green source -> RGB index 1 -> BGR index 1 after the cv2 swap.
    assert written[0, 0, 1] > 0, 'Green channel must carry the colorized value'
