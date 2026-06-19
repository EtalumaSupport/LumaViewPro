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
# Config-layer derivation: legacy settings -> resolved capture_depth/save_encoding
# ---------------------------------------------------------------------------


def test_config_helper_derives_image_mode_from_settings():
    """The settings config getter exposes the resolved image_mode + its derived
    capture_depth / save_encoding, including for an unmigrated legacy dict."""
    from modules.config_helpers import get_image_capture_config_from_settings

    cfg = get_image_capture_config_from_settings({})  # defaults: 8-bit
    assert cfg['image_mode'] == '8bit'
    assert cfg['capture_depth'] == 8
    assert cfg['save_encoding'] == '8bit'

    cfg = get_image_capture_config_from_settings(
        {'use_full_pixel_depth': True, 'false_color_16bit': True}
    )
    assert cfg['image_mode'] == '12bit_false_color_rgb'
    assert cfg['capture_depth'] == 12
    assert cfg['save_encoding'] == 'rgb'
    # The retired keys are not re-emitted by the getter.
    assert 'use_full_pixel_depth' not in cfg
    assert 'false_color_16bit' not in cfg


# ---------------------------------------------------------------------------
# Capability gate: which modes a camera can offer
# ---------------------------------------------------------------------------


def test_available_modes_8bit_only_camera():
    """A camera without Mono12/Mono12p (LS560/620/720 class) offers 8-bit only."""
    from modules.image_mode import available_mode_labels, available_modes, camera_supports_12bit

    assert camera_supports_12bit(['Mono8']) is False
    assert available_modes(['Mono8']) == ['8bit']
    assert available_mode_labels(['Mono8']) == ['8-bit']
    # Empty / None capability set is treated as 8-bit-only, never as "all".
    assert available_modes([]) == ['8bit']
    assert available_modes(None) == ['8bit']


def test_available_modes_12bit_camera():
    """A Mono12-capable camera offers all four modes in selector order."""
    from modules.image_mode import available_modes, camera_supports_12bit

    assert camera_supports_12bit(['Mono8', 'Mono10', 'Mono12', 'Mono12p']) is True
    assert available_modes(['Mono8', 'Mono12']) == [
        '8bit',
        '12bit_scientific',
        '12bit_scaled',
        '12bit_false_color_rgb',
    ]


def test_image_mode_label_round_trip():
    """Every mode has a label and the label maps back to the mode."""
    from modules.image_mode import IMAGE_MODE_LABELS, LABEL_TO_IMAGE_MODE, resolve_image_mode

    for mode, label in IMAGE_MODE_LABELS.items():
        resolve_image_mode(mode)  # every labeled value is a real mode
        assert LABEL_TO_IMAGE_MODE[label] == mode


def test_resolve_settings_image_mode_prefers_explicit_key():
    """An explicit image_mode key wins over the legacy keys -- the only way to
    reach 12bit_scaled, which no legacy combination maps to."""
    from modules.image_mode import resolve_settings_image_mode

    settings = {
        'image_mode': '12bit_scaled',
        'use_full_pixel_depth': True,
        'false_color_16bit': True,
    }
    assert resolve_settings_image_mode(settings) == '12bit_scaled'


def test_resolve_settings_image_mode_falls_back_to_legacy():
    """No image_mode key -> derive from the legacy keys (old installs)."""
    from modules.image_mode import resolve_settings_image_mode

    assert resolve_settings_image_mode({}) == '8bit'
    assert (
        resolve_settings_image_mode({'use_full_pixel_depth': True, 'false_color_16bit': True})
        == '12bit_false_color_rgb'
    )
    # An unrecognized image_mode value is ignored in favor of the legacy derivation.
    assert resolve_settings_image_mode({'image_mode': 'bogus', 'use_full_pixel_depth': True}) == (
        '12bit_scientific'
    )


def test_config_helper_prefers_image_mode_key():
    """The settings getter honors an explicit image_mode, making 12bit_scaled
    reachable in production."""
    from modules.config_helpers import get_image_capture_config_from_settings

    cfg = get_image_capture_config_from_settings({'image_mode': '12bit_scaled'})
    assert cfg['image_mode'] == '12bit_scaled'
    assert cfg['capture_depth'] == 12
    assert cfg['save_encoding'] == 'msb_aligned'


# ---------------------------------------------------------------------------
# On-load migration: fold legacy keys into image_mode, then drop them
# ---------------------------------------------------------------------------


def test_migrate_settings_dict_folds_legacy_and_drops():
    """An old install's two keys collapse to the right image_mode and the
    legacy keys are removed in place."""
    from modules.image_mode import migrate_settings_dict

    settings = {'use_full_pixel_depth': True, 'false_color_16bit': True, 'other': 1}
    assert migrate_settings_dict(settings) is True
    assert settings['image_mode'] == '12bit_false_color_rgb'
    assert 'use_full_pixel_depth' not in settings
    assert 'false_color_16bit' not in settings
    assert settings['other'] == 1


def test_migrate_settings_dict_keeps_explicit_image_mode():
    """An explicit image_mode wins over stray legacy keys, which still drop --
    the only way 12bit_scaled survives a migration."""
    from modules.image_mode import migrate_settings_dict

    settings = {'image_mode': '12bit_scaled', 'use_full_pixel_depth': True}
    assert migrate_settings_dict(settings) is True
    assert settings['image_mode'] == '12bit_scaled'
    assert 'use_full_pixel_depth' not in settings


def test_migrate_settings_dict_noop_when_already_migrated():
    """A dict with only image_mode and no legacy keys is left untouched."""
    from modules.image_mode import migrate_settings_dict

    settings = {'image_mode': '8bit'}
    assert migrate_settings_dict(settings) is False
    assert settings == {'image_mode': '8bit'}


# ---------------------------------------------------------------------------
# Save encoding: 12-bit scaled (MSB-aligned, lossless, recoverable)
# ---------------------------------------------------------------------------


def test_save_encoding_scaled_is_msb_aligned_and_recoverable(tmp_path):
    """12bit_scaled left-justifies 0..4095 to 0..65520 (x16) so dumb viewers
    render it bright, and the shift is exactly recoverable by >>4. The data now
    fills the 16-bit container and makes no narrower significant-bits claim, so
    our own read-back reports container width (16) and scales it bright -- not
    12, which would mis-scale left-justified data."""
    import modules.image_utils as image_utils
    from modules.image_utils import write_tiff

    out_path = tmp_path / 'scaled.tiff'
    data = np.full((8, 8), 4095, dtype=np.uint16)  # full-scale 12-bit

    write_tiff(
        data=data,
        file_loc=out_path,
        metadata=_metadata(out_path, channel='BF', significant_bits=12),
        ome=False,
        color='BF',
        significant_bits=12,
        save_encoding='msb_aligned',
    )

    arr = tf.imread(str(out_path))
    assert arr.dtype == np.uint16
    assert arr[0, 0] == 65520, 'full-scale 12-bit must left-justify to 65520 (4095 x 16)'
    assert (arr >> 4 == 4095).all(), 'x16 must be exactly recoverable by >>4'
    assert image_utils.read_tiff_significant_bits(out_path) == 16, (
        'left-justified data fills the container; read-back must report 16, not 12'
    )


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
        significant_bits=12,
        save_encoding='right_aligned',
    )

    with tf.TiffFile(str(out_path)) as t:
        arr = t.pages[0].asarray()
    assert arr[0, 0] == 4095, 'right-aligned must store the raw value, not scale it'


# ---------------------------------------------------------------------------
# Save encoding: RGB false color -- fluorescence widens, transmitted stays mono
# ---------------------------------------------------------------------------


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
        significant_bits=16,
        save_encoding='rgb',
    )

    result = tf.imread(str(out_path))
    assert result.shape == (8, 8, 3)
    assert result[0, 0, 2] == 42000 and result[0, 0, 0] == 0 and result[0, 0, 1] == 0


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
        significant_bits=16,
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


# ---------------------------------------------------------------------------
# Video frame TIFF primitive: the write_tiff video_frame branch honors
# save_encoding (regression guards -- already correct, no marker)
# ---------------------------------------------------------------------------


def test_write_tiff_video_frame_rgb_widens_uint16(tmp_path):
    """A 12-bit video frame saved with rgb encoding widens to 3-channel RGB --
    the underlying capability the manual + protocol Frames paths route through."""
    from modules.image_utils import write_tiff

    out_path = tmp_path / 'frame_rgb.tiff'
    data = np.full((8, 8), 42000, dtype=np.uint16)

    write_tiff(
        data=data,
        file_loc=out_path,
        metadata=_metadata(out_path, channel='Green'),
        ome=False,
        color='Green',
        video_frame=True,
        significant_bits=16,
        save_encoding='rgb',
    )

    arr = tf.imread(str(out_path))
    assert arr.shape == (8, 8, 3)
    assert arr[0, 0, 1] == 42000 and arr[0, 0, 0] == 0 and arr[0, 0, 2] == 0


def test_write_tiff_video_frame_8bit_has_no_palette(tmp_path):
    """The video_frame TIFF branch emits NO palette colormap, so an 8-bit
    fluorescence frame saved through it stays mono. This is the constraint that
    forces write_video_frame to bake 8-bit RGB rather than rely on a colormap."""
    from modules.image_utils import write_tiff

    out_path = tmp_path / 'frame_8bit.tiff'
    data = np.full((8, 8), 200, dtype=np.uint8)

    write_tiff(
        data=data,
        file_loc=out_path,
        metadata=_metadata(out_path, channel='Green'),
        ome=False,
        color='Green',
        video_frame=True,
        significant_bits=8,
        save_encoding='8bit',
    )

    arr = tf.imread(str(out_path))
    assert arr.ndim == 2, 'video_frame 8-bit write has no palette -- stays mono'


# ---------------------------------------------------------------------------
# write_video_frame helper -- the single canonical frame-save path (V2 target).
# xfail(strict=True) until the helper lands; the markers flip green at V2.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('save_encoding', 'capture_depth', 'in_dtype', 'fill', 'layer', 'fc_on', 'ndim', 'out_dtype'),
    [
        # 8-bit: fluorescence + false color bakes RGB (no palette on video frames)
        ('8bit', 8, np.uint8, 200, 'Green', True, 3, np.uint8),
        ('8bit', 8, np.uint8, 200, 'Green', False, 2, np.uint8),
        ('8bit', 8, np.uint8, 200, 'BF', False, 2, np.uint8),
        # 12-bit scientific + scaled are mono modes regardless of the layer toggle
        ('right_aligned', 12, np.uint16, 4095, 'Green', True, 2, np.uint16),
        ('right_aligned', 12, np.uint16, 4095, 'Green', False, 2, np.uint16),
        ('msb_aligned', 12, np.uint16, 4095, 'Green', True, 2, np.uint16),
        # 12-bit RGB: colorize fluorescence when the layer toggle is on;
        # mono when off (per-layer gate) or transmitted
        ('rgb', 12, np.uint16, 42000, 'Green', True, 3, np.uint16),
        ('rgb', 12, np.uint16, 42000, 'Green', False, 2, np.uint16),
        ('rgb', 12, np.uint16, 42000, 'BF', True, 2, np.uint16),
    ],
)
def test_write_video_frame_matrix(
    tmp_path, save_encoding, capture_depth, in_dtype, fill, layer, fc_on, ndim, out_dtype
):
    """One canonical helper produces the right shape + dtype for every
    (image_mode, per-layer false color, depth) combination."""
    from modules.image_save import write_video_frame

    out_path = tmp_path / 'frame.tiff'
    data = np.full((8, 8), fill, dtype=in_dtype)
    write_video_frame(
        frame=data,
        file_loc=out_path,
        metadata=_metadata(out_path, channel=layer),
        layer_color=layer,
        false_color_on=fc_on,
        save_encoding=save_encoding,
        capture_depth=capture_depth,
    )
    arr = tf.imread(str(out_path))
    assert arr.ndim == ndim
    assert arr.dtype == out_dtype


def test_write_video_frame_12bit_falsecolor_off_stays_mono(tmp_path):
    """The headline manual-record fix: in the RGB image mode, a layer whose
    false-color toggle is OFF saves mono uint16 -- the mode never force-colorizes
    a colorless choice."""
    from modules.image_save import write_video_frame

    out_path = tmp_path / 'off.tiff'
    data = np.full((8, 8), 42000, dtype=np.uint16)
    write_video_frame(
        frame=data,
        file_loc=out_path,
        metadata=_metadata(out_path, channel='Green'),
        layer_color='Green',
        false_color_on=False,
        save_encoding='rgb',
        capture_depth=12,
    )
    arr = tf.imread(str(out_path))
    assert arr.ndim == 2, 'false-color-off layer must stay mono even in RGB mode'


def test_write_video_frame_8bit_falsecolor_bakes_rgb(tmp_path):
    """8-bit fluorescence with false color on bakes 3-channel RGB (the
    video_frame TIFF write has no palette to lean on)."""
    from modules.image_save import write_video_frame

    out_path = tmp_path / 'baked.tiff'
    data = np.full((8, 8), 200, dtype=np.uint8)
    write_video_frame(
        frame=data,
        file_loc=out_path,
        metadata=_metadata(out_path, channel='Green'),
        layer_color='Green',
        false_color_on=True,
        save_encoding='8bit',
        capture_depth=8,
    )
    arr = tf.imread(str(out_path))
    assert arr.shape == (8, 8, 3)
    assert arr[0, 0, 1] == 200 and arr[0, 0, 0] == 0 and arr[0, 0, 2] == 0


def test_write_video_frame_stamps_significant_bits_for_scaled(tmp_path):
    """The helper stamps significant_bits from capture_depth, so msb_aligned
    actually left-justifies even when the caller's metadata omits it -- without
    the stamp, write_tiff would treat the payload as 16-bit and not scale."""
    import modules.image_utils as image_utils
    from modules.image_save import write_video_frame

    out_path = tmp_path / 'scaled_frame.tiff'
    data = np.full((8, 8), 4095, dtype=np.uint16)  # no significant_bits in metadata
    write_video_frame(
        frame=data,
        file_loc=out_path,
        metadata=_metadata(out_path, channel='BF'),
        layer_color='BF',
        false_color_on=False,
        save_encoding='msb_aligned',
        capture_depth=12,
    )
    arr = tf.imread(str(out_path))
    assert arr[0, 0] == 65520, 'capture_depth=12 must drive the x16 left-justify'
    assert (arr >> 4 == 4095).all(), 'x16 must be exactly recoverable'
    assert image_utils.read_tiff_significant_bits(out_path) == 16
