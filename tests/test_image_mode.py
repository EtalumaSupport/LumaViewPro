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

import pathlib

import numpy as np
import pytest
import tifffile as tf
from modules.layer_record import release_color
from modules.layer_record import release_colors
from tests.frame_records import recorded_record, write_recorded
from tests.stock_channels import LED0, LED1


def _metadata(path, channel=LED0, significant_bits=None):
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


@pytest.mark.parametrize(
    ('mode', 'depth', 'encoding'),
    [
        ('8bit', 8, '8bit'),
        ('12bit_false_color_rgb', 12, 'rgb'),
        ('12bit_scaled', 12, 'msb_aligned'),
    ],
)
def test_config_helper_reads_the_stored_image_mode(mode, depth, encoding):
    """The settings config getter exposes the stored image_mode and its derived
    capture_depth / save_encoding; 12bit_scaled is reachable only this way."""
    from modules.config_helpers import get_image_capture_config_from_settings
    from tests.settings_fixtures import complete_settings

    cfg = get_image_capture_config_from_settings(complete_settings(image_mode=mode))
    assert cfg.image_mode == mode
    assert cfg.capture_depth == depth
    assert cfg.save_encoding == encoding
    # The retired keys are not re-emitted by the getter.
    assert not hasattr(cfg, 'use_full_pixel_depth')
    assert not hasattr(cfg, 'false_color_16bit')


# ---------------------------------------------------------------------------
# The modes offered: every one, on every camera
# ---------------------------------------------------------------------------


def test_available_modes_are_every_mode_in_selector_order():
    """A mode is a save policy every camera honours, so all four are offered."""
    from modules.image_mode import available_modes

    assert available_modes() == [
        '8bit',
        '12bit_scientific',
        '12bit_scaled',
        '12bit_false_color_rgb',
    ]


def test_select_capture_pixel_format_ids_no_mono8():
    """An IDS sensor without a native Mono8 resolves 8-bit to its lowest
    native (Mono10) and 12-bit to its Mono12 native -- never a bare Mono8
    the sensor would reject."""
    from modules.image_mode import select_capture_pixel_format

    ids = ('Mono10g40IDS', 'Mono12g24IDS')
    assert select_capture_pixel_format(8, ids) == 'Mono10g40IDS'
    assert select_capture_pixel_format(12, ids) == 'Mono12g24IDS'


def test_select_capture_pixel_format_native_mono8():
    """A camera with a native Mono8 (Pylon / FX2) uses it for 8-bit and a
    true Mono12 for 12-bit."""
    from modules.image_mode import select_capture_pixel_format

    pylon = ('Mono8', 'Mono10', 'Mono12', 'Mono12p')
    assert select_capture_pixel_format(8, pylon) == 'Mono8'
    assert select_capture_pixel_format(12, pylon) == 'Mono12'
    # 8-bit-only sensor (FX2/MT9P031): both depths land on Mono8.
    assert select_capture_pixel_format(8, ('Mono8',)) == 'Mono8'
    assert select_capture_pixel_format(12, ('Mono8',)) == 'Mono8'


def test_select_capture_pixel_format_no_formats():
    """No reported formats -> None (caller skips the push)."""
    from modules.image_mode import select_capture_pixel_format

    assert select_capture_pixel_format(8, ()) is None
    assert select_capture_pixel_format(12, None) is None


@pytest.mark.parametrize(
    'feature_count, mode, expected',
    [
        # Active feature (count > 1) in an 8-bit mode -> warn.
        (4, '8bit', True),
        (2, '8bit', True),
        # Inactive feature -> never warn, regardless of mode.
        (1, '8bit', False),
        (None, '8bit', False),
        (0, '8bit', False),
        # Active feature but a 12-bit mode keeps the range -> no warning.
        (4, '12bit_scientific', False),
        (4, '12bit_scaled', False),
        (4, '12bit_false_color_rgb', False),
    ],
)
def test_depth_truncation_warning_active(feature_count, mode, expected):
    """The summing/binning depth-loss hint fires only when an accumulating
    feature is active AND the mode saves 8-bit (the accumulated range is
    downconverted on save)."""
    from modules.image_mode import depth_truncation_warning_active

    assert depth_truncation_warning_active(feature_count, mode) is expected


def test_migrate_settings_dict_warns_on_a_corrupt_value_beside_the_legacy_keys():
    """A present-but-unrecognized stored image_mode beside the legacy keys is a
    corrupt setting: the migration derives the mode from the legacy keys AND
    surfaces a warning so the derivation does not hide the data loss. A file
    with no image_mode at all (a pre-consolidation install) is a normal
    migration and derives silently."""
    from unittest import mock

    import modules.image_mode as image_mode

    with mock.patch.object(image_mode, 'logger') as mock_logger:
        settings = {'image_mode': 'bogus', 'use_full_pixel_depth': True}
        assert image_mode.migrate_settings_dict(settings) is True
        assert settings['image_mode'] == '12bit_scientific'
        assert mock_logger.warning.call_count == 1

    with mock.patch.object(image_mode, 'logger') as mock_logger:
        settings = {}
        assert image_mode.migrate_settings_dict(settings) is True
        assert settings['image_mode'] == '8bit'
        mock_logger.warning.assert_not_called()


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


@pytest.mark.parametrize('stored', ['bogus', None])
def test_an_unusable_mode_beside_the_legacy_keys_takes_the_choice_they_hold(stored):
    """The legacy keys are the person's choice; an unusable image_mode beside
    them is derived from them, not left for the load to reset to the shipped one."""
    from modules.image_mode import migrate_settings_dict

    settings = {'image_mode': stored, 'use_full_pixel_depth': True}
    assert migrate_settings_dict(settings) is True
    assert settings == {'image_mode': '12bit_scientific'}


def test_an_unknown_mode_with_no_legacy_keys_is_left_for_the_load_to_replace():
    """With nothing to derive a choice from, the migration leaves the value to
    the writer's rule, which replaces and reports it after the merge."""
    from modules.image_mode import migrate_settings_dict

    settings = {'image_mode': 'bogus'}
    assert migrate_settings_dict(settings) is False
    assert settings == {'image_mode': 'bogus'}


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
        color=release_color('BF'),
        significant_bits=12,
        save_encoding='msb_aligned',
        white_bits=None,
    )

    arr = tf.imread(str(out_path))
    assert arr.dtype == np.uint16
    assert arr[0, 0] == 65520, 'full-scale 12-bit must left-justify to 65520 (4095 x 16)'
    assert (arr >> 4 == 4095).all(), 'x16 must be exactly recoverable by >>4'
    assert image_utils.read_tiff_significant_bits(out_path) == 16, (
        'left-justified data fills the container; read-back must report 16, not 12'
    )


def test_record_frame_must_be_right_aligned_for_video_frame(sim_scope, tmp_path):
    """Real-path lock for the manual-record Frames path: whatever the record
    path hands to image_save.write_recorded_frame -- the SOLE depth encoder --
    must be the camera's right-aligned payload (0..4095 for 12-bit), not a
    left-justified one. A right-aligned frame round-trips to its true value;
    the left-justified frame the record path produced before the depth fix
    does NOT -- the save edge stamps significant_bits=12 over already-shifted
    pixels, so a reader recovers ~16x the value. The earlier matrix missed
    this by feeding the helper right-aligned input the old production path
    never delivered.

    The alignment contract is a property of what reaches the encoder, not of
    the carrier that delivers it: this named the memmap when a memmap fed the
    encoder, and holds unchanged now that the recording engine does."""
    import modules.image_utils as image_utils

    raw = np.zeros((8, 8), dtype=np.uint16)
    raw[0, 0] = 4095  # full-scale, right-aligned 12-bit
    raw[1, 1] = 2048

    # Post-fix record-path content: the camera payload, verbatim.
    correct = tmp_path / 'right_aligned.tiff'
    write_recorded(sim_scope, raw.copy(), correct, save_encoding='right_aligned')
    arr = tf.imread(str(correct))
    assert arr[0, 0] == 4095, 'right-aligned frame must round-trip the raw 12-bit value'
    assert arr[1, 1] == 2048
    assert image_utils.read_tiff_significant_bits(correct) == 12

    # Pre-fix record-path content: left-justified by 4 bits (12-bit payload shifted
    # to fill the 16-bit container). The save edge tags significant_bits=12 over
    # already-shifted pixels, so the recovered value is corrupted (65520, not
    # 4095). Locks WHY the frame must stay right-aligned.
    left_justified = raw << 4
    corrupt = tmp_path / 'left_justified.tiff'
    write_recorded(sim_scope, left_justified, corrupt, save_encoding='right_aligned')
    arr_bug = tf.imread(str(corrupt))
    assert arr_bug[0, 0] == 65520, 'left-justified frame corrupts the saved value (4095 -> 65520)'
    assert arr_bug[0, 0] != 4095


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
        color=release_color('BF'),
        significant_bits=12,
        save_encoding='right_aligned',
        white_bits=None,
    )

    with tf.TiffFile(str(out_path)) as t:
        arr = t.pages[0].asarray()
    assert arr[0, 0] == 4095, 'right-aligned must store the raw value, not scale it'


# ---------------------------------------------------------------------------
# Save encoding: RGB false color -- fluorescence widens, transmitted stays mono
# ---------------------------------------------------------------------------


def test_save_encoding_rgb_widens_fluorescence(tmp_path):
    """rgb encoding bakes the layer color into 3-channel RGB for a
    fluorescence layer. LED0 (blue) lands at index 2, value preserved, others zero."""
    from modules.image_utils import write_tiff

    out_path = tmp_path / 'blue_rgb.tiff'
    data = np.full((8, 8), 42000, dtype=np.uint16)

    write_tiff(
        data=data,
        file_loc=out_path,
        metadata=_metadata(out_path, channel=LED0),
        ome=False,
        color=release_color(LED0),
        significant_bits=16,
        save_encoding='rgb',
        white_bits=None,
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
        color=release_color('BF'),
        significant_bits=16,
        save_encoding='rgb',
        white_bits=None,
    )

    result = tf.imread(str(out_path))
    assert result.ndim == 2, 'BF must stay mono even under rgb encoding (no color to bake)'


def test_mp4_writers_pass_layer_color():
    """Both MP4 writers pass color= to VideoWriter. The manual-record path and
    the queue (protocol video) now carry MONO frames -- the inline RGB bakes were
    removed in favor of one colorization site per output -- so without color= at
    the encoder every false-colored MP4 would silently gray-encode. Both call
    sites need a live scope to exercise end to end, so assert on source."""
    repo = pathlib.Path(__file__).resolve().parent.parent
    # Both MP4 writers live in the recording controllers (manual, protocol
    # video step).
    manual_src = (repo / 'modules' / 'manual_recording.py').read_text()
    cap_src = (repo / 'modules' / 'protocol_recording.py').read_text()
    # Both colorize from the recorded layer's false-color signal, in the
    # colours the recording began with, and both encode grey when false
    # color is off -- one contract, one resolver. Whitespace is dropped, so a
    # reflow of the call does not read as a change.
    manual_flat = ''.join(manual_src.split())
    cap_flat = ''.join(cap_src.split())
    assert (
        'color=layer_record.shown_color(resolved_layer,false_color_on,plan.channel_colors)'
        in manual_flat
    )
    assert (
        "color=layer_record.shown_color(step['Color'],false_color_on,self._channel_colors)"
        in cap_flat
    )
    # The inline protocol-capture bake stays gone (one colorization authority).
    assert 'add_false_color' not in cap_src


# ---------------------------------------------------------------------------
# Video frame TIFF primitive: the write_tiff video_frame branch honors
# save_encoding (regression guards -- already correct, no marker)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# write_recorded_frame -- a recorded frame is written as a still is.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('save_encoding', 'in_dtype', 'fill', 'layer', 'fc_on', 'ndim', 'out_dtype'),
    [
        # 8-bit: a fluorescence frame with false color on keeps one channel and
        # carries its colour as a palette, as a still does
        ('8bit', np.uint8, 200, LED1, True, 2, np.uint8),
        ('8bit', np.uint8, 200, LED1, False, 2, np.uint8),
        ('8bit', np.uint8, 200, 'BF', False, 2, np.uint8),
        # 12-bit scientific + scaled are mono modes regardless of the layer toggle
        ('right_aligned', np.uint16, 4095, LED1, True, 2, np.uint16),
        ('right_aligned', np.uint16, 4095, LED1, False, 2, np.uint16),
        ('msb_aligned', np.uint16, 4095, LED1, True, 2, np.uint16),
        # 12-bit RGB: colorize fluorescence when the layer toggle is on;
        # mono when off (per-layer gate) or transmitted
        ('rgb', np.uint16, 42000, LED1, True, 3, np.uint16),
        ('rgb', np.uint16, 42000, LED1, False, 2, np.uint16),
        ('rgb', np.uint16, 42000, 'BF', True, 2, np.uint16),
    ],
)
def test_write_recorded_frame_matrix(
    sim_scope, tmp_path, save_encoding, in_dtype, fill, layer, fc_on, ndim, out_dtype
):
    """One canonical writer produces the right shape + dtype for every
    (image_mode, per-layer false color, depth) combination."""
    out_path = tmp_path / 'frame.tiff'
    data = np.full((8, 8), fill, dtype=in_dtype)
    write_recorded(
        sim_scope, data, out_path, channel=layer, false_color_on=fc_on, save_encoding=save_encoding
    )
    arr = tf.imread(str(out_path))
    assert arr.ndim == ndim
    assert arr.dtype == out_dtype


def test_write_recorded_frame_12bit_falsecolor_off_stays_mono(sim_scope, tmp_path):
    """The headline manual-record fix: in the RGB image mode, a layer whose
    false-color toggle is OFF saves mono uint16 -- the mode never force-colorizes
    a colorless choice."""
    out_path = tmp_path / 'off.tiff'
    data = np.full((8, 8), 42000, dtype=np.uint16)
    write_recorded(sim_scope, data, out_path, channel=LED1, save_encoding='rgb')
    arr = tf.imread(str(out_path))
    assert arr.ndim == 2, 'false-color-off layer must stay mono even in RGB mode'


def test_an_8bit_false_color_recorded_frame_is_a_palette_like_a_still(sim_scope, tmp_path):
    """8-bit fluorescence with false color on keeps its one channel of
    pixels and is coloured by a palette, as the still beside it is: the
    colour is how it is shown, never baked into what was measured."""
    out_path = tmp_path / 'palette.tiff'
    data = np.full((8, 8), 200, dtype=np.uint8)
    write_recorded(
        sim_scope, data, out_path, channel=LED1, false_color_on=True, save_encoding='8bit'
    )
    with tf.TiffFile(str(out_path)) as tif:
        page = tif.pages[0]
        assert page.photometric == tf.PHOTOMETRIC.PALETTE
        assert page.colormap is not None
        assert page.asarray()[0, 0] == 200


def test_write_recorded_frame_takes_its_depth_from_its_record(sim_scope, tmp_path):
    """The writer stamps the depth the frame was delivered at, from its
    record, so msb_aligned left-justifies a 12-bit frame -- without the
    depth, write_tiff would treat the payload as 16-bit and not scale."""
    import modules.image_utils as image_utils

    out_path = tmp_path / 'scaled_frame.tiff'
    data = np.full((8, 8), 4095, dtype=np.uint16)
    write_recorded(
        sim_scope,
        data,
        out_path,
        save_encoding='msb_aligned',
        record=recorded_record(frame_significant_bits=12),
    )
    arr = tf.imread(str(out_path))
    assert arr[0, 0] == 65520, 'a 12-bit frame must drive the x16 left-justify'
    assert (arr >> 4 == 4095).all(), 'x16 must be exactly recoverable'
    assert image_utils.read_tiff_significant_bits(out_path) == 16


def test_write_recorded_frame_rejects_unknown_save_encoding(sim_scope, tmp_path):
    """A typo'd save_encoding fails loud at the write boundary instead of falling
    through to a plain mono write -- the silent-wrong-file the unvalidated entry
    point allowed."""
    from modules.exceptions import CaptureError

    out_path = tmp_path / 'bad_encoding.tiff'
    data = np.full((8, 8), 4095, dtype=np.uint16)
    with pytest.raises(CaptureError, match='save encoding'):
        write_recorded(sim_scope, data, out_path, save_encoding='not_a_real_encoding')
    assert not out_path.exists(), 'a rejected encoding must not produce a file'


# ---------------------------------------------------------------------------
# save_encoding is a required, SSOT-derived value -- not an omittable kwarg.
#
# The runtime mirror of significant_bits: a save that does not state how its
# pixels land on disk silently right-aligns (dark) a scaled mode. These pin
# the coupling that makes an encoding-less save unrepresentable, the live-path
# counterpart to load_pixels returning (pixels, depth) as one pair.
# ---------------------------------------------------------------------------


def test_write_tiff_requires_save_encoding(tmp_path):
    """write_tiff cannot be called without save_encoding -- the depth-less-write
    guard, now extended to encoding. A defaulted None silently right-aligned a
    scaled payload."""
    from modules.image_utils import write_tiff

    out_path = tmp_path / 'no_encoding.tiff'
    data = np.full((4, 4), 4095, dtype=np.uint16)
    with pytest.raises(TypeError, match='save_encoding'):
        write_tiff(
            data=data,
            file_loc=out_path,
            metadata=_metadata(out_path, channel='BF', significant_bits=12),
            ome=False,
            color=release_color('BF'),
            significant_bits=12,
            white_bits=None,
        )


def test_save_image_requires_save_encoding():
    """save_image cannot be called without save_encoding -- binding fails before
    any work, so no live-capture call site can drop the image mode."""
    from modules.image_save import save_image

    with pytest.raises(TypeError, match='save_encoding'):
        save_image(
            None,
            array=np.zeros((4, 4), dtype=np.uint8),
            channel_colors=release_colors(),
            white_bits=None,
            fallback_focus=None,
        )


@pytest.mark.parametrize(
    ('dtype', 'expected'),
    [
        (np.uint8, '8bit'),
        (np.uint16, 'right_aligned'),
    ],
)
def test_encoding_for_array_derives_from_dtype(dtype, expected):
    """A derived data product (projection / stitch / composite) preserves its
    inputs verbatim: 8-bit saves 8bit, a right-aligned uint16 payload saves
    right_aligned. One canonical helper so no post-processor inlines the check."""
    from modules.image_mode import encoding_for_array

    assert encoding_for_array(np.zeros((2, 2), dtype=dtype)) == expected


def test_post_processor_uint16_save_is_verbatim(tmp_path):
    """A post-processor re-save through the derived encoding keeps the raw
    right-aligned value (no MSB shift) -- the helper preserves bytes, matching
    the verbatim pixels load_pixels hands back."""
    from modules.image_mode import encoding_for_array
    from modules.image_utils import write_tiff

    out_path = tmp_path / 'proj.tiff'
    data = np.full((8, 8), 4095, dtype=np.uint16)
    write_tiff(
        data=data,
        file_loc=out_path,
        metadata=_metadata(out_path, channel='BF', significant_bits=12),
        ome=False,
        color=release_color('BF'),
        significant_bits=12,
        save_encoding=encoding_for_array(data),
        white_bits=None,
    )
    arr = tf.imread(str(out_path))
    assert arr[0, 0] == 4095, 'derived uint16 output must stay right-aligned (verbatim)'


@pytest.mark.parametrize(
    ('mode', 'dtype', 'expected'),
    [
        # The RGB false-color mode widens a derived fluorescence product, the
        # same as a freshly captured frame -- a stitch/zproject is not silently
        # demoted to mono.
        ('12bit_false_color_rgb', np.uint16, 'rgb'),
        # The quantitative modes keep the derived product verbatim: scaled does
        # NOT msb-align a derived output (matches the prior behavior), and
        # scientific stays right-aligned.
        ('12bit_scaled', np.uint16, 'right_aligned'),
        ('12bit_scientific', np.uint16, 'right_aligned'),
        ('8bit', np.uint8, '8bit'),
    ],
)
def test_save_encoding_for_derived_output_honors_rgb_mode(mode, dtype, expected):
    """A derived data product preserves its pixels verbatim EXCEPT under the RGB
    mode, where a fluorescence product widens to false color like a capture."""
    from modules.image_mode import save_encoding_for_derived_output

    arr = np.zeros((2, 2), dtype=dtype)
    assert save_encoding_for_derived_output(arr, mode) == expected


def test_resolve_output_save_encoding_reads_image_mode_ssot():
    """The derived-output encoding comes from the live image_mode SSOT, so a
    stitch/zproject honors the user's false-color choice -- the behavior that
    used to hide in maybe_apply_false_color's settings read."""
    from unittest import mock

    import modules.derived_output_encoding as derived_output_encoding

    arr = np.zeros((4, 4), dtype=np.uint16)

    rgb_ctx = mock.MagicMock()
    rgb_ctx.settings = {'image_mode': '12bit_false_color_rgb'}
    with mock.patch('modules.app_context.ctx', rgb_ctx):
        assert derived_output_encoding.resolve_output_save_encoding(arr) == 'rgb'

    sci_ctx = mock.MagicMock()
    sci_ctx.settings = {'image_mode': '12bit_scientific'}
    with mock.patch('modules.app_context.ctx', sci_ctx):
        assert derived_output_encoding.resolve_output_save_encoding(arr) == 'right_aligned'


def test_resolve_output_save_encoding_degrades_without_context():
    """With no app context (headless / pre-init, the post-processor test path),
    the encoding falls back to the verbatim dtype-based value instead of crashing
    on a None settings lock -- there is no user image_mode to honor."""
    from unittest import mock

    import modules.derived_output_encoding as derived_output_encoding

    with mock.patch('modules.app_context.ctx', None):
        assert (
            derived_output_encoding.resolve_output_save_encoding(np.zeros((4, 4), dtype=np.uint8))
            == '8bit'
        )
        assert (
            derived_output_encoding.resolve_output_save_encoding(np.zeros((4, 4), dtype=np.uint16))
            == 'right_aligned'
        )


def test_derived_fluorescence_widens_to_rgb_under_false_color_mode(tmp_path):
    """Regression: a uint16 fluorescence stitch/zproject output saved through the
    derived-output encoding widens to 3-channel RGB when the user is in the RGB
    mode -- it must not silently save mono (the grayscale-in-Windows-Preview
    regression)."""
    from unittest import mock

    import modules.derived_output_encoding as derived_output_encoding
    import modules.image_utils as image_utils

    out_path = tmp_path / 'derived_blue.tiff'
    data = np.full((8, 8), 3000, dtype=np.uint16)

    rgb_ctx = mock.MagicMock()
    rgb_ctx.settings = {'image_mode': '12bit_false_color_rgb'}
    with mock.patch('modules.app_context.ctx', rgb_ctx):
        encoding = derived_output_encoding.resolve_output_save_encoding(data)
        image_utils.write_tiff(
            data=data,
            file_loc=out_path,
            metadata=_metadata(out_path, channel=LED0),
            ome=False,
            color=release_color(LED0),
            significant_bits=16,
            save_encoding=encoding,
            white_bits=None,
        )

    arr = tf.imread(str(out_path))
    assert arr.shape == (8, 8, 3), 'derived fluorescence under RGB mode must widen to 3-channel'
    assert arr[0, 0, 2] == 3000 and arr[0, 0, 0] == 0 and arr[0, 0, 1] == 0


def test_manual_capture_path_passes_save_encoding():
    """Structural lock: every save_image call in the manual capture path
    (modules/manual_capture.py, which the Capture button calls) forwards both the image mode and the acquiring channel.

    A refactor that drops save_encoding from the non-engineering branch was the
    original defect. Channel is locked the same way and for the same reason:
    omitting it is what stamped every manual capture brightfield, and a default
    answering for it makes the omission invisible at the call site."""
    import ast

    src = pathlib.Path(__file__).resolve().parents[1] / 'modules' / 'manual_capture.py'
    tree = ast.parse(src.read_text())

    save_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == 'save_image'
    ]
    assert save_calls, 'expected save_image calls in the manual capture'
    for call in save_calls:
        assert any(kw.arg == 'save_encoding' for kw in call.keywords), (
            f'a {call.func.id} call at line {call.lineno} omits save_encoding -- '
            'the manual live-capture path must forward the image mode'
        )
        assert any(kw.arg == 'channel' for kw in call.keywords), (
            f'a {call.func.id} call at line {call.lineno} omits channel -- '
            'the manual live-capture path must state what it imaged'
        )
        assert not any(kw.arg in ('color', 'true_color') for kw in call.keywords), (
            f'a {call.func.id} call at line {call.lineno} still passes a '
            'retired color argument alongside the channel'
        )
